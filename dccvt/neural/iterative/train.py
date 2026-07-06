"""Training CLI for iterative learned sparse refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from dccvt.neural.data.datasets import resolve_cache_files
from dccvt.neural.iterative.checkpointing import save_checkpoint, _assert_finite_model_parameters
from dccvt.neural.iterative.config import HybridIterRefineConfig, VALID_INITIALIZATION_MODES, load_iter_refine_config
from dccvt.neural.iterative.data import HybridIterRefineDataset, _initialization_from_batch
from dccvt.neural.iterative.model import DCCVTHybridIterRefineNet
from dccvt.neural.losses import hybrid_direct_mesh_loss
from dccvt.neural.utils import device_from_value, parse_mesh_ids, seed_everything, seed_worker

def save_resolved_config(path: Path, *, config: HybridIterRefineConfig, args: argparse.Namespace) -> None:
    """Save resolved model config and command-line arguments."""
    payload = {
        "model_config": config.to_dict(),
        "seed": int(args.seed),
        "args": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _apply_model_overrides(config: HybridIterRefineConfig, args: argparse.Namespace) -> HybridIterRefineConfig:
    values = config.to_dict()
    feature_dim_override = getattr(args, "feature_dim", None) if hasattr(args, "feature_dim") else None
    if feature_dim_override is not None and values.get("graph_hidden_dim") == config.feature_dim:
        values["graph_hidden_dim"] = feature_dim_override
    for arg_name, key in (
        ("initialization_mode", "initialization_mode"),
        ("hotspot_grid_n", "hotspot_grid_n"),
        ("base_grid_n", "base_grid_n"),
        ("feature_dim", "feature_dim"),
        ("encoder_layers", "encoder_layers"),
        ("decoder_layers", "decoder_layers"),
        ("slots_per_parent", "slots_per_parent"),
        ("max_parents_per_round", "max_parents_per_round"),
        ("num_refinement_rounds", "num_refinement_rounds"),
        ("child_offset_scale", "child_offset_scale"),
        ("sdf_residual_scale", "sdf_residual_scale"),
        ("graph_layers", "graph_layers"),
        ("graph_hidden_dim", "graph_hidden_dim"),
    ):
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                values[key] = value
    return HybridIterRefineConfig.from_dict(values)


def _resolve_resume_config(
    requested_config: HybridIterRefineConfig,
    resume_checkpoint: Optional[dict[str, Any]],
) -> HybridIterRefineConfig:
    if resume_checkpoint is None or not resume_checkpoint.get("model_config"):
        return requested_config
    checkpoint_config = HybridIterRefineConfig.from_dict(resume_checkpoint["model_config"])
    if checkpoint_config.initialization_mode != requested_config.initialization_mode:
        raise ValueError(
            "Cannot resume with a different initialization mode: "
            f"checkpoint={checkpoint_config.initialization_mode}, requested={requested_config.initialization_mode}"
        )
    if checkpoint_config.base_grid_n != requested_config.base_grid_n:
        raise ValueError(
            "Cannot resume with a different base grid: "
            f"checkpoint={checkpoint_config.base_grid_n}, requested={requested_config.base_grid_n}"
        )
    if checkpoint_config.surface_pair_count != requested_config.surface_pair_count:
        raise ValueError(
            "Cannot resume with a different near-surface pair count: "
            f"checkpoint={checkpoint_config.surface_pair_count}, requested={requested_config.surface_pair_count}"
        )
    return checkpoint_config


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train iterative learned sparse refinement with mesh loss only.")
    parser.add_argument("--config", default="configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p128.json")
    parser.add_argument("--cache-root", default="outputs/neural_hotspot_sdf/thingi32_g33")
    parser.add_argument("--local-udf-root", default=None)
    parser.add_argument("--allow-missing-local-features", action="store_true")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None)
    parser.add_argument("--checkpoint-dir", default="outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r1_p128/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-subsample", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6.4e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--w-mesh-chamfer", type=float, default=1000.0)
    parser.add_argument("--w-mesh-cvt", type=float, default=100.0)
    parser.add_argument("--w-mesh-sdfsmooth", type=float, default=100.0)
    parser.add_argument("--strict-mesh-loss", action="store_true")
    parser.add_argument("--strict-initialization", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--initialization-mode", choices=sorted(VALID_INITIALIZATION_MODES), default=None)
    parser.add_argument("--hotspot-grid-n", type=int, default=None)
    parser.add_argument("--base-grid-n", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--slots-per-parent", type=int, default=None)
    parser.add_argument("--max-parents-per-round", type=int, default=None)
    parser.add_argument("--num-refinement-rounds", type=int, default=None)
    parser.add_argument("--child-offset-scale", type=float, default=None)
    parser.add_argument("--sdf-residual-scale", type=float, default=None)
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--graph-hidden-dim", type=int, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_train_arg_parser().parse_args(argv)
    if args.batch_size != 1:
        raise ValueError("Iterative refinement training currently requires --batch-size 1")
    seed_everything(args.seed)
    device = device_from_value(args.device)

    requested_config = _apply_model_overrides(load_iter_refine_config(args.config), args)
    resume_checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    config = _resolve_resume_config(requested_config, resume_checkpoint)
    model = DCCVTHybridIterRefineNet(config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])

    cache_files = resolve_cache_files(
        args.cache_root,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        split_file=args.split_file,
    )
    dataset = HybridIterRefineDataset(
        cache_files,
        config=config,
        target_subsample=args.target_subsample,
        local_udf_root=args.local_udf_root,
        allow_missing_local_features=args.allow_missing_local_features,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(checkpoint_dir / "resolved_config.json", config=config, args=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if resume_checkpoint is not None:
        if args.resume_optimizer and "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    stop_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, stop_epoch):
        local_epoch = epoch - start_epoch + 1
        model.train()
        epoch_loss = 0.0
        epoch_stats: dict[str, float] = {}
        for batch in dataloader:
            input_grid = batch["input_grid"].to(device, non_blocking=True)
            sdf_grid = batch["sdf_grid"].to(device, non_blocking=True)
            target_points = batch["target_points"].to(device, non_blocking=True)
            local_target_points = batch["local_target_points"].to(device, non_blocking=True)
            local_udf_grid = None
            if config.local_udf_samples:
                local_udf_grid = batch["local_udf_grid"].to(device, non_blocking=True)
            initial_field = _initialization_from_batch(batch, device, input_grid.dtype)
            if not initial_field["valid"]:
                reason = initial_field["reason"]
                if args.strict_initialization:
                    raise RuntimeError(f"Invalid near-surface initialization: {reason}")
                epoch_stats["initialization_skipped_shapes"] = (
                    epoch_stats.get("initialization_skipped_shapes", 0.0) + 1.0
                )
                reason_key = f"initialization_skip_{reason}"
                epoch_stats[reason_key] = epoch_stats.get(reason_key, 0.0) + 1.0
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_grid,
                sdf_grid,
                initial_field=initial_field,
                target_points=local_target_points if config.local_knn_features else target_points,
                local_udf_grid=local_udf_grid,
            )
            loss, stats = hybrid_direct_mesh_loss(
                outputs,
                target_points,
                chamfer_weight=args.w_mesh_chamfer,
                cvt_weight=args.w_mesh_cvt,
                sdfsmooth_weight=args.w_mesh_sdfsmooth,
                strict=args.strict_mesh_loss,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {stats}")
            if loss.requires_grad:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}: {stats}")
                stats["gradient_norm"] = float(gradient_norm.detach().cpu())
                optimizer.step()
                _assert_finite_model_parameters(model)
            else:
                stats["mesh_no_grad_batch"] = 1.0

            epoch_loss += float(loss.detach().cpu())
            for key, value in stats.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value)
            epoch_stats["site_count"] = epoch_stats.get("site_count", 0.0) + float(outputs["sites"].shape[1])
            epoch_stats["local_udf_grid_n"] = epoch_stats.get("local_udf_grid_n", 0.0) + float(
                config.local_udf_grid_n if config.local_udf_samples else 0
            )
            epoch_stats["local_knn_features"] = epoch_stats.get("local_knn_features", 0.0) + float(
                config.local_knn_features
            )
            if config.local_knn_features:
                epoch_stats["local_target_point_count"] = epoch_stats.get("local_target_point_count", 0.0) + float(
                    local_target_points.shape[1]
                )
            if config.local_udf_samples:
                local_udf_valid = batch["local_udf_valid"]
                epoch_stats["local_udf_valid"] = epoch_stats.get("local_udf_valid", 0.0) + float(
                    local_udf_valid.reshape(-1)[0].item()
                )
            for round_index, round_data in enumerate(outputs["rounds"]):
                prefix = f"round_{round_index:02d}"
                epoch_stats[f"{prefix}_parent_count"] = epoch_stats.get(f"{prefix}_parent_count", 0.0) + float(
                    round_data["parent_indices"].shape[0]
                )
                epoch_stats[f"{prefix}_spawned_site_count"] = epoch_stats.get(
                    f"{prefix}_spawned_site_count", 0.0
                ) + float(round_data["spawned_sites"].shape[0])
                epoch_stats[f"{prefix}_rejected_spawn_count"] = epoch_stats.get(
                    f"{prefix}_rejected_spawn_count", 0.0
                ) + float(round_data["rejected_spawn_count"].item())
            initialization_diagnostics = outputs["initialization_diagnostics"]
            for key in (
                "initial_site_count",
                "surface_anchor_count",
                "positive_sdf_count",
                "negative_sdf_count",
                "minimum_site_distance",
            ):
                stat_key = f"initialization_{key}"
                epoch_stats[stat_key] = epoch_stats.get(stat_key, 0.0) + float(initialization_diagnostics[key])

        num_batches = max(len(dataloader), 1)
        epoch_loss /= num_batches
        epoch_stats = {key: value / num_batches for key, value in epoch_stats.items()}
        print(f"epoch={epoch} local_epoch={local_epoch}/{args.epochs} loss={epoch_loss:.6g} stats={epoch_stats}")

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            stats=epoch_stats,
        )
        if args.save_every > 0 and (local_epoch % args.save_every == 0 or local_epoch == args.epochs):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                stats=epoch_stats,
            )



train_main = main
