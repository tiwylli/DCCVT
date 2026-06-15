"""Train the hybrid direct PoNQ-DCCVT extractor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dccvt.neural.dataset import HybridDirectDataset, resolve_cache_files
from dccvt.neural.losses import hybrid_direct_mesh_loss, hybrid_direct_supervised_loss
from dccvt.neural.models import DCCVTHybridDirectNet, HybridDirectConfig, load_hybrid_direct_config


def _parse_mesh_ids(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible supervised training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_resolved_config(path: Path, *, config: HybridDirectConfig, args: argparse.Namespace) -> None:
    payload = {
        "model_config": config.to_dict(),
        "seed": int(args.seed),
        "channel_names": list(config.channel_names),
        "args": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: DCCVTHybridDirectNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    stats: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "seed": int(args.seed),
        "channel_names": list(model.config_obj.channel_names),
        "args": vars(args),
        "stats": stats,
    }
    torch.save(payload, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hybrid direct PoNQ-DCCVT extractor.")
    parser.add_argument("--config", default="configs/neural_hybrid_direct_v1.json")
    parser.add_argument("--cache-root", required=True, help="Directory of precomputed HotSpot SDF .npz caches.")
    parser.add_argument("--label-root", default="outputs/neural_labels/n32")
    parser.add_argument("--mesh-ids", default=None, help="Optional comma or space separated cache stems.")
    parser.add_argument("--split-file", default=None, help="Optional text file of cache stems.")
    parser.add_argument("--checkpoint-dir", default="outputs/neural_dccvt/hybrid_direct_v1/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--stage", choices=("supervised", "mesh"), default="supervised")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-subsample", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6.4e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=69)

    parser.add_argument("--label-upsampling", type=int, default=0)
    parser.add_argument("--label-state", default="final")
    parser.add_argument("--label-variant", default="projDCCVT")
    parser.add_argument("--label-w-cvt", type=float, default=100.0)
    parser.add_argument("--label-w-sdfsmooth", type=float, default=100.0)

    parser.add_argument("--w-site", type=float, default=1.0)
    parser.add_argument("--w-sdf", type=float, default=1.0)
    parser.add_argument("--w-sign", type=float, default=0.1)
    parser.add_argument("--w-residual", type=float, default=0.01)
    parser.add_argument("--sdf-near-weight", type=float, default=4.0)
    parser.add_argument("--sdf-near-tau", type=float, default=0.1)
    parser.add_argument("--sign-temperature", type=float, default=0.05)

    parser.add_argument("--w-mesh", type=float, default=1.0)
    parser.add_argument("--w-mesh-chamfer", type=float, default=1000.0)
    parser.add_argument("--w-mesh-cvt", type=float, default=100.0)
    parser.add_argument("--w-mesh-sdfsmooth", type=float, default=100.0)
    parser.add_argument(
        "--strict-mesh-loss",
        action="store_true",
        help="Abort if a mesh-stage batch item cannot produce a valid differentiable surface.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[DCCVTHybridDirectNet, Optional[dict]]:
    resume_checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    if resume_checkpoint is not None and resume_checkpoint.get("model_config"):
        config = HybridDirectConfig.from_dict(resume_checkpoint["model_config"])
    else:
        config = load_hybrid_direct_config(args.config)
    model = DCCVTHybridDirectNet(config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    return model, resume_checkpoint


def _build_dataset(args: argparse.Namespace, config: HybridDirectConfig) -> HybridDirectDataset:
    cache_files = resolve_cache_files(
        args.cache_root,
        mesh_ids=_parse_mesh_ids(args.mesh_ids),
        split_file=args.split_file,
    )
    return HybridDirectDataset(
        cache_files,
        label_root=args.label_root,
        target_subsample=args.target_subsample,
        upsampling=args.label_upsampling,
        label_state=args.label_state,
        label_variant=args.label_variant,
        label_w_cvt=args.label_w_cvt,
        label_w_sdfsmooth=args.label_w_sdfsmooth,
        point_udf_clip=config.point_udf_clip,
        point_confidence_sigma_scale=config.point_confidence_sigma_scale,
        channel_names=config.channel_names,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    seed_everything(args.seed)
    device = _device(args.device)

    model, resume_checkpoint = _load_model(args, device)
    dataset = _build_dataset(args, model.config_obj)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=data_generator,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(checkpoint_dir / "resolved_config.json", config=model.config_obj, args=args)

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
            label_sites = batch["label_sites"].to(device, non_blocking=True)
            label_sites_sdf = batch["label_sites_sdf"].to(device, non_blocking=True)
            target_points = batch["target_points"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_grid, sdf_grid)
            loss, stats = hybrid_direct_supervised_loss(
                outputs,
                label_sites,
                label_sites_sdf,
                site_weight=args.w_site,
                sdf_weight=args.w_sdf,
                sign_weight=args.w_sign,
                residual_weight=args.w_residual,
                sdf_near_weight=args.sdf_near_weight,
                sdf_near_tau=args.sdf_near_tau,
                sign_temperature=args.sign_temperature,
            )
            if args.stage == "mesh":
                mesh_loss, mesh_stats = hybrid_direct_mesh_loss(
                    outputs,
                    target_points,
                    chamfer_weight=args.w_mesh_chamfer,
                    cvt_weight=args.w_mesh_cvt,
                    sdfsmooth_weight=args.w_mesh_sdfsmooth,
                    strict=args.strict_mesh_loss,
                )
                loss = loss + args.w_mesh * mesh_loss
                stats.update(mesh_stats)

            loss.backward()
            optimizer.step()

            batch_loss = float(loss.detach().cpu())
            epoch_loss += batch_loss
            for key, value in stats.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value)

        num_batches = max(len(dataloader), 1)
        epoch_loss /= num_batches
        epoch_stats = {key: value / num_batches for key, value in epoch_stats.items()}
        print(f"epoch={epoch} local_epoch={local_epoch}/{args.epochs} loss={epoch_loss:.6g} stats={epoch_stats}")

        latest = checkpoint_dir / "latest.pt"
        save_checkpoint(latest, model=model, optimizer=optimizer, epoch=epoch, args=args, stats=epoch_stats)
        if args.save_every > 0 and (local_epoch % args.save_every == 0 or local_epoch == args.epochs):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                stats=epoch_stats,
            )


if __name__ == "__main__":
    main()
