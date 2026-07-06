"""Distributed mesh-loss training and evaluation for HybridPoNQ on ABC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dccvt.io_utils import save_obj_mesh
from dccvt.neural.abc.config import ABCHybridExperimentConfig, load_abc_hybrid_config, read_model_ids
from dccvt.neural.abc.data import ABCHybridDataset
from dccvt.neural.abc.modeling import build_abc_hybrid_model, deterministic_subset
from dccvt.neural.abc.udf import udf_sidecar_path, validate_udf_sidecar
from dccvt.neural.models import DCCVTHybridDirectNet


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("HybridPoNQ ABC mesh training requires CUDA")
    torch.cuda.set_device(local_rank)
    os.environ["DCCVT_DEVICE"] = f"cuda:{local_rank}"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _unwrap(model: torch.nn.Module) -> DCCVTHybridDirectNet:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if not isinstance(model, DCCVTHybridDirectNet):
        raise TypeError(f"Unexpected model type: {type(model)}")
    return model


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_ids(path: Path, model_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{model_id}\n" for model_id in model_ids), encoding="utf-8")


def _validate_udf_coverage(config: ABCHybridExperimentConfig, model_ids: Sequence[str]) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for model_id in model_ids:
        path = udf_sidecar_path(config.paths.udf_root, model_id)
        if not path.exists():
            missing.append(model_id)
            continue
        valid, reason = validate_udf_sidecar(path, config=config.udf, check_values=False)
        if not valid:
            invalid.append(f"{model_id}: {reason}")
    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing ({len(missing)}): {missing[:20]}")
        if invalid:
            details.append(f"invalid ({len(invalid)}): {invalid[:20]}")
        raise FileNotFoundError("ABC UDF sidecars are incomplete: " + "; ".join(details))


def _run_ids(
    config: ABCHybridExperimentConfig,
    run_name: str,
) -> tuple[list[str], list[str]]:
    train_ids = read_model_ids(config.paths.train_split)
    validation_ids = read_model_ids(config.paths.validation_split)
    if run_name == "pilot":
        train_ids = deterministic_subset(
            train_ids,
            config.dccvt_training.pilot_train_count,
            config.seed,
        )
        validation_ids = deterministic_subset(
            validation_ids,
            config.dccvt_training.pilot_validation_count,
            config.seed + 1,
        )
    return train_ids, validation_ids


def _proxy_ids(
    config: ABCHybridExperimentConfig,
    validation_ids: Sequence[str],
    run_name: str,
) -> list[str]:
    count = (
        config.dccvt_training.pilot_validation_count
        if run_name == "pilot"
        else config.dccvt_training.validation_proxy_count
    )
    return deterministic_subset(validation_ids, count, config.seed + 2)


def _rank_shard(ids: Sequence[str], rank: int, world_size: int, *, pad: bool) -> list[str]:
    shard = list(ids[rank::world_size])
    if not pad or world_size == 1:
        return shard
    target = (len(ids) + world_size - 1) // world_size
    if not shard:
        raise ValueError("A distributed rank received no ABC shapes")
    while len(shard) < target:
        shard.append(shard[0])
    return shard


def _build_loader(
    config: ABCHybridExperimentConfig,
    model_ids: Sequence[str],
    *,
    rank: int,
    world_size: int,
    shuffle: bool,
    deterministic_targets: bool,
) -> DataLoader:
    shard = _rank_shard(model_ids, rank, world_size, pad=shuffle)
    dataset = ABCHybridDataset(
        shard,
        hdf5_root=config.paths.hdf5_root,
        udf_root=config.paths.udf_root,
        target_sample_count=config.dccvt_training.target_sample_count,
        seed=config.seed,
        deterministic_targets=deterministic_targets,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed + rank)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=config.dccvt_training.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _infinite_batches(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _canonical_delaunay(model: torch.nn.Module, device: torch.device):
    from dccvt.geometry import compute_delaunay_simplices

    canonical_sites = _unwrap(model).canonical_sites.to(device=device)
    return compute_delaunay_simplices(canonical_sites)


def _regularized_mesh_loss(
    outputs: dict[str, torch.Tensor],
    target_points: torch.Tensor,
    config: ABCHybridExperimentConfig,
    *,
    strict: bool,
    delaunay_simplices,
) -> tuple[torch.Tensor, dict[str, float]]:
    from dccvt.neural.losses import hybrid_direct_mesh_loss

    settings = config.dccvt_training
    mesh_loss, stats = hybrid_direct_mesh_loss(
        outputs,
        target_points,
        chamfer_weight=settings.chamfer_weight,
        cvt_weight=settings.cvt_weight,
        sdfsmooth_weight=settings.sdf_smoothness_weight,
        strict=strict,
        delaunay_simplices=delaunay_simplices,
    )
    site_regularizer = outputs["site_delta"].pow(2).mean()
    sdf_regularizer = outputs["sdf_residual"].pow(2).mean()
    total = (
        mesh_loss
        + settings.site_displacement_weight * site_regularizer
        + settings.sdf_residual_weight * sdf_regularizer
    )
    stats["site_displacement_regularizer"] = float(site_regularizer.detach().cpu())
    stats["sdf_residual_regularizer"] = float(sdf_regularizer.detach().cpu())
    stats["total_loss"] = float(total.detach().cpu())
    return total, stats


def _evaluate_proxy(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ABCHybridExperimentConfig,
    *,
    device: torch.device,
    world_size: int,
    delaunay_simplices,
) -> dict[str, float]:
    model.eval()
    evaluation_model = _unwrap(model)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            input_grid = batch["input_grid"].to(device, non_blocking=True)
            sdf_grid = batch["sdf_grid"].to(device, non_blocking=True)
            target_points = batch["target_points"].to(device, non_blocking=True)
            outputs = evaluation_model(input_grid, sdf_grid)
            _, stats = _regularized_mesh_loss(
                outputs,
                target_points,
                config,
                strict=False,
                delaunay_simplices=delaunay_simplices,
            )
            chamfer = stats["mesh_chamfer"]
            used = stats["mesh_used_shapes"]
            if not np.isfinite(chamfer):
                totals[4] += max(used, 1.0)
                continue
            totals[0] += chamfer * used
            totals[1] += stats["site_displacement_regularizer"]
            totals[2] += stats["sdf_residual_regularizer"]
            totals[3] += used
            totals[4] += stats["mesh_skipped_shapes"]
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    used = max(float(totals[3].item()), 1.0)
    item_count = max(float(totals[3].item() + totals[4].item()), 1.0)
    model.train()
    return {
        "chamfer": float(totals[0].item() / used),
        "site_displacement_regularizer": float(totals[1].item() / item_count),
        "sdf_residual_regularizer": float(totals[2].item() / item_count),
        "used_shapes": float(totals[3].item()),
        "skipped_shapes": float(totals[4].item()),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ABCHybridExperimentConfig,
    args: argparse.Namespace,
    initialization: dict,
    validation: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": _unwrap(model).config(),
            "experiment_config": config.to_dict(),
            "seed": config.seed,
            "args": vars(args),
            "initialization": initialization,
            "validation": validation,
            "git_revision": _git_revision(),
        },
        path,
    )


def train(
    args: argparse.Namespace,
    config: ABCHybridExperimentConfig,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> Path:
    train_ids, validation_ids = _run_ids(config, args.run)
    proxy_ids = _proxy_ids(config, validation_ids, args.run)
    _validate_udf_coverage(config, list(dict.fromkeys(train_ids + proxy_ids)))

    run_root = config.paths.output_root / args.variant / args.run
    checkpoint_root = run_root / "checkpoints"
    if rank == 0:
        run_root.mkdir(parents=True, exist_ok=True)
        _write_ids(run_root / "splits" / "train.txt", train_ids)
        _write_ids(run_root / "splits" / "validation.txt", validation_ids)
        _write_ids(run_root / "splits" / "validation_proxy.txt", proxy_ids)
        resolved = {
            "config": config.to_dict(),
            "args": vars(args),
            "world_size": world_size,
            "git_revision": _git_revision(),
        }
        (run_root / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    _barrier(world_size)

    model, initialization = build_abc_hybrid_model(
        config.model,
        variant=args.variant,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.dccvt_training.learning_rate,
        weight_decay=config.dccvt_training.weight_decay,
    )
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.resume_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        initialization = checkpoint.get("initialization", initialization)

    wrapped: torch.nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    train_loader = _build_loader(
        config,
        train_ids,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        deterministic_targets=False,
    )
    validation_loader = _build_loader(
        config,
        proxy_ids,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        deterministic_targets=True,
    )
    train_batches = _infinite_batches(train_loader)
    training_delaunay = _canonical_delaunay(wrapped, device)
    max_steps = (
        config.dccvt_training.pilot_steps
        if args.run == "pilot"
        else config.dccvt_training.full_steps
    )

    baseline = _evaluate_proxy(
        wrapped,
        validation_loader,
        config,
        device=device,
        world_size=world_size,
        delaunay_simplices=training_delaunay,
    )
    if baseline["used_shapes"] <= 0 or not np.isfinite(baseline["chamfer"]):
        raise RuntimeError(f"Canonical proxy validation is invalid: {baseline}")
    initial_validation_step = start_step
    if rank == 0:
        (run_root / f"validation_step_{initial_validation_step:06d}.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"step={initial_validation_step} validation={baseline}", flush=True)
    best_chamfer = baseline["chamfer"]
    best_path = checkpoint_root / "best.pt"
    if rank == 0:
        if args.resume and best_path.exists():
            previous_best = torch.load(best_path, map_location="cpu")
            previous_validation = previous_best.get("validation", {})
            previous_chamfer = previous_validation.get("chamfer")
            if previous_chamfer is not None and np.isfinite(previous_chamfer):
                best_chamfer = float(previous_chamfer)
        else:
            _save_checkpoint(
                best_path,
                model=wrapped,
                optimizer=optimizer,
                step=initial_validation_step,
                config=config,
                args=args,
                initialization=initialization,
                validation=baseline,
            )
    _barrier(world_size)

    wrapped.train()
    for step in range(start_step + 1, max_steps + 1):
        batch = next(train_batches)
        input_grid = batch["input_grid"].to(device, non_blocking=True)
        sdf_grid = batch["sdf_grid"].to(device, non_blocking=True)
        target_points = batch["target_points"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = wrapped(input_grid, sdf_grid)
        loss, stats = _regularized_mesh_loss(
            outputs,
            target_points,
            config,
            strict=True,
            delaunay_simplices=training_delaunay,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {stats}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        stats["gradient_norm"] = float(gradient_norm.detach().cpu())

        if rank == 0:
            print(
                f"step={step}/{max_steps} model_id={batch['model_id'][0]} "
                f"loss={stats['total_loss']:.8g} chamfer={stats['mesh_chamfer']:.8g}",
                flush=True,
            )

        should_validate = (
            step % config.dccvt_training.validate_every_steps == 0 or step == max_steps
        )
        validation: dict = {}
        if should_validate:
            validation = _evaluate_proxy(
                wrapped,
                validation_loader,
                config,
                device=device,
                world_size=world_size,
                delaunay_simplices=training_delaunay,
            )
            if rank == 0:
                (run_root / f"validation_step_{step:06d}.json").write_text(
                    json.dumps(validation, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"step={step} validation={validation}", flush=True)
                if (
                    validation["used_shapes"] > 0
                    and np.isfinite(validation["chamfer"])
                    and validation["chamfer"] < best_chamfer
                ):
                    best_chamfer = validation["chamfer"]
                    _save_checkpoint(
                        best_path,
                        model=wrapped,
                        optimizer=optimizer,
                        step=step,
                        config=config,
                        args=args,
                        initialization=initialization,
                        validation=validation,
                    )

        should_checkpoint = (
            step % config.dccvt_training.checkpoint_every_steps == 0 or step == max_steps
        )
        if rank == 0 and should_checkpoint:
            _save_checkpoint(
                checkpoint_root / "latest.pt",
                model=wrapped,
                optimizer=optimizer,
                step=step,
                config=config,
                args=args,
                initialization=initialization,
                validation=validation,
            )
            _save_checkpoint(
                checkpoint_root / f"step_{step:06d}.pt",
                model=wrapped,
                optimizer=optimizer,
                step=step,
                config=config,
                args=args,
                initialization=initialization,
                validation=validation,
            )
        _barrier(world_size)

    _barrier(world_size)
    return best_path


def _load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device,
) -> DCCVTHybridDirectNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint lacks model_config: {checkpoint_path}")
    model = DCCVTHybridDirectNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _extract_one(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    output_path: Path,
) -> None:
    from dccvt.geometry import compute_delaunay_simplices
    from dccvt.mesh_ops import extract_cvt_mesh

    if sites.shape[0] < 5:
        raise RuntimeError("DCCVT extraction requires at least five sites")
    if not ((sites_sdf < 0).any() and (sites_sdf > 0).any()):
        raise RuntimeError("DCCVT extraction requires positive and negative SDF values")
    simplices = compute_delaunay_simplices(sites)
    vertices, faces = extract_cvt_mesh(sites, sites_sdf, simplices, True)
    if vertices.numel() == 0 or not faces:
        raise RuntimeError("DCCVT extraction produced an empty mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_obj_mesh(output_path, vertices.detach().cpu().numpy(), faces)


def _append_status(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _extract_method(
    model: DCCVTHybridDirectNet,
    config: ABCHybridExperimentConfig,
    model_ids: Sequence[str],
    *,
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    shard = _rank_shard(model_ids, rank, world_size, pad=False)
    dataset = ABCHybridDataset(
        shard,
        hdf5_root=config.paths.hdf5_root,
        udf_root=config.paths.udf_root,
        target_sample_count=1,
        seed=config.seed,
        deterministic_targets=True,
    )
    status_path = output_dir.parent / f"extraction_status_rank{rank}.jsonl"
    status_path.unlink(missing_ok=True)
    for item in dataset:
        model_id = item["model_id"]
        output_path = output_dir / f"{model_id}.obj"
        record = {"model_id": model_id, "output": str(output_path)}
        try:
            input_grid = item["input_grid"].unsqueeze(0).to(device)
            sdf_grid = item["sdf_grid"].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_grid, sdf_grid)
            _extract_one(outputs["sites"][0], outputs["sites_sdf"][0], output_path)
            record["status"] = "extracted"
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        _append_status(status_path, record)
        print(f"rank={rank} extraction={record}", flush=True)


def _run_metric_command(
    config: ABCHybridExperimentConfig,
    *,
    mesh_dir: Path,
    split_file: Path,
    output_prefix: Path,
) -> np.ndarray:
    eval_script = Path(__file__).resolve().parents[2] / "PoNQ-main" / "src" / "eval" / "eval_ABC.py"
    command = [
        str(config.paths.evaluation_python),
        str(eval_script),
        str(mesh_dir),
        "-gt_dir",
        str(config.paths.ground_truth_root),
        "--names-file",
        str(split_file),
        "--prediction-pattern",
        "{id}.obj",
        "--output",
        str(output_prefix),
        "--sample-count",
        str(config.evaluation.sample_count),
        "--seed",
        str(config.seed),
        "--n-jobs",
        str(config.evaluation.n_jobs),
    ]
    subprocess.run(command, cwd=eval_script.parents[2], check=True)
    return np.load(output_prefix.with_suffix(".npy"))


def _merge_extraction_status(evaluation_root: Path, world_size: int) -> dict:
    records = []
    for rank in range(world_size):
        path = evaluation_root / f"extraction_status_rank{rank}.jsonl"
        if not path.exists():
            continue
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    summary = {
        "shape_count": len(records),
        "extracted": sum(record["status"] == "extracted" for record in records),
        "failed": sum(record["status"] != "extracted" for record in records),
        "failures": [record for record in records if record["status"] != "extracted"],
    }
    (evaluation_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    mean = values.mean(axis=0)
    return {
        "cd1": float(mean[1]),
        "chamfer": float(mean[2]),
        "f1": float(mean[3]),
        "normal_consistency": float(mean[4]),
        "edge_chamfer": float(mean[5]),
        "edge_f1": float(mean[6]),
    }


def _qualification(
    config: ABCHybridExperimentConfig,
    *,
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    extraction_failures: int,
) -> dict:
    baseline_chamfer = baseline_metrics["chamfer"]
    if baseline_chamfer <= 0:
        raise ValueError(f"Canonical baseline Chamfer must be positive, got {baseline_chamfer}")
    chamfer_improvement = (baseline_chamfer - model_metrics["chamfer"]) / baseline_chamfer
    normal_regression = (
        baseline_metrics["normal_consistency"] - model_metrics["normal_consistency"]
    )
    edge_f1_regression = baseline_metrics["edge_f1"] - model_metrics["edge_f1"]
    checks = {
        "all_meshes_extracted": extraction_failures == 0,
        "chamfer_improvement": (
            chamfer_improvement >= config.evaluation.minimum_chamfer_improvement
        ),
        "normal_consistency_guardrail": (
            normal_regression
            <= config.evaluation.maximum_normal_consistency_regression
        ),
        "edge_f1_guardrail": (
            edge_f1_regression <= config.evaluation.maximum_edge_f1_regression
        ),
    }
    return {
        "qualifies": all(checks.values()),
        "checks": checks,
        "chamfer_improvement": chamfer_improvement,
        "normal_consistency_regression": normal_regression,
        "edge_f1_regression": edge_f1_regression,
        "thresholds": {
            "minimum_chamfer_improvement": config.evaluation.minimum_chamfer_improvement,
            "maximum_normal_consistency_regression": (
                config.evaluation.maximum_normal_consistency_regression
            ),
            "maximum_edge_f1_regression": config.evaluation.maximum_edge_f1_regression,
        },
    }


def evaluate(
    args: argparse.Namespace,
    config: ABCHybridExperimentConfig,
    checkpoint_path: Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    _, validation_ids = _run_ids(config, args.run)
    _validate_udf_coverage(config, validation_ids)
    run_root = config.paths.output_root / args.variant / args.run
    evaluation_root = run_root / "evaluation"
    split_file = evaluation_root / "validation_ids.txt"
    if rank == 0:
        evaluation_root.mkdir(parents=True, exist_ok=True)
        _write_ids(split_file, validation_ids)
    _barrier(world_size)

    model = _load_evaluation_model(checkpoint_path, device)
    model_root = evaluation_root / "model"
    _extract_method(
        model,
        config,
        validation_ids,
        output_dir=model_root / "meshes",
        rank=rank,
        world_size=world_size,
        device=device,
    )
    _barrier(world_size)

    baseline_root = evaluation_root / "canonical"
    if not args.skip_baseline:
        baseline, _ = build_abc_hybrid_model(config.model, variant="direct")
        baseline = baseline.to(device).eval()
        _extract_method(
            baseline,
            config,
            validation_ids,
            output_dir=baseline_root / "meshes",
            rank=rank,
            world_size=world_size,
            device=device,
        )
    _barrier(world_size)

    if rank != 0:
        return
    model_extraction = _merge_extraction_status(model_root, world_size)
    if model_extraction["failed"]:
        raise RuntimeError(f"Model extraction failed for {model_extraction['failed']} shapes")
    model_values = _run_metric_command(
        config,
        mesh_dir=model_root / "meshes",
        split_file=split_file,
        output_prefix=model_root / "metrics",
    )
    summary = {"model": _metric_summary(model_values), "checkpoint": str(checkpoint_path)}
    if not args.skip_baseline:
        baseline_extraction = _merge_extraction_status(baseline_root, world_size)
        if baseline_extraction["failed"]:
            raise RuntimeError(
                f"Canonical extraction failed for {baseline_extraction['failed']} shapes"
            )
        baseline_values = _run_metric_command(
            config,
            mesh_dir=baseline_root / "meshes",
            split_file=split_file,
            output_prefix=baseline_root / "metrics",
        )
        summary["canonical"] = _metric_summary(baseline_values)
        summary["qualification"] = _qualification(
            config,
            model_metrics=summary["model"],
            baseline_metrics=summary["canonical"],
            extraction_failures=model_extraction["failed"],
        )
    (evaluation_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate HybridPoNQ-DCCVT on ABC.")
    parser.add_argument("--config", default="configs/hybrid_ponq_abc_dccvt_v1.json")
    parser.add_argument("--variant", choices=("direct", "ponq_pretrained"), required=True)
    parser.add_argument("--run", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="train")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint used by --stage evaluate.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_abc_hybrid_config(args.config)
    rank, local_rank, world_size, device = _distributed_context()
    _seed_everything(config.seed + rank)
    checkpoint_path: Optional[Path] = Path(args.checkpoint).resolve() if args.checkpoint else None
    try:
        if args.stage in {"train", "all"}:
            checkpoint_path = train(
                args,
                config,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
            )
        if args.stage in {"evaluate", "all"}:
            if checkpoint_path is None:
                checkpoint_path = (
                    config.paths.output_root / args.variant / args.run / "checkpoints" / "best.pt"
                )
            evaluate(
                args,
                config,
                checkpoint_path,
                rank=rank,
                world_size=world_size,
                device=device,
            )
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
