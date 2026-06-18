"""Resumable multi-GPU exact UDF preprocessing for ABC point clouds."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
from typing import Optional, Sequence

import numpy as np
import torch

from dccvt.neural.abc_hybrid import (
    ABCHybridExperimentConfig,
    exact_point_udf_grid,
    load_abc_hybrid_config,
    read_model_ids,
    udf_sidecar_path,
    validate_udf_sidecar,
    write_udf_sidecar,
)


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(part) for part in value.replace(",", " ").split()]
    if not gpus:
        raise ValueError("At least one GPU must be provided")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU list contains duplicates: {gpus}")
    return gpus


def _selected_ids(config: ABCHybridExperimentConfig, split: str) -> list[str]:
    train_ids = read_model_ids(config.paths.train_split)
    validation_ids = read_model_ids(config.paths.validation_split)
    if split == "train":
        return train_ids
    if split == "validation":
        return validation_ids
    return list(dict.fromkeys(train_ids + validation_ids))


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _process_one(
    model_id: str,
    *,
    config: ABCHybridExperimentConfig,
    device: torch.device,
    resume: bool,
    verify_values: bool,
) -> tuple[str, str, float]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("ABC UDF preprocessing requires h5py") from exc

    output_path = udf_sidecar_path(config.paths.udf_root, model_id)
    if resume:
        valid, reason = validate_udf_sidecar(
            output_path,
            config=config.udf,
            check_values=verify_values,
        )
        if valid:
            return model_id, "skipped", 0.0
        if output_path.exists():
            print(f"{model_id}: regenerating invalid sidecar ({reason})", flush=True)

    source_path = config.paths.hdf5_root / f"{model_id}.hdf5"
    start = time.perf_counter()
    with h5py.File(source_path, "r") as source:
        points = np.asarray(source["pointcloud"][:], dtype=np.float32)
    if points.shape != (1_000_000, 3):
        raise ValueError(f"{source_path} pointcloud has shape {points.shape}, expected (1000000,3)")

    points_tensor = torch.from_numpy(points).to(device=device, non_blocking=False)
    with torch.no_grad():
        udf = exact_point_udf_grid(
            points_tensor,
            grid_n=config.udf.master_grid_n,
            coordinate_min=config.udf.coordinate_min,
            coordinate_max=config.udf.coordinate_max,
            query_chunk_size=config.udf.query_chunk_size,
        )
    udf_np = udf.cpu().numpy().astype(np.float32, copy=False)
    write_udf_sidecar(
        output_path,
        udf_np,
        source_point_count=points.shape[0],
        config=config.udf,
    )
    valid, reason = validate_udf_sidecar(output_path, config=config.udf, check_values=True)
    if not valid:
        raise RuntimeError(f"Sidecar validation failed after write: {reason}")
    return model_id, "written", time.perf_counter() - start


def _worker(
    worker_index: int,
    gpu: int,
    model_ids: Sequence[str],
    config_path: str,
    resume: bool,
    verify_values: bool,
    log_root: str,
    result_queue: mp.Queue,
) -> None:
    config = load_abc_hybrid_config(config_path)
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    failure_log = Path(log_root) / f"failures_gpu{gpu}.jsonl"
    counts = {"written": 0, "skipped": 0, "failed": 0}
    elapsed = 0.0
    for index, model_id in enumerate(model_ids, start=1):
        try:
            _, status, seconds = _process_one(
                model_id,
                config=config,
                device=device,
                resume=resume,
                verify_values=verify_values,
            )
            counts[status] += 1
            elapsed += seconds
            print(
                f"[gpu={gpu} worker={worker_index}] {index}/{len(model_ids)} "
                f"{model_id} {status} {seconds:.2f}s",
                flush=True,
            )
        except Exception as exc:
            counts["failed"] += 1
            _append_jsonl(
                failure_log,
                {
                    "model_id": model_id,
                    "gpu": gpu,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"[gpu={gpu}] {model_id} failed: {exc}", flush=True)
    result_queue.put({"gpu": gpu, "counts": counts, "elapsed_seconds": elapsed})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute exact 129^3 ABC point-cloud UDF sidecars.")
    parser.add_argument("--config", default="configs/hybrid_ponq_abc_dccvt_v1.json")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fast-resume-check",
        action="store_true",
        help="Validate schema and metadata but skip the aligned-value comparison for existing files.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Exact 129^3 UDF preprocessing requires CUDA")

    config_path = str(Path(args.config).resolve())
    config = load_abc_hybrid_config(config_path)
    gpus = _parse_gpus(args.gpus)
    invalid = [gpu for gpu in gpus if gpu < 0 or gpu >= torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Unavailable GPU indices: {invalid}")

    model_ids = _selected_ids(config, args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        model_ids = model_ids[: args.limit]
    shards = [model_ids[index:: len(gpus)] for index in range(len(gpus))]

    config.paths.udf_root.mkdir(parents=True, exist_ok=True)
    log_root = config.paths.udf_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config": config.to_dict(),
        "args": vars(args),
        "model_count": len(model_ids),
        "gpus": gpus,
        "shards": {str(gpu): len(shard) for gpu, shard in zip(gpus, shards)},
    }
    (log_root / "resolved_preprocess_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for worker_index, (gpu, shard) in enumerate(zip(gpus, shards)):
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                gpu,
                shard,
                config_path,
                args.resume,
                not args.fast_resume_check,
                str(log_root),
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=1.0))
        except Empty:
            break
    failed_workers = [process.pid for process in processes if process.exitcode != 0]
    totals = {
        key: sum(result["counts"][key] for result in results)
        for key in ("written", "skipped", "failed")
    }
    summary = {"totals": totals, "workers": results, "failed_worker_pids": failed_workers}
    (log_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if totals["failed"] or failed_workers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
