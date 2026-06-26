"""Exact point-UDF sidecars for iterative neural refinement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from dccvt.neural.dataset import resolve_cache_files
from dccvt.neural.grid import make_coord_grid, validate_grid_n


POINT_UDF_SIDECAR_VERSION = "point_udf_sidecar_v1"


def point_udf_sidecar_path(output_root: str | Path, mesh_id: str | Path) -> Path:
    """Return the sidecar path for a cache stem or mesh id."""
    return Path(output_root) / f"{Path(mesh_id).stem}.npz"


def exact_point_udf_grid(
    points: torch.Tensor,
    *,
    grid_n: int = 65,
    coordinate_min: float = -1.0,
    coordinate_max: float = 1.0,
    query_chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute exact nearest input-point distances on a dense cubic grid."""
    grid_n = validate_grid_n(grid_n)
    points = points.reshape(-1, 3)
    if points.numel() == 0:
        raise ValueError("Expected at least one point for point-UDF sidecar generation")
    if coordinate_min != -1.0 or coordinate_max != 1.0:
        axis = torch.linspace(
            float(coordinate_min),
            float(coordinate_max),
            grid_n,
            device=points.device,
            dtype=points.dtype,
        )
        try:
            xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
        except TypeError:
            xyz = torch.meshgrid(axis, axis, axis)
        queries = torch.stack(xyz, dim=-1).reshape(-1, 3)
    else:
        queries = make_coord_grid(grid_n, device=points.device, dtype=points.dtype)

    distances: list[torch.Tensor] = []
    for chunk in queries.split(int(query_chunk_size), dim=0):
        chunk_distances = torch.cdist(chunk.unsqueeze(0), points.unsqueeze(0), p=2).squeeze(0)
        distances.append(chunk_distances.min(dim=1).values.clamp_min(0.0))
    return torch.cat(distances, dim=0).reshape(grid_n, grid_n, grid_n)


def write_point_udf_sidecar(
    output_path: str | Path,
    udf_grid: np.ndarray,
    *,
    source_cache_path: str | Path,
    source_point_count: int,
    grid_n: int = 65,
    coordinate_min: float = -1.0,
    coordinate_max: float = 1.0,
    seed: int = 69,
    command_args: Optional[dict[str, Any]] = None,
) -> None:
    """Atomically write one compressed point-UDF sidecar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_n = validate_grid_n(grid_n)
    udf_grid = np.asarray(udf_grid, dtype=np.float32)
    if udf_grid.shape != (grid_n, grid_n, grid_n):
        raise ValueError(f"Expected {grid_n}^3 UDF grid, got {udf_grid.shape}")

    metadata = {
        "preprocessing_version": POINT_UDF_SIDECAR_VERSION,
        "grid_n": int(grid_n),
        "coordinate_min": float(coordinate_min),
        "coordinate_max": float(coordinate_max),
        "source_cache_path": str(source_cache_path),
        "source_point_count": int(source_point_count),
        "seed": int(seed),
        "command_args": command_args or {},
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        np.savez_compressed(
            temporary,
            **{
                f"{grid_n}_udf": udf_grid.astype(np.float32),
                "metadata": np.array(json.dumps(metadata, sort_keys=True)),
                "preprocessing_version": np.array(POINT_UDF_SIDECAR_VERSION),
                "grid_n": np.array(grid_n, dtype=np.int32),
                "coordinate_min": np.array(coordinate_min, dtype=np.float32),
                "coordinate_max": np.array(coordinate_max, dtype=np.float32),
                "source_cache_path": np.array(str(source_cache_path)),
                "source_point_count": np.array(int(source_point_count), dtype=np.int64),
                "seed": np.array(int(seed), dtype=np.int64),
            },
        )
        npz_temporary = temporary.with_suffix(temporary.suffix + ".npz")
        os.replace(npz_temporary if npz_temporary.exists() else temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(temporary.suffix + ".npz").unlink(missing_ok=True)


def validate_point_udf_sidecar(
    path: str | Path,
    *,
    grid_n: int = 65,
    check_values: bool = False,
) -> tuple[bool, str]:
    """Validate sidecar metadata and optionally UDF values."""
    path = Path(path)
    if not path.exists():
        return False, "missing sidecar"
    grid_n = validate_grid_n(grid_n)
    key = f"{grid_n}_udf"
    try:
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                return False, f"missing {key}"
            if data[key].shape != (grid_n, grid_n, grid_n):
                return False, f"invalid {key} shape"
            if data[key].dtype != np.float32:
                return False, f"invalid {key} dtype"
            if int(np.asarray(data["grid_n"]).item()) != grid_n:
                return False, "grid_n mismatch"
            if str(np.asarray(data["preprocessing_version"]).item()) != POINT_UDF_SIDECAR_VERSION:
                return False, "preprocessing version mismatch"
            if float(np.asarray(data["coordinate_min"]).item()) != -1.0:
                return False, "coordinate_min mismatch"
            if float(np.asarray(data["coordinate_max"]).item()) != 1.0:
                return False, "coordinate_max mismatch"
            if int(np.asarray(data["source_point_count"]).item()) < 1:
                return False, "invalid source point count"
            if check_values:
                udf = np.asarray(data[key], dtype=np.float32)
                if not np.isfinite(udf).all() or np.any(udf < 0.0):
                    return False, f"{key} contains invalid values"
    except Exception as exc:
        return False, repr(exc)
    return True, "ok"


def load_point_udf_sidecar(path: str | Path, *, grid_n: int = 65) -> np.ndarray:
    """Load and validate a point-UDF sidecar grid."""
    valid, reason = validate_point_udf_sidecar(path, grid_n=grid_n, check_values=False)
    if not valid:
        raise ValueError(f"Invalid point-UDF sidecar {path}: {reason}")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[f"{grid_n}_udf"], dtype=np.float32)


def precompute_point_udf_sidecar_for_cache(
    cache_path: str | Path,
    output_root: str | Path,
    *,
    grid_n: int = 65,
    query_chunk_size: int = 2048,
    device: torch.device | str = "cpu",
    overwrite: bool = False,
    seed: int = 69,
    command_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Precompute one sidecar from an existing HotSpot cache."""
    cache_path = Path(cache_path)
    output_path = point_udf_sidecar_path(output_root, cache_path.stem)
    if output_path.exists() and not overwrite:
        valid, reason = validate_point_udf_sidecar(output_path, grid_n=grid_n)
        if valid:
            return {
                "cache_path": str(cache_path),
                "sidecar_path": str(output_path),
                "status": "skipped_existing",
            }
        raise ValueError(f"Existing sidecar is invalid: {output_path}: {reason}")

    with np.load(cache_path, allow_pickle=False) as data:
        target_points = np.asarray(data["target_points"], dtype=np.float32).reshape(-1, 3)
    point_tensor = torch.from_numpy(target_points).to(device=device, dtype=torch.float32)
    udf_grid = exact_point_udf_grid(
        point_tensor,
        grid_n=grid_n,
        coordinate_min=-1.0,
        coordinate_max=1.0,
        query_chunk_size=query_chunk_size,
    )
    write_point_udf_sidecar(
        output_path,
        udf_grid.detach().cpu().numpy().astype(np.float32),
        source_cache_path=cache_path,
        source_point_count=target_points.shape[0],
        grid_n=grid_n,
        coordinate_min=-1.0,
        coordinate_max=1.0,
        seed=seed,
        command_args=command_args,
    )
    return {
        "cache_path": str(cache_path),
        "sidecar_path": str(output_path),
        "status": "written",
        "grid_n": int(grid_n),
        "source_point_count": int(target_points.shape[0]),
    }


def precompute_point_udf_sidecars(
    *,
    cache_root: str | Path,
    output_root: str | Path,
    split_file: str | Path | None = None,
    mesh_ids: Sequence[str] | None = None,
    grid_n: int = 65,
    query_chunk_size: int = 2048,
    device: torch.device | str = "cpu",
    overwrite: bool = False,
    fail_fast: bool = False,
    seed: int = 69,
    command_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Precompute exact point-UDF sidecars for a cache set."""
    cache_files = resolve_cache_files(cache_root, mesh_ids=mesh_ids, split_file=split_file)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "preprocessing_version": POINT_UDF_SIDECAR_VERSION,
        "grid_n": int(grid_n),
        "coordinate_min": -1.0,
        "coordinate_max": 1.0,
        "seed": int(seed),
        "args": command_args or {},
    }
    (output_root / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for cache_path in cache_files:
        try:
            result = precompute_point_udf_sidecar_for_cache(
                cache_path,
                output_root,
                grid_n=grid_n,
                query_chunk_size=query_chunk_size,
                device=device,
                overwrite=overwrite,
                seed=seed,
                command_args=command_args,
            )
            print(f"{Path(cache_path).stem}: {result['status']} {result['sidecar_path']}")
        except Exception as exc:
            result = {"cache_path": str(cache_path), "status": "failed", "error": repr(exc)}
            print(f"Failed point-UDF sidecar for {cache_path}: {exc}")
            if fail_fast:
                raise
        results.append(result)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved point-UDF sidecar summary: {summary_path}")
    return results


def _parse_mesh_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute exact point-UDF sidecars from HotSpot caches.")
    parser.add_argument("--cache-root", default="outputs/neural_hotspot_sdf/thingi32_g33")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids.")
    parser.add_argument("--output-root", default="outputs/neural_hotspot_sdf/thingi32_g65_point_udf")
    parser.add_argument("--grid-n", type=int, default=65)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    precompute_point_udf_sidecars(
        cache_root=args.cache_root,
        output_root=args.output_root,
        split_file=args.split_file,
        mesh_ids=_parse_mesh_ids(args.mesh_ids),
        grid_n=args.grid_n,
        query_chunk_size=args.query_chunk_size,
        device=device,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        seed=args.seed,
        command_args=vars(args),
    )


if __name__ == "__main__":
    main()
