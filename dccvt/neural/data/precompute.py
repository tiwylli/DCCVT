"""Precompute dense HotSpot SDF grids for PoNQ-style neural DCCVT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dccvt.argparse_utils import DEFAULTS
from dccvt.device import device, initialize_runtime
from dccvt.model_utils import load_hotspot_model
from dccvt.neural.grid import (
    default_near_surface_threshold,
    make_coord_grid,
    make_gt_activity_mask_np,
    make_near_surface_mask_np,
    validate_grid_n,
)


def _parse_mesh_ids(value: Optional[str]) -> list[str]:
    if not value:
        return list(DEFAULTS["mesh_ids"])
    return [part for part in value.replace(",", " ").split() if part]


def _read_mesh_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(part for part in line.replace(",", " ").split() if part)
    return ids


def _resolve_mesh_path(mesh_root: Path, mesh_id: str) -> Path:
    mesh_path = Path(mesh_id)
    if mesh_path.is_absolute():
        return mesh_path
    return mesh_root / mesh_path


def _resolve_hotspot_path(hotspot_root: Path, mesh_id: str, suffix: str) -> Path:
    hotspot_path = Path(mesh_id)
    if hotspot_path.is_absolute() and hotspot_path.suffix == ".pth":
        return hotspot_path
    if hotspot_path.suffix == ".pth":
        return hotspot_root / hotspot_path
    return hotspot_root / f"{hotspot_path.stem}{suffix}"


@torch.no_grad()
def sample_hotspot_sdf_grid(
    model: torch.nn.Module,
    *,
    grid_n: int,
    batch_size: int,
) -> np.ndarray:
    """Evaluate a HotSpot model on a dense normalized ``[-1, 1]^3`` grid."""
    coords = make_coord_grid(grid_n, device=device)
    values: list[torch.Tensor] = []
    model.eval()
    for start in range(0, coords.shape[0], batch_size):
        chunk = coords[start : start + batch_size]
        values.append(model(chunk).detach().reshape(-1).cpu())
    sdf = torch.cat(values, dim=0).numpy().astype(np.float32)
    return sdf.reshape(grid_n, grid_n, grid_n)


def build_hotspot_cache_record(
    *,
    mesh_path: str | Path,
    hotspot_weights_path: str | Path,
    mesh_id: str,
    grid_n: int = 33,
    sample_count: int = 200_000,
    max_amount_sites: int = 32,
    query_batch_size: int = 65_536,
    near_surface_threshold: Optional[float] = None,
) -> dict:
    """Create a cache record containing SDF grid, masks, and target samples."""
    grid_n = validate_grid_n(grid_n)
    if near_surface_threshold is None:
        near_surface_threshold = default_near_surface_threshold(grid_n)

    model, mnfld_points = load_hotspot_model(
        str(mesh_path),
        max_amount_sites=max_amount_sites,
        hotspot_weights_path=str(hotspot_weights_path),
        n_points=sample_count,
    )
    sdf_grid = sample_hotspot_sdf_grid(model, grid_n=grid_n, batch_size=query_batch_size)
    target_points = mnfld_points.squeeze(0).detach().cpu().numpy().astype(np.float32)
    near_surface_mask = make_near_surface_mask_np(sdf_grid, threshold=near_surface_threshold)
    gt_activity_mask = make_gt_activity_mask_np(target_points, grid_n)

    return {
        "mesh_id": np.array(mesh_id),
        "mesh_path": np.array(str(mesh_path)),
        "hotspot_weights_path": np.array(str(hotspot_weights_path)),
        "grid_n": np.array(grid_n, dtype=np.int32),
        "domain_min": np.array(-1.0, dtype=np.float32),
        "domain_max": np.array(1.0, dtype=np.float32),
        "near_surface_threshold": np.array(near_surface_threshold, dtype=np.float32),
        "sdf_grid": sdf_grid.astype(np.float32),
        "near_surface_mask": near_surface_mask.astype(bool),
        "gt_activity_mask": gt_activity_mask.astype(bool),
        "target_points": target_points.astype(np.float32),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute dense HotSpot SDF caches for neural DCCVT.")
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids. Defaults to DCCVT ids.")
    parser.add_argument("--mesh-ids-file", default=None, help="Optional text file of mesh ids or cache stems.")
    parser.add_argument("--mesh-root", default=str(Path(DEFAULTS["mesh"])), help="Root containing <mesh_id>.ply.")
    parser.add_argument("--hotspot-root", default=str(Path(DEFAULTS["trained_HotSpot"])), help="Root containing <mesh_id>.pth.")
    parser.add_argument("--hotspot-suffix", default=".pth")
    parser.add_argument("--output-root", default="outputs/neural_hotspot_sdf/g33")
    parser.add_argument("--grid-n", type=int, default=33)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--max-amount-sites", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=65_536)
    parser.add_argument("--near-surface-threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    initialize_runtime()

    if args.mesh_ids_file:
        mesh_ids = _read_mesh_ids_file(Path(args.mesh_ids_file))
    else:
        mesh_ids = _parse_mesh_ids(args.mesh_ids)

    mesh_root = Path(args.mesh_root)
    hotspot_root = Path(args.hotspot_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for mesh_id in mesh_ids:
        output_file = output_root / f"{Path(mesh_id).stem}.npz"
        if output_file.exists() and not args.overwrite:
            print(f"Skipping existing cache: {output_file}")
            continue

        mesh_path = _resolve_mesh_path(mesh_root, mesh_id)
        hotspot_path = _resolve_hotspot_path(hotspot_root, mesh_id, args.hotspot_suffix)
        print(f"Precomputing {mesh_id}: mesh={mesh_path} hotspot={hotspot_path}")
        record = build_hotspot_cache_record(
            mesh_path=mesh_path,
            hotspot_weights_path=hotspot_path,
            mesh_id=Path(mesh_id).stem,
            grid_n=args.grid_n,
            sample_count=args.sample_count,
            max_amount_sites=args.max_amount_sites,
            query_batch_size=args.query_batch_size,
            near_surface_threshold=args.near_surface_threshold,
        )
        np.savez_compressed(output_file, **record)
        print(
            f"Saved {output_file} "
            f"near_cells={int(record['near_surface_mask'].sum())} "
            f"gt_cells={int(record['gt_activity_mask'].sum())}"
        )


if __name__ == "__main__":
    main()
