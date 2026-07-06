"""Initialization export CLI for iterative refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch

from dccvt.neural.data.datasets import resolve_cache_files
from dccvt.neural.grid import build_hybrid_input_channels_np
from dccvt.neural.iterative.config import HybridIterRefineConfig, load_iter_refine_config
from dccvt.neural.iterative.initialization import build_hotspot_near_surface_initialization
from dccvt.neural.utils import cache_mesh_id, load_npz_cache, parse_mesh_ids, seed_everything

def _expected_initial_obj_path(
    output_dir: Path,
    *,
    state: str,
    variant: str,
    w_cvt: float,
    w_sdfsmooth: float,
) -> Path:
    return output_dir / f"DCCVT_0_{state}_{variant}_cvt{int(w_cvt)}_sdfsmooth{int(w_sdfsmooth)}.obj"


def _save_initialization_field(
    output_dir: Path,
    *,
    mesh_id: str,
    initialization: dict[str, Any],
    input_grid: np.ndarray,
    sdf_grid: np.ndarray,
    target_points: np.ndarray,
    config: HybridIterRefineConfig,
    seed: int,
    command_args: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = initialization["sites"].detach().cpu()
    sites_sdf = initialization["sites_sdf"].detach().cpu()
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites.shape[0]),
        "base_site_count": int(sites.shape[0]),
        "round_count": 0,
        "positive_sdf_count": int((sites_sdf > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf < 0).sum().item()),
        "initialization_valid": bool(initialization["valid"]),
        "initialization_reason": str(initialization["reason"]),
        "initialization": dict(initialization["diagnostics"]),
    }
    field_file = output_dir / f"{mesh_id}_hybrid_iter_refine_initial_field.npz"
    np.savez_compressed(
        field_file,
        sites=sites.numpy().astype(np.float32),
        sites_sdf=sites_sdf.numpy().astype(np.float32),
        base_sites=sites.numpy().astype(np.float32),
        base_sites_sdf=sites_sdf.numpy().astype(np.float32),
        background_sites=initialization["background_sites"].detach().cpu().numpy().astype(np.float32),
        background_sites_sdf=initialization["background_sdf"].detach().cpu().numpy().astype(np.float32),
        surface_anchors=initialization["surface_anchors"].detach().cpu().numpy().astype(np.float32),
        surface_sites=initialization["surface_sites"].detach().cpu().numpy().astype(np.float32),
        surface_sites_sdf=initialization["surface_sdf"].detach().cpu().numpy().astype(np.float32),
        input_grid=input_grid.astype(np.float32),
        sdf_grid=sdf_grid.astype(np.float32),
        target_points=target_points.astype(np.float32),
        diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
        resolved_config=np.array(json.dumps(config.to_dict(), sort_keys=True)),
        command_args=np.array(json.dumps(command_args, sort_keys=True)),
        seed=np.array(int(seed), dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )
    print(f"Saved iterative refinement initialization field: {field_file}")
    print(f"Diagnostics: {diagnostics}")
    return field_file


def extract_initialization_cache(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    config: HybridIterRefineConfig,
    seed: int = 69,
    extract: bool = True,
    overwrite: bool = False,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    command_args: Optional[dict[str, Any]] = None,
    state: str = "hybrid_iter_refine_initial",
) -> dict[str, Any]:
    """Export the HotSpot near-surface initialization for one cache."""
    output_path = Path(output_dir)
    int_obj = _expected_initial_obj_path(
        output_path,
        state=state,
        variant="intDCCVT",
        w_cvt=w_cvt,
        w_sdfsmooth=w_sdfsmooth,
    )
    proj_obj = _expected_initial_obj_path(
        output_path,
        state=state,
        variant="projDCCVT",
        w_cvt=w_cvt,
        w_sdfsmooth=w_sdfsmooth,
    )
    field_file = output_path / f"{Path(cache_path).stem}_hybrid_iter_refine_initial_field.npz"
    if not overwrite and field_file.exists() and (not extract or (int_obj.exists() and proj_obj.exists())):
        return {
            "cache_path": str(cache_path),
            "output_dir": str(output_path),
            "field_file": str(field_file),
            "status": "skipped_existing",
        }

    cache = load_npz_cache(cache_path)
    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = cache_mesh_id(cache, cache_path)
    if grid_n != config.hotspot_grid_n:
        raise ValueError(f"Cache grid_n={grid_n} does not match config hotspot_grid_n={config.hotspot_grid_n}")

    input_grid_np = build_hybrid_input_channels_np(
        sdf_grid_np,
        target_points_np,
        grid_n=grid_n,
        udf_clip=config.point_udf_clip,
        confidence_sigma_scale=config.point_confidence_sigma_scale,
        channel_names=config.channel_names,
    )
    initialization = build_hotspot_near_surface_initialization(sdf_grid_np, config)
    output_path.mkdir(parents=True, exist_ok=True)
    field_file = _save_initialization_field(
        output_path,
        mesh_id=mesh_id,
        initialization=initialization,
        input_grid=input_grid_np,
        sdf_grid=sdf_grid_np,
        target_points=target_points_np,
        config=config,
        seed=seed,
        command_args=command_args or {},
    )

    sites_cpu = initialization["sites"].detach().cpu()
    sites_sdf_cpu = initialization["sites_sdf"].detach().cpu()
    positive_count = int((sites_sdf_cpu > 0).sum().item())
    negative_count = int((sites_sdf_cpu < 0).sum().item())
    can_extract = (
        extract
        and bool(initialization["valid"])
        and sites_cpu.shape[0] >= 5
        and positive_count > 0
        and negative_count > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_path),
            upsampling=0,
            w_cvt=w_cvt,
            w_sdfsmooth=w_sdfsmooth,
        )
        extract_mesh(
            sites_cpu.to(dccvt_device),
            sites_sdf_cpu.to(dccvt_device),
            target_pc,
            0.0,
            args,
            state=state,
        )
    elif extract:
        print("Skipping DCCVT extraction: need valid initialization and both positive/negative SDF values.")

    return {
        "cache_path": str(cache_path),
        "output_dir": str(output_path),
        "field_file": str(field_file),
        "status": "extracted" if can_extract else "saved_field",
        "extracted": bool(can_extract),
        "site_count": int(sites_cpu.shape[0]),
        "initialization_valid": bool(initialization["valid"]),
        "initialization_reason": str(initialization["reason"]),
    }


def run_initialization_extraction(
    *,
    config: HybridIterRefineConfig,
    cache_root: str | Path,
    output_root: str | Path,
    split_file: str | Path | None = None,
    mesh_ids: Sequence[str] | None = None,
    seed: int = 69,
    extract: bool = True,
    overwrite: bool = False,
    fail_fast: bool = False,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    command_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Export initialization fields and optional meshes for a cache set."""
    seed_everything(seed)
    cache_files = resolve_cache_files(cache_root, mesh_ids=mesh_ids, split_file=split_file)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resolved_payload = {
        "model_config": config.to_dict(),
        "seed": int(seed),
        "args": command_args or {},
    }
    (output_root_path / "resolved_config.json").write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for cache_path in cache_files:
        mesh_id = Path(cache_path).stem
        output_dir = output_root_path / mesh_id
        try:
            result = extract_initialization_cache(
                cache_path,
                output_dir,
                config=config,
                seed=seed,
                extract=extract,
                overwrite=overwrite,
                w_cvt=w_cvt,
                w_sdfsmooth=w_sdfsmooth,
                command_args=command_args or {},
            )
        except Exception as exc:
            result = {
                "cache_path": str(cache_path),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"Failed iterative refinement initialization extraction for {mesh_id}: {exc}")
            if fail_fast:
                raise
        results.append(result)

    summary_path = output_root_path / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved iterative refinement initialization summary: {summary_path}")
    return results


def build_initial_extract_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract iterative-refinement HotSpot near-surface initialization.")
    parser.add_argument("--config", default="configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json")
    parser.add_argument("--cache-root", default="outputs/neural_hotspot_sdf/thingi32_g33")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids.")
    parser.add_argument("--output-root", default="outputs/neural_dccvt/hybrid_iter_refine_initial_v2_hotspot_point_udf")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_initial_extract_arg_parser().parse_args(argv)
    config = load_iter_refine_config(args.config)
    run_initialization_extraction(
        config=config,
        cache_root=args.cache_root,
        output_root=args.output_root,
        split_file=args.split_file,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        seed=args.seed,
        extract=not args.no_extract,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        command_args=vars(args),
    )


initial_extract_main = main
