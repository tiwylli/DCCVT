"""Inference and DCCVT extraction for the PoNQ-style neural site predictor."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from dccvt.neural.grid import make_gt_activity_mask_np, make_near_surface_mask_np, trilinear_interpolate_sdf
from dccvt.neural.models import DCCVTPoNQNet
from dccvt.neural.utils import device_from_value


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[DCCVTPoNQNet, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = DCCVTPoNQNet(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _load_cache(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        record = {key: data[key] for key in data.files}
    if "near_surface_mask" not in record:
        record["near_surface_mask"] = make_near_surface_mask_np(record["sdf_grid"])
    if "gt_activity_mask" not in record and "target_points" in record:
        record["gt_activity_mask"] = make_gt_activity_mask_np(record["target_points"], int(record["grid_n"]))
    return record


def _select_active_cells(
    activity: torch.Tensor,
    near_surface_mask: torch.Tensor,
    *,
    threshold: float,
    max_sites: Optional[int],
    k: int,
    fallback_cells: int,
    selection_mode: str,
) -> torch.Tensor:
    candidate_mask = near_surface_mask.bool()
    if selection_mode == "activity":
        active = (activity >= threshold) & candidate_mask
    elif selection_mode == "near-surface":
        active = candidate_mask.clone()
    elif selection_mode == "topk":
        if max_sites is None:
            raise ValueError("--selection-mode topk requires --max-sites")
        active = torch.zeros_like(candidate_mask, dtype=torch.bool)
        candidates = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            candidates = torch.arange(activity.numel(), device=activity.device)
        max_cells = max(1, int(np.ceil(max_sites / float(k))))
        keep = min(max_cells, candidates.numel())
        top = candidates[torch.topk(activity[candidates], k=keep, largest=True).indices]
        active[top] = True
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    if not active.any():
        candidates = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            candidates = torch.arange(activity.numel(), device=activity.device)
        keep = min(fallback_cells, candidates.numel())
        top = candidates[torch.topk(activity[candidates], k=keep, largest=True).indices]
        active = torch.zeros_like(candidate_mask, dtype=torch.bool)
        active[top] = True

    if max_sites is not None:
        max_cells = max(1, int(np.ceil(max_sites / float(k))))
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_idx.numel() > max_cells:
            top = active_idx[torch.topk(activity[active_idx], k=max_cells, largest=True).indices]
            reduced = torch.zeros_like(active, dtype=torch.bool)
            reduced[top] = True
            active = reduced
    return active


def run_inference(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    cache_path: Optional[str | Path] = None,
    cache_record: Optional[dict] = None,
    activity_threshold: float = 0.5,
    max_sites: Optional[int] = None,
    fallback_cells: int = 64,
    device_value: str = "auto",
    extract: bool = True,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    selection_mode: str = "activity",
) -> dict:
    device = device_from_value(device_value)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    if cache_record is None:
        if cache_path is None:
            raise ValueError("Either `cache_path` or `cache_record` is required")
        cache_record = _load_cache(cache_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf_grid_np = np.asarray(cache_record["sdf_grid"], dtype=np.float32)
    near_surface_np = np.asarray(cache_record["near_surface_mask"], dtype=bool)
    target_points_np = np.asarray(cache_record.get("target_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    mesh_id = str(np.asarray(cache_record.get("mesh_id", np.array("shape"))).item())

    sdf_grid = torch.from_numpy(sdf_grid_np[None, None, ...]).to(device)
    near_surface = torch.from_numpy(near_surface_np).to(device)
    with torch.no_grad():
        outputs = model(sdf_grid)
        activity = outputs["activity"][0]
        active_cells = _select_active_cells(
            activity,
            near_surface,
            threshold=activity_threshold,
            max_sites=max_sites,
            k=model.k,
            fallback_cells=fallback_cells,
            selection_mode=selection_mode,
        )
        sites = outputs["sites"][0, active_cells].reshape(-1, 3)
        sites_sdf = trilinear_interpolate_sdf(sdf_grid[0], sites).reshape(-1)

    sites_cpu = sites.detach().cpu()
    sites_sdf_cpu = sites_sdf.detach().cpu()
    active_cells_cpu = active_cells.detach().cpu()
    activity_cpu = activity.detach().cpu()
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites_cpu.shape[0]),
        "active_cell_count": int(active_cells_cpu.sum().item()),
        "positive_sdf_count": int((sites_sdf_cpu > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf_cpu < 0).sum().item()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "selection_mode": selection_mode,
    }

    prediction_file = output_dir / f"{mesh_id}_neural_dccvt_prediction.npz"
    np.savez_compressed(
        prediction_file,
        sites=sites_cpu.numpy().astype(np.float32),
        sites_sdf=sites_sdf_cpu.numpy().astype(np.float32),
        active_cell_mask=active_cells_cpu.numpy().astype(bool),
        activity=activity_cpu.numpy().astype(np.float32),
        near_surface_mask=near_surface_np.astype(bool),
        sdf_grid=sdf_grid_np.astype(np.float32),
        diagnostics=np.array(str(diagnostics)),
        mesh_id=np.array(mesh_id),
    )
    print(f"Saved neural prediction: {prediction_file}")
    print(f"Diagnostics: {diagnostics}")

    can_extract = (
        extract
        and sites_cpu.shape[0] >= 5
        and diagnostics["positive_sdf_count"] > 0
        and diagnostics["negative_sdf_count"] > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime()
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_dir),
            upsampling=0,
            w_cvt=w_cvt,
            w_sdfsmooth=w_sdfsmooth,
        )
        extract_mesh(sites.to(dccvt_device), sites_sdf.to(dccvt_device), target_pc, 0.0, args, state="neural")
    elif extract:
        print("Skipping DCCVT extraction: need at least 5 sites and both positive/negative SDF samples.")

    return {
        "prediction_file": str(prediction_file),
        "diagnostics": diagnostics,
        "extracted": bool(can_extract),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run neural DCCVT inference and optional extraction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default=None, help="Precomputed HotSpot SDF .npz cache.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--selection-mode",
        choices=("activity", "near-surface", "topk"),
        default="activity",
        help="How to choose active cells before DCCVT extraction.",
    )
    parser.add_argument("--max-sites", type=int, default=None)
    parser.add_argument("--fallback-cells", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)

    parser.add_argument("--mesh", default=None, help="Optional mesh path/stem for on-the-fly cache creation.")
    parser.add_argument("--hotspot-weights", default=None, help="Optional HotSpot .pth for on-the-fly cache creation.")
    parser.add_argument("--grid-n", type=int, default=33)
    parser.add_argument("--sample-count", type=int, default=200_000)
    parser.add_argument("--max-amount-sites", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=65_536)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cache_record = None
    if args.cache is None:
        if args.mesh is None or args.hotspot_weights is None:
            raise ValueError("Provide --cache, or provide both --mesh and --hotspot-weights.")
        from dccvt.neural.data.precompute import build_hotspot_cache_record

        cache_record = build_hotspot_cache_record(
            mesh_path=args.mesh,
            hotspot_weights_path=args.hotspot_weights,
            mesh_id=Path(args.mesh).stem,
            grid_n=args.grid_n,
            sample_count=args.sample_count,
            max_amount_sites=args.max_amount_sites,
            query_batch_size=args.query_batch_size,
        )

    run_inference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        cache_path=args.cache,
        cache_record=cache_record,
        activity_threshold=args.activity_threshold,
        max_sites=args.max_sites,
        fallback_cells=args.fallback_cells,
        device_value=args.device,
        extract=not args.no_extract,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        selection_mode=args.selection_mode,
    )


if __name__ == "__main__":
    main()
