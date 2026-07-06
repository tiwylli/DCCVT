"""Inference CLI for iterative learned sparse refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

from dccvt.neural.data.point_udf_sidecar import load_point_udf_sidecar, point_udf_sidecar_path
from dccvt.neural.grid import build_hybrid_input_channels_np
from dccvt.neural.iterative.config import HybridIterRefineConfig
from dccvt.neural.iterative.initialization import build_hotspot_near_surface_initialization
from dccvt.neural.iterative.model import DCCVTHybridIterRefineNet
from dccvt.neural.utils import cache_mesh_id, device_from_value, load_npz_cache, seed_everything

def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[DCCVTHybridIterRefineNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = HybridIterRefineConfig.from_dict(checkpoint["model_config"])
    model = DCCVTHybridIterRefineNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _save_prediction(
    output_dir: Path,
    *,
    mesh_id: str,
    outputs: dict[str, Any],
    input_grid: np.ndarray,
    sdf_grid: np.ndarray,
    target_points: np.ndarray,
    checkpoint: dict[str, Any],
    command_args: dict[str, Any],
    local_udf_path: str = "",
    local_udf_valid: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = outputs["sites"][0].detach().cpu()
    sites_sdf = outputs["sites_sdf"][0].detach().cpu()
    initialization_diagnostics = dict(outputs["initialization_diagnostics"])
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites.shape[0]),
        "base_site_count": int(outputs["base_sites"].shape[0]),
        "round_count": int(len(outputs["rounds"])),
        "positive_sdf_count": int((sites_sdf > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf < 0).sum().item()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "initialization_valid": bool(outputs["initialization_valid"]),
        "initialization_reason": str(outputs["initialization_reason"]),
        "initialization": initialization_diagnostics,
        "architecture": str(checkpoint["model_config"].get("architecture", "dense_cnn")),
        "local_feature_mode": str(checkpoint["model_config"].get("local_feature_mode", "none")),
        "local_udf_grid_n": int(checkpoint["model_config"].get("local_udf_grid_n", 0)),
        "local_udf_samples": bool(checkpoint["model_config"].get("local_udf_samples", False)),
        "local_udf_path": local_udf_path,
        "local_udf_valid": bool(local_udf_valid),
        "local_knn_features": bool(checkpoint["model_config"].get("local_knn_features", False)),
        "graph_layers": int(checkpoint["model_config"].get("graph_layers", 0)),
        "site_position_encoding": str(checkpoint["model_config"].get("site_position_encoding", "fourier")),
        "graph_edge_features": str(
            checkpoint["model_config"].get("graph_edge_features", "relative_xyz_distance_direction_sdf_delta")
        ),
    }
    arrays: dict[str, Any] = {
        "sites": sites.numpy().astype(np.float32),
        "sites_sdf": sites_sdf.numpy().astype(np.float32),
        "base_sites": outputs["base_sites"].detach().cpu().numpy().astype(np.float32),
        "base_sites_sdf": outputs["base_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "background_sites": outputs["background_sites"].detach().cpu().numpy().astype(np.float32),
        "background_sites_sdf": outputs["background_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "surface_anchors": outputs["surface_anchors"].detach().cpu().numpy().astype(np.float32),
        "surface_sites": outputs["surface_sites"].detach().cpu().numpy().astype(np.float32),
        "surface_sites_sdf": outputs["surface_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "input_grid": input_grid.astype(np.float32),
        "sdf_grid": sdf_grid.astype(np.float32),
        "target_points": target_points.astype(np.float32),
        "local_udf_path": np.array(local_udf_path),
        "local_udf_valid": np.array(bool(local_udf_valid)),
        "diagnostics": np.array(json.dumps(diagnostics, sort_keys=True)),
        "resolved_config": np.array(json.dumps(checkpoint["model_config"], sort_keys=True)),
        "command_args": np.array(json.dumps(command_args, sort_keys=True)),
        "seed": np.array(int(checkpoint.get("seed", command_args.get("seed", 69))), dtype=np.int64),
        "mesh_id": np.array(mesh_id),
    }
    for round_index, round_data in enumerate(outputs["rounds"]):
        prefix = f"round_{round_index:02d}"
        arrays[f"{prefix}_parent_indices"] = round_data["parent_indices"].detach().cpu().numpy().astype(np.int64)
        arrays[f"{prefix}_parent_scores"] = round_data["parent_scores"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_spawned_sites"] = round_data["spawned_sites"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_spawned_sdf"] = round_data["spawned_sdf"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_rejected_spawn_count"] = np.array(
            int(round_data["rejected_spawn_count"].item()), dtype=np.int64
        )

    prediction_file = output_dir / f"{mesh_id}_hybrid_iter_refine_prediction.npz"
    np.savez_compressed(prediction_file, **arrays)
    print(f"Saved iterative refinement prediction: {prediction_file}")
    print(f"Diagnostics: {diagnostics}")
    return prediction_file


def run_inference(
    *,
    checkpoint_path: str | Path,
    cache_path: str | Path,
    output_dir: str | Path,
    device_value: str = "auto",
    local_udf_root: str | Path | None = None,
    allow_missing_local_features: bool = False,
    extract: bool = True,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    seed: int = 69,
    command_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    device = device_from_value(device_value)
    seed_everything(seed)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    cache = load_npz_cache(cache_path)

    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())
    if grid_n != model.config_obj.hotspot_grid_n:
        raise ValueError(f"Cache grid_n={grid_n} does not match model hotspot_grid_n={model.config_obj.hotspot_grid_n}")

    input_grid_np = build_hybrid_input_channels_np(
        sdf_grid_np,
        target_points_np,
        grid_n=grid_n,
        udf_clip=model.config_obj.point_udf_clip,
        confidence_sigma_scale=model.config_obj.point_confidence_sigma_scale,
        channel_names=model.config_obj.channel_names,
    )
    input_grid = torch.from_numpy(input_grid_np[None, ...]).to(device)
    sdf_grid = torch.from_numpy(sdf_grid_np[None, None, ...]).to(device)
    target_points = torch.from_numpy(target_points_np[None, ...]).to(device)
    local_udf_grid = None
    local_udf_path = ""
    local_udf_valid = False
    if model.config_obj.local_udf_samples:
        if local_udf_root is None:
            if not allow_missing_local_features:
                raise ValueError("Checkpoint config requests local UDF samples; provide --local-udf-root")
            grid_n = model.config_obj.local_udf_grid_n
            local_udf_np = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        else:
            sidecar_path = point_udf_sidecar_path(local_udf_root, Path(cache_path).stem)
            local_udf_path = str(sidecar_path)
            if not sidecar_path.exists():
                if not allow_missing_local_features:
                    raise FileNotFoundError(f"Missing local point-UDF sidecar: {sidecar_path}")
                grid_n = model.config_obj.local_udf_grid_n
                local_udf_np = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
            else:
                local_udf_np = load_point_udf_sidecar(sidecar_path, grid_n=model.config_obj.local_udf_grid_n)
                local_udf_valid = True
        local_udf_grid = torch.from_numpy(local_udf_np[None, None, ...]).to(device)
    initial_field = build_hotspot_near_surface_initialization(sdf_grid_np, model.config_obj)
    for name in (
        "sites",
        "sites_sdf",
        "background_sites",
        "background_sdf",
        "surface_anchors",
        "surface_sites",
        "surface_sdf",
    ):
        initial_field[name] = initial_field[name].to(device=device, dtype=input_grid.dtype)
    with torch.no_grad():
        outputs = model(
            input_grid,
            sdf_grid,
            initial_field=initial_field,
            target_points=target_points,
            local_udf_grid=local_udf_grid,
        )

    output_path = Path(output_dir)
    prediction_file = _save_prediction(
        output_path,
        mesh_id=mesh_id,
        outputs=outputs,
        input_grid=input_grid_np,
        sdf_grid=sdf_grid_np,
        target_points=target_points_np,
        checkpoint=checkpoint,
        command_args=command_args or {},
        local_udf_path=local_udf_path,
        local_udf_valid=local_udf_valid,
    )

    sites_cpu = outputs["sites"][0].detach().cpu()
    sites_sdf_cpu = outputs["sites_sdf"][0].detach().cpu()
    can_extract = (
        extract
        and bool(outputs["initialization_valid"])
        and sites_cpu.shape[0] >= 5
        and int((sites_sdf_cpu > 0).sum().item()) > 0
        and int((sites_sdf_cpu < 0).sum().item()) > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_path),
            upsampling=model.config_obj.num_refinement_rounds,
            w_cvt=w_cvt,
            w_sdfsmooth=w_sdfsmooth,
        )
        extract_mesh(
            sites_cpu.to(dccvt_device),
            sites_sdf_cpu.to(dccvt_device),
            target_pc,
            0.0,
            args,
            state="hybrid_iter_refine",
        )
    elif extract:
        print("Skipping DCCVT extraction: need at least 5 sites and both positive/negative SDF values.")

    result = {
        "prediction_file": str(prediction_file),
        "extracted": bool(can_extract),
        "site_count": int(sites_cpu.shape[0]),
        "initialization_valid": bool(outputs["initialization_valid"]),
        "initialization_reason": str(outputs["initialization_reason"]),
    }
    result_path = output_path / "inference_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved iterative refinement inference result: {result_path}")
    return result


def build_infer_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run iterative learned sparse-refinement inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--local-udf-root", default=None)
    parser.add_argument("--allow-missing-local-features", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_infer_arg_parser().parse_args(argv)
    run_inference(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output_dir=args.output_dir,
        device_value=args.device,
        local_udf_root=args.local_udf_root,
        allow_missing_local_features=args.allow_missing_local_features,
        extract=not args.no_extract,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        seed=args.seed,
        command_args=vars(args),
    )



infer_main = main
