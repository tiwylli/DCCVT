"""Inference for the hybrid direct PoNQ-DCCVT extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from dccvt.neural.grid import build_hybrid_input_channels_np
from dccvt.neural.models import DCCVTHybridDirectNet, HybridDirectConfig
from dccvt.neural.utils import device_from_value, load_npz_cache


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[DCCVTHybridDirectNet, dict]:
    checkpoint = torch.load(path, map_location=device)
    config = HybridDirectConfig.from_dict(checkpoint["model_config"])
    model = DCCVTHybridDirectNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_inference(
    *,
    checkpoint_path: str | Path,
    cache_path: str | Path,
    output_dir: str | Path,
    device_value: str = "auto",
    extract: bool = True,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    seed: int = 69,
    command_args: Optional[dict] = None,
) -> dict:
    device = device_from_value(device_value)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    cache = load_npz_cache(cache_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())

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
    with torch.no_grad():
        outputs = model(input_grid, sdf_grid)

    sites_cpu = outputs["sites"][0].detach().cpu()
    sites_sdf_cpu = outputs["sites_sdf"][0].detach().cpu()
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites_cpu.shape[0]),
        "positive_sdf_count": int((sites_sdf_cpu > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf_cpu < 0).sum().item()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "seed": int(seed),
        "channel_names": list(model.config_obj.channel_names),
    }

    prediction_file = output_dir / f"{mesh_id}_hybrid_direct_prediction.npz"
    np.savez_compressed(
        prediction_file,
        sites=sites_cpu.numpy().astype(np.float32),
        sites_sdf=sites_sdf_cpu.numpy().astype(np.float32),
        site_delta=outputs["site_delta"][0].detach().cpu().numpy().astype(np.float32),
        sdf_residual=outputs["sdf_residual"][0].detach().cpu().numpy().astype(np.float32),
        hotspot_sdf_at_sites=outputs["hotspot_sdf_at_sites"][0].detach().cpu().numpy().astype(np.float32),
        canonical_sites=outputs["canonical_sites"].detach().cpu().numpy().astype(np.float32),
        input_grid=input_grid_np.astype(np.float32),
        sdf_grid=sdf_grid_np.astype(np.float32),
        target_points=target_points_np.astype(np.float32),
        resolved_config=np.array(json.dumps(model.config())),
        channel_names=np.array(list(model.config_obj.channel_names)),
        diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
        command_args=np.array(json.dumps(command_args or {}, sort_keys=True)),
        seed=np.array(seed, dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )
    print(f"Saved hybrid direct prediction: {prediction_file}")
    print(f"Diagnostics: {diagnostics}")

    can_extract = extract and sites_cpu.shape[0] >= 5 and diagnostics["positive_sdf_count"] > 0 and diagnostics["negative_sdf_count"] > 0
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_dir),
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
            state="hybrid_direct",
        )
    elif extract:
        print("Skipping DCCVT extraction: need at least 5 sites and both positive/negative SDF values.")

    return {
        "prediction_file": str(prediction_file),
        "diagnostics": diagnostics,
        "extracted": bool(can_extract),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid direct DCCVT inference and optional extraction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True, help="Precomputed HotSpot SDF .npz cache.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_inference(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output_dir=args.output_dir,
        device_value=args.device,
        extract=not args.no_extract,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        seed=args.seed,
        command_args=vars(args),
    )


if __name__ == "__main__":
    main()
