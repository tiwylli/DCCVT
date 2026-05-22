"""Inference bridge from neural predictions to DCCVT mesh extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from dccvt.neural.dataset import read_point_cloud, sample_points
from dccvt.neural.models import PointNetDCCVT


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> PointNetDCCVT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    if not model_config:
        # Backward compatibility for early n16 checkpoints created before model_config existed.
        model_config = {"num_centroids": 16}
    model = PointNetDCCVT(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_generators(
    *,
    checkpoint_path: str | Path,
    point_path: str | Path,
    num_points: int = 9600,
    seed: int = 0,
    device: str = "auto",
    report_sdf_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict sites and SDF values from one point cloud."""
    resolved_device = _resolve_device(device)
    model = load_checkpoint_model(checkpoint_path, resolved_device)
    points_np = sample_points(read_point_cloud(point_path), num_points, seed)
    points = torch.from_numpy(points_np).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        pred = model(points)
    sdf = pred["sites_sdf"][0].detach()
    if report_sdf_stats:
        _print_sdf_stats(sdf, "Predicted")
    return pred["sites"][0].detach(), sdf, points[0].detach()


def _resolve_hotspot_sdf(
    *,
    sites: torch.Tensor,
    mesh_path: str | Path,
    hotspot_path: str | Path,
    max_amount_sites: int,
) -> torch.Tensor:
    from dccvt.device import device as dccvt_device
    from dccvt.model_utils import load_hotspot_model, resolve_sdf_values

    model, _ = load_hotspot_model(
        mesh_path=str(mesh_path),
        max_amount_sites=max_amount_sites,
        hotspot_weights_path=str(hotspot_path),
    )
    model.eval()
    with torch.no_grad():
        sdf = resolve_sdf_values(model, sites.to(dccvt_device), verbose=True).detach()
    return sdf


def _print_sdf_stats(sdf: torch.Tensor, label: str) -> None:
    neg_count = int((sdf < 0).sum().item())
    zero_count = int((sdf == 0).sum().item())
    pos_count = int((sdf > 0).sum().item())
    print(
        f"{label} SDF stats: "
        f"min={sdf.min().item():.6f} max={sdf.max().item():.6f} "
        f"neg={neg_count} zero={zero_count} pos={pos_count}"
    )


def export_prediction_mesh(
    *,
    checkpoint_path: str | Path,
    point_path: str | Path,
    output_dir: str | Path,
    num_points: int = 9600,
    seed: int = 0,
    device: str = "auto",
    state: str = "pred",
    sdf_source: str = "predicted",
    mesh_path: Optional[str | Path] = None,
    hotspot_path: Optional[str | Path] = None,
    max_amount_sites: int = 32,
) -> dict:
    """Predict generators and write DCCVT mesh artifacts."""
    sites, sites_sdf, sampled_points = predict_generators(
        checkpoint_path=checkpoint_path,
        point_path=point_path,
        num_points=num_points,
        seed=seed,
        device=device,
        report_sdf_stats=sdf_source == "predicted",
    )

    from dccvt.device import device as dccvt_device
    from dccvt.device import initialize_runtime
    from dccvt.mesh_ops import extract_mesh

    initialize_runtime(seed)
    if sdf_source == "hotspot":
        if mesh_path is None or hotspot_path is None:
            raise ValueError("`mesh_path` and `hotspot_path` are required when sdf_source='hotspot'")
        sites_sdf = _resolve_hotspot_sdf(
            sites=sites,
            mesh_path=mesh_path,
            hotspot_path=hotspot_path,
            max_amount_sites=max_amount_sites,
        )
        _print_sdf_stats(sites_sdf, "HotSpot")
    elif sdf_source != "predicted":
        raise ValueError(f"Unsupported SDF source: {sdf_source!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(save_path=str(output_dir), upsampling=0, w_cvt=0, w_sdfsmooth=0)
    extract_mesh(
        sites.to(dccvt_device),
        sites_sdf.to(dccvt_device),
        sampled_points.unsqueeze(0).to(dccvt_device),
        0.0,
        args,
        state=state,
    )
    return {"sites": sites, "sites_sdf": sites_sdf, "output_dir": output_dir}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DCCVT neural generator inference and mesh extraction.")
    parser.add_argument("--checkpoint", required=True, help="Path to a neural checkpoint.")
    parser.add_argument("--point-cloud", required=True, help="Input PLY/OBJ point cloud or mesh.")
    parser.add_argument("--output-dir", required=True, help="Directory for predicted DCCVT mesh artifacts.")
    parser.add_argument("--num-points", type=int, default=9600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Network device. DCCVT extraction uses DCCVT_DEVICE if set.")
    parser.add_argument("--state", default="pred")
    parser.add_argument(
        "--sdf-source",
        choices=("predicted", "hotspot"),
        default="predicted",
        help="Use neural SDF predictions or evaluate a HotSpot model at predicted sites.",
    )
    parser.add_argument("--mesh", default=None, help="Mesh path used to load the HotSpot dataset when --sdf-source=hotspot.")
    parser.add_argument("--hotspot", default=None, help="HotSpot checkpoint path when --sdf-source=hotspot.")
    parser.add_argument("--max-amount-sites", type=int, default=32)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.sdf_source == "hotspot" and (args.mesh is None or args.hotspot is None):
        parser.error("--mesh and --hotspot are required when --sdf-source=hotspot")
    result = export_prediction_mesh(
        checkpoint_path=args.checkpoint,
        point_path=args.point_cloud,
        output_dir=args.output_dir,
        num_points=args.num_points,
        seed=args.seed,
        device=args.device,
        state=args.state,
        sdf_source=args.sdf_source,
        mesh_path=args.mesh,
        hotspot_path=args.hotspot,
        max_amount_sites=args.max_amount_sites,
    )
    print(f"wrote prediction artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
