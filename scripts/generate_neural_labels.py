#!/usr/bin/env python3
"""Generate fixed-size supervised labels for the DCCVT neural prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.argparse_utils import DEFAULTS


def _parse_mesh_ids(value: Optional[str]) -> list[str]:
    if not value:
        return list(DEFAULTS["mesh_ids"])
    return [part for part in value.replace(",", " ").split() if part]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DCCVT `.npz` labels for neural generator training.")
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids. Defaults to DCCVT ids.")
    parser.add_argument("--mesh-root", default=str(Path(DEFAULTS["mesh"]) ), help="Root containing <mesh_id>.ply/.obj.")
    parser.add_argument("--hotspot-root", default=str(Path(DEFAULTS["trained_HotSpot"]) / "thingi32"))
    parser.add_argument("--output-root", default="outputs/neural_labels/n32")
    parser.add_argument("--num-iterations", type=int, default=DEFAULTS["num_iterations"])
    parser.add_argument("--num-centroids", type=int, default=32)
    parser.add_argument("--max-amount-sites", type=int, default=DEFAULTS["max_amount_sites"])
    parser.add_argument("--w-chamfer", type=float, default=DEFAULTS["w_chamfer"])
    parser.add_argument("--w-cvt", type=float, default=DEFAULTS["w_cvt"])
    parser.add_argument("--w-sdfsmooth", type=float, default=DEFAULTS["w_sdfsmooth"])
    parser.add_argument("--lr-sites", type=float, default=DEFAULTS["lr_sites"])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--overwrite", action="store_true", help="Re-run meshes even if final labels exist.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from dccvt.api import run_mesh_from_params
    from dccvt.device import seed_everything

    mesh_ids = _parse_mesh_ids(args.mesh_ids)
    mesh_root = Path(args.mesh_root)
    hotspot_root = Path(args.hotspot_root)
    output_root = Path(args.output_root)

    for index, mesh_id in enumerate(mesh_ids):
        seed_everything(args.seed + index)
        output_dir = output_root / mesh_id
        final_npz = output_dir / f"DCCVT_0_final_projDCCVT_cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.npz"
        if final_npz.exists() and not args.overwrite:
            print(f"Skipping existing label: {final_npz}")
            continue

        print(f"Generating neural label for mesh {mesh_id} -> {output_dir}")
        run_mesh_from_params(
            mesh=str(mesh_root / mesh_id),
            trained_HotSpot=str(hotspot_root / f"{mesh_id}.pth"),
            output=str(output_dir),
            num_iterations=args.num_iterations,
            num_centroids=args.num_centroids,
            sample_near=0,
            max_amount_sites=args.max_amount_sites,
            video=False,
            w_cvt=args.w_cvt,
            w_sdfsmooth=args.w_sdfsmooth,
            w_chamfer=args.w_chamfer,
            upsampling=0,
            lr_sites=args.lr_sites,
            skip_existing=not args.overwrite,
        )


if __name__ == "__main__":
    main()
