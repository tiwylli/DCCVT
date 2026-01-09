"""Experiment runner utilities for per-mesh execution."""

import os
from time import time
from typing import Any, List

import torch

from dccvt import argparse_utils as config_utils
from dccvt.argparse_utils import parse_experiment_args
from dccvt.mesh_ops import extract_mesh
from dccvt.model_utils import init_sdf_from_model, init_sites_from_mnfld_points, load_hotspot_model
from dccvt.paths import make_dccvt_obj_path
from dccvt.training import run_dccvt_training


def run_single_mesh_experiment(arg_list: List[str]) -> None:
    """Run a single mesh experiment from a parsed argv list."""
    args = parse_experiment_args(arg_list, defaults=config_utils.DEFAULTS)
    args.save_path = f"{args.output}" if args.save_path is None else args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    use_chamfer = args.w_chamfer > 0
    use_training = use_chamfer or args.w_cvt > 0 or args.w_sdfsmooth > 0

    output_files = expected_output_files(args)
    if output_files and all(os.path.exists(path) for path in output_files):
        print(f"Skipping already processed mesh: {args.mesh}")
        return

    print("args: ", args)
    try:
        model, mnfld_points = load_hotspot_model(
            mesh_path=args.mesh,
            max_amount_sites=args.max_amount_sites,
            hotspot_weights_path=args.trained_HotSpot,
        )
        sites = init_sites_from_mnfld_points(
            mnfld_points=mnfld_points,
            num_centroids=args.num_centroids,
            sample_near=args.sample_near,
        )

        if use_chamfer:
            sdf = init_sdf_from_model(model, sites)
        else:
            sdf = model

        # Extract the initial mesh
        extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

        elapsed = 0.0
        if use_training:
            t0 = time()
            sites, sdf = run_dccvt_training(sites, sdf, mnfld_points, model, args)
            elapsed = time() - t0

        # Extract the final mesh
        extract_mesh(sites, sdf, mnfld_points, elapsed, args, state="final")
    except Exception as e:
        print(f"Error processing mesh {args.mesh}: {e}")
    else:
        print(f"Finished processing mesh: {args.mesh}")
        torch.cuda.empty_cache()


def expected_output_files(args: Any) -> List[str]:
    state = "final"
    outputs: List[str] = []
    if args.w_chamfer > 0:
        outputs.append(make_dccvt_obj_path(args, state, "projDCCVT"))
    return outputs
