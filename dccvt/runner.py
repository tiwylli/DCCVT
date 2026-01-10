"""Experiment runner utilities for per-mesh execution."""

import os
from time import time
from typing import Any, Dict, List

import torch

from dccvt import argparse_utils as config_utils
from dccvt.argparse_utils import parse_experiment_args
from dccvt.device import initialize_runtime
from dccvt.mesh_ops import extract_mesh
from dccvt.model_utils import init_sdf_from_model, init_sites_from_mnfld_points, load_hotspot_model
from dccvt.paths import make_dccvt_obj_path
from dccvt.training import run_dccvt_training


def run_mesh(args: Any, *, skip_existing: bool = True) -> Dict[str, Any]:
    """Run a single mesh experiment from a populated args namespace."""
    args.save_path = f"{args.output}" if args.save_path is None else args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    output_files = expected_output_files(args)
    if skip_existing and output_files and all(os.path.exists(path) for path in output_files):
        print(f"Skipping already processed mesh: {args.mesh}")
        return {
            "skipped": True,
            "args": args,
            "output_files": output_files,
            "output_dir": args.save_path,
        }

    print("args: ", args)
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

    sdf = init_sdf_from_model(model, sites)

    # Extract the initial mesh
    extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

    t0 = time()
    sites, sdf = run_dccvt_training(sites, sdf, mnfld_points, args)
    elapsed = time() - t0

    # Extract the final mesh
    extract_mesh(sites, sdf, mnfld_points, elapsed, args, state="final")

    print(f"Finished processing mesh: {args.mesh}")
    torch.cuda.empty_cache()

    return {
        "skipped": False,
        "args": args,
        "sites": sites,
        "sdf": sdf,
        "output_files": output_files,
        "output_dir": args.save_path,
        "elapsed": elapsed,
    }


def run_single_mesh_experiment(arg_list: List[str]) -> None:
    """Run a single mesh experiment from a parsed argv list."""
    initialize_runtime()
    args = parse_experiment_args(arg_list, defaults=config_utils.DEFAULTS)
    try:
        run_mesh(args, skip_existing=True)
    except Exception as e:
        print(f"Error processing mesh {args.mesh}: {e}")


def expected_output_files(args: Any) -> List[str]:
    state = "final"
    outputs: List[str] = []
    outputs.append(make_dccvt_obj_path(args, state, "projDCCVT"))
    return outputs
