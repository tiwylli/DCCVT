"""Experiment runner utilities for per-mesh execution."""

import os
from time import time
from typing import Any, List

import torch

from dccvt import config
from dccvt.alpha_shape import complex_alpha_sdf
from dccvt.argparse_utils import parse_experiment_args
from dccvt.mesh_ops import extract_mesh
from dccvt.model_utils import init_sdf_from_model, init_sites_from_mnfld_points, load_hotspot_model
from dccvt.paths import make_dccvt_obj_path, make_voromesh_obj_path
from dccvt.device import device
from dccvt.training import run_dccvt_training


def run_single_mesh_experiment(arg_list: List[str]) -> None:
    """Run a single mesh experiment from a parsed argv list."""
    args = parse_experiment_args(arg_list, defaults=config.DEFAULTS)
    args.save_path = f"{args.output}" if args.save_path is None else args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    use_chamfer = args.w_chamfer > 0
    use_voroloss = args.w_voroloss > 0

    # if not os.path.exists(f"{args.save_path}/marching_tetrahedra_{args.upsampling}_final_MT.obj"):
    output_obj_file = check_if_already_processed(args)
    if os.path.exists(output_obj_file):
        print(f"Skipping already processed mesh: {output_obj_file}")
    else:
        print("args: ", args)
        try:
            model, mnfld_points = load_hotspot_model(
                mesh_path=args.mesh,
                target_size=args.target_size,
                hotspot_weights_path=args.trained_HotSpot,
            )
            sites = init_sites_from_mnfld_points(
                mnfld_points=mnfld_points,
                num_centroids=args.num_centroids,
                sample_near=args.sample_near,
                input_dims=args.input_dims,
            )

            if use_chamfer:
                if args.sdf_type == "bounding_sphere":
                    # bounding sphere of mnfld points
                    radius = torch.norm(mnfld_points, dim=-1).max().item() + 0.1
                    sdf = torch.norm(sites - torch.zeros(3).to(device), dim=-1) - radius
                    sdf = sdf.detach().squeeze(-1).requires_grad_()
                    print("sdf:", sdf.shape, sdf.dtype, sdf.is_leaf)
                elif args.sdf_type == "complex_alpha":
                    sdf = complex_alpha_sdf(mnfld_points, sites)
                    print("sdf:", sdf.shape, sdf.dtype, sdf.is_leaf)
                else:
                    sdf = init_sdf_from_model(model, sites)
            else:
                sdf = model

            # Extract the initial mesh
            extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

            if use_chamfer or use_voroloss:
                t0 = time()
                sites, sdf = run_dccvt_training(sites, sdf, mnfld_points, model, args)
                ti = time() - t0

            # Extract the final mesh
            extract_mesh(sites, sdf, mnfld_points, ti, args, state="final")
        except Exception as e:
            print(f"Error processing mesh {args.mesh}: {e}")
        else:
            print(f"Finished processing mesh: {args.mesh}")
            torch.cuda.empty_cache()


def check_if_already_processed(args: Any) -> str:
    state = "final"
    if args.w_chamfer > 0:
        output_obj_file = make_dccvt_obj_path(args, state, "projDCCVT")
    if args.w_voroloss > 0:
        output_obj_file = make_voromesh_obj_path(args, state)
    if args.w_mc > 0:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        output_obj_file = make_dccvt_obj_path(args, state, "MT")
    return output_obj_file
