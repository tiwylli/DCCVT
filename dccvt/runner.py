"""Experiment runner utilities for per-mesh execution."""

import os
from time import time
from typing import Any, List

import torch

from dccvt import config
from dccvt.alpha_shape import complex_alpha_sdf
from dccvt.argparse_utils import define_options_parser
from dccvt.mesh_ops import extract_mesh
from dccvt.model_utils import init_sdf, init_sites, load_model
from dccvt.paths import build_dccvt_obj_path, build_voromesh_obj_path
from dccvt.runtime import device
from dccvt.training import train_DCCVT
def process_single_mesh(arg_list: List[str]) -> None:
    """Run a single mesh experiment from a parsed argv list."""
    args = define_options_parser(arg_list, defaults=config.DEFAULTS)
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
            model, mnfld_points = load_model(args.mesh, args.target_size, args.trained_HotSpot)
            sites = init_sites(mnfld_points, args.num_centroids, args.sample_near, args.input_dims)

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
                    sdf = init_sdf(model, sites)
            else:
                sdf = model

            # Extract the initial mesh
            extract_mesh(sites, sdf, mnfld_points, 0, args, state="init")

            if use_chamfer or use_voroloss:
                t0 = time()
                sites, sdf = train_DCCVT(sites, sdf, mnfld_points, model, args)
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
        output_obj_file = build_dccvt_obj_path(args, state, "projDCCVT")
    if args.w_voroloss > 0:
        output_obj_file = build_voromesh_obj_path(args, state)
    if args.w_mc > 0:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        output_obj_file = build_dccvt_obj_path(args, state, "MT")
    return output_obj_file
