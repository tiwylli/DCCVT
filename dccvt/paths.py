"""Output path helpers for generated meshes and artifacts."""

from typing import Any


def build_dccvt_obj_path(args: Any, state: str, variant: str) -> str:
    prefix = "marching_tetrahedra" if args.marching_tetrahedra else "DCCVT"
    return (
        f"{args.save_path}/{prefix}_{args.upsampling}_{state}_{variant}_"
        f"cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    )


def build_voromesh_obj_path(args: Any, state: str) -> str:
    return (
        f"{args.save_path}/voromesh_{args.num_centroids}_{state}_DCCVT_"
        f"cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    )
