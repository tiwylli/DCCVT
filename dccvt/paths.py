"""Output path helpers for generated meshes and artifacts."""

from typing import Any


def make_dccvt_obj_path(args: Any, state: str, variant: str) -> str:
    return (
        f"{args.save_path}/DCCVT_{args.upsampling}_{state}_{variant}_"
        f"cvt{int(args.w_cvt)}_sdfsmooth{int(args.w_sdfsmooth)}.obj"
    )
