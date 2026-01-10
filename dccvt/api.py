"""Public API helpers for running DCCVT programmatically."""

from __future__ import annotations

from typing import Iterable, List, Optional

from dccvt import argparse_utils as config_utils
from dccvt.argparse_utils import DEFAULTS, parse_experiment_args
from dccvt.io_utils import copy_experiment_script
from dccvt.runner import run_mesh


def run_mesh_from_params(
    *,
    mesh: str,
    trained_HotSpot: str,
    output: str,
    num_iterations: int = DEFAULTS["num_iterations"],
    num_centroids: int = DEFAULTS["num_centroids"],
    sample_near: int = DEFAULTS["sample_near"],
    max_amount_sites: int = DEFAULTS["max_amount_sites"],
    video: bool = DEFAULTS["video"],
    w_cvt: float = DEFAULTS["w_cvt"],
    w_sdfsmooth: float = DEFAULTS["w_sdfsmooth"],
    w_chamfer: float = DEFAULTS["w_chamfer"],
    upsampling: int = DEFAULTS["upsampling"],
    lr_sites: float = DEFAULTS["lr_sites"],
    save_path: Optional[str] = None,
    skip_existing: bool = True,
) -> dict:
    """Run DCCVT on a single mesh with explicit parameters."""
    defaults = DEFAULTS.copy()
    defaults.update(
        {
            "mesh": mesh,
            "trained_HotSpot": trained_HotSpot,
            "output": output,
            "num_iterations": num_iterations,
            "num_centroids": num_centroids,
            "sample_near": sample_near,
            "max_amount_sites": max_amount_sites,
            "video": video,
            "w_cvt": w_cvt,
            "w_sdfsmooth": w_sdfsmooth,
            "w_chamfer": w_chamfer,
            "upsampling": upsampling,
            "lr_sites": lr_sites,
            "save_path": save_path,
        }
    )
    args = parse_experiment_args([], defaults=defaults)
    return run_mesh(args, skip_existing=skip_existing)


def run_from_args_file(
    args_file: str,
    *,
    mesh_ids: Optional[Iterable[str]] = None,
    timestamp: Optional[str] = None,
    dry_run: bool = False,
    script_path: Optional[str] = None,
) -> Optional[List[List[str]]]:
    """Run a list of experiments from an args template file."""
    if timestamp:
        config_utils.update_timestamp(timestamp)

    merged_defaults = config_utils.DEFAULTS | {
        "timestamp": config_utils.timestamp,
        "ROOT_DIR": config_utils.ROOT_DIR,
    }
    arg_lists = config_utils.parse_args_template_file(args_file, defaults=merged_defaults, mesh_ids=mesh_ids)
    if dry_run:
        return arg_lists

    if script_path is not None:
        copy_experiment_script(arg_lists, script_path, config_utils.DEFAULTS["output"])

    for arg_list in arg_lists:
        args = parse_experiment_args(arg_list, defaults=config_utils.DEFAULTS)
        run_mesh(args, skip_existing=True)

    return None
