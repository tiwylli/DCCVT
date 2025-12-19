"""Argument parsing helpers for DCCVT experiments."""

import argparse
import os
import re
import shlex
from typing import Dict, Iterable, List, Optional

from dccvt import config


class _SafeDict(dict):
    def __missing__(self, key):
        # leave unknown placeholders intact, e.g. "{mesh_id}"
        return "{" + key + "}"


def load_arg_lists_from_file(path: str, defaults: Dict[str, object], mesh_ids: Optional[Iterable[str]] = None) -> List[List[str]]:
    """Load an args template file and return a list of argv-style lists."""
    if mesh_ids is None:
        mesh_ids = list(defaults.get("mesh_ids", []))

    arg_lists: List[List[str]] = []
    active_mesh_ids = list(mesh_ids)
    buf = ""

    def process_buffer(s: str) -> None:
        nonlocal arg_lists, active_mesh_ids
        s = s.strip()
        if not s:
            return

        # handle directives (not part of continued blocks)
        if s.lower().startswith("@mesh_ids"):
            _, rhs = s.split(":", 1)
            items = re.split(r"[,\s]+", rhs.strip())
            active_mesh_ids = [it for it in items if it]
            return

        # 1) safely format known {placeholders} from defaults
        templated = s.format_map(_SafeDict(**defaults))
        # 2) expand env vars like $HOME
        templated = os.path.expandvars(templated)

        # 3) fan out over {mesh_id} if present
        if "{mesh_id}" in templated:
            for mid in active_mesh_ids:
                filled = templated.replace("{mesh_id}", str(mid))
                arg_lists.append(shlex.split(filled))
        else:
            arg_lists.append(shlex.split(templated))

    with open(path, "r") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not buf and (not stripped or stripped.startswith("#")):
                continue

            # continuation if there's an odd number of trailing backslashes
            m = re.search(r"(\\+)$", line)
            trailing_bs = len(m.group(1)) if m else 0
            is_cont = trailing_bs % 2 == 1

            if is_cont:
                line = line[:-1]  # drop exactly one "\\" for continuation
                buf += line
                continue
            else:
                buf += line
                process_buffer(buf)  # complete logical line
                buf = ""

    if buf.strip():
        process_buffer(buf)

    return arg_lists


def _add_bool_arg(parser: argparse.ArgumentParser, flag: str, default: bool, help_text: str) -> None:
    parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_text)


def define_options_parser(
    arg_list: Optional[List[str]] = None, defaults: Optional[Dict[str, object]] = None
) -> argparse.Namespace:
    """Parse per-mesh experiment arguments from an argv list."""
    if defaults is None:
        defaults = config.DEFAULTS

    parser = argparse.ArgumentParser(description="DCCVT experiments")
    parser.add_argument("--input_dims", type=int, default=defaults["input_dims"], help="Dimensionality of the input")
    parser.add_argument("--output", type=str, default=defaults["output"], help="Output directory")
    parser.add_argument("--mesh", type=str, default=defaults["mesh"], help="Mesh directory")
    parser.add_argument(
        "--trained_HotSpot", type=str, default=defaults["trained_HotSpot"], help="Trained HotSpot model directory"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=defaults["num_iterations"], help="Number of iterations for optimization"
    )
    parser.add_argument("--num_centroids", type=int, default=defaults["num_centroids"], help="Number of centroids")
    parser.add_argument("--sample_near", type=int, default=defaults["sample_near"], help="Samples drawn near each site")
    parser.add_argument("--target_size", type=int, default=defaults["target_size"], help="Target size for sampling")
    _add_bool_arg(parser, "--clip", defaults["clip"], "Enable/disable clipping")
    parser.add_argument(
        "--grad_interpol",
        type=str,
        default=defaults["grad_interpol"],
        help="Gradient interpolation method: robust, hybrid, barycentric",
    )
    _add_bool_arg(
        parser, "--marching_tetrahedra", defaults["marching_tetrahedra"], "Enable/disable marching_tetrahedra"
    )
    _add_bool_arg(parser, "--true_cvt", defaults["true_cvt"], "Enable/disable true CVT loss")
    _add_bool_arg(parser, "--extract_optim", defaults["extract_optim"], "Enable/disable extraction optimization")
    parser.add_argument(
        "--sdf_type",
        type=str,
        default=defaults["sdf_type"],
        help="SDF type: hotspot, sphere, complex_alpha",
    )
    _add_bool_arg(parser, "--no_mp", defaults["no_mp"], "Enable/disable multiprocessing")
    _add_bool_arg(parser, "--ups_extraction", defaults["ups_extraction"], "Enable/disable upsampling extraction")
    _add_bool_arg(parser, "--build_mesh", False, "Enable/disable build mesh")
    _add_bool_arg(parser, "--video", False, "Enable/disable video output")
    parser.add_argument("--w_cvt", type=float, default=defaults["w_cvt"], help="Weight for CVT regularization")
    parser.add_argument(
        "--w_vertex_sdf_interpolation",
        type=float,
        default=defaults["w_vertex_sdf_interpolation"],
        help="Weight for vertex SDF interpolation",
    )
    parser.add_argument("--w_sdfsmooth", type=float, default=defaults["w_sdfsmooth"], help="Weight for SDF smoothing")
    parser.add_argument("--w_voroloss", type=float, default=defaults["w_voroloss"], help="Weight for Voronoi loss")
    parser.add_argument(
        "--w_chamfer", type=float, default=defaults["w_chamfer"], help="Weight for Chamfer distance on points"
    )
    # parser.add_argument("--w_bpa", type=float, default=defaults.get("w_bpa", 0), help="flag to use BPA instead of DCCVT")
    parser.add_argument("--w_mc", type=float, default=defaults["w_mc"], help="Weight for MC loss")
    parser.add_argument("--w_mt", type=float, default=defaults["w_mt"], help="Weight for MT loss")
    parser.add_argument("--upsampling", type=int, default=defaults["upsampling"], help="Upsampling factor")
    parser.add_argument(
        "--ups_method",
        type=str,
        default=defaults["ups_method"],
        help="Upsampling method either tet_frame or tet_random or random",
    )
    parser.add_argument("--lr_sites", type=float, default=defaults["lr_sites"], help="Learning rate for sites")
    parser.add_argument(
        "--save_path", type=str, default=None, help="(optional) full save path; if omitted, computed from other flags"
    )
    parser.add_argument(
        "--score",
        type=str,
        default=defaults["score"],
        help="Score computation [legacy, density, sqrt_curvature, cosine]",
    )
    return parser.parse_args(arg_list)
