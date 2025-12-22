"""CLI entrypoint wiring for the DCCVT experiment runner."""

import argparse
import re
import sys
from typing import Optional

from dccvt import config
from dccvt.argparse_utils import parse_args_template_file
from dccvt.io_utils import copy_experiment_script
from dccvt.runner import run_single_mesh_experiment
import dccvt.device  # initialize device + seeds


def main(script_path: Optional[str] = None) -> None:
    root = argparse.ArgumentParser(add_help=True)
    root.add_argument(
        "--args-file",
        type=str,
        default=None,
        help="Text file: one experiment template per line. Use {mesh_id} to expand.",
    )
    root.add_argument("--mesh-ids", type=str, default=None, help="Override mesh list (comma/space separated).")
    root.add_argument("--timestamp", type=str, default=None, help="Timestamp for the experiment.")
    root.add_argument("--dry-run", action="store_true", help="Print experiments and exit.")
    root_args, _ = root.parse_known_args()

    # Build the mesh list override if provided
    mesh_ids_override = None
    if root_args.mesh_ids:
        mesh_ids_override = [s for s in re.split(r"[,\s]+", root_args.mesh_ids.strip()) if s]
        print(f"Using mesh IDs override: {mesh_ids_override}")

    if root_args.timestamp:
        config.update_timestamp(root_args.timestamp)

    if root_args.args_file:
        # Provide DEFAULTS + timestamp to formatting
        merged_defaults = config.DEFAULTS | {"timestamp": config.timestamp, "ROOT_DIR": config.ROOT_DIR}
        arg_lists = parse_args_template_file(
            root_args.args_file, defaults=merged_defaults, mesh_ids=mesh_ids_override
        )
        if root_args.dry_run:
            for i, a in enumerate(arg_lists):
                print(f"[{i}] {a}")
            sys.exit(0)
    else:
        raise ValueError("Please provide an --args-file with experiment templates.")

    if script_path is None:
        script_path = __file__

    copy_experiment_script(arg_lists, script_path, config.DEFAULTS["output"])

    for arg_list in arg_lists:
        run_single_mesh_experiment(arg_list)
