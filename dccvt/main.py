"""CLI entrypoint wiring for the DCCVT experiment runner."""

import argparse
import re
import sys
from typing import Optional

from dccvt.api import run_from_args_file


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

    if not root_args.args_file:
        raise ValueError("Please provide an --args-file with experiment templates.")

    if script_path is None:
        script_path = __file__

    arg_lists = run_from_args_file(
        root_args.args_file,
        mesh_ids=mesh_ids_override,
        timestamp=root_args.timestamp,
        dry_run=root_args.dry_run,
        script_path=script_path,
    )
    if root_args.dry_run:
        for i, a in enumerate(arg_lists or []):
            print(f"[{i}] {a}")
        sys.exit(0)
