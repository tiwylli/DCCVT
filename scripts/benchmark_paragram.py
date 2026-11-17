#!/usr/bin/env python3
"""CLI for the Paragram versus gDel3D adjacency benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Importing the dccvt package selects a device eagerly. This benchmark manages
# its CUDA device explicitly and should still expose --help on non-GPU hosts.
os.environ.setdefault("DCCVT_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dccvt.paragram_benchmark import build_dataset_specs, run_benchmark, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Paragram adjacency construction with gDel3D-derived adjacency."
    )
    parser.add_argument("--backend", choices=("both", "paragram", "gdel3d"), default="both")
    parser.add_argument("--sizes", type=int, nargs="+", default=(8192, 36120, 100000, 262144, 1000000))
    parser.add_argument(
        "--distribution", choices=("uniform", "clustered"), action="append", dest="distributions"
    )
    parser.add_argument("--npz", action="append", default=[], help="Additional NPZ containing a 'sites' array")
    parser.add_argument(
        "--default-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include known saved 36k and 262k DCCVT site sets when present",
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--cluster-std", type=float, default=0.08)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warm-frames", type=int, default=3)
    parser.add_argument("--warm-displacement", type=float, nargs="+", default=(0.01, 0.05))
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--common-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert native outputs to canonical edges (disable for capacity-only scaling runs)",
    )
    parser.add_argument("--memory-timeout", type=float, default=600.0)
    parser.add_argument("--output", default="outputs/paragram_benchmark/results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.warm_frames < 1:
        raise SystemExit("repetitions and warm-frames must be positive; warmups cannot be negative")
    if any(value <= 0 for value in args.sizes):
        raise SystemExit("all dataset sizes must be positive")
    if any(value < 0 for value in args.warm_displacement):
        raise SystemExit("warm displacements cannot be negative")

    distributions = args.distributions or ("uniform", "clustered")
    backends = ("paragram", "gdel3d") if args.backend == "both" else (args.backend,)
    specs = build_dataset_specs(
        root=ROOT,
        sizes=args.sizes,
        distributions=distributions,
        npz_paths=args.npz,
        include_default_npz=args.default_npz,
        seed=args.seed,
        clusters=args.clusters,
        cluster_std=args.cluster_std,
    )
    result = run_benchmark(
        specs=specs,
        backends=backends,
        repetitions=args.repetitions,
        warmups=args.warmups,
        warm_start=args.warm_start,
        warm_frames=args.warm_frames,
        warm_displacements=args.warm_displacement,
        memory=args.memory,
        memory_timeout=args.memory_timeout,
        seed=args.seed,
        command=sys.argv,
        common_output=args.common_output,
    )
    write_json(result, args.output)
    print(f"Wrote {args.output} ({result['status']})")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
