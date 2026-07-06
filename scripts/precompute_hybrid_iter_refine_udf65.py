#!/usr/bin/env python3
"""Precompute 65^3 point-UDF sidecars for hybrid iterative refinement."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.data.point_udf_sidecar import main


if __name__ == "__main__":
    main()
