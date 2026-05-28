#!/usr/bin/env python3
"""Precompute dense HotSpot SDF grids for PoNQ-style neural DCCVT."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.precompute import main


if __name__ == "__main__":
    main()
