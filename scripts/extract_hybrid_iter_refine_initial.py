#!/usr/bin/env python3
"""Extract iterative-refinement HotSpot near-surface initialization."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.iter_refine import initial_extract_main


if __name__ == "__main__":
    initial_extract_main()
