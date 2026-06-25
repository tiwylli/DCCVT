#!/usr/bin/env python3
"""Train iterative learned sparse refinement with DCCVT mesh loss."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.iter_refine import train_main


if __name__ == "__main__":
    train_main()
