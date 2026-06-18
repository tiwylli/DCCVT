#!/usr/bin/env python3
"""Reproduce the PoNQ ABC encoder pretraining schedule."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PONQ_UTILS = ROOT / "PoNQ-main" / "src" / "utils"
if str(PONQ_UTILS) not in sys.path:
    sys.path.insert(0, str(PONQ_UTILS))

from abc_ddp_training import main


if __name__ == "__main__":
    main()
