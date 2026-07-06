#!/usr/bin/env python3
"""Train the PoNQ-style dense-SDF neural DCCVT site predictor."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.ponq.train import main


if __name__ == "__main__":
    main()
