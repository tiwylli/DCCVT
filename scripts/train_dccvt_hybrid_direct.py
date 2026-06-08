#!/usr/bin/env python3
"""Train the hybrid direct PoNQ-DCCVT extractor."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.hybrid_train import main


if __name__ == "__main__":
    main()
