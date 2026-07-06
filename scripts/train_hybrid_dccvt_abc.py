#!/usr/bin/env python3
"""Train or evaluate the 32^3 HybridPoNQ-DCCVT ABC experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.abc.cli import main


if __name__ == "__main__":
    main()
