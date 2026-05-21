#!/usr/bin/env python3
"""Run DCCVT neural generator inference and export meshes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.infer import main


if __name__ == "__main__":
    main()
