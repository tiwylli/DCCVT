#!/usr/bin/env python3
"""Run PoNQ-style neural DCCVT inference and export meshes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.ponq.infer import main


if __name__ == "__main__":
    main()
