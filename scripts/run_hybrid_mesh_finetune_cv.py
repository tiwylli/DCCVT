#!/usr/bin/env python3
"""Run the five-fold hybrid direct mesh-loss adaptation study."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.mesh_finetune_cv import main


if __name__ == "__main__":
    main()
