#!/usr/bin/env python3
"""Extract canonical-grid HotSpot initial DCCVT meshes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dccvt.neural.extraction.hybrid_initial import main


if __name__ == "__main__":
    main()
