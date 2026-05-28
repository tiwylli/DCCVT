"""Datasets for cached HotSpot SDF grids used by neural DCCVT."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(part for part in line.replace(",", " ").split() if part)
    return ids


def resolve_cache_files(
    cache_root: str | Path,
    *,
    mesh_ids: Optional[Iterable[str]] = None,
    split_file: Optional[str | Path] = None,
) -> list[Path]:
    """Resolve cached ``.npz`` files from ids, a split file, or all files."""
    cache_root = Path(cache_root)
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    ids: Optional[list[str]] = None
    if split_file is not None:
        ids = _read_ids(Path(split_file))
    elif mesh_ids is not None:
        ids = list(mesh_ids)

    if ids is None:
        files = sorted(cache_root.glob("*.npz"))
    else:
        files = []
        for mesh_id in ids:
            candidate = Path(mesh_id)
            if candidate.suffix == ".npz" and candidate.exists():
                files.append(candidate)
            else:
                files.append(cache_root / f"{candidate.stem}.npz")

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing neural cache files:\n" + "\n".join(missing))
    if not files:
        raise FileNotFoundError(f"No .npz cache files found in {cache_root}")
    return files


class HotspotSDFDataset(Dataset):
    """Load dense HotSpot SDF caches for PoNQ-style DCCVT training."""

    def __init__(
        self,
        files: Iterable[str | Path],
        *,
        target_subsample: Optional[int] = None,
    ) -> None:
        self.files = [Path(path) for path in files]
        self.target_subsample = target_subsample
        if not self.files:
            raise ValueError("HotspotSDFDataset requires at least one cache file")

    def __len__(self) -> int:
        return len(self.files)

    def _target_points(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        points = np.asarray(data["target_points"], dtype=np.float32).reshape(-1, 3)
        if self.target_subsample is None or points.shape[0] <= self.target_subsample:
            return points
        indices = np.random.choice(points.shape[0], self.target_subsample, replace=False)
        return points[indices]

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            sdf_grid = np.asarray(data["sdf_grid"], dtype=np.float32)
            near_surface_mask = np.asarray(data["near_surface_mask"], dtype=bool)
            gt_activity_mask = np.asarray(data["gt_activity_mask"], dtype=bool)
            target_points = self._target_points(data)
            grid_n = int(np.asarray(data["grid_n"]).item())
            mesh_id = str(np.asarray(data["mesh_id"]).item()) if "mesh_id" in data else path.stem

        return {
            "sdf_grid": torch.from_numpy(sdf_grid[None, ...]),
            "near_surface_mask": torch.from_numpy(near_surface_mask),
            "gt_activity_mask": torch.from_numpy(gt_activity_mask),
            "target_points": torch.from_numpy(target_points),
            "grid_n": torch.tensor(grid_n, dtype=torch.long),
            "mesh_id": mesh_id,
            "cache_path": str(path),
        }
