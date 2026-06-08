"""Datasets for cached HotSpot SDF grids used by neural DCCVT."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dccvt.neural.grid import build_hybrid_input_channels_np


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


def resolve_dccvt_label_file(
    label_root: str | Path,
    mesh_id: str,
    *,
    upsampling: int = 0,
    state: str = "final",
    variant: str = "projDCCVT",
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
) -> Path:
    """Resolve the optimized DCCVT label file for a mesh id."""
    filename = (
        f"DCCVT_{int(upsampling)}_{state}_{variant}_"
        f"cvt{int(w_cvt)}_sdfsmooth{int(w_sdfsmooth)}.npz"
    )
    path = Path(label_root) / str(mesh_id) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing DCCVT label file: {path}")
    return path


class HybridDirectDataset(Dataset):
    """Pair HotSpot SDF caches with full-field optimized DCCVT labels."""

    def __init__(
        self,
        cache_files: Iterable[str | Path],
        *,
        label_root: str | Path,
        target_subsample: Optional[int] = None,
        upsampling: int = 0,
        label_state: str = "final",
        label_variant: str = "projDCCVT",
        label_w_cvt: float = 100.0,
        label_w_sdfsmooth: float = 100.0,
        point_udf_clip: float = 4.0,
        point_confidence_sigma_scale: float = 1.5,
    ) -> None:
        self.files = [Path(path) for path in cache_files]
        self.label_root = Path(label_root)
        self.target_subsample = target_subsample
        self.upsampling = int(upsampling)
        self.label_state = label_state
        self.label_variant = label_variant
        self.label_w_cvt = float(label_w_cvt)
        self.label_w_sdfsmooth = float(label_w_sdfsmooth)
        self.point_udf_clip = float(point_udf_clip)
        self.point_confidence_sigma_scale = float(point_confidence_sigma_scale)
        if not self.files:
            raise ValueError("HybridDirectDataset requires at least one cache file")

    def __len__(self) -> int:
        return len(self.files)

    def _target_points(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        points = np.asarray(data["target_points"], dtype=np.float32).reshape(-1, 3)
        if self.target_subsample is None or points.shape[0] <= self.target_subsample:
            return points
        indices = np.random.choice(points.shape[0], self.target_subsample, replace=False)
        return points[indices]

    def __getitem__(self, index: int) -> dict:
        cache_path = self.files[index]
        with np.load(cache_path, allow_pickle=False) as data:
            sdf_grid = np.asarray(data["sdf_grid"], dtype=np.float32)
            target_points = self._target_points(data)
            grid_n = int(np.asarray(data["grid_n"]).item())
            mesh_id = str(np.asarray(data["mesh_id"]).item()) if "mesh_id" in data else cache_path.stem

        label_path = resolve_dccvt_label_file(
            self.label_root,
            mesh_id,
            upsampling=self.upsampling,
            state=self.label_state,
            variant=self.label_variant,
            w_cvt=self.label_w_cvt,
            w_sdfsmooth=self.label_w_sdfsmooth,
        )
        with np.load(label_path, allow_pickle=False) as label_data:
            label_sites = np.asarray(label_data["sites"], dtype=np.float32).reshape(-1, 3)
            label_sites_sdf = np.asarray(label_data["sites_sdf"], dtype=np.float32).reshape(-1)

        input_grid = build_hybrid_input_channels_np(
            sdf_grid,
            target_points,
            grid_n=grid_n,
            udf_clip=self.point_udf_clip,
            confidence_sigma_scale=self.point_confidence_sigma_scale,
        )

        return {
            "input_grid": torch.from_numpy(input_grid),
            "sdf_grid": torch.from_numpy(sdf_grid[None, ...]),
            "target_points": torch.from_numpy(target_points),
            "label_sites": torch.from_numpy(label_sites),
            "label_sites_sdf": torch.from_numpy(label_sites_sdf),
            "grid_n": torch.tensor(grid_n, dtype=torch.long),
            "mesh_id": mesh_id,
            "cache_path": str(cache_path),
            "label_path": str(label_path),
        }
