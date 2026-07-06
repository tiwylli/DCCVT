"""ABC HDF5 dataset for HybridPoNQ-DCCVT."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dccvt.neural.abc.udf import udf_sidecar_path, validate_udf_sidecar
from dccvt.neural.grid import build_hybrid_input_channels_np

class ABCHybridDataset(Dataset):
    """Load aligned ABC SDF/UDF grids and normalized surface targets."""

    def __init__(
        self,
        model_ids: Iterable[str],
        *,
        hdf5_root: str | Path,
        udf_root: str | Path,
        target_sample_count: int,
        seed: int,
        deterministic_targets: bool = False,
    ) -> None:
        self.model_ids = [Path(model_id).stem for model_id in model_ids]
        self.hdf5_root = Path(hdf5_root)
        self.udf_root = Path(udf_root)
        self.target_sample_count = int(target_sample_count)
        self.seed = int(seed)
        self.deterministic_targets = bool(deterministic_targets)
        if not self.model_ids:
            raise ValueError("ABCHybridDataset requires at least one model ID")
        if self.target_sample_count < 1:
            raise ValueError("target_sample_count must be positive")

    def __len__(self) -> int:
        return len(self.model_ids)

    def _target_indices(self, index: int, point_count: int) -> np.ndarray:
        replace = point_count < self.target_sample_count
        if self.deterministic_targets:
            rng = np.random.RandomState(self.seed + index)
            return rng.choice(point_count, self.target_sample_count, replace=replace)
        return np.random.choice(point_count, self.target_sample_count, replace=replace)

    def __getitem__(self, index: int) -> dict:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("ABC HybridPoNQ training requires h5py") from exc

        model_id = self.model_ids[index]
        source_path = self.hdf5_root / f"{model_id}.hdf5"
        sidecar_path = udf_sidecar_path(self.udf_root, model_id)
        with h5py.File(source_path, "r") as source:
            sdf = np.asarray(source["32_sdf"][:], dtype=np.float32) * 2.0
            points = np.asarray(source["pointcloud"][:], dtype=np.float32)
        with h5py.File(sidecar_path, "r") as sidecar:
            udf = np.asarray(sidecar["32_udf"][:], dtype=np.float32) * 2.0
        if sdf.shape != (33, 33, 33) or udf.shape != (33, 33, 33):
            raise ValueError(
                f"{model_id} requires 33^3 SDF/UDF grids, got {sdf.shape} and {udf.shape}"
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{model_id} pointcloud must have shape (N,3), got {points.shape}")
        if not np.isfinite(sdf).all() or not np.isfinite(udf).all():
            raise ValueError(f"{model_id} contains non-finite SDF/UDF values")

        indices = self._target_indices(index, points.shape[0])
        targets = np.ascontiguousarray(points[indices] * 2.0, dtype=np.float32)
        input_grid = np.stack((sdf, udf), axis=0)
        return {
            "input_grid": torch.from_numpy(input_grid),
            "sdf_grid": torch.from_numpy(sdf[None, ...]),
            "target_points": torch.from_numpy(targets),
            "model_id": model_id,
            "source_path": str(source_path),
            "udf_path": str(sidecar_path),
        }
