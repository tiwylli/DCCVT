"""Datasets and batch helpers for iterative refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dccvt.neural.data.point_udf_sidecar import load_point_udf_sidecar, point_udf_sidecar_path
from dccvt.neural.grid import build_hybrid_input_channels_np
from dccvt.neural.iterative.config import HybridIterRefineConfig
from dccvt.neural.iterative.initialization import build_hotspot_near_surface_initialization

class HybridIterRefineDataset(Dataset):
    """HotSpot cache dataset for mesh-loss-only iterative refinement."""

    def __init__(
        self,
        cache_files: Sequence[str | Path],
        *,
        config: HybridIterRefineConfig,
        target_subsample: Optional[int] = None,
        local_udf_root: str | Path | None = None,
        allow_missing_local_features: bool = False,
    ) -> None:
        self.files = [Path(path) for path in cache_files]
        self.config = config
        self.target_subsample = target_subsample
        self.local_udf_root = None if local_udf_root is None else Path(local_udf_root)
        self.allow_missing_local_features = bool(allow_missing_local_features)
        self._initialization_cache: dict[int, dict[str, Any]] = {}
        if not self.files:
            raise ValueError("HybridIterRefineDataset requires at least one cache file")
        if self.config.local_udf_samples and self.local_udf_root is None and not self.allow_missing_local_features:
            raise ValueError("Config requests local UDF samples; provide --local-udf-root or allow missing features")

    def __len__(self) -> int:
        return len(self.files)

    def _subsample_target_points(self, points: np.ndarray) -> np.ndarray:
        if self.target_subsample is None or points.shape[0] <= self.target_subsample:
            return points
        indices = np.random.choice(points.shape[0], self.target_subsample, replace=False)
        return points[indices]

    def _target_points(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        points = np.asarray(data["target_points"], dtype=np.float32).reshape(-1, 3)
        return self._subsample_target_points(points)

    def _initialization(self, index: int, sdf_grid: np.ndarray) -> dict[str, Any]:
        initialization = self._initialization_cache.get(index)
        if initialization is None:
            initialization = build_hotspot_near_surface_initialization(sdf_grid, self.config)
            self._initialization_cache[index] = initialization
        return initialization

    def _local_udf_grid(self, cache_path: Path) -> tuple[np.ndarray, str, bool]:
        if not self.config.local_udf_samples:
            return np.zeros((0,), dtype=np.float32), "", False
        if self.local_udf_root is None:
            if self.allow_missing_local_features:
                grid_n = self.config.local_udf_grid_n
                return np.zeros((grid_n, grid_n, grid_n), dtype=np.float32), "", False
            raise ValueError("Config requests local UDF samples but no local UDF root was provided")

        sidecar_path = point_udf_sidecar_path(self.local_udf_root, cache_path.stem)
        if not sidecar_path.exists():
            if self.allow_missing_local_features:
                grid_n = self.config.local_udf_grid_n
                return np.zeros((grid_n, grid_n, grid_n), dtype=np.float32), str(sidecar_path), False
            raise FileNotFoundError(f"Missing local point-UDF sidecar: {sidecar_path}")
        return (
            load_point_udf_sidecar(sidecar_path, grid_n=self.config.local_udf_grid_n),
            str(sidecar_path),
            True,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache_path = self.files[index]
        with np.load(cache_path, allow_pickle=False) as data:
            sdf_grid = np.asarray(data["sdf_grid"], dtype=np.float32)
            full_target_points = np.asarray(data["target_points"], dtype=np.float32).reshape(-1, 3)
            target_points = self._subsample_target_points(full_target_points)
            grid_n = int(np.asarray(data["grid_n"]).item())
            mesh_id = str(np.asarray(data["mesh_id"]).item()) if "mesh_id" in data else cache_path.stem
        if grid_n != self.config.hotspot_grid_n:
            raise ValueError(f"Cache grid_n={grid_n} does not match config hotspot_grid_n={self.config.hotspot_grid_n}")
        initialization = self._initialization(index, sdf_grid)
        input_grid = build_hybrid_input_channels_np(
            sdf_grid,
            target_points,
            grid_n=grid_n,
            udf_clip=self.config.point_udf_clip,
            confidence_sigma_scale=self.config.point_confidence_sigma_scale,
            channel_names=self.config.channel_names,
        )
        local_udf_grid, local_udf_path, local_udf_valid = self._local_udf_grid(cache_path)
        return {
            "input_grid": torch.from_numpy(input_grid),
            "sdf_grid": torch.from_numpy(sdf_grid[None, ...]),
            "target_points": torch.from_numpy(target_points),
            "local_target_points": torch.from_numpy(full_target_points if self.config.local_knn_features else target_points),
            "local_udf_grid": torch.from_numpy(local_udf_grid),
            "local_udf_path": local_udf_path,
            "local_udf_valid": torch.tensor(bool(local_udf_valid)),
            "grid_n": torch.tensor(grid_n, dtype=torch.long),
            "mesh_id": mesh_id,
            "cache_path": str(cache_path),
            "initial_sites": initialization["sites"],
            "initial_sites_sdf": initialization["sites_sdf"],
            "background_sites": initialization["background_sites"],
            "background_sdf": initialization["background_sdf"],
            "surface_anchors": initialization["surface_anchors"],
            "surface_sites": initialization["surface_sites"],
            "surface_sdf": initialization["surface_sdf"],
            "initialization_valid": torch.tensor(bool(initialization["valid"])),
            "initialization_reason": str(initialization["reason"]),
            "initialization_diagnostics": json.dumps(initialization["diagnostics"], sort_keys=True),
        }


def _initialization_from_batch(
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    diagnostics_value = batch["initialization_diagnostics"]
    diagnostics_text = diagnostics_value[0] if isinstance(diagnostics_value, (list, tuple)) else diagnostics_value
    reason_value = batch["initialization_reason"]
    reason = reason_value[0] if isinstance(reason_value, (list, tuple)) else str(reason_value)

    def tensor(name: str) -> torch.Tensor:
        value = batch[name]
        if value.dim() > 0 and value.shape[0] == 1:
            value = value[0]
        return value.to(device=device, dtype=dtype, non_blocking=True)

    valid_value = batch["initialization_valid"]
    valid = bool(valid_value.reshape(-1)[0].item()) if isinstance(valid_value, torch.Tensor) else bool(valid_value)
    return {
        "valid": valid,
        "reason": str(reason),
        "sites": tensor("initial_sites"),
        "sites_sdf": tensor("initial_sites_sdf"),
        "background_sites": tensor("background_sites"),
        "background_sdf": tensor("background_sdf"),
        "surface_anchors": tensor("surface_anchors"),
        "surface_sites": tensor("surface_sites"),
        "surface_sdf": tensor("surface_sdf"),
        "diagnostics": json.loads(str(diagnostics_text)),
    }

