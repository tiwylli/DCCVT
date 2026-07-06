"""Exact ABC point-cloud UDF sidecar utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from dccvt.neural.abc.config import ABCUDFConfig

def udf_sidecar_path(udf_root: str | Path, model_id: str) -> Path:
    """Return the UDF HDF5 sidecar path for one ABC shape."""
    return Path(udf_root) / f"{Path(model_id).stem}.hdf5"


def validate_udf_sidecar(
    path: str | Path,
    *,
    config: ABCUDFConfig,
    check_values: bool = True,
) -> tuple[bool, str]:
    """Validate sidecar schema, metadata, and aligned 33^3 values."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("ABC UDF preprocessing and training require h5py") from exc

    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as handle:
            if handle["128_udf"].shape != (129, 129, 129):
                return False, "invalid 128_udf shape"
            if handle["32_udf"].shape != (33, 33, 33):
                return False, "invalid 32_udf shape"
            if handle["128_udf"].dtype != np.float32 or handle["32_udf"].dtype != np.float32:
                return False, "UDF datasets must be float32"
            if str(handle.attrs.get("preprocessing_version", "")) != config.preprocessing_version:
                return False, "preprocessing version mismatch"
            if int(handle.attrs.get("source_point_count", -1)) != 1_000_000:
                return False, "source point count mismatch"
            if float(handle.attrs.get("coordinate_min", np.nan)) != config.coordinate_min:
                return False, "coordinate_min mismatch"
            if float(handle.attrs.get("coordinate_max", np.nan)) != config.coordinate_max:
                return False, "coordinate_max mismatch"
            if str(handle.attrs.get("downsample_rule", "")) != "128_udf[::4,::4,::4]":
                return False, "downsample rule mismatch"
            if check_values:
                udf128 = np.asarray(handle["128_udf"][:], dtype=np.float32)
                udf32 = np.asarray(handle["32_udf"][:], dtype=np.float32)
                if not np.isfinite(udf128).all() or np.any(udf128 < 0):
                    return False, "128_udf contains invalid values"
                if not np.isfinite(udf32).all() or np.any(udf32 < 0):
                    return False, "32_udf contains invalid values"
                aligned = udf128[::4, ::4, ::4]
                if not np.array_equal(udf32, aligned):
                    return False, "32_udf is not the exact stride-four view"
    except (KeyError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, "ok"


def exact_point_udf_grid(
    points: torch.Tensor,
    *,
    grid_n: int,
    coordinate_min: float,
    coordinate_max: float,
    query_chunk_size: int,
) -> torch.Tensor:
    """Compute exact nearest-sample UDF values on a dense vertex grid."""
    try:
        from pytorch3d.ops import knn_points
    except ImportError as exc:
        raise ImportError("Exact ABC UDF preprocessing requires PyTorch3D") from exc

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty point tensor with shape (N,3), got {points.shape}")
    axis = torch.linspace(
        coordinate_min,
        coordinate_max,
        grid_n,
        device=points.device,
        dtype=points.dtype,
    )
    try:
        xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        xyz = torch.meshgrid(axis, axis, axis)
    queries = torch.stack(xyz, dim=-1).reshape(-1, 3)

    distances: list[torch.Tensor] = []
    reference = points.unsqueeze(0)
    for chunk in queries.split(query_chunk_size, dim=0):
        squared = knn_points(chunk.unsqueeze(0), reference, K=1).dists[0, :, 0]
        distances.append(squared.clamp_min_(0).sqrt_())
    return torch.cat(distances).reshape(grid_n, grid_n, grid_n)


def write_udf_sidecar(
    output_path: str | Path,
    udf128: np.ndarray,
    *,
    source_point_count: int,
    config: ABCUDFConfig,
) -> None:
    """Atomically write one exact 129^3 UDF and its aligned 33^3 view."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("ABC UDF preprocessing requires h5py") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    udf128 = np.asarray(udf128, dtype=np.float32)
    if udf128.shape != (129, 129, 129):
        raise ValueError(f"Expected 129^3 UDF, got {udf128.shape}")
    udf32 = np.ascontiguousarray(udf128[::4, ::4, ::4])
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with h5py.File(temporary, "w") as handle:
            options = {
                "compression": config.compression,
                "compression_opts": config.compression_level,
                "shuffle": True,
            }
            handle.create_dataset("128_udf", data=udf128, **options)
            handle.create_dataset("32_udf", data=udf32, **options)
            handle.attrs["coordinate_min"] = config.coordinate_min
            handle.attrs["coordinate_max"] = config.coordinate_max
            handle.attrs["source_point_count"] = int(source_point_count)
            handle.attrs["preprocessing_version"] = config.preprocessing_version
            handle.attrs["downsample_rule"] = "128_udf[::4,::4,::4]"
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
