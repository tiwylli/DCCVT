"""Grid, mask, and interpolation utilities for neural DCCVT."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def validate_grid_n(grid_n: int) -> int:
    grid_n = int(grid_n)
    if grid_n < 2:
        raise ValueError(f"`grid_n` must be >= 2, got {grid_n}")
    return grid_n


def num_cells_from_grid(grid_n: int) -> int:
    return validate_grid_n(grid_n) - 1


def cell_size_from_grid(grid_n: int) -> float:
    return 2.0 / float(num_cells_from_grid(grid_n))


def default_near_surface_threshold(grid_n: int) -> float:
    """PoNQ-style surface band threshold for a DCCVT [-1, 1]^3 grid."""
    return cell_size_from_grid(grid_n) * math.sqrt(3.0)


def make_coord_grid(
    grid_n: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return dense grid coordinates with shape ``(grid_n**3, 3)``."""
    grid_n = validate_grid_n(grid_n)
    axis = torch.linspace(-1.0, 1.0, grid_n, device=device, dtype=dtype)
    try:
        xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        xyz = torch.meshgrid(axis, axis, axis)
    return torch.stack(xyz, dim=-1).reshape(-1, 3)


def make_cell_lower_corners(
    grid_n: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return lower-corner coordinates for the ``(grid_n - 1)^3`` cells."""
    grid_n = validate_grid_n(grid_n)
    cells = grid_n - 1
    cell_size = cell_size_from_grid(grid_n)
    axis = torch.linspace(-1.0, 1.0 - cell_size, cells, device=device, dtype=dtype)
    try:
        xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        xyz = torch.meshgrid(axis, axis, axis)
    return torch.stack(xyz, dim=-1).reshape(-1, 3)


def make_canonical_sites(
    grid_n: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the canonical DCCVT site grid for a ``grid_n`` SDF vertex grid.

    DCCVT initializes one site at each point of a ``(grid_n - 1)^3`` regular
    grid spanning ``[-1, 1]^3``. This differs from PoNQ-style cell centers.
    """
    grid_n = validate_grid_n(grid_n)
    cells = grid_n - 1
    axis = torch.linspace(-1.0, 1.0, cells, device=device, dtype=dtype)
    try:
        xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        xyz = torch.meshgrid(axis, axis, axis)
    return torch.stack(xyz, dim=-1).reshape(-1, 3)


def make_near_surface_mask_np(
    sdf_grid: np.ndarray,
    *,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Return a flattened PoNQ-style cell mask from SDF corner values."""
    sdf_grid = np.asarray(sdf_grid)
    if sdf_grid.ndim != 3 or len(set(sdf_grid.shape)) != 1:
        raise ValueError(f"`sdf_grid` must be cubic with shape (G, G, G), got {sdf_grid.shape}")
    grid_n = int(sdf_grid.shape[0])
    if threshold is None:
        threshold = default_near_surface_threshold(grid_n)

    close = np.abs(sdf_grid) < float(threshold)
    mask = (
        close[:-1, :-1, :-1]
        & close[1:, :-1, :-1]
        & close[:-1, 1:, :-1]
        & close[:-1, :-1, 1:]
        & close[1:, 1:, :-1]
        & close[1:, :-1, 1:]
        & close[:-1, 1:, 1:]
        & close[1:, 1:, 1:]
    )
    return mask.reshape(-1)


def make_gt_activity_mask_np(samples: np.ndarray, grid_n: int) -> np.ndarray:
    """Mark cells that contain normalized target samples in ``[-1, 1]^3``."""
    grid_n = validate_grid_n(grid_n)
    cells = grid_n - 1
    samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    mask = np.zeros((cells, cells, cells), dtype=bool)
    if samples.size == 0:
        return mask.reshape(-1)

    idx = np.floor((samples + 1.0) * 0.5 * cells).astype(np.int64)
    idx = np.clip(idx, 0, cells - 1)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask.reshape(-1)


def point_udf_grid(
    points: torch.Tensor,
    *,
    grid_n: int,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return nearest-neighbor unsigned distances from grid vertices to points."""
    grid_n = validate_grid_n(grid_n)
    points = points.reshape(-1, 3)
    if points.numel() == 0:
        raise ValueError("`points` must contain at least one 3D point")

    coords = make_coord_grid(grid_n, device=points.device, dtype=points.dtype)
    distances: list[torch.Tensor] = []
    for chunk in coords.split(chunk_size, dim=0):
        dists = torch.cdist(chunk.unsqueeze(0), points.unsqueeze(0), p=2).squeeze(0)
        distances.append(dists.min(dim=1).values)
    return torch.cat(distances, dim=0).reshape(grid_n, grid_n, grid_n)


def build_hybrid_input_channels(
    sdf_grid: torch.Tensor,
    target_points: torch.Tensor,
    *,
    grid_n: Optional[int] = None,
    udf_clip: float = 4.0,
    confidence_sigma_scale: float = 1.5,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Build HotSpot plus point-cloud zero-level evidence channels.

    Channels are ``sdf``, ``abs_sdf``, normalized clipped point UDF, and point
    confidence. The point UDF is normalized by the SDF grid cell size.
    """
    if sdf_grid.dim() == 4 and sdf_grid.shape[0] == 1:
        sdf_grid = sdf_grid[0]
    if sdf_grid.dim() != 3 or len(set(sdf_grid.shape)) != 1:
        raise ValueError(f"`sdf_grid` must have shape (G,G,G) or (1,G,G,G), got {sdf_grid.shape}")

    resolved_grid_n = validate_grid_n(grid_n or int(sdf_grid.shape[0]))
    if sdf_grid.shape[0] != resolved_grid_n:
        raise ValueError(f"SDF grid shape {sdf_grid.shape} does not match grid_n={resolved_grid_n}")

    target_points = target_points.to(device=sdf_grid.device, dtype=sdf_grid.dtype)
    udf = point_udf_grid(target_points, grid_n=resolved_grid_n, chunk_size=chunk_size)
    cell_size = cell_size_from_grid(resolved_grid_n)
    normalized_udf = (udf / cell_size).clamp(min=0.0, max=float(udf_clip))
    sigma = float(confidence_sigma_scale) * cell_size
    confidence = torch.exp(-0.5 * (udf / sigma).pow(2))
    return torch.stack((sdf_grid, sdf_grid.abs(), normalized_udf, confidence), dim=0)


def build_hybrid_input_channels_np(
    sdf_grid: np.ndarray,
    target_points: np.ndarray,
    *,
    grid_n: Optional[int] = None,
    udf_clip: float = 4.0,
    confidence_sigma_scale: float = 1.5,
    chunk_size: int = 2048,
) -> np.ndarray:
    """NumPy wrapper for ``build_hybrid_input_channels``."""
    sdf_tensor = torch.from_numpy(np.asarray(sdf_grid, dtype=np.float32))
    points_tensor = torch.from_numpy(np.asarray(target_points, dtype=np.float32).reshape(-1, 3))
    channels = build_hybrid_input_channels(
        sdf_tensor,
        points_tensor,
        grid_n=grid_n,
        udf_clip=udf_clip,
        confidence_sigma_scale=confidence_sigma_scale,
        chunk_size=chunk_size,
    )
    return channels.detach().cpu().numpy().astype(np.float32)


def _prepare_grid_and_points(
    sdf_grid: torch.Tensor,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    squeezed_points = points.dim() == 2
    if sdf_grid.dim() == 5:
        if sdf_grid.shape[1] != 1:
            raise ValueError(f"5D `sdf_grid` must have one channel, got {sdf_grid.shape}")
        sdf_grid = sdf_grid[:, 0]
    elif sdf_grid.dim() == 3:
        sdf_grid = sdf_grid.unsqueeze(0)
    elif sdf_grid.dim() != 4:
        raise ValueError(f"`sdf_grid` must have shape (G,G,G), (B,G,G,G), or (B,1,G,G,G), got {sdf_grid.shape}")

    if points.dim() == 2:
        points = points.unsqueeze(0)
    elif points.dim() != 3:
        raise ValueError(f"`points` must have shape (N,3) or (B,N,3), got {points.shape}")

    if sdf_grid.shape[0] == 1 and points.shape[0] > 1:
        sdf_grid = sdf_grid.expand(points.shape[0], -1, -1, -1)
    if sdf_grid.shape[0] != points.shape[0]:
        raise ValueError(f"Batch mismatch between grid {sdf_grid.shape} and points {points.shape}")
    return sdf_grid, points, squeezed_points


def trilinear_interpolate_sdf(
    sdf_grid: torch.Tensor,
    points: torch.Tensor,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Differentiably sample a dense SDF grid at normalized ``[-1, 1]^3`` points.

    The integer cell indices are piecewise constant, while the interpolation
    weights remain differentiable with respect to ``points`` inside each cell.
    """
    sdf_grid, points, squeezed_points = _prepare_grid_and_points(sdf_grid, points)
    if sdf_grid.shape[-3] != sdf_grid.shape[-2] or sdf_grid.shape[-2] != sdf_grid.shape[-1]:
        raise ValueError(f"`sdf_grid` must be cubic, got {sdf_grid.shape}")

    grid_n = sdf_grid.shape[-1]
    coords = points.clamp(-1.0, 1.0) if clamp else points
    scaled = (coords + 1.0) * 0.5 * float(grid_n - 1)
    i0 = torch.floor(scaled).to(torch.long).clamp(0, grid_n - 1)
    i1 = (i0 + 1).clamp(0, grid_n - 1)
    frac = (scaled - i0.to(scaled.dtype)).clamp(0.0, 1.0)

    flat = sdf_grid.reshape(sdf_grid.shape[0], -1)

    def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        linear = ix * grid_n * grid_n + iy * grid_n + iz
        return flat.gather(1, linear)

    x0, y0, z0 = i0.unbind(dim=-1)
    x1, y1, z1 = i1.unbind(dim=-1)
    wx, wy, wz = frac.unbind(dim=-1)

    c000 = gather(x0, y0, z0)
    c100 = gather(x1, y0, z0)
    c010 = gather(x0, y1, z0)
    c001 = gather(x0, y0, z1)
    c110 = gather(x1, y1, z0)
    c101 = gather(x1, y0, z1)
    c011 = gather(x0, y1, z1)
    c111 = gather(x1, y1, z1)

    c00 = c000 * (1.0 - wx) + c100 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    values = c0 * (1.0 - wz) + c1 * wz
    return values.squeeze(0) if squeezed_points else values
