"""HotSpot near-surface initialization for iterative refinement."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import torch

from dccvt.neural.grid import make_canonical_sites, trilinear_interpolate_sdf
from dccvt.neural.iterative.config import HybridIterRefineConfig

def _jittered_background_sites(config: HybridIterRefineConfig) -> torch.Tensor:
    sites = make_canonical_sites(config.base_grid_n, dtype=torch.float32).cpu()
    if config.background_jitter_scale > 0.0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.bootstrap_seed)
        noise = torch.randn(sites.shape, generator=generator, dtype=sites.dtype)
        sites = sites + noise * config.background_jitter_scale
    return sites.clamp(-1.0 + 1e-4, 1.0 - 1e-4)


def _crossing_cell_indices(sdf_grid: torch.Tensor) -> torch.Tensor:
    corners = torch.stack(
        [
            sdf_grid[:-1, :-1, :-1],
            sdf_grid[1:, :-1, :-1],
            sdf_grid[:-1, 1:, :-1],
            sdf_grid[:-1, :-1, 1:],
            sdf_grid[1:, 1:, :-1],
            sdf_grid[1:, :-1, 1:],
            sdf_grid[:-1, 1:, 1:],
            sdf_grid[1:, 1:, 1:],
        ],
        dim=0,
    )
    crossing = (corners.amin(dim=0) < 0.0) & (corners.amax(dim=0) > 0.0)
    return torch.nonzero(crossing, as_tuple=False)


def _estimate_grid_gradient(sdf_grid: torch.Tensor, points: torch.Tensor, step: float) -> torch.Tensor:
    gradients: list[torch.Tensor] = []
    for axis in range(3):
        delta = torch.zeros_like(points)
        delta[:, axis] = step
        positive = trilinear_interpolate_sdf(sdf_grid, (points + delta).clamp(-1.0, 1.0)).reshape(-1)
        negative = trilinear_interpolate_sdf(sdf_grid, (points - delta).clamp(-1.0, 1.0)).reshape(-1)
        gradients.append((positive - negative) / (2.0 * step))
    return torch.stack(gradients, dim=1)


def _project_to_crossing_cells(
    sdf_grid: torch.Tensor,
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    steps: int,
    cell_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = points
    gradient_step = cell_size * 0.25
    for _ in range(steps):
        values = trilinear_interpolate_sdf(sdf_grid, projected).reshape(-1)
        gradients = _estimate_grid_gradient(sdf_grid, projected, gradient_step)
        denominator = gradients.square().sum(dim=1).clamp(min=1e-12)
        update = values[:, None] * gradients / denominator[:, None]
        update_norm = update.norm(dim=1, keepdim=True).clamp(min=1e-12)
        update = update * torch.clamp(cell_size / update_norm, max=1.0)
        projected = torch.maximum(torch.minimum(projected - update, upper), lower)
    gradients = _estimate_grid_gradient(sdf_grid, projected, gradient_step)
    return projected, gradients


def _farthest_point_order(points: torch.Tensor, count: int, min_distance: float) -> torch.Tensor:
    if points.numel() == 0 or count <= 0:
        return torch.empty((0,), dtype=torch.long)
    count = min(int(count), points.shape[0])
    points_np = points.numpy()
    center = points_np.mean(axis=0, keepdims=True)
    current = int(np.argmax(np.square(points_np - center).sum(axis=1)))
    selected = np.empty((count,), dtype=np.int64)
    min_dist_sq = np.full((points.shape[0],), np.inf, dtype=np.float32)
    threshold_sq = float(min_distance) ** 2
    for selected_count in range(count):
        selected[selected_count] = current
        distance_sq = np.square(points_np - points_np[current]).sum(axis=1)
        np.minimum(min_dist_sq, distance_sq, out=min_dist_sq)
        min_dist_sq[current] = -1.0
        next_index = int(np.argmax(min_dist_sq))
        if selected_count + 1 < count and float(min_dist_sq[next_index]) < threshold_sq:
            return torch.from_numpy(selected[: selected_count + 1].copy())
        current = next_index
    return torch.from_numpy(selected[: selected_count + 1].copy())


def _select_spaced_pairs(
    anchors: torch.Tensor,
    pairs: torch.Tensor,
    background_sites: torch.Tensor,
    *,
    target_count: int,
    min_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if anchors.numel() == 0 or target_count <= 0:
        return anchors.new_empty((0, 3)), pairs.new_empty((0, 2, 3))

    order_count = min(anchors.shape[0], max(target_count * 3, target_count))
    order = _farthest_point_order(anchors, order_count, min_distance)
    if min_distance <= 0.0:
        chosen = order[:target_count]
        return anchors[chosen], pairs[chosen]

    cell_size = float(min_distance)
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}

    def key(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor((point + 1.0) / cell_size).astype(np.int64).tolist())

    def can_insert(point: np.ndarray) -> bool:
        point_key = key(point)
        for offset in product((-1, 0, 1), repeat=3):
            neighbor_key = tuple(point_key[axis] + offset[axis] for axis in range(3))
            for existing in buckets.get(neighbor_key, ()):
                if float(np.linalg.norm(point - existing)) < cell_size:
                    return False
        return True

    def insert(point: np.ndarray) -> None:
        buckets.setdefault(key(point), []).append(point)

    for point in background_sites.numpy():
        insert(point)

    chosen: list[int] = []
    pairs_np = pairs.numpy()
    for index in order.tolist():
        first, second = pairs_np[index]
        if not can_insert(first) or not can_insert(second):
            continue
        if float(np.linalg.norm(first - second)) < cell_size:
            continue
        insert(first)
        insert(second)
        chosen.append(index)
        if len(chosen) == target_count:
            break
    chosen_tensor = torch.as_tensor(chosen, dtype=torch.long)
    return anchors[chosen_tensor], pairs[chosen_tensor]


def _minimum_pairwise_distance(points: torch.Tensor, chunk_size: int = 512) -> float:
    if points.shape[0] < 2:
        return float("inf")
    minimum = float("inf")
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        distances = torch.cdist(chunk, points)
        row_indices = torch.arange(chunk.shape[0])
        distances[row_indices, row_indices + start] = float("inf")
        minimum = min(minimum, float(distances.min().item()))
    return minimum


def _canonical_initialization(
    background_sites: torch.Tensor,
    background_sdf: torch.Tensor,
) -> dict[str, Any]:
    empty = torch.empty((0, 3), dtype=torch.float32)
    diagnostics = {
        "initialization_mode": "canonical",
        "valid": True,
        "reason": "ok",
        "background_site_count": int(background_sites.shape[0]),
        "surface_anchor_count": 0,
        "surface_site_count": 0,
        "initial_site_count": int(background_sites.shape[0]),
        "positive_sdf_count": int((background_sdf > 0).sum().item()),
        "negative_sdf_count": int((background_sdf < 0).sum().item()),
        "unique_site_count": int(torch.unique(background_sites, dim=0).shape[0]),
        "minimum_site_distance": _minimum_pairwise_distance(background_sites),
        "candidate_multiplier": 0,
        "candidate_count": 0,
        "rejected_candidate_count": 0,
    }
    return {
        "valid": True,
        "reason": "ok",
        "sites": background_sites,
        "sites_sdf": background_sdf,
        "background_sites": background_sites,
        "background_sdf": background_sdf,
        "surface_anchors": empty,
        "surface_sites": empty,
        "surface_sdf": torch.empty((0,), dtype=torch.float32),
        "diagnostics": diagnostics,
    }


def _surface_pair_candidates(
    sdf: torch.Tensor,
    crossing_cells: torch.Tensor,
    config: HybridIterRefineConfig,
    *,
    target_anchor_count: int,
    multiplier: int,
    pass_index: int,
    cell_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cell_lower = -1.0 + crossing_cells.to(torch.float32) * cell_size
    cell_upper = cell_lower + cell_size
    requested_candidates = target_anchor_count * multiplier
    samples_per_cell = max(1, int(np.ceil(requested_candidates / crossing_cells.shape[0])))
    repeated_lower = cell_lower.repeat_interleave(samples_per_cell, dim=0)
    repeated_upper = cell_upper.repeat_interleave(samples_per_cell, dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.bootstrap_seed + pass_index)
    unit_samples = torch.rand(repeated_lower.shape, generator=generator)
    candidates = repeated_lower + unit_samples * (repeated_upper - repeated_lower)
    projected, gradients = _project_to_crossing_cells(
        sdf,
        candidates,
        repeated_lower,
        repeated_upper,
        steps=config.projection_steps,
        cell_size=cell_size,
    )
    gradient_norm = gradients.norm(dim=1, keepdim=True)
    normals = gradients / gradient_norm.clamp(min=1e-12)
    pairs = torch.stack(
        [
            projected - config.surface_pair_offset * normals,
            projected + config.surface_pair_offset * normals,
        ],
        dim=1,
    )
    pair_sdf = trilinear_interpolate_sdf(sdf, pairs.reshape(-1, 3)).reshape(-1, 2)
    finite = torch.isfinite(projected).all(dim=1) & torch.isfinite(pairs).flatten(1).all(dim=1)
    finite &= torch.isfinite(pair_sdf).all(dim=1) & (gradient_norm[:, 0] > 1e-8)
    in_domain = (pairs >= -1.0).flatten(1).all(dim=1) & (pairs <= 1.0).flatten(1).all(dim=1)
    opposite_signs = pair_sdf[:, 0] * pair_sdf[:, 1] < 0.0
    valid = finite & in_domain & opposite_signs
    return projected[valid], pairs[valid]


def _choose_projected_surface_pairs(
    sdf: torch.Tensor,
    crossing_cells: torch.Tensor,
    background_sites: torch.Tensor,
    config: HybridIterRefineConfig,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    target_anchor_count = config.surface_pair_count // 2
    chosen_anchors = torch.empty((0, 3), dtype=torch.float32)
    chosen_pairs = torch.empty((0, 2, 3), dtype=torch.float32)
    candidate_count = 0
    used_multiplier = 0
    if crossing_cells.numel() == 0 or target_anchor_count == 0:
        return chosen_anchors, chosen_pairs, candidate_count, used_multiplier

    cell_size = 2.0 / float(config.hotspot_grid_n - 1)
    for pass_index, multiplier in enumerate(config.bootstrap_candidate_multipliers):
        valid_anchors, valid_pairs = _surface_pair_candidates(
            sdf,
            crossing_cells,
            config,
            target_anchor_count=target_anchor_count,
            multiplier=multiplier,
            pass_index=pass_index,
            cell_size=cell_size,
        )
        candidate_count = int(valid_anchors.shape[0])
        chosen_anchors, chosen_pairs = _select_spaced_pairs(
            valid_anchors,
            valid_pairs,
            background_sites,
            target_count=target_anchor_count,
            min_distance=config.bootstrap_min_distance,
        )
        used_multiplier = multiplier
        if chosen_anchors.shape[0] >= target_anchor_count:
            break
    return chosen_anchors, chosen_pairs, candidate_count, used_multiplier


def _near_surface_initialization_result(
    sdf: torch.Tensor,
    background_sites: torch.Tensor,
    background_sdf: torch.Tensor,
    crossing_cells: torch.Tensor,
    chosen_anchors: torch.Tensor,
    chosen_pairs: torch.Tensor,
    candidate_count: int,
    used_multiplier: int,
    config: HybridIterRefineConfig,
) -> dict[str, Any]:
    surface_sites = chosen_pairs.reshape(-1, 3)
    surface_sdf = trilinear_interpolate_sdf(sdf, surface_sites).reshape(-1)
    sites = torch.cat([background_sites, surface_sites], dim=0)
    sites_sdf = torch.cat([background_sdf, surface_sdf], dim=0)
    valid = chosen_anchors.shape[0] >= config.min_surface_anchors
    if crossing_cells.numel() == 0:
        reason = "no_sign_changing_cells"
    elif not valid:
        reason = "insufficient_valid_surface_pairs"
    else:
        reason = "ok"
    diagnostics = {
        "initialization_mode": config.initialization_mode,
        "valid": bool(valid),
        "reason": reason,
        "crossing_cell_count": int(crossing_cells.shape[0]),
        "requested_surface_anchor_count": int(config.surface_pair_count // 2),
        "surface_anchor_count": int(chosen_anchors.shape[0]),
        "surface_site_count": int(surface_sites.shape[0]),
        "background_site_count": int(background_sites.shape[0]),
        "initial_site_count": int(sites.shape[0]),
        "positive_sdf_count": int((sites_sdf > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf < 0).sum().item()),
        "unique_site_count": int(torch.unique(sites, dim=0).shape[0]),
        "minimum_site_distance": _minimum_pairwise_distance(sites),
        "candidate_multiplier": int(used_multiplier),
        "candidate_count": int(candidate_count),
        "rejected_candidate_count": int(max(candidate_count - chosen_anchors.shape[0], 0)),
    }
    return {
        "valid": bool(valid),
        "reason": reason,
        "sites": sites,
        "sites_sdf": sites_sdf,
        "background_sites": background_sites,
        "background_sdf": background_sdf,
        "surface_anchors": chosen_anchors,
        "surface_sites": surface_sites,
        "surface_sdf": surface_sdf,
        "diagnostics": diagnostics,
    }


def build_hotspot_near_surface_initialization(
    sdf_grid: torch.Tensor | np.ndarray,
    config: HybridIterRefineConfig,
) -> dict[str, Any]:
    """Build a deterministic HotSpot-only initialization field on CPU."""
    sdf = torch.as_tensor(sdf_grid, dtype=torch.float32).squeeze().cpu()
    if sdf.ndim != 3 or len(set(sdf.shape)) != 1:
        raise ValueError(f"Expected a cubic HotSpot SDF grid, got {tuple(sdf.shape)}")
    if sdf.shape[0] != config.hotspot_grid_n:
        raise ValueError(
            f"HotSpot grid shape {tuple(sdf.shape)} does not match hotspot_grid_n={config.hotspot_grid_n}"
        )

    background_sites = _jittered_background_sites(config)
    background_sdf = trilinear_interpolate_sdf(sdf, background_sites).reshape(-1)
    if config.initialization_mode == "canonical":
        return _canonical_initialization(background_sites, background_sdf)

    crossing_cells = _crossing_cell_indices(sdf)
    chosen_anchors, chosen_pairs, candidate_count, used_multiplier = _choose_projected_surface_pairs(
        sdf,
        crossing_cells,
        background_sites,
        config,
    )
    return _near_surface_initialization_result(
        sdf,
        background_sites,
        background_sdf,
        crossing_cells,
        chosen_anchors,
        chosen_pairs,
        candidate_count,
        used_multiplier,
        config,
    )

