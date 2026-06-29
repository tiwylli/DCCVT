"""Iterative learned sparse refinement trained through DCCVT mesh loss."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from itertools import product
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dccvt.neural.dataset import resolve_cache_files
from dccvt.neural.grid import (
    HYBRID_DIRECT_CHANNELS,
    build_hybrid_input_channels_np,
    make_canonical_sites,
    trilinear_interpolate_sdf,
    validate_grid_n,
    validate_hybrid_channel_names,
)
from dccvt.neural.losses import hybrid_direct_mesh_loss
from dccvt.neural.point_udf_sidecar import load_point_udf_sidecar, point_udf_sidecar_path

VALID_INITIALIZATION_MODES = frozenset(("canonical", "hotspot_near_surface"))
VALID_LOCAL_FEATURE_MODES = frozenset(("none", "udf65_knn_stats"))
VALID_ARCHITECTURES = frozenset(("dense_cnn", "delaunay_gcnn"))
VALID_SITE_POSITION_ENCODINGS = frozenset(("fourier",))
VALID_GRAPH_EDGE_FEATURES = frozenset(("relative_xyz_distance_direction_sdf_delta",))
GRAPH_EDGE_FEATURE_DIM = 8


@dataclass
class HybridIterRefineConfig:
    """Configuration for iterative learned sparse refinement."""

    config_version: int = 2
    hotspot_grid_n: int = 33
    initialization_mode: str = "hotspot_near_surface"
    base_grid_n: int = 9
    background_jitter_scale: float = 0.005
    surface_pair_count: int = 3236
    min_surface_anchors: int = 128
    projection_steps: int = 3
    surface_pair_offset: float = 0.03125
    bootstrap_min_distance: float = 0.005
    bootstrap_seed: int = 69
    bootstrap_candidate_multipliers: tuple[int, ...] = (4, 8, 16)
    input_channels: int | None = None
    feature_dim: int = 128
    encoder_layers: int = 5
    decoder_layers: int = 2
    slots_per_parent: int = 4
    max_parents_per_round: int = 128
    num_refinement_rounds: int = 1
    child_stencil_scale: float = 0.015625
    child_offset_scale: float = 0.03125
    sdf_residual_scale: float = 0.0625
    spawn_min_distance: float = 0.0025
    point_udf_clip: float = 4.0
    point_confidence_sigma_scale: float = 1.5
    local_udf_grid_n: int = 0
    local_udf_samples: bool = False
    local_knn_features: bool = False
    local_knn_k: int = 8
    local_knn_radius: float = 0.0625
    local_feature_mode: str = "none"
    parent_selection: str = "procedural_zero_crossing_curvature"
    training_objective: str = "mesh_loss_only"
    channel_names: tuple[str, ...] = field(default_factory=lambda: HYBRID_DIRECT_CHANNELS)
    architecture: str = "dense_cnn"
    graph_layers: int = 3
    graph_hidden_dim: int | None = None
    site_position_encoding: str = "fourier"
    site_position_num_frequencies: int = 4
    graph_edge_features: str = "relative_xyz_distance_direction_sdf_delta"

    def __post_init__(self) -> None:
        self.config_version = int(self.config_version)
        self.hotspot_grid_n = validate_grid_n(self.hotspot_grid_n)
        self.initialization_mode = str(self.initialization_mode)
        self.base_grid_n = validate_grid_n(self.base_grid_n)
        self.background_jitter_scale = float(self.background_jitter_scale)
        self.surface_pair_count = int(self.surface_pair_count)
        self.min_surface_anchors = int(self.min_surface_anchors)
        self.projection_steps = int(self.projection_steps)
        self.surface_pair_offset = float(self.surface_pair_offset)
        self.bootstrap_min_distance = float(self.bootstrap_min_distance)
        self.bootstrap_seed = int(self.bootstrap_seed)
        self.bootstrap_candidate_multipliers = tuple(int(value) for value in self.bootstrap_candidate_multipliers)
        self.channel_names = validate_hybrid_channel_names(self.channel_names)
        if self.input_channels is None:
            self.input_channels = len(self.channel_names)
        else:
            self.input_channels = int(self.input_channels)
        self.feature_dim = int(self.feature_dim)
        self.encoder_layers = int(self.encoder_layers)
        self.decoder_layers = int(self.decoder_layers)
        self.slots_per_parent = int(self.slots_per_parent)
        self.max_parents_per_round = int(self.max_parents_per_round)
        self.num_refinement_rounds = int(self.num_refinement_rounds)
        self.child_stencil_scale = float(self.child_stencil_scale)
        self.child_offset_scale = float(self.child_offset_scale)
        self.sdf_residual_scale = float(self.sdf_residual_scale)
        self.spawn_min_distance = float(self.spawn_min_distance)
        self.point_udf_clip = float(self.point_udf_clip)
        self.point_confidence_sigma_scale = float(self.point_confidence_sigma_scale)
        self.local_udf_grid_n = int(self.local_udf_grid_n)
        self.local_udf_samples = bool(self.local_udf_samples)
        self.local_knn_features = bool(self.local_knn_features)
        self.local_knn_k = int(self.local_knn_k)
        self.local_knn_radius = float(self.local_knn_radius)
        self.local_feature_mode = str(self.local_feature_mode)
        self.parent_selection = str(self.parent_selection)
        self.training_objective = str(self.training_objective)
        self.architecture = str(self.architecture)
        self.graph_layers = int(self.graph_layers)
        if self.graph_hidden_dim is None:
            self.graph_hidden_dim = self.feature_dim
        else:
            self.graph_hidden_dim = int(self.graph_hidden_dim)
        self.site_position_encoding = str(self.site_position_encoding)
        self.site_position_num_frequencies = int(self.site_position_num_frequencies)
        self.graph_edge_features = str(self.graph_edge_features)

        if self.input_channels != len(self.channel_names):
            raise ValueError(
                f"input_channels={self.input_channels} does not match {len(self.channel_names)} channel names"
            )
        if self.config_version not in (1, 2):
            raise ValueError(f"Unsupported config_version: {self.config_version}")
        if self.initialization_mode not in VALID_INITIALIZATION_MODES:
            raise ValueError(f"Unknown initialization_mode: {self.initialization_mode}")
        if self.background_jitter_scale < 0.0:
            raise ValueError("background_jitter_scale must be non-negative")
        if self.surface_pair_count < 0 or self.surface_pair_count % 2 != 0:
            raise ValueError("surface_pair_count must be a non-negative even number")
        if self.min_surface_anchors < 1:
            raise ValueError("min_surface_anchors must be positive")
        if self.projection_steps < 1:
            raise ValueError("projection_steps must be positive")
        if self.surface_pair_offset <= 0.0:
            raise ValueError("surface_pair_offset must be positive")
        if self.bootstrap_min_distance < 0.0:
            raise ValueError("bootstrap_min_distance must be non-negative")
        if not self.bootstrap_candidate_multipliers or any(value < 1 for value in self.bootstrap_candidate_multipliers):
            raise ValueError("bootstrap_candidate_multipliers must contain positive integers")
        if self.slots_per_parent < 1:
            raise ValueError("slots_per_parent must be positive")
        if self.config_version >= 2 and self.slots_per_parent > 4:
            raise ValueError("Version 2 supports at most four tetrahedral child slots per parent")
        if self.max_parents_per_round < 0:
            raise ValueError("max_parents_per_round must be non-negative")
        if self.num_refinement_rounds < 0:
            raise ValueError("num_refinement_rounds must be non-negative")
        if self.child_stencil_scale < 0.0 or self.child_offset_scale < 0.0:
            raise ValueError("child stencil and offset scales must be non-negative")
        if self.sdf_residual_scale < 0.0 or self.spawn_min_distance < 0.0:
            raise ValueError("SDF residual and spawn distance scales must be non-negative")
        if self.local_feature_mode not in VALID_LOCAL_FEATURE_MODES:
            raise ValueError(f"Unknown local_feature_mode: {self.local_feature_mode}")
        if self.local_feature_mode == "udf65_knn_stats":
            if not self.local_udf_samples or not self.local_knn_features:
                raise ValueError("local_feature_mode='udf65_knn_stats' requires local UDF samples and KNN features")
            if self.local_udf_grid_n != 65:
                raise ValueError("local_feature_mode='udf65_knn_stats' requires local_udf_grid_n=65")
        if self.local_udf_samples:
            self.local_udf_grid_n = validate_grid_n(self.local_udf_grid_n)
        elif self.local_udf_grid_n < 0:
            raise ValueError("local_udf_grid_n must be non-negative")
        if self.local_knn_k < 1:
            raise ValueError("local_knn_k must be positive")
        if self.local_knn_radius <= 0.0:
            raise ValueError("local_knn_radius must be positive")
        if self.parent_selection != "procedural_zero_crossing_curvature":
            raise ValueError(f"Unknown parent_selection: {self.parent_selection}")
        if self.training_objective != "mesh_loss_only":
            raise ValueError(f"Unknown training_objective: {self.training_objective}")
        if self.architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if self.graph_layers < 0:
            raise ValueError("graph_layers must be non-negative")
        if self.graph_hidden_dim < 1:
            raise ValueError("graph_hidden_dim must be positive")
        if self.site_position_encoding not in VALID_SITE_POSITION_ENCODINGS:
            raise ValueError(f"Unknown site_position_encoding: {self.site_position_encoding}")
        if self.site_position_num_frequencies < 0:
            raise ValueError("site_position_num_frequencies must be non-negative")
        if self.graph_edge_features not in VALID_GRAPH_EDGE_FEATURES:
            raise ValueError(f"Unknown graph_edge_features: {self.graph_edge_features}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridIterRefineConfig":
        values = dict(data)
        if "config_version" not in values:
            values["config_version"] = 1
            values.setdefault("initialization_mode", "canonical")
            values.setdefault("background_jitter_scale", 0.0)
            values.setdefault("child_stencil_scale", 0.0)
            values.setdefault("spawn_min_distance", 0.0)
        if "channel_names" in values:
            values["channel_names"] = tuple(values["channel_names"])
        if "bootstrap_candidate_multipliers" in values:
            values["bootstrap_candidate_multipliers"] = tuple(values["bootstrap_candidate_multipliers"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channel_names"] = list(self.channel_names)
        data["bootstrap_candidate_multipliers"] = list(self.bootstrap_candidate_multipliers)
        return data

    @property
    def local_feature_dim(self) -> int:
        """Return the per-parent local feature width added to the decoder."""
        width = 0
        if self.local_udf_samples:
            width += 1 + self.slots_per_parent
        if self.local_knn_features:
            width += 7
        return width

    @property
    def site_position_feature_dim(self) -> int:
        """Return the encoded site-position feature width."""
        return 3 + 3 * 2 * self.site_position_num_frequencies

    @property
    def graph_node_input_dim(self) -> int:
        """Return the per-site graph input feature width."""
        width = int(self.input_channels) + 1 + self.site_position_feature_dim
        if self.local_udf_samples:
            width += 1
        if self.local_knn_features:
            width += 7
        return width


def load_iter_refine_config(path: str | Path | None = None) -> HybridIterRefineConfig:
    """Load an iterative-refinement JSON config, or return defaults."""
    if path is None:
        return HybridIterRefineConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return HybridIterRefineConfig.from_dict(data)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training/inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_mesh_ids(value: str | None) -> list[str] | None:
    """Parse comma- or whitespace-separated mesh ids from a CLI value."""
    if value is None:
        return None
    return [part for part in value.replace(",", " ").split() if part]


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


def _build_neighbors_from_simplices(simplices: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    tets = torch.as_tensor(simplices, device=device).long()
    if tets.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    edges = torch.cat(
        [
            tets[:, [0, 1]],
            tets[:, [1, 2]],
            tets[:, [2, 3]],
            tets[:, [3, 0]],
            tets[:, [0, 2]],
            tets[:, [1, 3]],
        ],
        dim=0,
    )
    neighbors, _ = torch.sort(edges, dim=1)
    return torch.unique(neighbors, dim=0)


def build_directed_edges_from_simplices(
    simplices: np.ndarray | torch.Tensor,
    *,
    num_sites: int,
    device: torch.device,
) -> torch.Tensor:
    """Return unique bidirectional Delaunay graph edges from tetrahedra."""
    neighbors = _build_neighbors_from_simplices(simplices, device)
    if neighbors.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    if int(neighbors.min().item()) < 0 or int(neighbors.max().item()) >= int(num_sites):
        raise ValueError("Delaunay simplex indices are outside the site range")
    directed = torch.cat([neighbors, neighbors[:, [1, 0]]], dim=0)
    return torch.unique(directed, dim=0)


def _neighbor_counts(neighbors: torch.Tensor, num_sites: int, device: torch.device) -> torch.Tensor:
    ones = torch.ones((neighbors.shape[0],), device=device)
    counts = torch.zeros((num_sites,), device=device)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    return counts


def _min_neighbor_distances(sites: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]
    edge_len = torch.norm(edge_vec, dim=1)
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((sites.shape[0],), float("inf"), device=sites.device)
    return min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")


def _curvature_score(
    neighbors: torch.Tensor,
    grad_est: torch.Tensor,
    num_sites: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)
    counts = _neighbor_counts(neighbors, num_sites, device).clamp(min=1.0)
    dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1) * 0.8 + 0.2
    scores = torch.zeros(num_sites, device=device)
    scores = scores.index_add(0, neighbors[:, 0], dn2)
    scores = scores.index_add(0, neighbors[:, 1], dn2)
    return scores / counts


def _zero_crossing_sites(neighbors: torch.Tensor, sdf_values: torch.Tensor) -> torch.Tensor:
    sdf_i = sdf_values[neighbors[:, 0]]
    sdf_j = sdf_values[neighbors[:, 1]]
    mask = sdf_i * sdf_j <= 0
    return torch.unique(neighbors[mask].reshape(-1))


def _select_unique_to_budget(
    indices: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.numel() == 0 or budget <= 0:
        return indices.new_empty((0,)), scores.new_empty((0,))
    count = min(int(budget), int(indices.numel()))
    order = torch.topk(scores, k=count, largest=True, sorted=True).indices
    return indices[order], scores[order]


def fourier_site_position_encoding(sites: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    """Encode normalized site coordinates with low-frequency Fourier features."""
    sites = sites.reshape(-1, 3)
    parts = [sites]
    for exponent in range(int(num_frequencies)):
        frequency = float(2**exponent)
        phase = torch.pi * frequency * sites
        parts.extend([torch.sin(phase), torch.cos(phase)])
    return torch.cat(parts, dim=1)


def delaunay_edge_features(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    directed_edges: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return relative geometry and SDF-delta features for directed graph edges."""
    if directed_edges.numel() == 0:
        return sites.new_empty((0, GRAPH_EDGE_FEATURE_DIM))
    src = directed_edges[:, 0]
    dst = directed_edges[:, 1]
    delta = sites[dst] - sites[src]
    distance = delta.norm(dim=1, keepdim=True).clamp_min(float(eps))
    direction = delta / distance
    sdf_delta = sites_sdf.reshape(-1)[dst, None] - sites_sdf.reshape(-1)[src, None]
    features = torch.cat([delta, distance, direction, sdf_delta], dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)


def local_knn_parent_features(
    parent_sites: torch.Tensor,
    target_points: torch.Tensor,
    *,
    k: int,
    radius: float,
) -> torch.Tensor:
    """Return local input-point statistics for each refinement parent."""
    parent_sites = parent_sites.reshape(-1, 3)
    target_points = target_points.reshape(-1, 3).to(device=parent_sites.device, dtype=parent_sites.dtype)
    if parent_sites.numel() == 0:
        return parent_sites.new_empty((0, 7))
    if target_points.numel() == 0:
        return parent_sites.new_zeros((parent_sites.shape[0], 7))

    k = min(int(k), int(target_points.shape[0]))
    radius = float(radius)
    distances = torch.cdist(parent_sites.unsqueeze(0), target_points.unsqueeze(0), p=2).squeeze(0)
    knn_dist, knn_idx = torch.topk(distances, k=k, largest=False, sorted=True)
    knn_points = target_points[knn_idx]
    offsets = knn_points - parent_sites[:, None, :]
    nearest_distance = knn_dist[:, :1] / radius
    mean_offset = offsets.mean(dim=1) / radius
    mean_distance = knn_dist.mean(dim=1, keepdim=True) / radius
    radius_density = (distances <= radius).sum(dim=1, keepdim=True).to(parent_sites.dtype) / max(int(k), 1)

    centered = offsets - offsets.mean(dim=1, keepdim=True)
    if k > 1:
        covariance = centered.transpose(1, 2) @ centered / float(k - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        anisotropy = (eigenvalues[:, -1:] - eigenvalues[:, :1]) / eigenvalues[:, -1:].clamp_min(1e-12)
    else:
        anisotropy = parent_sites.new_zeros((parent_sites.shape[0], 1))

    features = torch.cat(
        [
            nearest_distance.clamp(max=8.0),
            mean_offset.clamp(min=-8.0, max=8.0),
            mean_distance.clamp(max=8.0),
            radius_density.clamp(max=8.0),
            anisotropy.clamp(min=0.0, max=1.0),
        ],
        dim=1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)


def select_procedural_refinement_parents(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    *,
    max_parents: int,
    simplices: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Select up to a fixed budget of unique zero-crossing Delaunay sites."""
    if max_parents <= 0 or sites.shape[0] < 5:
        empty = torch.empty((0,), dtype=torch.long, device=sites.device)
        return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": np.empty((0, 4))}
    if not ((sites_sdf.min() < 0) and (sites_sdf.max() > 0)):
        empty = torch.empty((0,), dtype=torch.long, device=sites.device)
        return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": np.empty((0, 4))}

    from dccvt.geometry import compute_delaunay_simplices
    from dccvt.sdf_gradients import compute_sdf_gradients_sites_tets

    with torch.no_grad():
        if simplices is None:
            simplices = compute_delaunay_simplices(sites.detach())
        else:
            simplices = np.asarray(simplices)
        if simplices.size == 0:
            empty = torch.empty((0,), dtype=torch.long, device=sites.device)
            return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": simplices}
        neighbors = _build_neighbors_from_simplices(simplices, sites.device)
        zc_sites = _zero_crossing_sites(neighbors, sites_sdf.detach().reshape(-1))
        if zc_sites.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=sites.device)
            return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": simplices}

        tets = torch.as_tensor(simplices, device=sites.device).long()
        grad_est, _, _ = compute_sdf_gradients_sites_tets(sites.detach(), sites_sdf.detach().reshape(-1), tets)
        min_dists = _min_neighbor_distances(sites.detach(), neighbors)
        curv = _curvature_score(neighbors, grad_est, sites.shape[0], sites.device, eps)
        dist_scale = torch.median(min_dists[zc_sites]).clamp(min=eps)
        curv_scale = torch.median(curv[zc_sites]).clamp(min=eps)
        scores = (min_dists[zc_sites] / dist_scale) * (curv[zc_sites] / curv_scale)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        parent_indices, parent_scores = _select_unique_to_budget(zc_sites, scores, int(max_parents))
    return {
        "parent_indices": parent_indices.long(),
        "parent_scores": parent_scores,
        "simplices": simplices,
    }


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


class DelaunayGraphMessageLayer(nn.Module):
    """Simple mean-aggregation message passing over directed Delaunay edges."""

    def __init__(self, hidden_dim: int, edge_dim: int = GRAPH_EDGE_FEATURE_DIM) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        directed_edges: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if directed_edges.numel() == 0:
            aggregated = torch.zeros_like(node_features)
        else:
            src = directed_edges[:, 0]
            dst = directed_edges[:, 1]
            message_input = torch.cat([node_features[src], node_features[dst], edge_features], dim=1)
            messages = self.message_mlp(message_input)
            aggregated = torch.zeros_like(node_features)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros((node_features.shape[0], 1), device=node_features.device, dtype=node_features.dtype)
            degree_ones = torch.ones((dst.shape[0], 1), device=node_features.device, dtype=node_features.dtype)
            degree.index_add_(0, dst, degree_ones)
            aggregated = aggregated / degree.clamp_min(1.0)
        update = self.update_mlp(torch.cat([node_features, aggregated], dim=1))
        return node_features + update


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


class DCCVTHybridIterRefineNet(nn.Module):
    """Iteratively spawn learned refinement sites from procedural DCCVT parents."""

    def __init__(self, config: HybridIterRefineConfig | dict | None = None, **overrides) -> None:
        super().__init__()
        if config is None:
            config_obj = HybridIterRefineConfig(**overrides)
        elif isinstance(config, dict):
            config_obj = HybridIterRefineConfig.from_dict({**config, **overrides})
        else:
            config_obj = HybridIterRefineConfig.from_dict({**config.to_dict(), **overrides})
        self.config_obj = config_obj

        if config_obj.architecture == "dense_cnn":
            encoder: list[nn.Module] = [
                nn.Conv3d(config_obj.input_channels, config_obj.feature_dim, kernel_size=2, stride=1, bias=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ]
            for _ in range(config_obj.encoder_layers):
                encoder.extend(
                    [
                        nn.Conv3d(config_obj.feature_dim, config_obj.feature_dim, kernel_size=3, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.01, inplace=True),
                    ]
                )
            self.encoder = nn.Sequential(*encoder)
            parent_feature_dim = config_obj.feature_dim
        else:
            self.graph_input = nn.Sequential(
                nn.Linear(config_obj.graph_node_input_dim, config_obj.graph_hidden_dim),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            self.graph_layers = nn.ModuleList(
                [
                    DelaunayGraphMessageLayer(config_obj.graph_hidden_dim, GRAPH_EDGE_FEATURE_DIM)
                    for _ in range(config_obj.graph_layers)
                ]
            )
            parent_feature_dim = config_obj.graph_hidden_dim

        parent_dim = parent_feature_dim + 4 + config_obj.local_feature_dim
        decoder: list[nn.Module] = []
        for _ in range(config_obj.decoder_layers):
            decoder.extend(
                [
                    nn.Linear(parent_dim, parent_dim),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ]
            )
        output_layer = nn.Linear(parent_dim, config_obj.slots_per_parent * 4)
        if config_obj.config_version >= 2:
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        decoder.append(output_layer)
        self.refine_decoder = nn.Sequential(*decoder)
        tetrahedral_directions = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        tetrahedral_directions = tetrahedral_directions / tetrahedral_directions.norm(dim=1, keepdim=True)
        if config_obj.config_version == 1:
            tetrahedral_directions = torch.zeros((config_obj.slots_per_parent, 3), dtype=torch.float32)
        self.register_buffer(
            "child_stencil",
            tetrahedral_directions[: config_obj.slots_per_parent],
            persistent=False,
        )

    def config(self) -> dict[str, Any]:
        return self.config_obj.to_dict()

    def _sample_features(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return features.new_empty((0, features.shape[1]))
        grid_points = points[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3).clamp(-1.0, 1.0)
        sampled = F.grid_sample(features, grid_points, mode="bilinear", align_corners=True)
        return sampled[0, :, :, 0, 0].transpose(0, 1)

    def _sample_scalar_grid(self, scalar_grid: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return points.new_empty((0,))
        if scalar_grid.dim() == 3:
            scalar_grid = scalar_grid[None, None, ...]
        elif scalar_grid.dim() == 4:
            scalar_grid = scalar_grid[:, None, ...] if scalar_grid.shape[0] == 1 else scalar_grid[None, ...]
        elif scalar_grid.dim() != 5:
            raise ValueError(f"Expected scalar grid with 3, 4, or 5 dims, got {scalar_grid.shape}")
        if scalar_grid.shape[0] != 1 or scalar_grid.shape[1] != 1:
            raise ValueError(f"Expected scalar grid shape (1,1,G,G,G), got {scalar_grid.shape}")
        grid_points = points[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3).clamp(-1.0, 1.0)
        sampled = F.grid_sample(scalar_grid, grid_points, mode="bilinear", align_corners=True)
        return sampled[0, 0, :, 0, 0]

    def _local_udf_parent_features(
        self,
        parent_sites: torch.Tensor,
        local_udf_grid: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config_obj.local_udf_samples:
            return parent_sites.new_empty((parent_sites.shape[0], 0))
        if local_udf_grid is None or local_udf_grid.numel() == 0:
            raise ValueError("Config requests local UDF samples but no local UDF grid was provided")

        stencil = self.child_stencil.to(device=parent_sites.device, dtype=parent_sites.dtype)
        stencil = stencil[None, :, :] * self.config_obj.child_stencil_scale
        sample_points = torch.cat(
            [parent_sites[:, None, :], parent_sites[:, None, :] + stencil],
            dim=1,
        ).reshape(-1, 3)
        values = self._sample_scalar_grid(local_udf_grid.to(device=parent_sites.device, dtype=parent_sites.dtype), sample_points)
        cell_size = 2.0 / float(self.config_obj.local_udf_grid_n - 1)
        values = (values / cell_size).clamp(min=0.0, max=self.config_obj.point_udf_clip)
        return values.reshape(parent_sites.shape[0], 1 + self.config_obj.slots_per_parent)

    def _local_udf_site_features(
        self,
        sites: torch.Tensor,
        local_udf_grid: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config_obj.local_udf_samples:
            return sites.new_empty((sites.shape[0], 0))
        if local_udf_grid is None or local_udf_grid.numel() == 0:
            raise ValueError("Config requests local UDF samples but no local UDF grid was provided")
        values = self._sample_scalar_grid(local_udf_grid.to(device=sites.device, dtype=sites.dtype), sites)
        cell_size = 2.0 / float(self.config_obj.local_udf_grid_n - 1)
        return (values[:, None] / cell_size).clamp(min=0.0, max=self.config_obj.point_udf_clip)

    def _local_parent_features(
        self,
        parent_sites: torch.Tensor,
        target_points: torch.Tensor | None,
        local_udf_grid: torch.Tensor | None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.config_obj.local_udf_samples:
            parts.append(self._local_udf_parent_features(parent_sites, local_udf_grid))
        if self.config_obj.local_knn_features:
            if target_points is None:
                raise ValueError("Config requests local KNN features but no target points were provided")
            if target_points.dim() == 3:
                target_points = target_points[0]
            parts.append(
                local_knn_parent_features(
                    parent_sites,
                    target_points,
                    k=self.config_obj.local_knn_k,
                    radius=self.config_obj.local_knn_radius,
                )
            )
        if not parts:
            return parent_sites.new_empty((parent_sites.shape[0], 0))
        return torch.cat(parts, dim=1)

    def _graph_site_features(
        self,
        input_grid: torch.Tensor,
        sites: torch.Tensor,
        sites_sdf: torch.Tensor,
        simplices: np.ndarray,
        target_points: torch.Tensor | None,
        local_udf_grid: torch.Tensor | None,
    ) -> torch.Tensor:
        sampled_channels = self._sample_features(input_grid, sites)
        position_features = fourier_site_position_encoding(
            sites,
            self.config_obj.site_position_num_frequencies,
        )
        parts = [sampled_channels, sites_sdf.reshape(-1, 1), position_features]
        if self.config_obj.local_udf_samples:
            parts.append(self._local_udf_site_features(sites, local_udf_grid))
        if self.config_obj.local_knn_features:
            if target_points is None:
                raise ValueError("Config requests local KNN features but no target points were provided")
            if target_points.dim() == 3:
                target_points = target_points[0]
            parts.append(
                local_knn_parent_features(
                    sites,
                    target_points,
                    k=self.config_obj.local_knn_k,
                    radius=self.config_obj.local_knn_radius,
                )
            )
        node_features = self.graph_input(torch.cat(parts, dim=1))
        directed_edges = build_directed_edges_from_simplices(
            simplices,
            num_sites=sites.shape[0],
            device=sites.device,
        )
        edge_features = delaunay_edge_features(sites, sites_sdf, directed_edges)
        for layer in self.graph_layers:
            node_features = layer(node_features, directed_edges, edge_features)
        return node_features

    def _filter_spawned_sites(
        self,
        spawned_sites: torch.Tensor,
        spawned_sdf: torch.Tensor,
        existing_sites: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        min_distance = self.config_obj.spawn_min_distance
        if spawned_sites.numel() == 0 or min_distance <= 0.0:
            return spawned_sites, spawned_sdf, 0
        with torch.no_grad():
            detached = spawned_sites.detach()
            keep = torch.ones(detached.shape[0], dtype=torch.bool, device=detached.device)
            if existing_sites.numel() > 0:
                keep &= torch.cdist(detached, existing_sites.detach()).amin(dim=1) >= min_distance
            accepted_indices: list[int] = []
            for index in torch.nonzero(keep, as_tuple=False).reshape(-1).tolist():
                if accepted_indices:
                    accepted = detached[torch.as_tensor(accepted_indices, device=detached.device)]
                    if float(torch.norm(detached[index] - accepted, dim=1).amin().item()) < min_distance:
                        keep[index] = False
                        continue
                accepted_indices.append(index)
        rejected = int((~keep).sum().item())
        return spawned_sites[keep], spawned_sdf[keep], rejected

    def _spawn_from_parents(
        self,
        features: torch.Tensor,
        sdf_grid: torch.Tensor,
        sites: torch.Tensor,
        sites_sdf: torch.Tensor,
        parent_indices: torch.Tensor,
        target_points: torch.Tensor | None,
        local_udf_grid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if parent_indices.numel() == 0:
            return sites.new_empty((0, 3)), sites_sdf.new_empty((0,)), 0
        parent_sites = sites[parent_indices]
        parent_sdf = sites_sdf[parent_indices].unsqueeze(1)
        parent_features = self._sample_features(features, parent_sites)
        return self._spawn_from_parent_features(
            parent_features,
            sdf_grid,
            sites,
            sites_sdf,
            parent_indices,
            target_points,
            local_udf_grid,
        )

    def _spawn_from_parent_features(
        self,
        parent_features: torch.Tensor,
        sdf_grid: torch.Tensor,
        sites: torch.Tensor,
        sites_sdf: torch.Tensor,
        parent_indices: torch.Tensor,
        target_points: torch.Tensor | None,
        local_udf_grid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if parent_indices.numel() == 0:
            return sites.new_empty((0, 3)), sites_sdf.new_empty((0,)), 0
        parent_sites = sites[parent_indices]
        parent_sdf = sites_sdf[parent_indices].unsqueeze(1)
        local_features = self._local_parent_features(parent_sites, target_points, local_udf_grid)
        decoder_input = torch.cat([parent_features, parent_sites, parent_sdf, local_features], dim=1)
        decoded = self.refine_decoder(decoder_input)
        decoded = decoded.reshape(parent_indices.numel(), self.config_obj.slots_per_parent, 4)

        stencil = self.child_stencil.to(device=sites.device, dtype=sites.dtype)
        stencil = stencil[None, :, :] * self.config_obj.child_stencil_scale
        offsets = stencil + torch.tanh(decoded[..., :3]) * self.config_obj.child_offset_scale
        residuals = torch.tanh(decoded[..., 3]) * self.config_obj.sdf_residual_scale
        spawned_sites = (parent_sites[:, None, :] + offsets).reshape(-1, 3).clamp(-1.0, 1.0)
        hotspot_sdf = trilinear_interpolate_sdf(sdf_grid, spawned_sites).reshape(-1)
        spawned_sdf = hotspot_sdf + residuals.reshape(-1)
        return self._filter_spawned_sites(spawned_sites, spawned_sdf, sites)

    def forward(
        self,
        input_grid: torch.Tensor,
        hotspot_sdf_grid: torch.Tensor | None = None,
        initial_field: Optional[dict[str, Any]] = None,
        target_points: torch.Tensor | None = None,
        local_udf_grid: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if input_grid.dim() == 4:
            input_grid = input_grid.unsqueeze(0)
        if input_grid.dim() != 5:
            raise ValueError(f"Expected input shape (B,C,G,G,G) or (C,G,G,G), got {input_grid.shape}")
        if input_grid.shape[0] != 1:
            raise ValueError("DCCVTHybridIterRefineNet currently supports batch_size=1 because Delaunay topology is variable")
        if input_grid.shape[1] != self.config_obj.input_channels:
            raise ValueError(f"Expected {self.config_obj.input_channels} input channels, got {input_grid.shape}")
        if len(set(input_grid.shape[-3:])) != 1:
            raise ValueError(f"Expected cubic input grid, got {input_grid.shape}")

        if hotspot_sdf_grid is None:
            hotspot_sdf_grid = input_grid[:, 0]
        if hotspot_sdf_grid.dim() == 5:
            hotspot_sdf_grid = hotspot_sdf_grid[:, 0]
        if hotspot_sdf_grid.dim() != 4 or hotspot_sdf_grid.shape[0] != 1:
            raise ValueError(f"Expected HotSpot SDF shape (1,G,G,G) or (1,1,G,G,G), got {hotspot_sdf_grid.shape}")

        dense_features = self.encoder(input_grid) if self.config_obj.architecture == "dense_cnn" else None
        if initial_field is None:
            initial_field = build_hotspot_near_surface_initialization(
                hotspot_sdf_grid[0].detach().cpu(),
                self.config_obj,
            )

        def initial_tensor(name: str) -> torch.Tensor:
            return initial_field[name].to(device=input_grid.device, dtype=input_grid.dtype)

        sites = initial_tensor("sites")
        sites_sdf = initial_tensor("sites_sdf")
        background_sites = initial_tensor("background_sites")
        background_sdf = initial_tensor("background_sdf")
        surface_anchors = initial_tensor("surface_anchors")
        surface_sites = initial_tensor("surface_sites")
        surface_sdf = initial_tensor("surface_sdf")
        rounds: list[dict[str, torch.Tensor]] = []

        for round_index in range(self.config_obj.num_refinement_rounds if initial_field["valid"] else 0):
            parent_data = select_procedural_refinement_parents(
                sites.detach(),
                sites_sdf.detach(),
                max_parents=self.config_obj.max_parents_per_round,
            )
            parent_indices = parent_data["parent_indices"]
            assert isinstance(parent_indices, torch.Tensor)
            parent_scores = parent_data["parent_scores"]
            assert isinstance(parent_scores, torch.Tensor)
            if parent_indices.numel() == 0:
                spawned_sites, spawned_sdf, rejected_spawn_count = sites.new_empty((0, 3)), sites_sdf.new_empty((0,)), 0
            elif self.config_obj.architecture == "dense_cnn":
                assert dense_features is not None
                spawned_sites, spawned_sdf, rejected_spawn_count = self._spawn_from_parents(
                    dense_features,
                    hotspot_sdf_grid,
                    sites,
                    sites_sdf,
                    parent_indices,
                    target_points,
                    local_udf_grid,
                )
            else:
                simplices = parent_data["simplices"]
                assert isinstance(simplices, np.ndarray)
                graph_features = self._graph_site_features(
                    input_grid,
                    sites,
                    sites_sdf,
                    simplices,
                    target_points,
                    local_udf_grid,
                )
                spawned_sites, spawned_sdf, rejected_spawn_count = self._spawn_from_parent_features(
                    graph_features[parent_indices],
                    hotspot_sdf_grid,
                    sites,
                    sites_sdf,
                    parent_indices,
                    target_points,
                    local_udf_grid,
                )
            rounds.append(
                {
                    "round_index": torch.tensor(round_index, device=input_grid.device),
                    "parent_indices": parent_indices,
                    "parent_scores": parent_scores,
                    "spawned_sites": spawned_sites,
                    "spawned_sdf": spawned_sdf,
                    "rejected_spawn_count": torch.tensor(rejected_spawn_count, device=input_grid.device),
                }
            )
            if spawned_sites.numel() == 0:
                break
            sites = torch.cat([sites, spawned_sites], dim=0)
            sites_sdf = torch.cat([sites_sdf, spawned_sdf], dim=0)

        return {
            "sites": sites.unsqueeze(0),
            "sites_sdf": sites_sdf.unsqueeze(0),
            "base_sites": initial_tensor("sites"),
            "base_sites_sdf": initial_tensor("sites_sdf"),
            "background_sites": background_sites,
            "background_sites_sdf": background_sdf,
            "surface_anchors": surface_anchors,
            "surface_sites": surface_sites,
            "surface_sites_sdf": surface_sdf,
            "initialization_valid": bool(initial_field["valid"]),
            "initialization_reason": str(initial_field["reason"]),
            "initialization_diagnostics": dict(initial_field["diagnostics"]),
            "rounds": rounds,
        }


def run_iterative_refinement(
    model: DCCVTHybridIterRefineNet,
    input_grid: torch.Tensor,
    hotspot_sdf_grid: torch.Tensor,
    initial_field: Optional[dict[str, Any]] = None,
    target_points: torch.Tensor | None = None,
    local_udf_grid: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run the iterative refinement model and return extraction-ready fields."""
    return model(
        input_grid,
        hotspot_sdf_grid,
        initial_field=initial_field,
        target_points=target_points,
        local_udf_grid=local_udf_grid,
    )


def save_checkpoint(
    path: Path,
    *,
    model: DCCVTHybridIterRefineNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    stats: dict[str, float],
) -> None:
    """Save iterative-refinement training state."""
    _assert_finite_model_parameters(model)
    payload = {
        "config_version": int(model.config_obj.config_version),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "seed": int(args.seed),
        "args": vars(args),
        "stats": stats,
    }
    torch.save(payload, path)


def _nonfinite_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            names.append(name)
    return names


def _assert_finite_model_parameters(model: nn.Module) -> None:
    names = _nonfinite_parameter_names(model)
    if names:
        shown = ", ".join(names[:8])
        suffix = "" if len(names) <= 8 else f", ... ({len(names)} total)"
        raise RuntimeError(f"Non-finite model parameters detected: {shown}{suffix}")


def save_resolved_config(path: Path, *, config: HybridIterRefineConfig, args: argparse.Namespace) -> None:
    """Save resolved model config and command-line arguments."""
    payload = {
        "model_config": config.to_dict(),
        "seed": int(args.seed),
        "args": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _apply_model_overrides(config: HybridIterRefineConfig, args: argparse.Namespace) -> HybridIterRefineConfig:
    values = config.to_dict()
    feature_dim_override = getattr(args, "feature_dim", None) if hasattr(args, "feature_dim") else None
    if feature_dim_override is not None and values.get("graph_hidden_dim") == config.feature_dim:
        values["graph_hidden_dim"] = feature_dim_override
    for arg_name, key in (
        ("initialization_mode", "initialization_mode"),
        ("hotspot_grid_n", "hotspot_grid_n"),
        ("base_grid_n", "base_grid_n"),
        ("feature_dim", "feature_dim"),
        ("encoder_layers", "encoder_layers"),
        ("decoder_layers", "decoder_layers"),
        ("slots_per_parent", "slots_per_parent"),
        ("max_parents_per_round", "max_parents_per_round"),
        ("num_refinement_rounds", "num_refinement_rounds"),
        ("child_offset_scale", "child_offset_scale"),
        ("sdf_residual_scale", "sdf_residual_scale"),
        ("graph_layers", "graph_layers"),
        ("graph_hidden_dim", "graph_hidden_dim"),
    ):
        if hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                values[key] = value
    return HybridIterRefineConfig.from_dict(values)


def _resolve_resume_config(
    requested_config: HybridIterRefineConfig,
    resume_checkpoint: Optional[dict[str, Any]],
) -> HybridIterRefineConfig:
    if resume_checkpoint is None or not resume_checkpoint.get("model_config"):
        return requested_config
    checkpoint_config = HybridIterRefineConfig.from_dict(resume_checkpoint["model_config"])
    if checkpoint_config.initialization_mode != requested_config.initialization_mode:
        raise ValueError(
            "Cannot resume with a different initialization mode: "
            f"checkpoint={checkpoint_config.initialization_mode}, requested={requested_config.initialization_mode}"
        )
    if checkpoint_config.base_grid_n != requested_config.base_grid_n:
        raise ValueError(
            "Cannot resume with a different base grid: "
            f"checkpoint={checkpoint_config.base_grid_n}, requested={requested_config.base_grid_n}"
        )
    if checkpoint_config.surface_pair_count != requested_config.surface_pair_count:
        raise ValueError(
            "Cannot resume with a different near-surface pair count: "
            f"checkpoint={checkpoint_config.surface_pair_count}, requested={requested_config.surface_pair_count}"
        )
    return checkpoint_config


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train iterative learned sparse refinement with mesh loss only.")
    parser.add_argument("--config", default="configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p128.json")
    parser.add_argument("--cache-root", default="outputs/neural_hotspot_sdf/thingi32_g33")
    parser.add_argument("--local-udf-root", default=None)
    parser.add_argument("--allow-missing-local-features", action="store_true")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None)
    parser.add_argument("--checkpoint-dir", default="outputs/neural_dccvt/hybrid_iter_refine_v2_hotspot_point_udf_r1_p128/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-subsample", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6.4e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--w-mesh-chamfer", type=float, default=1000.0)
    parser.add_argument("--w-mesh-cvt", type=float, default=100.0)
    parser.add_argument("--w-mesh-sdfsmooth", type=float, default=100.0)
    parser.add_argument("--strict-mesh-loss", action="store_true")
    parser.add_argument("--strict-initialization", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--initialization-mode", choices=sorted(VALID_INITIALIZATION_MODES), default=None)
    parser.add_argument("--hotspot-grid-n", type=int, default=None)
    parser.add_argument("--base-grid-n", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--slots-per-parent", type=int, default=None)
    parser.add_argument("--max-parents-per-round", type=int, default=None)
    parser.add_argument("--num-refinement-rounds", type=int, default=None)
    parser.add_argument("--child-offset-scale", type=float, default=None)
    parser.add_argument("--sdf-residual-scale", type=float, default=None)
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--graph-hidden-dim", type=int, default=None)
    return parser


def train_main(argv: Optional[list[str]] = None) -> None:
    args = build_train_arg_parser().parse_args(argv)
    if args.batch_size != 1:
        raise ValueError("Iterative refinement training currently requires --batch-size 1")
    seed_everything(args.seed)
    device = _device(args.device)

    requested_config = _apply_model_overrides(load_iter_refine_config(args.config), args)
    resume_checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    config = _resolve_resume_config(requested_config, resume_checkpoint)
    model = DCCVTHybridIterRefineNet(config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])

    cache_files = resolve_cache_files(
        args.cache_root,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        split_file=args.split_file,
    )
    dataset = HybridIterRefineDataset(
        cache_files,
        config=config,
        target_subsample=args.target_subsample,
        local_udf_root=args.local_udf_root,
        allow_missing_local_features=args.allow_missing_local_features,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(checkpoint_dir / "resolved_config.json", config=config, args=args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if resume_checkpoint is not None:
        if args.resume_optimizer and "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    stop_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, stop_epoch):
        local_epoch = epoch - start_epoch + 1
        model.train()
        epoch_loss = 0.0
        epoch_stats: dict[str, float] = {}
        for batch in dataloader:
            input_grid = batch["input_grid"].to(device, non_blocking=True)
            sdf_grid = batch["sdf_grid"].to(device, non_blocking=True)
            target_points = batch["target_points"].to(device, non_blocking=True)
            local_target_points = batch["local_target_points"].to(device, non_blocking=True)
            local_udf_grid = None
            if config.local_udf_samples:
                local_udf_grid = batch["local_udf_grid"].to(device, non_blocking=True)
            initial_field = _initialization_from_batch(batch, device, input_grid.dtype)
            if not initial_field["valid"]:
                reason = initial_field["reason"]
                if args.strict_initialization:
                    raise RuntimeError(f"Invalid near-surface initialization: {reason}")
                epoch_stats["initialization_skipped_shapes"] = (
                    epoch_stats.get("initialization_skipped_shapes", 0.0) + 1.0
                )
                reason_key = f"initialization_skip_{reason}"
                epoch_stats[reason_key] = epoch_stats.get(reason_key, 0.0) + 1.0
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_grid,
                sdf_grid,
                initial_field=initial_field,
                target_points=local_target_points if config.local_knn_features else target_points,
                local_udf_grid=local_udf_grid,
            )
            loss, stats = hybrid_direct_mesh_loss(
                outputs,
                target_points,
                chamfer_weight=args.w_mesh_chamfer,
                cvt_weight=args.w_mesh_cvt,
                sdfsmooth_weight=args.w_mesh_sdfsmooth,
                strict=args.strict_mesh_loss,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {stats}")
            if loss.requires_grad:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}: {stats}")
                stats["gradient_norm"] = float(gradient_norm.detach().cpu())
                optimizer.step()
                _assert_finite_model_parameters(model)
            else:
                stats["mesh_no_grad_batch"] = 1.0

            epoch_loss += float(loss.detach().cpu())
            for key, value in stats.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value)
            epoch_stats["site_count"] = epoch_stats.get("site_count", 0.0) + float(outputs["sites"].shape[1])
            epoch_stats["local_udf_grid_n"] = epoch_stats.get("local_udf_grid_n", 0.0) + float(
                config.local_udf_grid_n if config.local_udf_samples else 0
            )
            epoch_stats["local_knn_features"] = epoch_stats.get("local_knn_features", 0.0) + float(
                config.local_knn_features
            )
            if config.local_knn_features:
                epoch_stats["local_target_point_count"] = epoch_stats.get("local_target_point_count", 0.0) + float(
                    local_target_points.shape[1]
                )
            if config.local_udf_samples:
                local_udf_valid = batch["local_udf_valid"]
                epoch_stats["local_udf_valid"] = epoch_stats.get("local_udf_valid", 0.0) + float(
                    local_udf_valid.reshape(-1)[0].item()
                )
            for round_index, round_data in enumerate(outputs["rounds"]):
                prefix = f"round_{round_index:02d}"
                epoch_stats[f"{prefix}_parent_count"] = epoch_stats.get(f"{prefix}_parent_count", 0.0) + float(
                    round_data["parent_indices"].shape[0]
                )
                epoch_stats[f"{prefix}_spawned_site_count"] = epoch_stats.get(
                    f"{prefix}_spawned_site_count", 0.0
                ) + float(round_data["spawned_sites"].shape[0])
                epoch_stats[f"{prefix}_rejected_spawn_count"] = epoch_stats.get(
                    f"{prefix}_rejected_spawn_count", 0.0
                ) + float(round_data["rejected_spawn_count"].item())
            initialization_diagnostics = outputs["initialization_diagnostics"]
            for key in (
                "initial_site_count",
                "surface_anchor_count",
                "positive_sdf_count",
                "negative_sdf_count",
                "minimum_site_distance",
            ):
                stat_key = f"initialization_{key}"
                epoch_stats[stat_key] = epoch_stats.get(stat_key, 0.0) + float(initialization_diagnostics[key])

        num_batches = max(len(dataloader), 1)
        epoch_loss /= num_batches
        epoch_stats = {key: value / num_batches for key, value in epoch_stats.items()}
        print(f"epoch={epoch} local_epoch={local_epoch}/{args.epochs} loss={epoch_loss:.6g} stats={epoch_stats}")

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            stats=epoch_stats,
        )
        if args.save_every > 0 and (local_epoch % args.save_every == 0 or local_epoch == args.epochs):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                stats=epoch_stats,
            )


def _load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[DCCVTHybridIterRefineNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = HybridIterRefineConfig.from_dict(checkpoint["model_config"])
    model = DCCVTHybridIterRefineNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _cache_mesh_id(cache: dict[str, np.ndarray], cache_path: str | Path) -> str:
    return str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())


def _expected_initial_obj_path(
    output_dir: Path,
    *,
    state: str,
    variant: str,
    w_cvt: float,
    w_sdfsmooth: float,
) -> Path:
    return output_dir / f"DCCVT_0_{state}_{variant}_cvt{int(w_cvt)}_sdfsmooth{int(w_sdfsmooth)}.obj"


def _save_initialization_field(
    output_dir: Path,
    *,
    mesh_id: str,
    initialization: dict[str, Any],
    input_grid: np.ndarray,
    sdf_grid: np.ndarray,
    target_points: np.ndarray,
    config: HybridIterRefineConfig,
    seed: int,
    command_args: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = initialization["sites"].detach().cpu()
    sites_sdf = initialization["sites_sdf"].detach().cpu()
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites.shape[0]),
        "base_site_count": int(sites.shape[0]),
        "round_count": 0,
        "positive_sdf_count": int((sites_sdf > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf < 0).sum().item()),
        "initialization_valid": bool(initialization["valid"]),
        "initialization_reason": str(initialization["reason"]),
        "initialization": dict(initialization["diagnostics"]),
    }
    field_file = output_dir / f"{mesh_id}_hybrid_iter_refine_initial_field.npz"
    np.savez_compressed(
        field_file,
        sites=sites.numpy().astype(np.float32),
        sites_sdf=sites_sdf.numpy().astype(np.float32),
        base_sites=sites.numpy().astype(np.float32),
        base_sites_sdf=sites_sdf.numpy().astype(np.float32),
        background_sites=initialization["background_sites"].detach().cpu().numpy().astype(np.float32),
        background_sites_sdf=initialization["background_sdf"].detach().cpu().numpy().astype(np.float32),
        surface_anchors=initialization["surface_anchors"].detach().cpu().numpy().astype(np.float32),
        surface_sites=initialization["surface_sites"].detach().cpu().numpy().astype(np.float32),
        surface_sites_sdf=initialization["surface_sdf"].detach().cpu().numpy().astype(np.float32),
        input_grid=input_grid.astype(np.float32),
        sdf_grid=sdf_grid.astype(np.float32),
        target_points=target_points.astype(np.float32),
        diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
        resolved_config=np.array(json.dumps(config.to_dict(), sort_keys=True)),
        command_args=np.array(json.dumps(command_args, sort_keys=True)),
        seed=np.array(int(seed), dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )
    print(f"Saved iterative refinement initialization field: {field_file}")
    print(f"Diagnostics: {diagnostics}")
    return field_file


def extract_initialization_cache(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    config: HybridIterRefineConfig,
    seed: int = 69,
    extract: bool = True,
    overwrite: bool = False,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    command_args: Optional[dict[str, Any]] = None,
    state: str = "hybrid_iter_refine_initial",
) -> dict[str, Any]:
    """Export the HotSpot near-surface initialization for one cache."""
    output_path = Path(output_dir)
    int_obj = _expected_initial_obj_path(
        output_path,
        state=state,
        variant="intDCCVT",
        w_cvt=w_cvt,
        w_sdfsmooth=w_sdfsmooth,
    )
    proj_obj = _expected_initial_obj_path(
        output_path,
        state=state,
        variant="projDCCVT",
        w_cvt=w_cvt,
        w_sdfsmooth=w_sdfsmooth,
    )
    field_file = output_path / f"{Path(cache_path).stem}_hybrid_iter_refine_initial_field.npz"
    if not overwrite and field_file.exists() and (not extract or (int_obj.exists() and proj_obj.exists())):
        return {
            "cache_path": str(cache_path),
            "output_dir": str(output_path),
            "field_file": str(field_file),
            "status": "skipped_existing",
        }

    cache = _load_cache(cache_path)
    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = _cache_mesh_id(cache, cache_path)
    if grid_n != config.hotspot_grid_n:
        raise ValueError(f"Cache grid_n={grid_n} does not match config hotspot_grid_n={config.hotspot_grid_n}")

    input_grid_np = build_hybrid_input_channels_np(
        sdf_grid_np,
        target_points_np,
        grid_n=grid_n,
        udf_clip=config.point_udf_clip,
        confidence_sigma_scale=config.point_confidence_sigma_scale,
        channel_names=config.channel_names,
    )
    initialization = build_hotspot_near_surface_initialization(sdf_grid_np, config)
    output_path.mkdir(parents=True, exist_ok=True)
    field_file = _save_initialization_field(
        output_path,
        mesh_id=mesh_id,
        initialization=initialization,
        input_grid=input_grid_np,
        sdf_grid=sdf_grid_np,
        target_points=target_points_np,
        config=config,
        seed=seed,
        command_args=command_args or {},
    )

    sites_cpu = initialization["sites"].detach().cpu()
    sites_sdf_cpu = initialization["sites_sdf"].detach().cpu()
    positive_count = int((sites_sdf_cpu > 0).sum().item())
    negative_count = int((sites_sdf_cpu < 0).sum().item())
    can_extract = (
        extract
        and bool(initialization["valid"])
        and sites_cpu.shape[0] >= 5
        and positive_count > 0
        and negative_count > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_path),
            upsampling=0,
            w_cvt=w_cvt,
            w_sdfsmooth=w_sdfsmooth,
        )
        extract_mesh(
            sites_cpu.to(dccvt_device),
            sites_sdf_cpu.to(dccvt_device),
            target_pc,
            0.0,
            args,
            state=state,
        )
    elif extract:
        print("Skipping DCCVT extraction: need valid initialization and both positive/negative SDF values.")

    return {
        "cache_path": str(cache_path),
        "output_dir": str(output_path),
        "field_file": str(field_file),
        "status": "extracted" if can_extract else "saved_field",
        "extracted": bool(can_extract),
        "site_count": int(sites_cpu.shape[0]),
        "initialization_valid": bool(initialization["valid"]),
        "initialization_reason": str(initialization["reason"]),
    }


def run_initialization_extraction(
    *,
    config: HybridIterRefineConfig,
    cache_root: str | Path,
    output_root: str | Path,
    split_file: str | Path | None = None,
    mesh_ids: Sequence[str] | None = None,
    seed: int = 69,
    extract: bool = True,
    overwrite: bool = False,
    fail_fast: bool = False,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    command_args: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Export initialization fields and optional meshes for a cache set."""
    seed_everything(seed)
    cache_files = resolve_cache_files(cache_root, mesh_ids=mesh_ids, split_file=split_file)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resolved_payload = {
        "model_config": config.to_dict(),
        "seed": int(seed),
        "args": command_args or {},
    }
    (output_root_path / "resolved_config.json").write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    for cache_path in cache_files:
        mesh_id = Path(cache_path).stem
        output_dir = output_root_path / mesh_id
        try:
            result = extract_initialization_cache(
                cache_path,
                output_dir,
                config=config,
                seed=seed,
                extract=extract,
                overwrite=overwrite,
                w_cvt=w_cvt,
                w_sdfsmooth=w_sdfsmooth,
                command_args=command_args or {},
            )
        except Exception as exc:
            result = {
                "cache_path": str(cache_path),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"Failed iterative refinement initialization extraction for {mesh_id}: {exc}")
            if fail_fast:
                raise
        results.append(result)

    summary_path = output_root_path / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved iterative refinement initialization summary: {summary_path}")
    return results


def _save_prediction(
    output_dir: Path,
    *,
    mesh_id: str,
    outputs: dict[str, Any],
    input_grid: np.ndarray,
    sdf_grid: np.ndarray,
    target_points: np.ndarray,
    checkpoint: dict[str, Any],
    command_args: dict[str, Any],
    local_udf_path: str = "",
    local_udf_valid: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = outputs["sites"][0].detach().cpu()
    sites_sdf = outputs["sites_sdf"][0].detach().cpu()
    initialization_diagnostics = dict(outputs["initialization_diagnostics"])
    diagnostics = {
        "mesh_id": mesh_id,
        "site_count": int(sites.shape[0]),
        "base_site_count": int(outputs["base_sites"].shape[0]),
        "round_count": int(len(outputs["rounds"])),
        "positive_sdf_count": int((sites_sdf > 0).sum().item()),
        "negative_sdf_count": int((sites_sdf < 0).sum().item()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "initialization_valid": bool(outputs["initialization_valid"]),
        "initialization_reason": str(outputs["initialization_reason"]),
        "initialization": initialization_diagnostics,
        "architecture": str(checkpoint["model_config"].get("architecture", "dense_cnn")),
        "local_feature_mode": str(checkpoint["model_config"].get("local_feature_mode", "none")),
        "local_udf_grid_n": int(checkpoint["model_config"].get("local_udf_grid_n", 0)),
        "local_udf_samples": bool(checkpoint["model_config"].get("local_udf_samples", False)),
        "local_udf_path": local_udf_path,
        "local_udf_valid": bool(local_udf_valid),
        "local_knn_features": bool(checkpoint["model_config"].get("local_knn_features", False)),
        "graph_layers": int(checkpoint["model_config"].get("graph_layers", 0)),
        "site_position_encoding": str(checkpoint["model_config"].get("site_position_encoding", "fourier")),
        "graph_edge_features": str(
            checkpoint["model_config"].get("graph_edge_features", "relative_xyz_distance_direction_sdf_delta")
        ),
    }
    arrays: dict[str, Any] = {
        "sites": sites.numpy().astype(np.float32),
        "sites_sdf": sites_sdf.numpy().astype(np.float32),
        "base_sites": outputs["base_sites"].detach().cpu().numpy().astype(np.float32),
        "base_sites_sdf": outputs["base_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "background_sites": outputs["background_sites"].detach().cpu().numpy().astype(np.float32),
        "background_sites_sdf": outputs["background_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "surface_anchors": outputs["surface_anchors"].detach().cpu().numpy().astype(np.float32),
        "surface_sites": outputs["surface_sites"].detach().cpu().numpy().astype(np.float32),
        "surface_sites_sdf": outputs["surface_sites_sdf"].detach().cpu().numpy().astype(np.float32),
        "input_grid": input_grid.astype(np.float32),
        "sdf_grid": sdf_grid.astype(np.float32),
        "target_points": target_points.astype(np.float32),
        "local_udf_path": np.array(local_udf_path),
        "local_udf_valid": np.array(bool(local_udf_valid)),
        "diagnostics": np.array(json.dumps(diagnostics, sort_keys=True)),
        "resolved_config": np.array(json.dumps(checkpoint["model_config"], sort_keys=True)),
        "command_args": np.array(json.dumps(command_args, sort_keys=True)),
        "seed": np.array(int(checkpoint.get("seed", command_args.get("seed", 69))), dtype=np.int64),
        "mesh_id": np.array(mesh_id),
    }
    for round_index, round_data in enumerate(outputs["rounds"]):
        prefix = f"round_{round_index:02d}"
        arrays[f"{prefix}_parent_indices"] = round_data["parent_indices"].detach().cpu().numpy().astype(np.int64)
        arrays[f"{prefix}_parent_scores"] = round_data["parent_scores"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_spawned_sites"] = round_data["spawned_sites"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_spawned_sdf"] = round_data["spawned_sdf"].detach().cpu().numpy().astype(np.float32)
        arrays[f"{prefix}_rejected_spawn_count"] = np.array(
            int(round_data["rejected_spawn_count"].item()), dtype=np.int64
        )

    prediction_file = output_dir / f"{mesh_id}_hybrid_iter_refine_prediction.npz"
    np.savez_compressed(prediction_file, **arrays)
    print(f"Saved iterative refinement prediction: {prediction_file}")
    print(f"Diagnostics: {diagnostics}")
    return prediction_file


def run_inference(
    *,
    checkpoint_path: str | Path,
    cache_path: str | Path,
    output_dir: str | Path,
    device_value: str = "auto",
    local_udf_root: str | Path | None = None,
    allow_missing_local_features: bool = False,
    extract: bool = True,
    w_cvt: float = 100.0,
    w_sdfsmooth: float = 100.0,
    seed: int = 69,
    command_args: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    device = _device(device_value)
    seed_everything(seed)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    cache = _load_cache(cache_path)

    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())
    if grid_n != model.config_obj.hotspot_grid_n:
        raise ValueError(f"Cache grid_n={grid_n} does not match model hotspot_grid_n={model.config_obj.hotspot_grid_n}")

    input_grid_np = build_hybrid_input_channels_np(
        sdf_grid_np,
        target_points_np,
        grid_n=grid_n,
        udf_clip=model.config_obj.point_udf_clip,
        confidence_sigma_scale=model.config_obj.point_confidence_sigma_scale,
        channel_names=model.config_obj.channel_names,
    )
    input_grid = torch.from_numpy(input_grid_np[None, ...]).to(device)
    sdf_grid = torch.from_numpy(sdf_grid_np[None, None, ...]).to(device)
    target_points = torch.from_numpy(target_points_np[None, ...]).to(device)
    local_udf_grid = None
    local_udf_path = ""
    local_udf_valid = False
    if model.config_obj.local_udf_samples:
        if local_udf_root is None:
            if not allow_missing_local_features:
                raise ValueError("Checkpoint config requests local UDF samples; provide --local-udf-root")
            grid_n = model.config_obj.local_udf_grid_n
            local_udf_np = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
        else:
            sidecar_path = point_udf_sidecar_path(local_udf_root, Path(cache_path).stem)
            local_udf_path = str(sidecar_path)
            if not sidecar_path.exists():
                if not allow_missing_local_features:
                    raise FileNotFoundError(f"Missing local point-UDF sidecar: {sidecar_path}")
                grid_n = model.config_obj.local_udf_grid_n
                local_udf_np = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
            else:
                local_udf_np = load_point_udf_sidecar(sidecar_path, grid_n=model.config_obj.local_udf_grid_n)
                local_udf_valid = True
        local_udf_grid = torch.from_numpy(local_udf_np[None, None, ...]).to(device)
    initial_field = build_hotspot_near_surface_initialization(sdf_grid_np, model.config_obj)
    for name in (
        "sites",
        "sites_sdf",
        "background_sites",
        "background_sdf",
        "surface_anchors",
        "surface_sites",
        "surface_sdf",
    ):
        initial_field[name] = initial_field[name].to(device=device, dtype=input_grid.dtype)
    with torch.no_grad():
        outputs = model(
            input_grid,
            sdf_grid,
            initial_field=initial_field,
            target_points=target_points,
            local_udf_grid=local_udf_grid,
        )

    output_path = Path(output_dir)
    prediction_file = _save_prediction(
        output_path,
        mesh_id=mesh_id,
        outputs=outputs,
        input_grid=input_grid_np,
        sdf_grid=sdf_grid_np,
        target_points=target_points_np,
        checkpoint=checkpoint,
        command_args=command_args or {},
        local_udf_path=local_udf_path,
        local_udf_valid=local_udf_valid,
    )

    sites_cpu = outputs["sites"][0].detach().cpu()
    sites_sdf_cpu = outputs["sites_sdf"][0].detach().cpu()
    can_extract = (
        extract
        and bool(outputs["initialization_valid"])
        and sites_cpu.shape[0] >= 5
        and int((sites_sdf_cpu > 0).sum().item()) > 0
        and int((sites_sdf_cpu < 0).sum().item()) > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_path),
            upsampling=model.config_obj.num_refinement_rounds,
            w_cvt=w_cvt,
            w_sdfsmooth=w_sdfsmooth,
        )
        extract_mesh(
            sites_cpu.to(dccvt_device),
            sites_sdf_cpu.to(dccvt_device),
            target_pc,
            0.0,
            args,
            state="hybrid_iter_refine",
        )
    elif extract:
        print("Skipping DCCVT extraction: need at least 5 sites and both positive/negative SDF values.")

    result = {
        "prediction_file": str(prediction_file),
        "extracted": bool(can_extract),
        "site_count": int(sites_cpu.shape[0]),
        "initialization_valid": bool(outputs["initialization_valid"]),
        "initialization_reason": str(outputs["initialization_reason"]),
    }
    result_path = output_path / "inference_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved iterative refinement inference result: {result_path}")
    return result


def build_infer_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run iterative learned sparse-refinement inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--local-udf-root", default=None)
    parser.add_argument("--allow-missing-local-features", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)
    return parser


def infer_main(argv: Optional[list[str]] = None) -> None:
    args = build_infer_arg_parser().parse_args(argv)
    run_inference(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output_dir=args.output_dir,
        device_value=args.device,
        local_udf_root=args.local_udf_root,
        allow_missing_local_features=args.allow_missing_local_features,
        extract=not args.no_extract,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        seed=args.seed,
        command_args=vars(args),
    )


def build_initial_extract_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract iterative-refinement HotSpot near-surface initialization.")
    parser.add_argument("--config", default="configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json")
    parser.add_argument("--cache-root", default="outputs/neural_hotspot_sdf/thingi32_g33")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids.")
    parser.add_argument("--output-root", default="outputs/neural_dccvt/hybrid_iter_refine_initial_v2_hotspot_point_udf")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--w-cvt", type=float, default=100.0)
    parser.add_argument("--w-sdfsmooth", type=float, default=100.0)
    return parser


def initial_extract_main(argv: Optional[list[str]] = None) -> None:
    args = build_initial_extract_arg_parser().parse_args(argv)
    config = load_iter_refine_config(args.config)
    run_initialization_extraction(
        config=config,
        cache_root=args.cache_root,
        output_root=args.output_root,
        split_file=args.split_file,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        seed=args.seed,
        extract=not args.no_extract,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        w_cvt=args.w_cvt,
        w_sdfsmooth=args.w_sdfsmooth,
        command_args=vars(args),
    )
