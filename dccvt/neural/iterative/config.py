"""Configuration for iterative neural DCCVT refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from dccvt.neural.grid import HYBRID_DIRECT_CHANNELS, validate_grid_n, validate_hybrid_channel_names

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

