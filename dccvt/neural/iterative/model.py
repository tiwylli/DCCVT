"""Iterative neural refinement model."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dccvt.neural.grid import trilinear_interpolate_sdf
from dccvt.neural.iterative.config import HybridIterRefineConfig
from dccvt.neural.iterative.graph import (
    GRAPH_EDGE_FEATURE_DIM,
    build_directed_edges_from_simplices,
    delaunay_edge_features,
    fourier_site_position_encoding,
    local_knn_parent_features,
    select_procedural_refinement_parents,
)
from dccvt.neural.iterative.initialization import build_hotspot_near_surface_initialization

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
