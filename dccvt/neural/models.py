"""PoNQ-style dense SDF CNNs for DCCVT Voronoi prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from dccvt.neural.grid import (
    cell_size_from_grid,
    make_canonical_sites,
    make_cell_lower_corners,
    trilinear_interpolate_sdf,
    validate_grid_n,
)


HYBRID_DIRECT_CHANNELS = ("hotspot_sdf", "abs_hotspot_sdf", "point_udf", "point_confidence")


@dataclass
class HybridDirectConfig:
    """Typed configuration for the hybrid direct DCCVT extractor."""

    grid_n: int = 33
    input_channels: int = 4
    feature_dim: int = 128
    encoder_layers: int = 5
    decoder_layers: int = 3
    site_delta_scale: float = 0.30
    sdf_residual_scale: float = 0.50
    point_udf_clip: float = 4.0
    point_confidence_sigma_scale: float = 1.5
    channel_names: tuple[str, ...] = field(default_factory=lambda: HYBRID_DIRECT_CHANNELS)

    def __post_init__(self) -> None:
        self.grid_n = validate_grid_n(self.grid_n)
        self.input_channels = int(self.input_channels)
        self.feature_dim = int(self.feature_dim)
        self.encoder_layers = int(self.encoder_layers)
        self.decoder_layers = int(self.decoder_layers)
        self.site_delta_scale = float(self.site_delta_scale)
        self.sdf_residual_scale = float(self.sdf_residual_scale)
        self.point_udf_clip = float(self.point_udf_clip)
        self.point_confidence_sigma_scale = float(self.point_confidence_sigma_scale)
        self.channel_names = tuple(self.channel_names)
        if self.input_channels != len(self.channel_names):
            raise ValueError(
                f"input_channels={self.input_channels} does not match "
                f"{len(self.channel_names)} channel names"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "HybridDirectConfig":
        values = dict(data)
        if "channel_names" in values:
            values["channel_names"] = tuple(values["channel_names"])
        return cls(**values)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["channel_names"] = list(self.channel_names)
        return data


def load_hybrid_direct_config(path: str | Path | None = None) -> HybridDirectConfig:
    """Load a hybrid direct config JSON file, or return defaults."""
    if path is None:
        return HybridDirectConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return HybridDirectConfig.from_dict(data)


class ResNetBlock(nn.Module):
    """Small 1x1x1 residual block matching the PoNQ decoder style."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.conv_2 = nn.Conv3d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.conv_1(x), negative_slope=0.01, inplace=True)
        y = self.conv_2(y)
        return F.leaky_relu(x + y, negative_slope=0.01, inplace=True)


class CellDecoder(nn.Module):
    """Decode per-cell values from a dense cell-feature grid."""

    def __init__(
        self,
        out_features: int,
        *,
        k: int = 1,
        feature_dim: int = 128,
        decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ResNetBlock(feature_dim) for _ in range(decoder_layers)]
        layers.append(nn.Conv3d(feature_dim, out_features * k, kernel_size=1, bias=True))
        self.decoder = nn.Sequential(*layers)
        self.out_features = int(out_features)
        self.k = int(k)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(features)
        batch = decoded.shape[0]
        decoded = decoded.reshape(batch, self.out_features * self.k, -1).permute(0, 2, 1)
        return decoded.reshape(batch, -1, self.k, self.out_features)


class DCCVTPoNQNet(nn.Module):
    """Site-only PoNQ-style model adapted for DCCVT extraction.

    The network consumes a dense SDF vertex grid and predicts ``K`` DCCVT sites
    for each grid cell plus one activity logit per cell. SDF values for the
    sites are intentionally not predicted by this model; they are sampled from
    the HotSpot SDF grid downstream.
    """

    def __init__(
        self,
        *,
        grid_n: int = 33,
        k: int = 4,
        feature_dim: int = 128,
        encoder_layers: int = 5,
        decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        self.grid_n = validate_grid_n(grid_n)
        self.k = int(k)
        self.feature_dim = int(feature_dim)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)

        encoder: list[nn.Module] = [
            nn.Conv3d(1, self.feature_dim, kernel_size=2, stride=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ]
        for _ in range(self.encoder_layers):
            encoder.extend(
                [
                    nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=True),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                ]
            )
        self.encoder = nn.Sequential(*encoder)
        self.site_head = CellDecoder(
            3,
            k=self.k,
            feature_dim=self.feature_dim,
            decoder_layers=self.decoder_layers,
        )
        self.activity_head = CellDecoder(
            1,
            k=1,
            feature_dim=self.feature_dim,
            decoder_layers=self.decoder_layers,
        )
        self.register_buffer("cell_lower_corners", make_cell_lower_corners(self.grid_n), persistent=False)

    def config(self) -> Dict[str, int]:
        return {
            "grid_n": self.grid_n,
            "k": self.k,
            "feature_dim": self.feature_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
        }

    def _cell_geometry(self, features: torch.Tensor) -> tuple[torch.Tensor, float]:
        cell_res = int(features.shape[-1])
        grid_n = cell_res + 1
        expected_cells = cell_res**3
        if (
            grid_n == self.grid_n
            and self.cell_lower_corners.device == features.device
            and self.cell_lower_corners.dtype == features.dtype
            and self.cell_lower_corners.shape[0] == expected_cells
        ):
            corners = self.cell_lower_corners
        else:
            corners = make_cell_lower_corners(grid_n, device=features.device, dtype=features.dtype)
        return corners, cell_size_from_grid(grid_n)

    def forward(self, sdf_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        if sdf_grid.dim() == 4:
            sdf_grid = sdf_grid.unsqueeze(1)
        if sdf_grid.dim() != 5 or sdf_grid.shape[1] != 1:
            raise ValueError(f"Expected SDF input shape (B,1,G,G,G) or (B,G,G,G), got {sdf_grid.shape}")
        if len(set(sdf_grid.shape[-3:])) != 1:
            raise ValueError(f"Expected cubic SDF grid, got {sdf_grid.shape}")

        features = self.encoder(sdf_grid)
        corners, cell_size = self._cell_geometry(features)

        raw_offsets = self.site_head(features)
        offset_fraction = torch.sigmoid(raw_offsets)
        sites = corners[None, :, None, :] + offset_fraction * cell_size

        activity_logits = self.activity_head(features).squeeze(-1).squeeze(-1)
        activity = torch.sigmoid(activity_logits)
        return {
            "sites": sites,
            "raw_offsets": raw_offsets,
            "offset_fraction": offset_fraction,
            "activity_logits": activity_logits,
            "activity": activity,
            "cell_lower_corners": corners,
        }


class DCCVTHybridDirectNet(nn.Module):
    """Hybrid PoNQ-style direct predictor for full DCCVT site and SDF fields."""

    def __init__(self, config: HybridDirectConfig | dict | None = None, **overrides) -> None:
        super().__init__()
        if config is None:
            config_obj = HybridDirectConfig(**overrides)
        elif isinstance(config, dict):
            config_obj = HybridDirectConfig.from_dict({**config, **overrides})
        else:
            config_obj = HybridDirectConfig.from_dict({**config.to_dict(), **overrides})
        self.config_obj = config_obj

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
        self.site_delta_head = CellDecoder(
            3,
            k=1,
            feature_dim=config_obj.feature_dim,
            decoder_layers=config_obj.decoder_layers,
        )
        self.sdf_residual_head = CellDecoder(
            1,
            k=1,
            feature_dim=config_obj.feature_dim,
            decoder_layers=config_obj.decoder_layers,
        )
        self.register_buffer("canonical_sites", make_canonical_sites(config_obj.grid_n), persistent=False)

    @property
    def grid_n(self) -> int:
        return self.config_obj.grid_n

    def config(self) -> dict:
        return self.config_obj.to_dict()

    def _canonical_sites_for_features(self, features: torch.Tensor) -> torch.Tensor:
        cell_res = int(features.shape[-1])
        grid_n = cell_res + 1
        expected_sites = cell_res**3
        if (
            grid_n == self.grid_n
            and self.canonical_sites.device == features.device
            and self.canonical_sites.dtype == features.dtype
            and self.canonical_sites.shape[0] == expected_sites
        ):
            return self.canonical_sites
        return make_canonical_sites(grid_n, device=features.device, dtype=features.dtype)

    def forward(self, input_grid: torch.Tensor, hotspot_sdf_grid: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if input_grid.dim() == 4:
            input_grid = input_grid.unsqueeze(0)
        if input_grid.dim() != 5:
            raise ValueError(f"Expected input shape (B,C,G,G,G) or (C,G,G,G), got {input_grid.shape}")
        if input_grid.shape[1] != self.config_obj.input_channels:
            raise ValueError(
                f"Expected {self.config_obj.input_channels} channels, got input shape {input_grid.shape}"
            )
        if len(set(input_grid.shape[-3:])) != 1:
            raise ValueError(f"Expected cubic input grid, got {input_grid.shape}")

        if hotspot_sdf_grid is None:
            hotspot_sdf_grid = input_grid[:, 0]
        if hotspot_sdf_grid.dim() == 5:
            hotspot_sdf_grid = hotspot_sdf_grid[:, 0]
        if hotspot_sdf_grid.dim() != 4:
            raise ValueError(f"Expected HotSpot SDF shape (B,G,G,G) or (B,1,G,G,G), got {hotspot_sdf_grid.shape}")

        features = self.encoder(input_grid)
        canonical_sites = self._canonical_sites_for_features(features)

        raw_site_delta = self.site_delta_head(features).squeeze(2)
        site_delta = torch.tanh(raw_site_delta) * self.config_obj.site_delta_scale
        sites = canonical_sites[None, :, :] + site_delta

        raw_sdf_residual = self.sdf_residual_head(features).squeeze(2).squeeze(-1)
        sdf_residual = torch.tanh(raw_sdf_residual) * self.config_obj.sdf_residual_scale
        hotspot_sdf_at_sites = trilinear_interpolate_sdf(hotspot_sdf_grid, sites)
        sites_sdf = hotspot_sdf_at_sites + sdf_residual

        return {
            "sites": sites,
            "sites_sdf": sites_sdf,
            "raw_site_delta": raw_site_delta,
            "site_delta": site_delta,
            "raw_sdf_residual": raw_sdf_residual,
            "sdf_residual": sdf_residual,
            "hotspot_sdf_at_sites": hotspot_sdf_at_sites,
            "canonical_sites": canonical_sites,
        }
