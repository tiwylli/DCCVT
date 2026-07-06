"""Hybrid direct PoNQ-DCCVT model."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from dccvt.neural.grid import cell_size_from_grid, make_canonical_sites, make_cell_lower_corners, trilinear_interpolate_sdf
from dccvt.neural.models.blocks import CellDecoder
from dccvt.neural.models.config import HybridDirectConfig

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
