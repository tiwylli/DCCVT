"""PoNQ-style dense SDF CNN for predicting DCCVT Voronoi sites."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from dccvt.neural.grid import cell_size_from_grid, make_cell_lower_corners, validate_grid_n


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
