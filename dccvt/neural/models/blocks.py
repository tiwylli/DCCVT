"""Shared neural network blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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
