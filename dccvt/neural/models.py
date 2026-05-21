"""Small neural models for point-cloud to DCCVT generator prediction."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def make_template_grid(num_centroids: int = 32, domain_limit: float = 1.0) -> torch.Tensor:
    """Create a deterministic `(num_centroids ** 3, 3)` template grid."""
    axis = torch.linspace(-domain_limit, domain_limit, int(num_centroids), dtype=torch.float32)
    try:
        grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        grid = torch.meshgrid(axis, axis, axis)
    return torch.stack(grid, dim=-1).reshape(-1, 3)


class PointNetDCCVT(nn.Module):
    """PointNet-style encoder with a template-grid decoder for DCCVT sites and SDF."""

    def __init__(
        self,
        *,
        num_centroids: int = 32,
        point_feature_dim: int = 256,
        global_feature_dim: int = 512,
        decoder_hidden_dim: int = 256,
        offset_scale: float = 0.25,
        domain_limit: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_centroids = int(num_centroids)
        self.num_sites = self.num_centroids**3
        self.offset_scale = float(offset_scale)
        self.domain_limit = float(domain_limit)

        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, point_feature_dim, kernel_size=1),
            nn.BatchNorm1d(point_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(point_feature_dim * 2, global_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_feature_dim, global_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(global_feature_dim + 3, decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden_dim, 4),
        )

        template = make_template_grid(self.num_centroids, self.domain_limit)
        self.register_buffer("template_sites", template, persistent=False)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape (B, P, 3), got {tuple(points.shape)}")
        features = self.point_encoder(points.transpose(1, 2))
        max_features = features.max(dim=2).values
        mean_features = features.mean(dim=2)
        return self.global_encoder(torch.cat([max_features, mean_features], dim=1))

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_features = self.encode(points)
        batch_size = points.shape[0]

        template = self.template_sites.to(dtype=points.dtype, device=points.device)
        template_b = template.unsqueeze(0).expand(batch_size, -1, -1)
        global_b = global_features.unsqueeze(1).expand(-1, self.num_sites, -1)

        decoded = self.decoder(torch.cat([template_b, global_b], dim=-1))
        offsets = self.offset_scale * torch.tanh(decoded[..., :3])
        sites = template_b + offsets
        sites_sdf = decoded[..., 3]
        return {
            "sites": sites,
            "sites_sdf": sites_sdf,
            "offsets": offsets,
            "template_sites": template_b,
        }

    def config(self) -> dict:
        return {
            "num_centroids": self.num_centroids,
            "offset_scale": self.offset_scale,
            "domain_limit": self.domain_limit,
        }
