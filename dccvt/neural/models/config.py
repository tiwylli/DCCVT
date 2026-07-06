"""Configuration for neural DCCVT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from dccvt.neural.grid import HYBRID_DIRECT_CHANNELS, validate_grid_n, validate_hybrid_channel_names

@dataclass
class HybridDirectConfig:
    """Typed configuration for the hybrid direct DCCVT extractor."""

    grid_n: int = 33
    input_channels: int | None = None
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
        self.channel_names = validate_hybrid_channel_names(self.channel_names)
        if self.input_channels is None:
            self.input_channels = len(self.channel_names)
        else:
            self.input_channels = int(self.input_channels)
        self.feature_dim = int(self.feature_dim)
        self.encoder_layers = int(self.encoder_layers)
        self.decoder_layers = int(self.decoder_layers)
        self.site_delta_scale = float(self.site_delta_scale)
        self.sdf_residual_scale = float(self.sdf_residual_scale)
        self.point_udf_clip = float(self.point_udf_clip)
        self.point_confidence_sigma_scale = float(self.point_confidence_sigma_scale)
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
