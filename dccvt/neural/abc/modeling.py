"""Model construction and initialization for ABC HybridPoNQ-DCCVT."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Optional, Sequence

import torch

from dccvt.neural.models import DCCVTHybridDirectNet, DCCVTPoNQNet, HybridDirectConfig

def zero_initialize_dccvt_heads(model: DCCVTHybridDirectNet) -> None:
    """Initialize direct DCCVT heads to the canonical SDF field."""
    with torch.no_grad():
        for head in (model.site_delta_head, model.sdf_residual_head):
            for parameter in head.parameters():
                parameter.zero_()


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("PoNQ checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(key, str) for key in checkpoint):
        return checkpoint  # Legacy PoNQ checkpoints are plain state dictionaries.
    raise ValueError("Could not find a model state dictionary in PoNQ checkpoint")


def initialize_from_ponq_encoder(
    model: DCCVTHybridDirectNet,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    """Transfer the PoNQ SDF encoder and zero the new UDF input channel."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = _checkpoint_state_dict(checkpoint)
    encoder_state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.removeprefix("module.").startswith("encoder.")
    }
    if "encoder.0.weight" not in encoder_state or "encoder.0.bias" not in encoder_state:
        raise ValueError("PoNQ checkpoint does not contain the input convolution")

    target_state = model.state_dict()
    copied = 0
    with torch.no_grad():
        source_weight = encoder_state["encoder.0.weight"]
        target_weight = target_state["encoder.0.weight"]
        if source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[2:] != target_weight.shape[2:]:
            raise ValueError(
                f"Incompatible PoNQ input convolution {source_weight.shape} for {target_weight.shape}"
            )
        target_weight[:, 0].copy_(source_weight[:, 0])
        target_weight[:, 1].zero_()
        target_state["encoder.0.bias"].copy_(encoder_state["encoder.0.bias"])
        copied += 2

        for key, value in encoder_state.items():
            if key in {"encoder.0.weight", "encoder.0.bias"}:
                continue
            if key not in target_state:
                continue
            if target_state[key].shape != value.shape:
                raise ValueError(f"Incompatible PoNQ encoder tensor {key}: {value.shape}")
            target_state[key].copy_(value)
            copied += 1

    zero_initialize_dccvt_heads(model)
    return {"copied_tensors": copied, "encoder_tensors": len(encoder_state)}


def build_abc_hybrid_model(
    config: HybridDirectConfig,
    *,
    variant: str,
    encoder_checkpoint: Optional[str | Path] = None,
) -> tuple[DCCVTHybridDirectNet, dict]:
    """Construct one comparison variant with canonical DCCVT outputs."""
    if variant not in {"direct", "ponq_pretrained"}:
        raise ValueError(f"Unknown ABC HybridPoNQ variant: {variant}")
    model = DCCVTHybridDirectNet(config)
    metadata: dict = {"variant": variant}
    if variant == "ponq_pretrained":
        if encoder_checkpoint is None:
            raise ValueError("ponq_pretrained requires --encoder-checkpoint")
        metadata.update(initialize_from_ponq_encoder(model, encoder_checkpoint))
        metadata["encoder_checkpoint"] = str(Path(encoder_checkpoint).resolve())
    else:
        zero_initialize_dccvt_heads(model)
    return model, metadata


def deterministic_subset(ids: Sequence[str], count: int, seed: int) -> list[str]:
    """Select a fixed seeded subset without changing source split order."""
    if count >= len(ids):
        return list(ids)
    rng = random.Random(seed)
    selected = set(rng.sample(list(ids), count))
    return [model_id for model_id in ids if model_id in selected]
