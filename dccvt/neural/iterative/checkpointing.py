"""Checkpoint helpers for iterative refinement."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from dccvt.neural.iterative.model import DCCVTHybridIterRefineNet

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

