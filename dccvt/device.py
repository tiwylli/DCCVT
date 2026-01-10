"""Runtime initialization: device selection and deterministic seeds."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    env_device = os.environ.get("DCCVT_DEVICE")
    if env_device:
        return torch.device(env_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError("CUDA is not available and no device specified via DCCVT_DEVICE environment variable")


device = _select_device()
_initialized = False

def seed_everything(seed: int = 69) -> None:
    """Seed RNGs for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def initialize_runtime(seed: int = 69) -> torch.device:
    """Initialize device logging and deterministic seeds."""
    global _initialized
    if _initialized:
        return device
    if device.type == "cuda":
        logger.info("Using device: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using device: %s", device)
    seed_everything(seed)
    _initialized = True
    return device
