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
    return torch.device("cpu")


device = _select_device()

if device.type == "cuda":
    logger.info("Using device: %s", torch.cuda.get_device_name(device))
else:
    logger.info("Using device: %s", device)

# Improve reproducibility
torch.manual_seed(69)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)
