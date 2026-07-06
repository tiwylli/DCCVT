"""Shared utilities for neural DCCVT command-line workflows."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible neural workflows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch dataloader worker from PyTorch's worker seed."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_mesh_ids(value: str | None) -> list[str] | None:
    """Parse comma- or whitespace-separated mesh ids from a CLI value."""
    if value is None:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def read_mesh_ids(path: str | Path) -> list[str]:
    """Read non-empty mesh ids from a text file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def device_from_value(value: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device string."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_npz_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load an `.npz` cache into memory without pickle support."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def cache_mesh_id(cache: dict[str, np.ndarray], cache_path: str | Path) -> str:
    """Return the cache mesh id, falling back to the file stem."""
    return str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())
