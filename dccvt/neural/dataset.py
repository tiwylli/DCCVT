"""Datasets for supervised DCCVT generator prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh


DEFAULT_LABEL_PATTERN = "DCCVT_0_final_projDCCVT_*.npz"


@dataclass(frozen=True)
class DCCVTLabelRecord:
    mesh_id: str
    label_path: Path
    point_path: Path


def _stable_int(mesh_id: str) -> int:
    digits = "".join(ch for ch in mesh_id if ch.isdigit())
    if digits:
        return int(digits)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(mesh_id))


def read_point_cloud(path: str | Path) -> np.ndarray:
    """Load points from a PLY/OBJ mesh or point cloud."""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        clouds = [
            np.asarray(geom.vertices, dtype=np.float32)
            for geom in loaded.geometry.values()
            if hasattr(geom, "vertices") and len(geom.vertices) > 0
        ]
        if not clouds:
            raise ValueError(f"No vertices found in scene: {path}")
        points = np.concatenate(clouds, axis=0)
    elif hasattr(loaded, "vertices") and len(loaded.vertices) > 0:
        points = np.asarray(loaded.vertices, dtype=np.float32)
    else:
        raise ValueError(f"No vertices found in point cloud or mesh: {path}")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape} from {path}")
    return points.astype(np.float32, copy=False)


def sample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Return a deterministic fixed-size point sample."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) == 0:
        raise ValueError("Cannot sample from an empty point cloud")

    rng = np.random.default_rng(seed)
    replace = len(points) < num_points
    indices = rng.choice(len(points), size=num_points, replace=replace)
    return points[indices].astype(np.float32, copy=False)


def find_label_path(label_root: str | Path, mesh_id: str, pattern: str = DEFAULT_LABEL_PATTERN) -> Path:
    """Find the final DCCVT label bundle for one mesh id."""
    root = Path(label_root)
    candidates = sorted((root / mesh_id).glob(pattern))
    if not candidates:
        candidates = sorted(root.glob(f"**/{mesh_id}/{pattern}"))
    if not candidates:
        raise FileNotFoundError(f"No label bundle for mesh id {mesh_id!r} under {root}")
    return candidates[0]


def discover_mesh_ids(label_root: str | Path, pattern: str = DEFAULT_LABEL_PATTERN) -> List[str]:
    """Discover mesh ids from a label root organized as <root>/<mesh_id>/<bundle>.npz."""
    root = Path(label_root)
    mesh_ids = sorted({path.parent.name for path in root.glob(f"*/{pattern}")})
    return mesh_ids


def build_records(
    *,
    mesh_ids: Iterable[str],
    mesh_root: str | Path,
    label_root: str | Path,
    label_pattern: str = DEFAULT_LABEL_PATTERN,
    prefer_label_target: bool = True,
) -> List[DCCVTLabelRecord]:
    """Create dataset records for mesh ids with paired point clouds and DCCVT labels."""
    records: List[DCCVTLabelRecord] = []
    mesh_root = Path(mesh_root)
    for mesh_id in mesh_ids:
        label_path = find_label_path(label_root, mesh_id, label_pattern)
        target_path = label_path.parent / "target.ply"
        if prefer_label_target and target_path.exists():
            point_path = target_path
        else:
            point_path = mesh_root / f"{mesh_id}.ply"
            if not point_path.exists():
                point_path = mesh_root / f"{mesh_id}.obj"
        if not point_path.exists():
            raise FileNotFoundError(f"No point cloud or mesh found for mesh id {mesh_id!r}: {point_path}")
        records.append(DCCVTLabelRecord(mesh_id=mesh_id, label_path=label_path, point_path=point_path))
    return records


class DCCVTGeneratorDataset(Dataset):
    """Point-cloud to fixed-size `(sites, sites_sdf)` supervision dataset."""

    def __init__(
        self,
        *,
        label_root: str | Path,
        mesh_root: str | Path = "mesh/thingi32",
        mesh_ids: Optional[Iterable[str]] = None,
        num_points: int = 9600,
        num_centroids: int = 32,
        label_pattern: str = DEFAULT_LABEL_PATTERN,
        prefer_label_target: bool = True,
        seed: int = 0,
    ) -> None:
        self.label_root = Path(label_root)
        self.mesh_root = Path(mesh_root)
        self.num_points = num_points
        self.num_centroids = num_centroids
        self.expected_sites = num_centroids**3
        self.seed = seed

        if mesh_ids is None:
            mesh_ids = discover_mesh_ids(self.label_root, label_pattern)
        mesh_ids = list(mesh_ids)
        if not mesh_ids:
            raise ValueError(f"No mesh ids found under label root: {self.label_root}")

        self.records = build_records(
            mesh_ids=mesh_ids,
            mesh_root=self.mesh_root,
            label_root=self.label_root,
            label_pattern=label_pattern,
            prefer_label_target=prefer_label_target,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        points = read_point_cloud(record.point_path)
        points = sample_points(points, self.num_points, self.seed + _stable_int(record.mesh_id))

        bundle = np.load(record.label_path, allow_pickle=True)
        sites = bundle["sites"].astype(np.float32, copy=False)
        sites_sdf = bundle["sites_sdf"].astype(np.float32, copy=False)
        if sites.shape != (self.expected_sites, 3):
            raise ValueError(
                f"Expected sites shape {(self.expected_sites, 3)} for {record.label_path}, got {sites.shape}"
            )
        if sites_sdf.shape != (self.expected_sites,):
            raise ValueError(
                f"Expected sites_sdf shape {(self.expected_sites,)} for {record.label_path}, got {sites_sdf.shape}"
            )

        return {
            "mesh_id": record.mesh_id,
            "points": torch.from_numpy(points),
            "target_sites": torch.from_numpy(sites),
            "target_sdf": torch.from_numpy(sites_sdf),
            "label_path": str(record.label_path),
            "point_path": str(record.point_path),
        }
