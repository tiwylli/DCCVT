"""ABC SDF/UDF data and initialization utilities for HybridPoNQ-DCCVT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dccvt.neural.models import DCCVTHybridDirectNet, HybridDirectConfig


ROOT = Path(__file__).resolve().parents[2]
UDF_PREPROCESSING_VERSION = "abc_udf_129_stride4_v1"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_model_ids(path: str | Path) -> list[str]:
    """Read shape IDs from a PoNQ split while preserving file order."""
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(Path(part).stem for part in line.replace(",", " ").split())
    if not ids:
        raise ValueError(f"No model IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate model IDs found in {path}")
    return ids


@dataclass(frozen=True)
class ABCPathsConfig:
    """Filesystem inputs and output roots for the ABC experiment."""

    hdf5_root: Path
    udf_root: Path
    train_split: Path
    validation_split: Path
    output_root: Path
    ground_truth_root: Path
    ponq_output_root: Path
    evaluation_python: Path


@dataclass(frozen=True)
class ABCUDFConfig:
    """Exact point-cloud UDF preprocessing settings."""

    master_grid_n: int
    model_grid_n: int
    coordinate_min: float
    coordinate_max: float
    query_chunk_size: int
    compression: str
    compression_level: int
    preprocessing_version: str


@dataclass(frozen=True)
class PoNQPhaseConfig:
    """One phase of the reproduced PoNQ schedule."""

    epochs: int
    sample_count: int
    learning_rate: float


@dataclass(frozen=True)
class PoNQTrainingConfig:
    """PoNQ pretraining settings used before encoder transfer."""

    global_batch_size: int
    weight_decay: float
    beta1: float
    beta2: float
    amsgrad: bool
    k: int
    loss_weights: tuple[float, ...]
    phases: tuple[PoNQPhaseConfig, ...]


@dataclass(frozen=True)
class DCCVTTrainingConfig:
    """Step-based DCCVT mesh-training settings."""

    learning_rate: float
    weight_decay: float
    target_sample_count: int
    pilot_train_count: int
    pilot_validation_count: int
    pilot_steps: int
    full_steps: int
    validation_proxy_count: int
    validate_every_steps: int
    checkpoint_every_steps: int
    num_workers: int
    chamfer_weight: float
    site_displacement_weight: float
    sdf_residual_weight: float
    cvt_weight: float
    sdf_smoothness_weight: float
    delaunay_mode: str


@dataclass(frozen=True)
class ABCEvaluationConfig:
    """Extraction, metric, and pilot qualification settings."""

    sample_count: int
    n_jobs: int
    minimum_chamfer_improvement: float
    maximum_normal_consistency_regression: float
    maximum_edge_f1_regression: float


@dataclass(frozen=True)
class ABCHybridExperimentConfig:
    """Fully resolved HybridPoNQ ABC experiment configuration."""

    experiment_name: str
    seed: int
    paths: ABCPathsConfig
    udf: ABCUDFConfig
    model: HybridDirectConfig
    ponq_training: PoNQTrainingConfig
    dccvt_training: DCCVTTrainingConfig
    evaluation: ABCEvaluationConfig

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data["paths"].items():
            data["paths"][key] = str(value)
        data["model"] = self.model.to_dict()
        return data


def load_abc_hybrid_config(path: str | Path) -> ABCHybridExperimentConfig:
    """Load and validate the typed HybridPoNQ ABC JSON config."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    paths_data = data["paths"]
    udf_data = data["udf"]
    ponq_data = data["ponq_training"]
    train_data = data["dccvt_training"]
    eval_data = data["evaluation"]
    config = ABCHybridExperimentConfig(
        experiment_name=str(data["experiment_name"]),
        seed=int(data["seed"]),
        paths=ABCPathsConfig(
            hdf5_root=_resolve_path(paths_data["hdf5_root"]),
            udf_root=_resolve_path(paths_data["udf_root"]),
            train_split=_resolve_path(paths_data["train_split"]),
            validation_split=_resolve_path(paths_data["validation_split"]),
            output_root=_resolve_path(paths_data["output_root"]),
            ground_truth_root=_resolve_path(paths_data["ground_truth_root"]),
            ponq_output_root=_resolve_path(paths_data["ponq_output_root"]),
            evaluation_python=_resolve_path(paths_data["evaluation_python"]),
        ),
        udf=ABCUDFConfig(
            master_grid_n=int(udf_data["master_grid_n"]),
            model_grid_n=int(udf_data["model_grid_n"]),
            coordinate_min=float(udf_data["coordinate_min"]),
            coordinate_max=float(udf_data["coordinate_max"]),
            query_chunk_size=int(udf_data["query_chunk_size"]),
            compression=str(udf_data["compression"]),
            compression_level=int(udf_data["compression_level"]),
            preprocessing_version=str(udf_data["preprocessing_version"]),
        ),
        model=HybridDirectConfig.from_dict(data["model"]),
        ponq_training=PoNQTrainingConfig(
            global_batch_size=int(ponq_data["global_batch_size"]),
            weight_decay=float(ponq_data["weight_decay"]),
            beta1=float(ponq_data["beta1"]),
            beta2=float(ponq_data["beta2"]),
            amsgrad=bool(ponq_data["amsgrad"]),
            k=int(ponq_data["k"]),
            loss_weights=tuple(float(value) for value in ponq_data["loss_weights"]),
            phases=tuple(
                PoNQPhaseConfig(
                    epochs=int(phase["epochs"]),
                    sample_count=int(phase["sample_count"]),
                    learning_rate=float(phase["learning_rate"]),
                )
                for phase in ponq_data["phases"]
            ),
        ),
        dccvt_training=DCCVTTrainingConfig(
            learning_rate=float(train_data["learning_rate"]),
            weight_decay=float(train_data["weight_decay"]),
            target_sample_count=int(train_data["target_sample_count"]),
            pilot_train_count=int(train_data["pilot_train_count"]),
            pilot_validation_count=int(train_data["pilot_validation_count"]),
            pilot_steps=int(train_data["pilot_steps"]),
            full_steps=int(train_data["full_steps"]),
            validation_proxy_count=int(train_data["validation_proxy_count"]),
            validate_every_steps=int(train_data["validate_every_steps"]),
            checkpoint_every_steps=int(train_data["checkpoint_every_steps"]),
            num_workers=int(train_data["num_workers"]),
            chamfer_weight=float(train_data["chamfer_weight"]),
            site_displacement_weight=float(train_data["site_displacement_weight"]),
            sdf_residual_weight=float(train_data["sdf_residual_weight"]),
            cvt_weight=float(train_data["cvt_weight"]),
            sdf_smoothness_weight=float(train_data["sdf_smoothness_weight"]),
            delaunay_mode=str(train_data["delaunay_mode"]),
        ),
        evaluation=ABCEvaluationConfig(
            sample_count=int(eval_data["sample_count"]),
            n_jobs=int(eval_data["n_jobs"]),
            minimum_chamfer_improvement=float(eval_data["minimum_chamfer_improvement"]),
            maximum_normal_consistency_regression=float(
                eval_data["maximum_normal_consistency_regression"]
            ),
            maximum_edge_f1_regression=float(eval_data["maximum_edge_f1_regression"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: ABCHybridExperimentConfig) -> None:
    if config.udf.master_grid_n != 129 or config.udf.model_grid_n != 33:
        raise ValueError("ABC UDF grids must be 129^3 master and 33^3 model inputs")
    if (
        config.udf.coordinate_min != -0.5
        or config.udf.coordinate_max != 0.5
        or config.udf.preprocessing_version != UDF_PREPROCESSING_VERSION
    ):
        raise ValueError("ABC UDF preprocessing must use the versioned [-0.5,0.5]^3 convention")
    if (config.udf.master_grid_n - 1) % (config.udf.model_grid_n - 1) != 0:
        raise ValueError("UDF master and model grids are not exactly aligned")
    if config.model.grid_n != 33:
        raise ValueError("HybridPoNQ ABC model grid_n must be 33")
    if config.model.channel_names != ("hotspot_sdf", "point_udf"):
        raise ValueError("HybridPoNQ ABC model channels must be SDF followed by UDF")
    if len(config.ponq_training.loss_weights) != 6:
        raise ValueError("PoNQ training requires six loss weights")
    if len(config.ponq_training.phases) != 3:
        raise ValueError("PoNQ reproduction requires exactly three phases")
    if config.ponq_training.global_batch_size < 1:
        raise ValueError("PoNQ global_batch_size must be positive")
    if config.dccvt_training.pilot_steps < 1 or config.dccvt_training.full_steps < 1:
        raise ValueError("DCCVT pilot and full step counts must be positive")
    if config.dccvt_training.cvt_weight != 0.0:
        raise ValueError("ABC v1 keeps the CVT loss disabled")
    if config.dccvt_training.sdf_smoothness_weight != 0.0:
        raise ValueError("ABC v1 keeps the SDF smoothness loss disabled")
    if config.dccvt_training.delaunay_mode != "canonical_fixed":
        raise ValueError("ABC v1 requires canonical_fixed training Delaunay topology")
    required_inputs = (
        config.paths.hdf5_root,
        config.paths.train_split,
        config.paths.validation_split,
    )
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ABC experiment inputs:\n" + "\n".join(missing))


def udf_sidecar_path(udf_root: str | Path, model_id: str) -> Path:
    """Return the UDF HDF5 sidecar path for one ABC shape."""
    return Path(udf_root) / f"{Path(model_id).stem}.hdf5"


def validate_udf_sidecar(
    path: str | Path,
    *,
    config: ABCUDFConfig,
    check_values: bool = True,
) -> tuple[bool, str]:
    """Validate sidecar schema, metadata, and aligned 33^3 values."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("ABC UDF preprocessing and training require h5py") from exc

    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as handle:
            if handle["128_udf"].shape != (129, 129, 129):
                return False, "invalid 128_udf shape"
            if handle["32_udf"].shape != (33, 33, 33):
                return False, "invalid 32_udf shape"
            if handle["128_udf"].dtype != np.float32 or handle["32_udf"].dtype != np.float32:
                return False, "UDF datasets must be float32"
            if str(handle.attrs.get("preprocessing_version", "")) != config.preprocessing_version:
                return False, "preprocessing version mismatch"
            if int(handle.attrs.get("source_point_count", -1)) != 1_000_000:
                return False, "source point count mismatch"
            if float(handle.attrs.get("coordinate_min", np.nan)) != config.coordinate_min:
                return False, "coordinate_min mismatch"
            if float(handle.attrs.get("coordinate_max", np.nan)) != config.coordinate_max:
                return False, "coordinate_max mismatch"
            if str(handle.attrs.get("downsample_rule", "")) != "128_udf[::4,::4,::4]":
                return False, "downsample rule mismatch"
            if check_values:
                udf128 = np.asarray(handle["128_udf"][:], dtype=np.float32)
                udf32 = np.asarray(handle["32_udf"][:], dtype=np.float32)
                if not np.isfinite(udf128).all() or np.any(udf128 < 0):
                    return False, "128_udf contains invalid values"
                if not np.isfinite(udf32).all() or np.any(udf32 < 0):
                    return False, "32_udf contains invalid values"
                aligned = udf128[::4, ::4, ::4]
                if not np.array_equal(udf32, aligned):
                    return False, "32_udf is not the exact stride-four view"
    except (KeyError, OSError, ValueError) as exc:
        return False, str(exc)
    return True, "ok"


def exact_point_udf_grid(
    points: torch.Tensor,
    *,
    grid_n: int,
    coordinate_min: float,
    coordinate_max: float,
    query_chunk_size: int,
) -> torch.Tensor:
    """Compute exact nearest-sample UDF values on a dense vertex grid."""
    try:
        from pytorch3d.ops import knn_points
    except ImportError as exc:
        raise ImportError("Exact ABC UDF preprocessing requires PyTorch3D") from exc

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected non-empty point tensor with shape (N,3), got {points.shape}")
    axis = torch.linspace(
        coordinate_min,
        coordinate_max,
        grid_n,
        device=points.device,
        dtype=points.dtype,
    )
    try:
        xyz = torch.meshgrid(axis, axis, axis, indexing="ij")
    except TypeError:
        xyz = torch.meshgrid(axis, axis, axis)
    queries = torch.stack(xyz, dim=-1).reshape(-1, 3)

    distances: list[torch.Tensor] = []
    reference = points.unsqueeze(0)
    for chunk in queries.split(query_chunk_size, dim=0):
        squared = knn_points(chunk.unsqueeze(0), reference, K=1).dists[0, :, 0]
        distances.append(squared.clamp_min_(0).sqrt_())
    return torch.cat(distances).reshape(grid_n, grid_n, grid_n)


def write_udf_sidecar(
    output_path: str | Path,
    udf128: np.ndarray,
    *,
    source_point_count: int,
    config: ABCUDFConfig,
) -> None:
    """Atomically write one exact 129^3 UDF and its aligned 33^3 view."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("ABC UDF preprocessing requires h5py") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    udf128 = np.asarray(udf128, dtype=np.float32)
    if udf128.shape != (129, 129, 129):
        raise ValueError(f"Expected 129^3 UDF, got {udf128.shape}")
    udf32 = np.ascontiguousarray(udf128[::4, ::4, ::4])
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with h5py.File(temporary, "w") as handle:
            options = {
                "compression": config.compression,
                "compression_opts": config.compression_level,
                "shuffle": True,
            }
            handle.create_dataset("128_udf", data=udf128, **options)
            handle.create_dataset("32_udf", data=udf32, **options)
            handle.attrs["coordinate_min"] = config.coordinate_min
            handle.attrs["coordinate_max"] = config.coordinate_max
            handle.attrs["source_point_count"] = int(source_point_count)
            handle.attrs["preprocessing_version"] = config.preprocessing_version
            handle.attrs["downsample_rule"] = "128_udf[::4,::4,::4]"
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


class ABCHybridDataset(Dataset):
    """Load aligned ABC SDF/UDF grids and normalized surface targets."""

    def __init__(
        self,
        model_ids: Iterable[str],
        *,
        hdf5_root: str | Path,
        udf_root: str | Path,
        target_sample_count: int,
        seed: int,
        deterministic_targets: bool = False,
    ) -> None:
        self.model_ids = [Path(model_id).stem for model_id in model_ids]
        self.hdf5_root = Path(hdf5_root)
        self.udf_root = Path(udf_root)
        self.target_sample_count = int(target_sample_count)
        self.seed = int(seed)
        self.deterministic_targets = bool(deterministic_targets)
        if not self.model_ids:
            raise ValueError("ABCHybridDataset requires at least one model ID")
        if self.target_sample_count < 1:
            raise ValueError("target_sample_count must be positive")

    def __len__(self) -> int:
        return len(self.model_ids)

    def _target_indices(self, index: int, point_count: int) -> np.ndarray:
        replace = point_count < self.target_sample_count
        if self.deterministic_targets:
            rng = np.random.RandomState(self.seed + index)
            return rng.choice(point_count, self.target_sample_count, replace=replace)
        return np.random.choice(point_count, self.target_sample_count, replace=replace)

    def __getitem__(self, index: int) -> dict:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("ABC HybridPoNQ training requires h5py") from exc

        model_id = self.model_ids[index]
        source_path = self.hdf5_root / f"{model_id}.hdf5"
        sidecar_path = udf_sidecar_path(self.udf_root, model_id)
        with h5py.File(source_path, "r") as source:
            sdf = np.asarray(source["32_sdf"][:], dtype=np.float32) * 2.0
            points = np.asarray(source["pointcloud"][:], dtype=np.float32)
        with h5py.File(sidecar_path, "r") as sidecar:
            udf = np.asarray(sidecar["32_udf"][:], dtype=np.float32) * 2.0
        if sdf.shape != (33, 33, 33) or udf.shape != (33, 33, 33):
            raise ValueError(
                f"{model_id} requires 33^3 SDF/UDF grids, got {sdf.shape} and {udf.shape}"
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{model_id} pointcloud must have shape (N,3), got {points.shape}")
        if not np.isfinite(sdf).all() or not np.isfinite(udf).all():
            raise ValueError(f"{model_id} contains non-finite SDF/UDF values")

        indices = self._target_indices(index, points.shape[0])
        targets = np.ascontiguousarray(points[indices] * 2.0, dtype=np.float32)
        input_grid = np.stack((sdf, udf), axis=0)
        return {
            "input_grid": torch.from_numpy(input_grid),
            "sdf_grid": torch.from_numpy(sdf[None, ...]),
            "target_points": torch.from_numpy(targets),
            "model_id": model_id,
            "source_path": str(source_path),
            "udf_path": str(sidecar_path),
        }


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
    rng = np.random.RandomState(seed)
    selected = set(int(index) for index in rng.choice(len(ids), count, replace=False))
    return [model_id for index, model_id in enumerate(ids) if index in selected]
