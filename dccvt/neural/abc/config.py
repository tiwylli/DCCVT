"""Configuration and split helpers for ABC HybridPoNQ-DCCVT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]

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
