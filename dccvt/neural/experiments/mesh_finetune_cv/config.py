"""Configuration loading for the hybrid-direct mesh fine-tuning study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TRAIN_SCRIPT = ROOT / "scripts" / "train_dccvt_hybrid_direct.py"
INFER_SCRIPT = ROOT / "scripts" / "infer_dccvt_hybrid_direct.py"
EVAL_ROOT = ROOT / "PoNQ-main"
EVAL_SCRIPT = EVAL_ROOT / "src" / "eval" / "eval_HOTSPOT.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "neural_dccvt" / "hybrid_direct_mesh_finetune_cv"
MESH_FILENAME = "DCCVT_0_hybrid_direct_{variant}_cvt{w_cvt}_sdfsmooth{w_sdfsmooth}.obj"
METRIC_COLUMNS = ("index", "cd1", "cd2", "f1", "nc", "ecd2", "ef1")

@dataclass(frozen=True)
class FoldSplit:
    """One deterministic train/test split."""

    index: int
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


@dataclass(frozen=True)
class LossVariant:
    """Mesh-loss weights for one adaptation run."""

    name: str
    w_mesh: float
    w_chamfer: float
    w_cvt: float
    w_sdfsmooth: float


@dataclass(frozen=True)
class EvaluationConfig:
    """Mesh extraction and metric settings."""

    mesh_variant: str
    sample_count: int
    seed: int
    modes: tuple[str, ...]
    w_cvt: float
    w_sdfsmooth: float


@dataclass(frozen=True)
class QualificationConfig:
    """Go/no-go thresholds for the adaptation study."""

    minimum_improved_folds: int
    minimum_improved_shapes: int
    maximum_nc_regression: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved mesh fine-tuning experiment configuration."""

    experiment_name: str
    source_ids_file: Path
    model_config: Path
    cache_root: Path
    label_root: Path
    starting_checkpoint: Path
    starting_checkpoint_epoch: int
    ground_truth_root: Path
    train_python: Path
    eval_python: Path
    fold_count: int
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    seed: int
    save_every: int
    target_subsample: int | None
    label: dict
    supervised_loss: dict
    variants: tuple[LossVariant, ...]
    evaluation: EvaluationConfig
    qualification: QualificationConfig

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "source_ids_file",
            "model_config",
            "cache_root",
            "label_root",
            "starting_checkpoint",
            "ground_truth_root",
            "train_python",
            "eval_python",
        ):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class CommandJob:
    """One external command and its log/output identity."""

    name: str
    command: tuple[str, ...]
    cwd: Path
    log_path: Path | None = None


def _resolve_path(value: str, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _required_mapping(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate the adaptation-study JSON config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    variants_data = data.get("variants")
    if not isinstance(variants_data, list) or not variants_data:
        raise ValueError("variants must be a non-empty list")
    variants = tuple(
        LossVariant(
            name=str(item["name"]),
            w_mesh=float(item["w_mesh"]),
            w_chamfer=float(item["w_chamfer"]),
            w_cvt=float(item["w_cvt"]),
            w_sdfsmooth=float(item["w_sdfsmooth"]),
        )
        for item in variants_data
    )
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("variant names must be unique")

    evaluation_data = _required_mapping(data, "evaluation")
    modes = tuple(str(mode) for mode in evaluation_data.get("modes", ()))
    valid_modes = {"ponq_thingi", "raw", "bbox_aligned"}
    if not modes or not set(modes).issubset(valid_modes):
        raise ValueError(f"evaluation.modes must be selected from {sorted(valid_modes)}")

    qualification_data = _required_mapping(data, "qualification")
    config = ExperimentConfig(
        experiment_name=str(data["experiment_name"]),
        source_ids_file=_resolve_path(data["source_ids_file"]),
        model_config=_resolve_path(data["model_config"]),
        cache_root=_resolve_path(data["cache_root"]),
        label_root=_resolve_path(data["label_root"]),
        starting_checkpoint=_resolve_path(data["starting_checkpoint"]),
        starting_checkpoint_epoch=int(data["starting_checkpoint_epoch"]),
        ground_truth_root=_resolve_path(data["ground_truth_root"]),
        train_python=_resolve_path(data["train_python"]),
        eval_python=_resolve_path(data["eval_python"]),
        fold_count=int(data["fold_count"]),
        epochs=int(data["epochs"]),
        batch_size=int(data["batch_size"]),
        learning_rate=float(data["learning_rate"]),
        num_workers=int(data["num_workers"]),
        seed=int(data["seed"]),
        save_every=int(data["save_every"]),
        target_subsample=(
            None if data.get("target_subsample") is None else int(data["target_subsample"])
        ),
        label=_required_mapping(data, "label"),
        supervised_loss=_required_mapping(data, "supervised_loss"),
        variants=variants,
        evaluation=EvaluationConfig(
            mesh_variant=str(evaluation_data["mesh_variant"]),
            sample_count=int(evaluation_data["sample_count"]),
            seed=int(evaluation_data["seed"]),
            modes=modes,
            w_cvt=float(evaluation_data["w_cvt"]),
            w_sdfsmooth=float(evaluation_data["w_sdfsmooth"]),
        ),
        qualification=QualificationConfig(
            minimum_improved_folds=int(qualification_data["minimum_improved_folds"]),
            minimum_improved_shapes=int(qualification_data["minimum_improved_shapes"]),
            maximum_nc_regression=float(qualification_data["maximum_nc_regression"]),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if config.fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.starting_checkpoint_epoch < 0:
        raise ValueError("starting_checkpoint_epoch must be non-negative")
    if config.evaluation.sample_count < 1:
        raise ValueError("evaluation.sample_count must be positive")
    if config.evaluation.mesh_variant not in {"intDCCVT", "projDCCVT"}:
        raise ValueError("evaluation.mesh_variant must be intDCCVT or projDCCVT")

    required_paths = (
        config.source_ids_file,
        config.model_config,
        config.cache_root,
        config.label_root,
        config.starting_checkpoint,
        config.ground_truth_root,
        config.train_python,
        config.eval_python,
        TRAIN_SCRIPT,
        INFER_SCRIPT,
        EVAL_SCRIPT,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment inputs:\n" + "\n".join(missing))

