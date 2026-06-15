"""Five-fold mesh-loss adaptation experiment orchestration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
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


def read_model_ids(path: str | Path) -> list[str]:
    """Read model IDs while preserving source-file order."""
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(Path(line).stem)
    if not ids:
        raise ValueError(f"No model IDs found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate model IDs found in {path}")
    return ids


def assign_folds(model_ids: Sequence[str], fold_count: int) -> tuple[FoldSplit, ...]:
    """Assign IDs by source index modulo fold count."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(model_ids) < fold_count:
        raise ValueError("model count must be at least fold_count")

    folds = []
    for fold_index in range(fold_count):
        test_ids = tuple(model_id for index, model_id in enumerate(model_ids) if index % fold_count == fold_index)
        test_set = set(test_ids)
        train_ids = tuple(model_id for model_id in model_ids if model_id not in test_set)
        folds.append(FoldSplit(index=fold_index, train_ids=train_ids, test_ids=test_ids))
    return tuple(folds)


def _write_ids(path: Path, model_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{model_id}\n" for model_id in model_ids), encoding="utf-8")


def fold_train_file(output_root: Path, fold_index: int) -> Path:
    return output_root / "splits" / f"fold_{fold_index}_train.txt"


def fold_test_file(output_root: Path, fold_index: int) -> Path:
    return output_root / "splits" / f"fold_{fold_index}_test.txt"


def checkpoint_dir(output_root: Path, variant: str, fold_index: int) -> Path:
    return output_root / "runs" / variant / f"fold_{fold_index}" / "checkpoints"


def inference_dir(output_root: Path, method: str, model_id: str, fold_index: int | None = None) -> Path:
    root = output_root / "inference" / method
    if fold_index is not None:
        root = root / f"fold_{fold_index}"
    return root / model_id


def evaluation_mesh_dir(output_root: Path, method: str, mesh_variant: str) -> Path:
    return output_root / "eval_meshes" / f"{method}_{mesh_variant}"


def extracted_mesh_path(output_dir: Path, config: ExperimentConfig) -> Path:
    filename = MESH_FILENAME.format(
        variant=config.evaluation.mesh_variant,
        w_cvt=int(config.evaluation.w_cvt),
        w_sdfsmooth=int(config.evaluation.w_sdfsmooth),
    )
    return output_dir / filename


def prepare_experiment(config: ExperimentConfig, output_root: Path) -> tuple[FoldSplit, ...]:
    """Persist resolved config and deterministic split files."""
    model_ids = read_model_ids(config.source_ids_file)
    folds = assign_folds(model_ids, config.fold_count)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    resolved["output_root"] = str(output_root.resolve())
    resolved["folds"] = [
        {"index": fold.index, "train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)}
        for fold in folds
    ]
    (output_root / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for fold in folds:
        _write_ids(fold_train_file(output_root, fold.index), fold.train_ids)
        _write_ids(fold_test_file(output_root, fold.index), fold.test_ids)
    return folds


def _select_folds(folds: Sequence[FoldSplit], selected: Sequence[int] | None) -> tuple[FoldSplit, ...]:
    if selected is None:
        return tuple(folds)
    by_index = {fold.index: fold for fold in folds}
    unknown = sorted(set(selected) - set(by_index))
    if unknown:
        raise ValueError(f"Unknown fold indices: {unknown}")
    return tuple(by_index[index] for index in selected)


def _select_variants(
    variants: Sequence[LossVariant],
    selected: Sequence[str] | None,
) -> tuple[LossVariant, ...]:
    if selected is None:
        return tuple(variants)
    by_name = {variant.name: variant for variant in variants}
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return tuple(by_name[name] for name in selected)


def build_training_jobs(
    config: ExperimentConfig,
    output_root: Path,
    *,
    folds: Sequence[FoldSplit],
    variants: Sequence[LossVariant],
    force: bool = False,
) -> list[CommandJob]:
    """Build the ten fold/variant training jobs."""
    jobs = []
    label = config.label
    supervised = config.supervised_loss
    for variant in variants:
        for fold in folds:
            run_checkpoint_dir = checkpoint_dir(output_root, variant.name, fold.index)
            expected_final = run_checkpoint_dir / (
                f"epoch_{config.starting_checkpoint_epoch + config.epochs:04d}.pt"
            )
            if expected_final.exists() and not force:
                continue
            if run_checkpoint_dir.exists() and any(run_checkpoint_dir.iterdir()) and not force:
                raise FileExistsError(
                    f"Partial or incompatible training output exists at {run_checkpoint_dir}; "
                    "use --force to restart from the configured supervised checkpoint"
                )
            command = [
                str(config.train_python),
                str(TRAIN_SCRIPT),
                "--config",
                str(config.model_config),
                "--cache-root",
                str(config.cache_root),
                "--label-root",
                str(config.label_root),
                "--split-file",
                str(fold_train_file(output_root, fold.index)),
                "--checkpoint-dir",
                str(run_checkpoint_dir),
                "--resume",
                str(config.starting_checkpoint),
                "--stage",
                "mesh",
                "--epochs",
                str(config.epochs),
                "--batch-size",
                str(config.batch_size),
                "--lr",
                str(config.learning_rate),
                "--device",
                "cuda",
                "--num-workers",
                str(config.num_workers),
                "--seed",
                str(config.seed),
                "--save-every",
                str(config.save_every),
                "--label-upsampling",
                str(label["upsampling"]),
                "--label-state",
                str(label["state"]),
                "--label-variant",
                str(label["variant"]),
                "--label-w-cvt",
                str(label["w_cvt"]),
                "--label-w-sdfsmooth",
                str(label["w_sdfsmooth"]),
                "--w-site",
                str(supervised["w_site"]),
                "--w-sdf",
                str(supervised["w_sdf"]),
                "--w-sign",
                str(supervised["w_sign"]),
                "--w-residual",
                str(supervised["w_residual"]),
                "--sdf-near-weight",
                str(supervised["sdf_near_weight"]),
                "--sdf-near-tau",
                str(supervised["sdf_near_tau"]),
                "--sign-temperature",
                str(supervised["sign_temperature"]),
                "--w-mesh",
                str(variant.w_mesh),
                "--w-mesh-chamfer",
                str(variant.w_chamfer),
                "--w-mesh-cvt",
                str(variant.w_cvt),
                "--w-mesh-sdfsmooth",
                str(variant.w_sdfsmooth),
                "--strict-mesh-loss",
            ]
            if config.target_subsample is not None:
                command.extend(["--target-subsample", str(config.target_subsample)])
            jobs.append(
                CommandJob(
                    name=f"{variant.name}/fold_{fold.index}/checkpoints",
                    command=tuple(command),
                    cwd=ROOT,
                    log_path=run_checkpoint_dir / "train.log",
                )
            )
    return jobs


def _inference_command(
    config: ExperimentConfig,
    *,
    checkpoint: Path,
    model_id: str,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        str(config.train_python),
        str(INFER_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(config.cache_root / f"{model_id}.npz"),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
        "--seed",
        str(config.seed),
        "--w-cvt",
        str(config.evaluation.w_cvt),
        "--w-sdfsmooth",
        str(config.evaluation.w_sdfsmooth),
    )


def build_inference_jobs(
    config: ExperimentConfig,
    output_root: Path,
    *,
    folds: Sequence[FoldSplit],
    variants: Sequence[LossVariant],
    include_baseline: bool = True,
    force: bool = False,
) -> list[CommandJob]:
    """Build baseline and out-of-fold inference jobs."""
    jobs = []
    selected_ids = [model_id for fold in folds for model_id in fold.test_ids]
    if include_baseline:
        for model_id in selected_ids:
            output_dir = inference_dir(output_root, "baseline", model_id)
            if extracted_mesh_path(output_dir, config).exists() and not force:
                continue
            jobs.append(
                CommandJob(
                    name=f"baseline/{model_id}",
                    command=_inference_command(
                        config,
                        checkpoint=config.starting_checkpoint,
                        model_id=model_id,
                        output_dir=output_dir,
                    ),
                    cwd=ROOT,
                    log_path=output_dir / "infer.log",
                )
            )

    for variant in variants:
        for fold in folds:
            fold_checkpoint = checkpoint_dir(output_root, variant.name, fold.index) / "latest.pt"
            for model_id in fold.test_ids:
                output_dir = inference_dir(output_root, variant.name, model_id, fold.index)
                if extracted_mesh_path(output_dir, config).exists() and not force:
                    continue
                jobs.append(
                    CommandJob(
                        name=f"{variant.name}/fold_{fold.index}/{model_id}",
                        command=_inference_command(
                            config,
                            checkpoint=fold_checkpoint,
                            model_id=model_id,
                            output_dir=output_dir,
                        ),
                        cwd=ROOT,
                        log_path=output_dir / "infer.log",
                    )
                )
    return jobs


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def create_evaluation_views(
    config: ExperimentConfig,
    output_root: Path,
    *,
    folds: Sequence[FoldSplit],
    variants: Sequence[LossVariant],
    include_baseline: bool = True,
) -> None:
    """Create flat <mesh_id>.obj views consumed by eval_HOTSPOT.py."""
    if include_baseline:
        baseline_view = evaluation_mesh_dir(output_root, "baseline", config.evaluation.mesh_variant)
        for fold in folds:
            for model_id in fold.test_ids:
                source = extracted_mesh_path(inference_dir(output_root, "baseline", model_id), config)
                if not source.exists():
                    raise FileNotFoundError(source)
                _replace_symlink(baseline_view / f"{model_id}.obj", source)

    for variant in variants:
        view = evaluation_mesh_dir(output_root, variant.name, config.evaluation.mesh_variant)
        for fold in folds:
            for model_id in fold.test_ids:
                source = extracted_mesh_path(
                    inference_dir(output_root, variant.name, model_id, fold.index),
                    config,
                )
                if not source.exists():
                    raise FileNotFoundError(source)
                _replace_symlink(view / f"{model_id}.obj", source)


def selected_ids_file(output_root: Path, folds: Sequence[FoldSplit]) -> Path:
    indices = "_".join(str(fold.index) for fold in folds)
    return output_root / "splits" / f"selected_folds_{indices}.txt"


def build_evaluation_jobs(
    config: ExperimentConfig,
    output_root: Path,
    *,
    folds: Sequence[FoldSplit],
    variants: Sequence[LossVariant],
    include_baseline: bool = True,
) -> list[CommandJob]:
    """Build deterministic full-view evaluator jobs."""
    ids_path = selected_ids_file(output_root, folds)
    methods = ["baseline"] if include_baseline else []
    methods.extend(variant.name for variant in variants)
    results_dir = output_root / "evaluation"
    jobs = []
    for method in methods:
        pred_dir = evaluation_mesh_dir(output_root, method, config.evaluation.mesh_variant)
        command = (
            str(config.eval_python),
            str(EVAL_SCRIPT),
            str(pred_dir),
            "-gt_dir",
            str(config.ground_truth_root),
            "-all_models",
            str(ids_path),
            "-pred_suffix",
            ".obj",
            "-mode",
            "all",
            "-sample_num",
            str(config.evaluation.sample_count),
            "-seed",
            str(config.evaluation.seed),
            "-results_dir",
            str(results_dir),
        )
        jobs.append(
            CommandJob(
                name=f"evaluate/{method}",
                command=command,
                cwd=EVAL_ROOT,
                log_path=results_dir / f"{method}.log",
            )
        )
    return jobs


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_jobs(jobs: Sequence[CommandJob], *, dry_run: bool) -> None:
    """Run commands sequentially with reproducible per-job logs."""
    for job in jobs:
        print(f"[{job.name}] {_command_text(job.command)}", flush=True)
        if dry_run:
            continue
        if job.log_path is None:
            subprocess.run(job.command, cwd=job.cwd, check=True)
            continue
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        with job.log_path.open("w", encoding="utf-8") as log:
            log.write(_command_text(job.command) + "\n")
            log.flush()
            subprocess.run(
                job.command,
                cwd=job.cwd,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )


def _load_metrics(results_dir: Path, method_dir_name: str, mode: str, model_ids: Sequence[str]) -> dict[str, dict]:
    path = results_dir / f"results_{method_dir_name}_{mode}.npy"
    values = np.load(path)
    if values.shape != (len(model_ids), len(METRIC_COLUMNS)):
        raise ValueError(f"Unexpected metric shape in {path}: {values.shape}")
    metrics = {}
    for row in values:
        index = int(row[0])
        model_id = model_ids[index]
        metrics[model_id] = {
            "cd1": float(row[1]),
            "cd2": float(row[2]),
            "cd_x1e5": float(row[2] * 1e5),
            "f1": float(row[3]),
            "nc": float(row[4]),
            "ecd": float(row[5] * 1e2),
            "ef1": float(row[6]),
        }
    return metrics


def summarize_results(
    config: ExperimentConfig,
    output_root: Path,
    *,
    folds: Sequence[FoldSplit],
    variants: Sequence[LossVariant],
) -> dict:
    """Write paired per-shape/fold summaries and the go/no-go decision."""
    model_ids = [model_id for fold in folds for model_id in fold.test_ids]
    fold_by_id = {model_id: fold.index for fold in folds for model_id in fold.test_ids}
    results_dir = output_root / "evaluation"
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    mesh_variant = config.evaluation.mesh_variant

    methods = ("baseline", *(variant.name for variant in variants))
    all_metrics: dict[str, dict[str, dict[str, dict]]] = {}
    for method in methods:
        method_dir_name = f"{method}_{mesh_variant}"
        all_metrics[method] = {
            mode: _load_metrics(results_dir, method_dir_name, mode, model_ids)
            for mode in config.evaluation.modes
        }

    per_shape_path = summary_dir / "per_shape_metrics.csv"
    with per_shape_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "fold", "model_id", "mode", "cd1", "cd2", "cd_x1e5", "f1", "nc", "ecd", "ef1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            for mode in config.evaluation.modes:
                for model_id in model_ids:
                    writer.writerow(
                        {
                            "method": method,
                            "fold": fold_by_id[model_id],
                            "model_id": model_id,
                            "mode": mode,
                            **all_metrics[method][mode][model_id],
                        }
                    )

    baseline_bbox = all_metrics["baseline"]["bbox_aligned"]
    baseline_strict = all_metrics["baseline"]["ponq_thingi"]
    fold_rows = []
    decisions = {}
    for variant in variants:
        variant_bbox = all_metrics[variant.name]["bbox_aligned"]
        variant_strict = all_metrics[variant.name]["ponq_thingi"]
        improved_shapes = sum(
            variant_bbox[model_id]["cd2"] < baseline_bbox[model_id]["cd2"]
            for model_id in model_ids
        )
        improved_folds = 0
        for fold in folds:
            baseline_cd = float(np.mean([baseline_bbox[model_id]["cd_x1e5"] for model_id in fold.test_ids]))
            variant_cd = float(np.mean([variant_bbox[model_id]["cd_x1e5"] for model_id in fold.test_ids]))
            improved = variant_cd < baseline_cd
            improved_folds += int(improved)
            fold_rows.append(
                {
                    "variant": variant.name,
                    "fold": fold.index,
                    "shape_count": len(fold.test_ids),
                    "baseline_bbox_cd_x1e5": baseline_cd,
                    "variant_bbox_cd_x1e5": variant_cd,
                    "bbox_cd_delta_x1e5": variant_cd - baseline_cd,
                    "improved": int(improved),
                }
            )

        baseline_nc = float(np.mean([baseline_bbox[model_id]["nc"] for model_id in model_ids]))
        variant_nc = float(np.mean([variant_bbox[model_id]["nc"] for model_id in model_ids]))
        variant_bbox_cd = float(np.mean([variant_bbox[model_id]["cd_x1e5"] for model_id in model_ids]))
        variant_strict_cd = float(np.mean([variant_strict[model_id]["cd_x1e5"] for model_id in model_ids]))
        extraction_failures = sum(
            not extracted_mesh_path(
                inference_dir(output_root, variant.name, model_id, fold_by_id[model_id]),
                config,
            ).exists()
            for model_id in model_ids
        )
        qualifies = (
            improved_folds >= config.qualification.minimum_improved_folds
            and improved_shapes >= config.qualification.minimum_improved_shapes
            and variant_nc >= baseline_nc - config.qualification.maximum_nc_regression
            and extraction_failures == 0
        )
        decisions[variant.name] = {
            "qualifies": qualifies,
            "improved_folds": improved_folds,
            "improved_shapes": improved_shapes,
            "bbox_cd_x1e5": variant_bbox_cd,
            "bbox_nc": variant_nc,
            "strict_cd_x1e5": variant_strict_cd,
            "extraction_failures": extraction_failures,
        }

    fold_summary_path = summary_dir / "fold_summary.csv"
    with fold_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    qualifying = [name for name, decision in decisions.items() if decision["qualifies"]]
    qualifying.sort(
        key=lambda name: (
            decisions[name]["bbox_cd_x1e5"],
            -decisions[name]["bbox_nc"],
            decisions[name]["strict_cd_x1e5"],
        )
    )
    decision_summary = {
        "study_type": "adaptation",
        "limitation": (
            "Held-out shapes were excluded only from mesh fine-tuning; the shared supervised "
            "starting checkpoint was trained on all 31 shapes."
        ),
        "baseline": {
            "bbox_cd_x1e5": float(np.mean([baseline_bbox[model_id]["cd_x1e5"] for model_id in model_ids])),
            "bbox_nc": float(np.mean([baseline_bbox[model_id]["nc"] for model_id in model_ids])),
            "strict_cd_x1e5": float(np.mean([baseline_strict[model_id]["cd_x1e5"] for model_id in model_ids])),
        },
        "variants": decisions,
        "recommended_variant": qualifying[0] if qualifying else None,
    }
    (summary_dir / "decision_summary.json").write_text(
        json.dumps(decision_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return decision_summary


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid direct mesh-loss adaptation study.")
    parser.add_argument("--config", default="configs/neural_hybrid_mesh_finetune_cv.json")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "infer", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold indices.")
    parser.add_argument("--variants", default=None, help="Optional comma-separated variant names.")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed training or inference outputs.")
    parser.add_argument("--parallel", action="store_true", help="Run training jobs across available GPUs.")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    output_root = Path(args.output_root).resolve()
    all_folds = assign_folds(read_model_ids(config.source_ids_file), config.fold_count)
    folds = _select_folds(all_folds, _parse_csv_ints(args.folds))
    variants = _select_variants(config.variants, _parse_csv_strings(args.variants))
    include_baseline = not args.no_baseline

    if args.stage in {"prepare", "all"}:
        if args.dry_run:
            print(f"[prepare] output_root={output_root}")
        else:
            prepare_experiment(config, output_root)
    elif not (output_root / "resolved_config.json").exists():
        if args.dry_run:
            print(f"[prepare-required] output_root={output_root}")
        else:
            prepare_experiment(config, output_root)

    if args.stage in {"train", "all"}:
        jobs = build_training_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            force=args.force,
        )
        if args.parallel:
            from scripts.run_hybrid_direct_channel_ablation import run_parallel

            run_parallel(
                [(job.name, list(job.command)) for job in jobs],
                output_root=output_root / "runs",
                devices=args.devices,
                min_free_gb=args.min_free_gb,
                poll_seconds=args.poll_seconds,
                max_jobs=args.max_jobs,
                dry_run=args.dry_run,
            )
        else:
            run_jobs(jobs, dry_run=args.dry_run)

    if args.stage in {"infer", "all"}:
        jobs = build_inference_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            include_baseline=include_baseline,
            force=args.force,
        )
        run_jobs(jobs, dry_run=args.dry_run)
        if not args.dry_run:
            create_evaluation_views(
                config,
                output_root,
                folds=folds,
                variants=variants,
                include_baseline=include_baseline,
            )

    if args.stage in {"evaluate", "all"}:
        ids_path = selected_ids_file(output_root, folds)
        if args.dry_run:
            print(f"[evaluation-ids] {ids_path}")
        else:
            _write_ids(ids_path, (model_id for fold in folds for model_id in fold.test_ids))
            create_evaluation_views(
                config,
                output_root,
                folds=folds,
                variants=variants,
                include_baseline=include_baseline,
            )
        jobs = build_evaluation_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            include_baseline=include_baseline,
        )
        run_jobs(jobs, dry_run=args.dry_run)

    if args.stage in {"summarize", "all"}:
        if not include_baseline:
            raise SystemExit("Summarization requires the baseline")
        if args.dry_run:
            print(f"[summarize] {output_root / 'summary'}")
        else:
            summary = summarize_results(config, output_root, folds=folds, variants=variants)
            print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
