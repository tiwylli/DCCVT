"""External command construction for mesh fine-tuning."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Sequence

from dccvt.neural.experiments.mesh_finetune_cv.config import (
    EVAL_ROOT,
    EVAL_SCRIPT,
    INFER_SCRIPT,
    ROOT,
    TRAIN_SCRIPT,
    CommandJob,
    ExperimentConfig,
    LossVariant,
)
from dccvt.neural.experiments.mesh_finetune_cv.folds import (
    FoldSplit,
    checkpoint_dir,
    evaluation_mesh_dir,
    extracted_mesh_path,
    fold_test_file,
    fold_train_file,
    inference_dir,
)

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
