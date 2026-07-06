"""Fold assignment and output path conventions for mesh fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import json

from dccvt.neural.experiments.mesh_finetune_cv.config import ExperimentConfig, FoldSplit, MESH_FILENAME

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
