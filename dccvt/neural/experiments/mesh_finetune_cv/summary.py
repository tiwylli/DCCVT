"""Metric summary generation for mesh fine-tuning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from dccvt.neural.experiments.mesh_finetune_cv.config import ExperimentConfig, LossVariant, METRIC_COLUMNS
from dccvt.neural.experiments.mesh_finetune_cv.folds import FoldSplit, extracted_mesh_path, inference_dir

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
