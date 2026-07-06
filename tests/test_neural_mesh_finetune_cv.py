from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.losses import hybrid_direct_mesh_loss
from dccvt.neural.experiments.mesh_finetune_cv import (
    ROOT,
    QualificationConfig,
    assign_folds,
    build_training_jobs,
    checkpoint_dir,
    evaluation_mesh_dir,
    extracted_mesh_path,
    inference_dir,
    load_experiment_config,
    summarize_results,
)


CONFIG_PATH = ROOT / "configs" / "neural_hybrid_mesh_finetune_cv.json"


def test_mesh_finetune_fold_assignment_preserves_source_order():
    model_ids = [f"mesh_{index}" for index in range(31)]
    folds = assign_folds(model_ids, 5)

    assert [len(fold.test_ids) for fold in folds] == [7, 6, 6, 6, 6]
    assert folds[0].test_ids == tuple(model_ids[0::5])
    assert folds[1].test_ids == tuple(model_ids[1::5])
    assert folds[0].train_ids == tuple(model_id for index, model_id in enumerate(model_ids) if index % 5 != 0)
    assert sorted(model_id for fold in folds for model_id in fold.test_ids) == sorted(model_ids)


def test_mesh_finetune_config_and_training_commands(tmp_path):
    config = load_experiment_config(CONFIG_PATH)
    folds = assign_folds(["a", "b", "c", "d", "e"], 5)
    variant = next(item for item in config.variants if item.name == "chamfer_only")
    jobs = build_training_jobs(
        config,
        tmp_path,
        folds=(folds[0],),
        variants=(variant,),
    )

    assert config.fold_count == 5
    assert [item.name for item in config.variants] == ["chamfer_only", "composite"]
    assert len(jobs) == 1
    command = list(jobs[0].command)
    assert command[command.index("--resume") + 1] == str(config.starting_checkpoint)
    assert "--resume-optimizer" not in command
    assert "--strict-mesh-loss" in command
    assert command[command.index("--w-mesh-cvt") + 1] == "0.0"
    assert command[command.index("--w-mesh-sdfsmooth") + 1] == "0.0"
    assert jobs[0].log_path == checkpoint_dir(tmp_path, "chamfer_only", 0) / "train.log"


def test_mesh_finetune_config_rejects_invalid_fold_count(tmp_path):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["fold_count"] = 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="fold_count"):
        load_experiment_config(path)


def test_mesh_finetune_output_naming():
    config = load_experiment_config(CONFIG_PATH)
    output_dir = Path("/tmp/mesh-finetune/inference/composite/fold_0/313444")

    assert extracted_mesh_path(output_dir, config).name == (
        "DCCVT_0_hybrid_direct_intDCCVT_cvt100_sdfsmooth100.obj"
    )
    assert evaluation_mesh_dir(Path("/tmp/out"), "composite", "intDCCVT") == (
        Path("/tmp/out/eval_meshes/composite_intDCCVT")
    )


def test_mesh_loss_default_skips_invalid_signs_and_strict_mode_fails():
    outputs = {
        "sites": torch.zeros(1, 8, 3, requires_grad=True),
        "sites_sdf": torch.ones(1, 8, requires_grad=True),
    }
    target_points = torch.zeros(1, 4, 3)

    loss, stats = hybrid_direct_mesh_loss(outputs, target_points, strict=False)

    assert loss.item() == 0.0
    assert stats["mesh_used_shapes"] == 0.0
    assert stats["mesh_skipped_shapes"] == 1.0
    with pytest.raises(RuntimeError, match="positive and negative"):
        hybrid_direct_mesh_loss(outputs, target_points, strict=True)


def _load_hotspot_eval_module():
    path = ROOT / "PoNQ-main" / "src" / "eval" / "eval_HOTSPOT.py"
    spec = importlib.util.spec_from_file_location("dccvt_test_eval_hotspot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hotspot_evaluation_seed_is_deterministic(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    module = _load_hotspot_eval_module()
    module.EDGE_SAMPLE_NUM = 1000

    gt_path = tmp_path / "gt.obj"
    pred_path = tmp_path / "pred.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(gt_path)
    prediction = trimesh.creation.box(extents=(0.9, 1.0, 1.1))
    prediction.apply_translation((0.02, -0.01, 0.03))
    prediction.export(pred_path)

    item = (0, "box", gt_path, pred_path, "bbox_aligned", 500, 69)
    first = module.evaluate_one(item)
    second = module.evaluate_one(item)

    assert np.array_equal(first, second)


def test_mesh_finetune_summary_applies_qualification_rules(tmp_path):
    config = load_experiment_config(CONFIG_PATH)
    config = replace(
        config,
        qualification=QualificationConfig(
            minimum_improved_folds=4,
            minimum_improved_shapes=4,
            maximum_nc_regression=0.01,
        ),
    )
    folds = assign_folds(["a", "b", "c", "d", "e"], 5)
    variant = next(item for item in config.variants if item.name == "chamfer_only")
    results_dir = tmp_path / "evaluation"
    results_dir.mkdir(parents=True)

    for method, cd2 in (("baseline", 0.002), ("chamfer_only", 0.001)):
        rows = np.array(
            [[index, 0.01, cd2, 0.2, 0.8, 0.03, 0.1] for index in range(5)],
            dtype=np.float64,
        )
        for mode in config.evaluation.modes:
            np.save(
                results_dir / f"results_{method}_intDCCVT_{mode}.npy",
                rows,
            )

    for fold in folds:
        model_id = fold.test_ids[0]
        mesh_path = extracted_mesh_path(
            inference_dir(tmp_path, variant.name, model_id, fold.index),
            config,
        )
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.touch()

    summary = summarize_results(config, tmp_path, folds=folds, variants=(variant,))

    assert summary["variants"]["chamfer_only"]["qualifies"] is True
    assert summary["variants"]["chamfer_only"]["improved_folds"] == 5
    assert summary["variants"]["chamfer_only"]["improved_shapes"] == 5
    assert summary["recommended_variant"] == "chamfer_only"
    assert (tmp_path / "summary" / "per_shape_metrics.csv").exists()
    assert (tmp_path / "summary" / "fold_summary.csv").exists()
