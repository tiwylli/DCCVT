import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.iter_refine import (
    DCCVTHybridIterRefineNet,
    HybridIterRefineConfig,
    _resolve_resume_config,
    _save_initialization_field,
    _save_prediction,
    build_hotspot_near_surface_initialization,
    build_train_arg_parser,
    load_iter_refine_config,
    select_procedural_refinement_parents,
)


def _linear_sdf_grid(grid_n: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, grid_n)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return x + 0.25 * y - 0.1 * z


def _small_config(rounds: int = 1) -> HybridIterRefineConfig:
    return HybridIterRefineConfig(
        hotspot_grid_n=17,
        base_grid_n=5,
        surface_pair_count=32,
        min_surface_anchors=8,
        projection_steps=2,
        bootstrap_candidate_multipliers=(2, 4),
        input_channels=1,
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        slots_per_parent=2,
        max_parents_per_round=3,
        num_refinement_rounds=rounds,
        child_offset_scale=0.02,
        sdf_residual_scale=0.05,
        spawn_min_distance=0.001,
        channel_names=("hotspot_sdf",),
    )


def test_parent_selection_returns_unique_indices_up_to_budget():
    pytest.importorskip("pygdel3d")
    sites = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, 4),
            torch.linspace(-1.0, 1.0, 4),
            torch.linspace(-1.0, 1.0, 4),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    sites_sdf = sites[:, 0]

    selected = select_procedural_refinement_parents(sites, sites_sdf, max_parents=1000)
    parent_indices = selected["parent_indices"]
    parent_scores = selected["parent_scores"]

    assert 0 < parent_indices.shape[0] <= sites.shape[0]
    assert parent_scores.shape == parent_indices.shape
    assert torch.unique(parent_indices).shape == parent_indices.shape
    assert parent_indices.dtype == torch.long
    assert int(parent_indices.min()) >= 0
    assert int(parent_indices.max()) < sites.shape[0]
    assert torch.isfinite(parent_scores).all()


def test_parent_selection_returns_empty_when_no_zero_crossing_exists():
    pytest.importorskip("pygdel3d")
    sites = torch.rand(16, 3) * 2.0 - 1.0
    sites_sdf = torch.ones(16)

    selected = select_procedural_refinement_parents(sites, sites_sdf, max_parents=4)

    assert selected["parent_indices"].numel() == 0
    assert selected["parent_scores"].numel() == 0


def test_near_surface_initialization_is_deterministic_and_sign_balanced():
    config = _small_config(rounds=0)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)

    first = build_hotspot_near_surface_initialization(sdf_grid, config)
    second = build_hotspot_near_surface_initialization(sdf_grid, config)

    assert first["valid"] is True
    assert first["sites"].shape == (96, 3)
    assert first["background_sites"].shape == (64, 3)
    assert first["surface_anchors"].shape == (16, 3)
    assert first["surface_sites"].shape == (32, 3)
    assert torch.equal(first["sites"], second["sites"])
    assert int((first["surface_sdf"] < 0).sum()) == 16
    assert int((first["surface_sdf"] > 0).sum()) == 16
    assert first["diagnostics"]["unique_site_count"] == 96
    assert first["diagnostics"]["minimum_site_distance"] >= config.bootstrap_min_distance


def test_default_initialization_count_matches_v0_budget():
    config = HybridIterRefineConfig(input_channels=1, channel_names=("hotspot_sdf",))
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)

    initialization = build_hotspot_near_surface_initialization(sdf_grid, config)

    assert initialization["valid"] is True
    assert initialization["background_sites"].shape == (512, 3)
    assert initialization["surface_sites"].shape == (3236, 3)
    assert initialization["sites"].shape == (3748, 3)
    assert initialization["sites"].shape[0] + 128 * 4 == 4260


def test_two_channel_comparison_configs_load_expected_budgets():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json": (0, 128),
        "configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p128.json": (1, 128),
        "configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r2_p128.json": (2, 128),
        "configs/neural_hybrid_iter_refine_v2_hotspot_point_udf_r1_p256.json": (1, 256),
    }

    for relative_path, (rounds, parent_budget) in expected.items():
        config = load_iter_refine_config(root / relative_path)

        assert config.input_channels == 2
        assert config.channel_names == ("hotspot_sdf", "point_udf")
        assert config.num_refinement_rounds == rounds
        assert config.max_parents_per_round == parent_budget


def test_iter_refine_initialization_export_has_zero_rounds_and_full_default_site_count(tmp_path):
    config = load_iter_refine_config(
        Path(__file__).resolve().parents[1]
        / "configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json"
    )
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    initialization = build_hotspot_near_surface_initialization(sdf_grid, config)
    target_points = np.zeros((4, 3), dtype=np.float32)
    input_grid = np.zeros(
        (config.input_channels, config.hotspot_grid_n, config.hotspot_grid_n, config.hotspot_grid_n),
        dtype=np.float32,
    )

    field_file = _save_initialization_field(
        tmp_path,
        mesh_id="unit_mesh",
        initialization=initialization,
        input_grid=input_grid,
        sdf_grid=sdf_grid.numpy().astype(np.float32),
        target_points=target_points,
        config=config,
        seed=69,
        command_args={"seed": 69},
    )

    with np.load(field_file, allow_pickle=False) as data:
        assert data["sites"].shape == (3748, 3)
        assert data["sites_sdf"].shape == (3748,)
        assert data["background_sites"].shape == (512, 3)
        assert data["surface_sites"].shape == (3236, 3)
        diagnostics = json.loads(str(data["diagnostics"]))
        resolved_config = json.loads(str(data["resolved_config"]))

    assert diagnostics["round_count"] == 0
    assert diagnostics["site_count"] == 3748
    assert diagnostics["base_site_count"] == 3748
    assert diagnostics["initialization"]["initial_site_count"] == 3748
    assert resolved_config["input_channels"] == 2
    assert resolved_config["channel_names"] == ["hotspot_sdf", "point_udf"]


def test_near_surface_initialization_reports_no_crossing():
    config = _small_config(rounds=0)
    initialization = build_hotspot_near_surface_initialization(
        torch.ones((config.hotspot_grid_n,) * 3),
        config,
    )

    assert initialization["valid"] is False
    assert initialization["reason"] == "no_sign_changing_cells"
    assert initialization["surface_sites"].numel() == 0
    assert build_train_arg_parser().parse_args(["--strict-initialization"]).strict_initialization is True


def test_iter_refine_forward_one_round_appends_finite_spawned_sites():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = _small_config(rounds=1)
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]

    outputs = model(input_grid, sdf_grid[None, None, ...])

    assert outputs["base_sites"].shape == (96, 3)
    assert outputs["sites"].shape[1] == 96 + 3 * 2
    assert outputs["sites_sdf"].shape == (1, 102)
    assert len(outputs["rounds"]) == 1
    assert outputs["rounds"][0]["spawned_sites"].shape == (6, 3)
    assert torch.isfinite(outputs["sites"]).all()
    assert torch.isfinite(outputs["sites_sdf"]).all()
    assert float(outputs["sites"].min()) >= -1.0
    assert float(outputs["sites"].max()) <= 1.0
    spawned = outputs["rounds"][0]["spawned_sites"].reshape(3, 2, 3)
    assert torch.all(torch.norm(spawned[:, 0] - spawned[:, 1], dim=1) > 0.0)


def test_iter_refine_forward_two_rounds_has_monotonic_site_growth():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = _small_config(rounds=2)
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]

    outputs = model(input_grid, sdf_grid[None, None, ...])

    assert len(outputs["rounds"]) == 2
    assert outputs["sites"].shape[1] >= 96 + 3 * 2
    assert outputs["sites"].shape[1] <= 96 + 2 * 3 * 2


def test_spawn_filter_rejects_existing_and_duplicate_sites():
    config = _small_config(rounds=0)
    config.spawn_min_distance = 0.01
    model = DCCVTHybridIterRefineNet(config)
    candidates = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1001, 0.0, 0.0]])
    candidate_sdf = torch.tensor([0.0, 0.1, 0.1])
    existing = torch.tensor([[0.0, 0.0, 0.0]])

    sites, sites_sdf, rejected = model._filter_spawned_sites(candidates, candidate_sdf, existing)

    assert sites.shape == (1, 3)
    assert sites_sdf.shape == (1,)
    assert rejected == 2


def test_legacy_config_and_resume_mode_compatibility():
    legacy = HybridIterRefineConfig.from_dict({"hotspot_grid_n": 33, "base_grid_n": 17})
    requested = HybridIterRefineConfig()

    assert legacy.config_version == 1
    assert legacy.initialization_mode == "canonical"
    assert legacy.background_jitter_scale == 0.0
    assert legacy.child_stencil_scale == 0.0
    with pytest.raises(ValueError, match="different initialization mode"):
        _resolve_resume_config(requested, {"model_config": legacy.to_dict()})


def test_iter_refine_prediction_export_contains_round_metadata(tmp_path):
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = _small_config(rounds=1)
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]
    outputs = model(input_grid, sdf_grid[None, None, ...])
    checkpoint = {"model_config": config.to_dict(), "epoch": 0, "seed": 123}
    target_points = np.zeros((4, 3), dtype=np.float32)

    prediction_file = _save_prediction(
        tmp_path,
        mesh_id="unit_mesh",
        outputs=outputs,
        input_grid=input_grid[0].numpy().astype(np.float32),
        sdf_grid=sdf_grid.numpy().astype(np.float32),
        target_points=target_points,
        checkpoint=checkpoint,
        command_args={"seed": 123},
    )

    with np.load(prediction_file, allow_pickle=False) as data:
        assert data["sites"].shape[1] == 3
        assert data["background_sites"].shape == (64, 3)
        assert data["surface_anchors"].shape == (16, 3)
        assert data["surface_sites"].shape == (32, 3)
        assert data["round_00_parent_indices"].shape == (3,)
        assert data["round_00_spawned_sites"].shape == (6, 3)
        assert int(data["round_00_rejected_spawn_count"]) == 0
        diagnostics = json.loads(str(data["diagnostics"]))
        resolved_config = json.loads(str(data["resolved_config"]))

    assert diagnostics["mesh_id"] == "unit_mesh"
    assert diagnostics["round_count"] == 1
    assert diagnostics["initialization_valid"] is True
    assert diagnostics["initialization"]["initial_site_count"] == 96
    assert resolved_config["training_objective"] == "mesh_loss_only"
