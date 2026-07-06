import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.iterative.config import HybridIterRefineConfig, load_iter_refine_config
from dccvt.neural.iterative.graph import (
    build_directed_edges_from_simplices,
    delaunay_edge_features,
    fourier_site_position_encoding,
    local_knn_parent_features,
    select_procedural_refinement_parents,
)
from dccvt.neural.iterative.infer import run_inference
from dccvt.neural.iterative.initial_extract import extract_initialization_cache
from dccvt.neural.iterative.initialization import build_hotspot_near_surface_initialization
from dccvt.neural.iterative.model import DCCVTHybridIterRefineNet
from dccvt.neural.iterative.train import build_train_arg_parser, main as train_main
from dccvt.neural.losses import hybrid_direct_mesh_loss
from dccvt.neural.data.point_udf_sidecar import (
    exact_point_udf_grid,
    load_point_udf_sidecar,
    validate_point_udf_sidecar,
    write_point_udf_sidecar,
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
        "configs/neural_hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128.json": (2, 128),
        "configs/neural_hybrid_iter_refine_v4_delaunay_gcnn_udf65_knn_r2_p128.json": (2, 128),
    }

    for relative_path, (rounds, parent_budget) in expected.items():
        config = load_iter_refine_config(root / relative_path)

        assert config.input_channels == 2
        assert config.channel_names == ("hotspot_sdf", "point_udf")
        assert config.num_refinement_rounds == rounds
        assert config.max_parents_per_round == parent_budget
        if "v3" in relative_path or "v4" in relative_path:
            assert config.base_grid_n == 17
            assert (config.base_grid_n - 1) ** 3 == 4096
            assert config.surface_pair_count == 3236
            assert (config.base_grid_n - 1) ** 3 + config.surface_pair_count == 7332
            assert (
                (config.base_grid_n - 1) ** 3
                + config.surface_pair_count
                + config.num_refinement_rounds * config.max_parents_per_round * config.slots_per_parent
                == 8356
            )
            assert config.local_feature_mode == "udf65_knn_stats"
            assert config.local_udf_grid_n == 65
            assert config.local_udf_samples is True
            assert config.local_knn_features is True
            assert config.local_knn_k == 8
            assert config.local_knn_radius == pytest.approx(0.0625)
        if "v4" in relative_path:
            assert config.architecture == "delaunay_gcnn"
            assert config.graph_layers == 3
            assert config.graph_hidden_dim == config.feature_dim
            assert config.site_position_encoding == "fourier"
            assert config.site_position_num_frequencies == 4
            assert config.graph_edge_features == "relative_xyz_distance_direction_sdf_delta"


def test_fourier_site_position_encoding_is_deterministic_and_finite():
    sites = torch.tensor([[0.0, 0.5, -1.0], [1.0, -0.25, 0.25]], dtype=torch.float32)

    encoded = fourier_site_position_encoding(sites, num_frequencies=4)
    second = fourier_site_position_encoding(sites, num_frequencies=4)

    assert encoded.shape == (2, 27)
    assert torch.equal(encoded, second)
    assert torch.allclose(encoded[:, :3], sites)
    assert torch.isfinite(encoded).all()


def test_directed_delaunay_edges_and_edge_features_are_bidirectional():
    sites = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sites_sdf = torch.tensor([-1.0, 0.5, 0.25, -0.25], dtype=torch.float32)
    simplices = np.array([[0, 1, 2, 3]], dtype=np.int64)

    edges = build_directed_edges_from_simplices(simplices, num_sites=sites.shape[0], device=sites.device)
    features = delaunay_edge_features(sites, sites_sdf, edges)

    assert edges.shape == (12, 2)
    assert features.shape == (12, 8)
    assert torch.isfinite(features).all()

    forward_index = torch.nonzero((edges[:, 0] == 0) & (edges[:, 1] == 1), as_tuple=False).reshape(-1)[0]
    reverse_index = torch.nonzero((edges[:, 0] == 1) & (edges[:, 1] == 0), as_tuple=False).reshape(-1)[0]
    assert torch.allclose(features[forward_index, :3], -features[reverse_index, :3])
    assert torch.allclose(features[forward_index, 3:4], features[reverse_index, 3:4])
    assert torch.allclose(features[forward_index, 4:7], -features[reverse_index, 4:7])
    assert torch.allclose(features[forward_index, 7:], -features[reverse_index, 7:])


def test_point_udf_sidecar_matches_bruteforce_distances_and_metadata(tmp_path):
    points = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    udf = exact_point_udf_grid(points, grid_n=65, query_chunk_size=4096)
    coords = torch.linspace(-1.0, 1.0, 65)
    probe = torch.tensor(
        [
            [0, 0, 0],
            [32, 32, 32],
            [64, 64, 64],
        ],
        dtype=torch.long,
    )
    probe_points = torch.stack([coords[probe[:, 0]], coords[probe[:, 1]], coords[probe[:, 2]]], dim=1)
    expected = torch.cdist(probe_points.unsqueeze(0), points.unsqueeze(0)).squeeze(0).amin(dim=1)

    assert torch.allclose(udf[probe[:, 0], probe[:, 1], probe[:, 2]], expected, atol=1e-6)

    sidecar_path = tmp_path / "unit.npz"
    write_point_udf_sidecar(
        sidecar_path,
        udf.numpy(),
        source_cache_path=tmp_path / "unit_cache.npz",
        source_point_count=points.shape[0],
        grid_n=65,
        seed=123,
        command_args={"seed": 123},
    )

    valid, reason = validate_point_udf_sidecar(sidecar_path, grid_n=65, check_values=True)
    loaded = load_point_udf_sidecar(sidecar_path, grid_n=65)

    assert valid is True, reason
    assert loaded.shape == (65, 65, 65)
    with np.load(sidecar_path, allow_pickle=False) as data:
        assert data["65_udf"].shape == (65, 65, 65)
        assert int(data["grid_n"]) == 65
        assert int(data["source_point_count"]) == 2
        metadata = json.loads(str(data["metadata"]))
    assert metadata["coordinate_min"] == -1.0
    assert metadata["coordinate_max"] == 1.0
    assert metadata["source_point_count"] == 2


def test_local_knn_parent_features_are_finite_with_fewer_than_k_points():
    parent_sites = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32)
    target_points = torch.tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=torch.float32)

    features = local_knn_parent_features(parent_sites, target_points, k=8, radius=0.25)

    assert features.shape == (2, 7)
    assert torch.isfinite(features).all()
    assert torch.all(features[:, 0] >= 0.0)
    assert torch.all((features[:, -1] >= 0.0) & (features[:, -1] <= 1.0))


def test_iter_refine_initialization_export_has_zero_rounds_and_full_default_site_count(tmp_path):
    config = load_iter_refine_config(
        Path(__file__).resolve().parents[1]
        / "configs/neural_hybrid_iter_refine_initial_v2_hotspot_point_udf.json"
    )
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    target_points = np.zeros((4, 3), dtype=np.float32)
    cache_path = tmp_path / "unit_mesh.npz"
    np.savez_compressed(
        cache_path,
        sdf_grid=sdf_grid.numpy().astype(np.float32),
        target_points=target_points,
        grid_n=np.array(config.hotspot_grid_n, dtype=np.int64),
        mesh_id=np.array("unit_mesh"),
    )

    result = extract_initialization_cache(
        cache_path,
        tmp_path / "out",
        config=config,
        seed=69,
        extract=False,
        command_args={"seed": 69},
    )
    field_file = Path(result["field_file"])

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


def test_delaunay_gcnn_forward_one_round_appends_finite_spawned_sites():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = HybridIterRefineConfig.from_dict(
        {
            **_small_config(rounds=1).to_dict(),
            "architecture": "delaunay_gcnn",
            "graph_layers": 2,
            "graph_hidden_dim": 8,
            "site_position_num_frequencies": 2,
        }
    )
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]

    outputs = model(input_grid, sdf_grid[None, None, ...])

    assert outputs["base_sites"].shape == (96, 3)
    assert outputs["sites"].shape[1] == 96 + 3 * 2
    assert outputs["sites_sdf"].shape == (1, 102)
    assert len(outputs["rounds"]) == 1
    assert outputs["rounds"][0]["parent_indices"].shape == (3,)
    assert outputs["rounds"][0]["spawned_sites"].shape == (6, 3)
    assert torch.isfinite(outputs["sites"]).all()
    assert torch.isfinite(outputs["sites_sdf"]).all()
    assert outputs["sites"].requires_grad is True


def test_delaunay_gcnn_zero_initialized_decoder_uses_stencil_children():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = HybridIterRefineConfig.from_dict(
        {
            **_small_config(rounds=1).to_dict(),
            "architecture": "delaunay_gcnn",
            "graph_layers": 1,
            "graph_hidden_dim": 8,
            "site_position_num_frequencies": 1,
            "max_parents_per_round": 1,
            "spawn_min_distance": 0.0,
        }
    )
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]

    outputs = model(input_grid, sdf_grid[None, None, ...])

    parent_index = int(outputs["rounds"][0]["parent_indices"][0].item())
    parent_site = outputs["base_sites"][parent_index]
    spawned = outputs["rounds"][0]["spawned_sites"].reshape(1, config.slots_per_parent, 3)[0]
    expected = parent_site[None, :] + model.child_stencil[: config.slots_per_parent] * config.child_stencil_scale
    assert torch.allclose(spawned, expected.clamp(-1.0, 1.0), atol=1e-6)


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


def test_v3_forward_uses_local_features_and_grows_default_initialization():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    root = Path(__file__).resolve().parents[1]
    base = load_iter_refine_config(
        root / "configs/neural_hybrid_iter_refine_v3_hotspot_point_udf_udf65_knn_r2_p128.json"
    )
    config = HybridIterRefineConfig.from_dict(
        {
            **base.to_dict(),
            "base_grid_n": 5,
            "surface_pair_count": 32,
            "min_surface_anchors": 8,
            "bootstrap_candidate_multipliers": (2, 4),
            "feature_dim": 4,
            "encoder_layers": 0,
            "decoder_layers": 1,
            "max_parents_per_round": 1,
            "num_refinement_rounds": 1,
        }
    )
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    point_udf = torch.zeros_like(sdf_grid)
    input_grid = torch.stack([sdf_grid, point_udf], dim=0)[None, ...]
    local_udf_grid = torch.zeros((1, 1, 65, 65, 65), dtype=torch.float32)
    target_points = torch.tensor(
        [
            [-0.5, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )[None, ...]

    outputs = model(
        input_grid,
        sdf_grid[None, None, ...],
        target_points=target_points,
        local_udf_grid=local_udf_grid,
    )

    assert outputs["base_sites"].shape == (96, 3)
    assert outputs["sites"].shape[1] > 96
    assert outputs["sites"].requires_grad is True
    assert len(outputs["rounds"]) == 1
    assert outputs["rounds"][0]["parent_indices"].shape == (1,)
    assert outputs["rounds"][0]["spawned_sites"].shape[0] > 0


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


def test_train_main_rejects_legacy_resume_mode_mismatch(tmp_path):
    legacy = HybridIterRefineConfig.from_dict({"hotspot_grid_n": 33, "base_grid_n": 17})
    requested = HybridIterRefineConfig()
    config_path = tmp_path / "requested.json"
    checkpoint_path = tmp_path / "legacy.pt"
    config_path.write_text(json.dumps(requested.to_dict()), encoding="utf-8")
    torch.save({"model_config": legacy.to_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="different initialization mode"):
        train_main(["--config", str(config_path), "--resume", str(checkpoint_path), "--device", "cpu", "--epochs", "0"])


def test_train_main_rejects_different_base_grid_for_same_initialization(tmp_path):
    requested = HybridIterRefineConfig.from_dict(
        {"initialization_mode": "hotspot_near_surface", "base_grid_n": 17}
    )
    checkpoint_config = HybridIterRefineConfig.from_dict(
        {"initialization_mode": "hotspot_near_surface", "base_grid_n": 9}
    )
    config_path = tmp_path / "requested.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    config_path.write_text(json.dumps(requested.to_dict()), encoding="utf-8")
    torch.save({"model_config": checkpoint_config.to_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="different base grid"):
        train_main(["--config", str(config_path), "--resume", str(checkpoint_path), "--device", "cpu", "--epochs", "0"])


def test_iter_refine_cvt_loss_has_finite_gradients():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = _small_config(rounds=1)
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    input_grid = sdf_grid[None, None, ...]
    outputs = model(input_grid, sdf_grid[None, None, ...])
    target_points = torch.tensor(
        [[[-0.75, 0.0, 0.0], [-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    loss, _ = hybrid_direct_mesh_loss(
        outputs,
        target_points,
        chamfer_weight=0.0,
        cvt_weight=100.0,
        sdfsmooth_weight=0.0,
        strict=True,
    )
    loss.backward()

    assert torch.isfinite(loss).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_iter_refine_prediction_export_contains_round_metadata(tmp_path):
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    config = _small_config(rounds=1)
    model = DCCVTHybridIterRefineNet(config)
    sdf_grid = _linear_sdf_grid(config.hotspot_grid_n)
    target_points = np.zeros((4, 3), dtype=np.float32)
    cache_path = tmp_path / "unit_mesh.npz"
    checkpoint_path = tmp_path / "checkpoint.pt"
    np.savez_compressed(
        cache_path,
        sdf_grid=sdf_grid.numpy().astype(np.float32),
        target_points=target_points,
        grid_n=np.array(config.hotspot_grid_n, dtype=np.int64),
        mesh_id=np.array("unit_mesh"),
    )
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "seed": 123,
        },
        checkpoint_path,
    )

    result = run_inference(
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        output_dir=tmp_path / "prediction",
        device_value="cpu",
        extract=False,
        seed=123,
        command_args={"seed": 123},
    )
    prediction_file = Path(result["prediction_file"])

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
