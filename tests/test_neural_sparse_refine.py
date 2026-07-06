import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.extraction.sparse_refine import (
    HybridSparseRefineConfig,
    build_sparse_base_field,
    extract_sparse_refine_cache,
    refine_sparse_field,
)


def _linear_sdf_grid(grid_n: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, grid_n)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return x + 0.25 * y - 0.1 * z


def test_sparse_base_field_samples_hotspot_sdf_on_sparse_canonical_sites():
    grid_n = 33
    base_grid_n = 17
    sdf_grid = _linear_sdf_grid(grid_n)

    sites, sites_sdf = build_sparse_base_field(
        sdf_grid,
        base_grid_n=base_grid_n,
        hotspot_grid_n=grid_n,
    )

    assert sites.shape == (4096, 3)
    assert sites_sdf.shape == (4096,)
    assert torch.isfinite(sites).all()
    assert torch.isfinite(sites_sdf).all()
    assert float(sites.min()) >= -1.0
    assert float(sites.max()) <= 1.0
    expected = sites[:, 0] + 0.25 * sites[:, 1] - 0.1 * sites[:, 2]
    assert torch.allclose(sites_sdf, expected, atol=1e-6)


def test_refine_sparse_field_spawns_finite_sites_inside_domain():
    pytest.importorskip("pygdel3d")
    torch.manual_seed(69)
    sdf_grid = _linear_sdf_grid(17)
    base_sites, base_sites_sdf = build_sparse_base_field(
        sdf_grid,
        base_grid_n=5,
        hotspot_grid_n=17,
    )

    refined_sites, refined_sdf, diagnostics = refine_sparse_field(
        sdf_grid,
        base_sites,
        base_sites_sdf,
        upsampling_rounds=1,
        growth_cap=0.25,
        clamp_domain=True,
    )

    assert refined_sites.shape[0] > base_sites.shape[0]
    assert refined_sdf.shape == (refined_sites.shape[0],)
    assert diagnostics["spawned_site_count"] == refined_sites.shape[0] - base_sites.shape[0]
    assert diagnostics["completed_upsampling_rounds"] == 1
    assert torch.isfinite(refined_sites).all()
    assert torch.isfinite(refined_sdf).all()
    assert float(refined_sites.min()) >= -1.0
    assert float(refined_sites.max()) <= 1.0


def test_extract_sparse_refine_cache_saves_field_metadata_without_mesh_extraction(tmp_path):
    grid_n = 33
    mesh_id = "unit_mesh"
    cache_path = tmp_path / f"{mesh_id}.npz"
    sdf_grid = _linear_sdf_grid(grid_n).numpy().astype(np.float32)
    target_points = np.array(
        [
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_path,
        sdf_grid=sdf_grid,
        target_points=target_points,
        grid_n=np.array(grid_n, dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )
    config = HybridSparseRefineConfig(
        cache_root=str(tmp_path),
        output_root=str(tmp_path / "out"),
        base_grid_n=5,
        hotspot_grid_n=grid_n,
        upsampling_rounds=0,
        seed=123,
    )

    result = extract_sparse_refine_cache(
        cache_path,
        tmp_path / "out" / mesh_id,
        config=config,
        extract=False,
    )

    assert result["status"] == "field_saved"
    field_file = result["field_file"]
    with np.load(field_file, allow_pickle=False) as data:
        assert data["base_sites"].shape == (64, 3)
        assert data["sites"].shape == (64, 3)
        assert data["sites_sdf"].shape == (64,)
        assert int(data["seed"]) == 123
        diagnostics = json.loads(str(data["diagnostics"]))
        resolved_config = json.loads(str(data["resolved_config"]))

    assert diagnostics["mesh_id"] == mesh_id
    assert diagnostics["base_grid_n"] == 5
    assert diagnostics["spawned_site_count"] == 0
    assert diagnostics["positive_sdf_count"] > 0
    assert diagnostics["negative_sdf_count"] > 0
    assert resolved_config["refinement_mode"] == "procedural_upsample"
