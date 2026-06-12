import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.hybrid_initial import build_initial_hotspot_field


def test_initial_hotspot_field_samples_linear_sdf_on_canonical_sites():
    grid_n = 33
    axis = torch.linspace(-1.0, 1.0, grid_n)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    sdf_grid = x + 2.0 * y - 0.5 * z

    sites, sites_sdf = build_initial_hotspot_field(sdf_grid, grid_n=grid_n)

    assert sites.shape == (32768, 3)
    assert sites_sdf.shape == (32768,)
    assert torch.allclose(sites[0], torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(sites[-1], torch.tensor([1.0, 1.0, 1.0]))
    expected = sites[:, 0] + 2.0 * sites[:, 1] - 0.5 * sites[:, 2]
    assert torch.allclose(sites_sdf, expected, atol=1e-6)
