import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.grid import make_gt_activity_mask_np, make_near_surface_mask_np, trilinear_interpolate_sdf
from dccvt.neural.models import DCCVTPoNQNet


def test_model_forward_shapes_and_domain():
    model = DCCVTPoNQNet(grid_n=5, k=2, feature_dim=8, encoder_layers=1, decoder_layers=1)
    sdf = torch.zeros(3, 1, 5, 5, 5)
    out = model(sdf)
    assert out["sites"].shape == (3, 64, 2, 3)
    assert out["activity_logits"].shape == (3, 64)
    assert torch.isfinite(out["sites"]).all()
    assert (out["sites"].abs() <= 1.0).all()


def test_trilinear_interpolate_sdf_matches_linear_x_field():
    grid_n = 5
    axis = torch.linspace(-1.0, 1.0, grid_n)
    x, _, _ = torch.meshgrid(axis, axis, axis, indexing="ij")
    points = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.25, -0.5], [1.0, 1.0, 1.0]])
    values = trilinear_interpolate_sdf(x, points)
    assert torch.allclose(values, points[:, 0], atol=1e-6)


def test_masks_are_flattened_cell_masks():
    sdf = np.zeros((4, 4, 4), dtype=np.float32)
    near = make_near_surface_mask_np(sdf)
    assert near.shape == (27,)
    assert near.all()

    samples = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    gt = make_gt_activity_mask_np(samples, grid_n=4)
    assert gt.shape == (27,)
    assert gt.sum() == 2
