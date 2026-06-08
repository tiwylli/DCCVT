import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.grid import build_hybrid_input_channels, make_canonical_sites
from dccvt.neural.hybrid_infer import _load_checkpoint
from dccvt.neural.models import DCCVTHybridDirectNet, HybridDirectConfig


def test_canonical_sites_match_dccvt_grid_ordering():
    sites = make_canonical_sites(5)
    assert sites.shape == (64, 3)
    assert torch.allclose(sites[0], torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(sites[-1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(sites[1], torch.tensor([-1.0, -1.0, -1.0 / 3.0]), atol=1e-6)


def test_hybrid_input_channels_encode_point_zero_level():
    sdf = torch.zeros(3, 3, 3)
    points = torch.tensor([[-1.0, -1.0, -1.0]])
    channels = build_hybrid_input_channels(sdf, points, grid_n=3, confidence_sigma_scale=1.0)

    assert channels.shape == (4, 3, 3, 3)
    assert channels[2, 0, 0, 0] == 0.0
    assert channels[3, 0, 0, 0] == 1.0
    assert channels[2, -1, -1, -1] > channels[2, 0, 0, 0]
    assert channels[3, -1, -1, -1] < channels[3, 0, 0, 0]


def test_hybrid_direct_forward_shapes_and_residual_composition():
    config = HybridDirectConfig(grid_n=5, feature_dim=8, encoder_layers=1, decoder_layers=1)
    model = DCCVTHybridDirectNet(config)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    axis = torch.linspace(-1.0, 1.0, 5)
    x, _, _ = torch.meshgrid(axis, axis, axis, indexing="ij")
    input_grid = torch.stack((x, x.abs(), torch.zeros_like(x), torch.ones_like(x)), dim=0).unsqueeze(0)
    out = model(input_grid)

    assert out["sites"].shape == (1, 64, 3)
    assert out["sites_sdf"].shape == (1, 64)
    assert torch.allclose(out["site_delta"], torch.zeros_like(out["site_delta"]))
    assert torch.allclose(out["sdf_residual"], torch.zeros_like(out["sdf_residual"]))
    assert torch.allclose(out["sites_sdf"][0], out["canonical_sites"][:, 0], atol=1e-6)


def test_hybrid_direct_checkpoint_reload(tmp_path):
    config = HybridDirectConfig(grid_n=5, feature_dim=8, encoder_layers=1, decoder_layers=1)
    model = DCCVTHybridDirectNet(config)
    checkpoint = tmp_path / "hybrid.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "seed": 123,
            "channel_names": list(config.channel_names),
            "args": {},
        },
        checkpoint,
    )

    loaded, payload = _load_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded.config() == model.config()
    assert payload["seed"] == 123
