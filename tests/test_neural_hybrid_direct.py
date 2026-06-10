import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.grid import build_hybrid_input_channels, make_canonical_sites
from dccvt.neural.hybrid_infer import _load_checkpoint
from dccvt.neural.models import DCCVTHybridDirectNet, HybridDirectConfig
from scripts.run_hybrid_direct_channel_ablation import (
    ABLATIONS,
    GpuInfo,
    _prepare_parallel_training_args,
    build_commands,
    build_parallel_dry_run_assignments,
    filter_available_gpus,
    parse_nvidia_smi_output,
    resolve_device_ids,
)


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


def test_hybrid_input_channels_can_select_and_order_channels():
    sdf = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3) - 13.0
    points = torch.tensor([[-1.0, -1.0, -1.0]])
    channels = build_hybrid_input_channels(
        sdf,
        points,
        grid_n=3,
        confidence_sigma_scale=1.0,
        channel_names=("hotspot_sdf", "point_udf", "abs_hotspot_sdf"),
    )

    assert channels.shape == (3, 3, 3, 3)
    assert torch.allclose(channels[0], sdf)
    assert torch.allclose(channels[2], sdf.abs())
    assert channels[1, 0, 0, 0] == 0.0
    assert channels[1, -1, -1, -1] > channels[1, 0, 0, 0]


def test_hybrid_input_channels_support_hotspot_sdf_only():
    sdf = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    channels = build_hybrid_input_channels(
        sdf,
        torch.empty(0, 3),
        grid_n=3,
        channel_names=("hotspot_sdf",),
    )

    assert channels.shape == (1, 3, 3, 3)
    assert torch.allclose(channels[0], sdf)


def test_hybrid_input_channels_support_hotspot_sdf_and_point_udf():
    sdf = torch.zeros(3, 3, 3)
    points = torch.tensor([[-1.0, -1.0, -1.0]])
    channels = build_hybrid_input_channels(
        sdf,
        points,
        grid_n=3,
        channel_names=("hotspot_sdf", "point_udf"),
    )

    assert channels.shape == (2, 3, 3, 3)
    assert torch.allclose(channels[0], sdf)
    assert channels[1, 0, 0, 0] == 0.0
    assert channels[1, -1, -1, -1] > 0.0


def test_hybrid_direct_config_derives_and_validates_input_channels():
    config = HybridDirectConfig(
        grid_n=5,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )

    assert config.input_channels == 2
    assert config.to_dict()["input_channels"] == 2

    with pytest.raises(ValueError, match="does not match"):
        HybridDirectConfig(input_channels=4, channel_names=("hotspot_sdf", "point_udf"))
    with pytest.raises(ValueError, match="first"):
        HybridDirectConfig(channel_names=("point_udf", "hotspot_sdf"))
    with pytest.raises(ValueError, match="Unknown"):
        HybridDirectConfig(channel_names=("hotspot_sdf", "bad_channel"))


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


def test_hybrid_direct_forward_accepts_reduced_channel_config():
    config = HybridDirectConfig(
        grid_n=5,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    model = DCCVTHybridDirectNet(config)
    input_grid = torch.zeros(1, 2, 5, 5, 5)
    out = model(input_grid)

    assert out["sites"].shape == (1, 64, 3)
    assert out["sites_sdf"].shape == (1, 64)


def test_hybrid_direct_checkpoint_reload(tmp_path):
    config = HybridDirectConfig(
        grid_n=5,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
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
    assert loaded.config_obj.channel_names == config.channel_names
    assert payload["seed"] == 123


def test_channel_ablation_runner_builds_expected_commands(tmp_path):
    commands = build_commands(
        output_root=tmp_path,
        training_args=["--cache-root", "cache", "--epochs", "1"],
    )

    assert [run_name for run_name, _ in commands] == [run_name for run_name, _ in ABLATIONS]
    assert all("--config" in command for _, command in commands)
    assert all("--checkpoint-dir" in command for _, command in commands)
    assert commands[0][1][-4:] == ["--cache-root", "cache", "--epochs", "1"]

    with pytest.raises(SystemExit, match="--config"):
        build_commands(output_root=tmp_path, training_args=["--config", "other.json"])


def test_channel_ablation_runner_parses_nvidia_smi_output():
    gpus = parse_nvidia_smi_output(
        """
        0, 49140, 655
        1, 49140, 46571
        """
    )

    assert gpus == [
        GpuInfo(index=0, memory_total_mb=49140, memory_used_mb=655),
        GpuInfo(index=1, memory_total_mb=49140, memory_used_mb=46571),
    ]
    assert gpus[0].memory_free_mb == 48485


def test_channel_ablation_runner_filters_available_gpus():
    gpus = [
        GpuInfo(index=0, memory_total_mb=49140, memory_used_mb=655),
        GpuInfo(index=1, memory_total_mb=49140, memory_used_mb=46571),
        GpuInfo(index=2, memory_total_mb=49140, memory_used_mb=1000),
    ]

    available = filter_available_gpus(gpus, allowed_ids=[0, 1], min_free_gb=20.0)

    assert [gpu.index for gpu in available] == [0]


def test_channel_ablation_runner_respects_cuda_visible_devices():
    assert resolve_device_ids("auto", env={"CUDA_VISIBLE_DEVICES": "0,3"}) == [0, 3]
    assert resolve_device_ids("1,2", env={"CUDA_VISIBLE_DEVICES": "0,3"}) == [1, 2]


def test_channel_ablation_runner_rejects_cpu_device_in_parallel_mode():
    with pytest.raises(SystemExit, match="cpu"):
        _prepare_parallel_training_args(["--cache-root", "cache", "--device", "cpu"])
    with pytest.raises(SystemExit, match="cpu"):
        _prepare_parallel_training_args(["--cache-root", "cache", "--device=cpu"])

    assert _prepare_parallel_training_args(["--cache-root", "cache"])[-2:] == ["--device", "cuda"]


def test_channel_ablation_runner_builds_parallel_dry_run_assignments(tmp_path):
    commands = build_commands(
        output_root=tmp_path,
        training_args=["--cache-root", "cache", "--device", "cuda"],
    )
    gpus = [
        GpuInfo(index=0, memory_total_mb=49140, memory_used_mb=655),
        GpuInfo(index=1, memory_total_mb=49140, memory_used_mb=1000),
    ]

    assignments = build_parallel_dry_run_assignments(
        commands,
        gpus=gpus,
        allowed_ids=[0, 1],
        min_free_gb=20.0,
        max_jobs=2,
    )

    assert len(assignments) == len(ABLATIONS)
    assert [gpu_index for _, gpu_index, _ in assignments[:4]] == [0, 1, 0, 1]
    assert assignments[0][0] == "hotspot_sdf"
