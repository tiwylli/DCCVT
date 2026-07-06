from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dccvt.neural.abc import (
    ABCHybridDataset,
    ABCUDFConfig,
    build_abc_hybrid_model,
    exact_point_udf_grid,
    initialize_from_ponq_encoder,
    validate_udf_sidecar,
    write_udf_sidecar,
)
from dccvt.neural.models import DCCVTHybridDirectNet, DCCVTPoNQNet, HybridDirectConfig


def _udf_config() -> ABCUDFConfig:
    return ABCUDFConfig(
        master_grid_n=129,
        model_grid_n=33,
        coordinate_min=-0.5,
        coordinate_max=0.5,
        query_chunk_size=32,
        compression="gzip",
        compression_level=1,
        preprocessing_version="abc_udf_129_stride4_v1",
    )


def test_udf_sidecar_stores_exact_aligned_33_grid(tmp_path):
    pytest.importorskip("h5py")
    udf128 = np.arange(129**3, dtype=np.float32).reshape(129, 129, 129)
    path = tmp_path / "shape.hdf5"

    write_udf_sidecar(
        path,
        udf128,
        source_point_count=1_000_000,
        config=_udf_config(),
    )

    valid, reason = validate_udf_sidecar(path, config=_udf_config())
    assert valid, reason
    import h5py

    with h5py.File(path, "r") as handle:
        assert handle["128_udf"].shape == (129, 129, 129)
        assert handle["32_udf"].shape == (33, 33, 33)
        assert np.array_equal(handle["32_udf"][:], udf128[::4, ::4, ::4])


def test_exact_point_udf_matches_brute_force_distances():
    pytest.importorskip("pytorch3d")
    points = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    udf = exact_point_udf_grid(
        points,
        grid_n=3,
        coordinate_min=-0.5,
        coordinate_max=0.5,
        query_chunk_size=5,
    )
    axis = torch.linspace(-0.5, 0.5, 3)
    coordinates = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    expected = torch.cdist(coordinates, points).min(dim=1).values.reshape(3, 3, 3)

    assert torch.allclose(udf, expected)


def test_abc_dataset_scales_sdf_udf_and_points_to_dccvt_domain(tmp_path):
    h5py = pytest.importorskip("h5py")
    source_root = tmp_path / "source"
    udf_root = tmp_path / "udf"
    source_root.mkdir()
    udf_root.mkdir()
    model_id = "00000001"

    with h5py.File(source_root / f"{model_id}.hdf5", "w") as handle:
        handle.create_dataset("32_sdf", data=np.full((33, 33, 33), -0.25, np.float32))
        handle.create_dataset(
            "pointcloud",
            data=np.array(
                [
                    [-0.5, 0.0, 0.5],
                    [0.25, -0.25, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            ),
        )
    with h5py.File(udf_root / f"{model_id}.hdf5", "w") as handle:
        handle.create_dataset("32_udf", data=np.full((33, 33, 33), 0.125, np.float32))

    dataset = ABCHybridDataset(
        [model_id],
        hdf5_root=source_root,
        udf_root=udf_root,
        target_sample_count=4,
        seed=69,
        deterministic_targets=True,
    )
    item = dataset[0]

    assert item["input_grid"].shape == (2, 33, 33, 33)
    assert torch.all(item["input_grid"][0] == -0.5)
    assert torch.all(item["input_grid"][1] == 0.25)
    assert item["target_points"].abs().max() <= 1.0
    assert torch.any(torch.all(item["target_points"] == torch.tensor([-1.0, 0.0, 1.0]), dim=1))


def test_direct_variant_has_canonical_zero_output_heads():
    config = HybridDirectConfig(
        grid_n=5,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    model, metadata = build_abc_hybrid_model(config, variant="direct")
    sdf = torch.linspace(-1.0, 1.0, 5).view(5, 1, 1).expand(5, 5, 5)
    inputs = torch.stack((sdf, sdf.abs()), dim=0).unsqueeze(0)
    outputs = model(inputs, sdf.unsqueeze(0))

    assert metadata["variant"] == "direct"
    assert torch.count_nonzero(outputs["site_delta"]) == 0
    assert torch.count_nonzero(outputs["sdf_residual"]) == 0
    assert torch.allclose(outputs["sites"], outputs["canonical_sites"].unsqueeze(0))


def test_abc_model_maps_33_vertex_grid_to_32_cubed_sites():
    config = HybridDirectConfig(
        grid_n=33,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    model, _ = build_abc_hybrid_model(config, variant="direct")
    inputs = torch.zeros(1, 2, 33, 33, 33)
    outputs = model(inputs)

    assert outputs["sites"].shape == (1, 32768, 3)
    assert outputs["sites_sdf"].shape == (1, 32768)


def test_ponq_encoder_transfer_copies_sdf_channel_and_zeros_udf(tmp_path):
    ponq = DCCVTPoNQNet(
        grid_n=5,
        k=2,
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    with torch.no_grad():
        for index, parameter in enumerate(ponq.encoder.parameters(), start=1):
            parameter.fill_(float(index))
    checkpoint = tmp_path / "ponq.pt"
    torch.save(ponq.state_dict(), checkpoint)

    config = HybridDirectConfig(
        grid_n=5,
        channel_names=("hotspot_sdf", "point_udf"),
        feature_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    hybrid = DCCVTHybridDirectNet(config)
    metadata = initialize_from_ponq_encoder(hybrid, checkpoint)

    assert metadata["copied_tensors"] == len(ponq.encoder.state_dict())
    assert torch.equal(hybrid.encoder[0].weight[:, 0], ponq.encoder[0].weight[:, 0])
    assert torch.count_nonzero(hybrid.encoder[0].weight[:, 1]) == 0
    assert torch.equal(hybrid.encoder[2].weight, ponq.encoder[2].weight)
    assert all(
        torch.count_nonzero(parameter) == 0
        for head in (hybrid.site_delta_head, hybrid.sdf_residual_head)
        for parameter in head.parameters()
    )
