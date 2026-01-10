"""Model and initialization helpers for DCCVT."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn

from dccvt.device import device

_HOTSPOT_PATH = Path(__file__).resolve().parents[1] / "3rdparty" / "HotSpot"
_HOTSPOT_SAMPLES_SCALE = 150


def _ensure_hotspot_on_path() -> None:
    if not _HOTSPOT_PATH.exists():
        raise FileNotFoundError(f"HotSpot dependency not found at {_HOTSPOT_PATH}")
    hotspot_path = str(_HOTSPOT_PATH)
    if hotspot_path not in sys.path:
        sys.path.append(hotspot_path)


def _resolve_hotspot_inputs(mesh_path: str, hotspot_weights_path: str) -> Tuple[Path, Path]:
    mesh_ply = Path(mesh_path).with_suffix(".ply")
    weights_path = Path(hotspot_weights_path)
    if not mesh_ply.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_ply}")
    if not weights_path.exists():
        raise FileNotFoundError(f"HotSpot weights not found: {weights_path}")
    return mesh_ply, weights_path


def _resolve_sdf_values_impl(model: Any, sites: torch.Tensor, *, verbose: bool = False) -> torch.Tensor:
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        if verbose:
            print("Using SDFGrid")
        return model.sdf(sites)
    if isinstance(model, torch.Tensor):
        if verbose:
            print("Using Tensor")
        return model.to(device)
    if verbose:
        print("Using nn.Module / callable model")
    return model(sites).detach()


def resolve_sdf_values(model: Any, sites: torch.Tensor, *, verbose: bool = False) -> torch.Tensor:
    """Resolve SDF values from a grid, tensor, or callable model."""
    return _resolve_sdf_values_impl(model, sites, verbose=verbose).squeeze()


def load_hotspot_model(mesh_path: str, max_amount_sites: int, hotspot_weights_path: str) -> Tuple[nn.Module, torch.Tensor]:
    """Load a HotSpot model and return the model and manifold points."""
    _ensure_hotspot_on_path()
    mesh_ply, weights_path = _resolve_hotspot_inputs(mesh_path, hotspot_weights_path)
    try:
        from dataset import shape_3d
        import models.Net as Net
    except ImportError as exc:
        raise ImportError(
            f"HotSpot dependencies not found at {_HOTSPOT_PATH}. "
            "Ensure the 3rdparty/HotSpot subtree is available."
        ) from exc
    train_set = shape_3d.ReconDataset(
        file_path=str(mesh_ply),
        n_points=max_amount_sites * max_amount_sites * _HOTSPOT_SAMPLES_SCALE,
        n_samples=10001,
        grid_res=256,
        grid_range=1.1,
        sample_type="uniform_central_gaussian",
        sampling_std=0.5,
        n_random_samples=7500,
        resample=True,
        compute_sal_dist_gt=False,
        scale_method="mean",
    )
    model = Net.Network(
        latent_size=0,  # args.latent_size,
        in_dim=3,
        decoder_hidden_dim=128,  # args.decoder_hidden_dim,
        nl="sine",  # args.nl,
        encoder_type="none",  # args.encoder_type,
        decoder_n_hidden_layers=5,  # args.decoder_n_hidden_layers,
        neuron_type="quadratic",  # args.neuron_type,
        init_type="mfgi",  # args.init_type,
        sphere_init_params=[1.6, 0.1],  # args.sphere_init_params,
        n_repeat_period=30,  # args.n_repeat_period,
    )
    model.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    test_data = next(iter(test_dataloader))
    mnfld_points = test_data["mnfld_points"].to(device)
    mnfld_points.requires_grad_()
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    return model, mnfld_points


def init_sites_from_mnfld_points(
    mnfld_points: torch.Tensor, num_centroids: int, sample_near: int
) -> torch.Tensor:
    """Initialize Voronoi sites for optimization."""
    noise_scale = 0.005
    domain_limit = 1
    x = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
    y = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
    z = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
    try:
        meshgrid = torch.meshgrid(x, y, z, indexing="ij")
    except TypeError:
        meshgrid = torch.meshgrid(x, y, z)
    meshgrid = torch.stack(meshgrid, dim=3).view(-1, 3)
    meshgrid += torch.randn_like(meshgrid) * noise_scale

    sites = meshgrid.to(device, dtype=torch.float32).requires_grad_(True)
    # add mnfld points with random noise to sites
    N = mnfld_points.squeeze(0).shape[0]
    if sample_near > 0:
        num_samples = sample_near
        idx = torch.randint(0, N, (num_samples,))
        sampled = mnfld_points.squeeze(0)[idx]
        perturbed = sampled + (torch.rand_like(sampled) - 0.5) * noise_scale
        sites = torch.cat((sites, perturbed), dim=0)
    # make sites a leaf tensor
    sites = sites.detach().requires_grad_()
    return sites


def init_sdf_from_model(model: nn.Module, sites: torch.Tensor) -> torch.Tensor:
    """Initialize SDF values at sites from the model."""
    sdf_values = model(sites)
    sdf_values = sdf_values.detach().squeeze(-1).requires_grad_()
    return sdf_values
