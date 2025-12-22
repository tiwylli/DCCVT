"""Model and initialization helpers for DCCVT."""

import sys
from typing import Tuple

import torch
from torch import nn

from dccvt.runtime import device

sys.path.append("3rdparty/HotSpot")
from dataset import shape_3d
import models.Net as Net


def resolve_sdf_values(model, sites, *, verbose: bool = False) -> torch.Tensor:
    """Resolve SDF values from a grid, tensor, or callable model."""
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        if verbose:
            print("Using SDFGrid")
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        if verbose:
            print("Using Tensor")
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        if verbose:
            print("Using nn.Module / callable model")
        sdf_values = model(sites).detach()
    return sdf_values.squeeze()


def load_hotspot_model(mesh_path: str, target_size: int, hotspot_weights_path: str) -> Tuple[nn.Module, torch.Tensor]:
    """Load a HotSpot model and return the model and manifold points."""
    loss_type = "igr_w_heat"
    loss_weights = [350, 0, 0, 1, 0, 0, 20]
    train_set = shape_3d.ReconDataset(
        file_path=mesh_path + ".ply",
        n_points=target_size * target_size * 150,  # 15000, #args.n_points,
        n_samples=10001,  # args.n_iterations,
        grid_res=256,  # args.grid_res,
        grid_range=1.1,  # args.grid_range,
        sample_type="uniform_central_gaussian",  # args.nonmnfld_sample_type,
        sampling_std=0.5,  # args.nonmnfld_sample_std,
        n_random_samples=7500,  # args.n_random_samples,
        resample=True,
        compute_sal_dist_gt=(True if "sal" in loss_type and loss_weights[5] > 0 else False),
        scale_method="mean",  # "mean" #args.pcd_scale_method,
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
    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")
    model.load_state_dict(torch.load(hotspot_weights_path, weights_only=True, map_location=map_location))
    return model, mnfld_points


def init_sites_from_mnfld_points(
    mnfld_points: torch.Tensor, num_centroids: int, sample_near: int, input_dims: int
) -> torch.Tensor:
    """Initialize Voronoi sites for optimization."""
    noise_scale = 0.005
    domain_limit = 1
    if input_dims == 2:
        # throw error not yet implemented
        raise NotImplementedError("2D not yet implemented")
    elif input_dims == 3:
        x = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        y = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        z = torch.linspace(-domain_limit, domain_limit, int(round(num_centroids)))
        meshgrid = torch.meshgrid(x, y, z)
        meshgrid = torch.stack(meshgrid, dim=3).view(-1, 3)
        meshgrid += torch.randn_like(meshgrid) * noise_scale

    sites = meshgrid.to(device, dtype=torch.float32).requires_grad_(True)
    # add mnfld points with random noise to sites
    N = mnfld_points.squeeze(0).shape[0]
    if sample_near > 0:
        # num_samples = sample_near**input_dims - num_centroids**input_dims
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

