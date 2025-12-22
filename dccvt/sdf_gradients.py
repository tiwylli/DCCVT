"""SDF gradient, curvature, and regularization utilities."""

import numpy as np
import torch


def sdf_space_grad_pytorch_diego_sites_tets(sites, sdf, tets):
    """
    Compute the spatial gradient of the SDF at each site (vertex) and each tetrahedron.

    Args:
        sites: (N, 3) tensor of 3D vertex coordinates.
        sdf: (N,) tensor of SDF values at each vertex.
        tets: (M, 4) tensor of tetrahedral indices.

    Returns:
        grad_sdf: (N, 3) estimated SDF gradient at each site (vertex-wise average of surrounding tet gradients)
        grad_sdf_tet: (M, 3) estimated SDF gradient inside each tetrahedron (constant per tet)
    """
    M = tets.shape[0]
    tet_ids = tets
    a, b, c, d = (
        sites[tet_ids[:, 0]],
        sites[tet_ids[:, 1]],
        sites[tet_ids[:, 2]],
        sites[tet_ids[:, 3]],
    )
    sdf_a, sdf_b, sdf_c, sdf_d = (
        sdf[tet_ids[:, 0]],
        sdf[tet_ids[:, 1]],
        sdf[tet_ids[:, 2]],
        sdf[tet_ids[:, 3]],
    )

    center = (a + b + c + d) / 4
    sdf_center = (sdf_a + sdf_b + sdf_c + sdf_d) / 4

    volume = volume_tetrahedron(a, b, c, d)  # (M,)

    X = torch.stack([a, b, c, d], dim=1)  # (M, 4, 3)
    dX = X - center[:, None, :]  # (M, 4, 3)
    dX_T = dX.transpose(1, 2)  # (M, 3, 4)

    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    Ginv = torch.linalg.pinv(G)  # (M, 3, 3)

    W = torch.einsum("mij,mnj->mni", Ginv, dX)  # (M, 4, 3)

    sdf_stack = torch.stack([sdf_a, sdf_b, sdf_c, sdf_d], dim=1)  # (M, 4)
    sdf_diff = sdf_stack - sdf_center[:, None]  # (M, 4)

    grad_sdf_tet = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)

    for i in range(4):
        ids = tet_ids[:, i]  # (M,)
        grad_contrib = grad_sdf_tet * volume[:, None]  # (M, 3)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    grad_sdf /= weights_tot.clamp(min=1e-8).unsqueeze(1)

    return grad_sdf, grad_sdf_tet, W


def volume_tetrahedron(a, b, c, d):
    ad = a - d
    bd = b - d
    cd = c - d
    n = torch.linalg.cross(bd, cd, dim=-1)
    return torch.abs((ad * n).sum(dim=-1)) / 6.0


def sdf_space_grad_pytorch_diego(sites, sdf, tets):
    # sites: (N, 3)
    # sdf: (N,)
    # tets: (M, 4)

    M = tets.shape[0]
    tet_ids = tets
    a, b, c, d = (
        sites[tet_ids[:, 0]],
        sites[tet_ids[:, 1]],
        sites[tet_ids[:, 2]],
        sites[tet_ids[:, 3]],
    )
    sdf_a, sdf_b, sdf_c, sdf_d = (
        sdf[tet_ids[:, 0]],
        sdf[tet_ids[:, 1]],
        sdf[tet_ids[:, 2]],
        sdf[tet_ids[:, 3]],
    )

    center = (a + b + c + d) / 4
    sdf_center = (sdf_a + sdf_b + sdf_c + sdf_d) / 4

    volume = volume_tetrahedron(a, b, c, d)

    # Build dX: (M, 4, 3)
    X = torch.stack([a, b, c, d], dim=1)  # (M, 4, 3)
    dX = X - center[:, None, :]

    dX_T = dX.transpose(1, 2)  # (M, 3, 4)
    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    # Inverse G: (M, 3, 3)
    Ginv = torch.linalg.pinv(G)  # stable pseudo-inverse for singular cases
    # Ginv = torch.linalg.inv(G)  # faster, uses LU

    # Weights: Ginv @ dX^T -> (M, 3, 4)
    # W = torch.einsum('mij,mkj->mki', Ginv, dX)  # (M, 4, 3)
    W = torch.einsum("mij,mnj->mni", Ginv, dX)

    sdf_stack = torch.stack([sdf_a, sdf_b, sdf_c, sdf_d], dim=1)  # (M, 4)
    sdf_diff = sdf_stack - sdf_center[:, None]

    elem = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)

    for i in range(4):
        ids = tet_ids[:, i]
        grad_contrib = elem * volume[:, None]  # (M, 3)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    # Avoid division by zero (e.g., isolated vertices)
    weights_tot_clamped = weights_tot.clamp(min=1e-8).unsqueeze(1)  # (N, 1)
    grad_sdf /= weights_tot_clamped

    return grad_sdf

def smoothed_heaviside(phi, eps_H):
    H = torch.zeros_like(phi)
    mask1 = phi < -eps_H
    mask2 = phi > eps_H
    mask3 = (~mask1) & (~mask2)
    phi_clip = phi[mask3]
    H[mask1] = 0
    H[mask2] = 1
    H[mask3] = 0.5 + phi_clip / (2 * eps_H) + (1 / (2 * np.pi)) * torch.sin(np.pi * phi_clip / eps_H)
    return H


def tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, tets, eps_H) -> torch.Tensor:
    if eps_H is None:
        eps_H = estimate_eps_H(sites, tets)  # adaptive bandwidth
    sdf_H = smoothed_heaviside(sites_sdf, eps_H)  # (M,)
    sdf_H_a = sdf_H[tets[:, 0]]
    sdf_H_b = sdf_H[tets[:, 1]]
    sdf_H_c = sdf_H[tets[:, 2]]
    sdf_H_d = sdf_H[tets[:, 3]]
    sdf_H_stack = torch.stack([sdf_H_a, sdf_H_b, sdf_H_c, sdf_H_d], dim=1)  # (M, 4)

    sdf_H_center = sdf_H_stack.mean(dim=1, keepdim=True)  # (M, 1)
    sdf_H_diff = sdf_H_stack - sdf_H_center  # (M, 4)

    grad_H_tet = torch.einsum("mi,mij->mj", sdf_H_diff, W)  # (M, 3)
    grad_norm = grad_H_tet.norm(dim=1)  # (M, 1)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]
    volume = volume_tetrahedron(a, b, c, d)  # (M,)
    # trim 5% biggest volumes
    volume = torch.where(volume > torch.quantile(volume, 0.95), torch.tensor(0.0, device=sites.device), volume)
    penalties = torch.mean(volume * grad_norm)
    # penalties = torch.mean(grad_norm)

    # return torch.mean(volume * grad_norm)
    return penalties


def discrete_tet_volume_eikonal_loss(sites, sites_sdf_grad, tets: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularization loss.

    Args:
        sites_sdf_grad: Tensor of shape (N, 3) containing ∇φ at each site.
        variant: 'a' for E1a: ½ mean((||∇φ|| - 1)²)
    Returns:
        A scalar tensor containing the eikonal loss.
    """
    grad_a = sites_sdf_grad[tets[:, 0]]  # (M,3)
    grad_b = sites_sdf_grad[tets[:, 1]]  # (M,3)
    grad_c = sites_sdf_grad[tets[:, 2]]  # (M,3)
    grad_d = sites_sdf_grad[tets[:, 3]]  # (M,3)

    grad_a_error = ((grad_a**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_b_error = ((grad_b**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_c_error = ((grad_c**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_d_error = ((grad_d**2).sum(dim=-1) - 1) ** 2  # (M,)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]

    volume = volume_tetrahedron(a, b, c, d)

    loss = 0.5 * torch.mean(volume * (grad_a_error + grad_b_error + grad_c_error + grad_d_error))  # (M,)

    return loss


def estimate_eps_H(sites, tets, multiplier=1.5):
    # Get all unique edges
    comb = torch.combinations(torch.arange(4), r=2)  # (6,2)
    edges = tets[:, comb]  # (M, 6, 2)
    edges = edges.reshape(-1, 2)  # (6M, 2)

    v0 = sites[edges[:, 0]]
    v1 = sites[edges[:, 1]]
    edge_lengths = torch.norm(v0 - v1, dim=1)

    # remove top 5% longest edges
    edge_lengths = torch.where(
        edge_lengths > torch.quantile(edge_lengths, 0.95), torch.tensor(0.0, device=sites.device), edge_lengths
    )

    avg_len = edge_lengths.mean()
    return multiplier * avg_len
