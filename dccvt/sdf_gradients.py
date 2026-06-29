"""SDF gradient, curvature, and regularization utilities."""

from __future__ import annotations

from typing import Tuple

import torch


def _least_squares_weights(G: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
    """Solve the batched least-squares systems, retaining SVD near singularities."""
    # G = dX.T @ dX is symmetric positive semidefinite. A direct solve is much
    # cheaper than an SVD pseudoinverse for normal Delaunay tetrahedra, but the
    # pseudoinverse is still needed for the small number of nearly-flat tets.
    with torch.no_grad():
        finfo = torch.finfo(G.dtype)
        scale = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1).pow(3)
        normalized_det = torch.linalg.det(G).abs() / scale.clamp_min(finfo.tiny)
        ill_conditioned = normalized_det < 100 * finfo.eps

    W = torch.empty_like(dX)
    well_conditioned = ~ill_conditioned
    W[well_conditioned] = torch.linalg.solve(
        G[well_conditioned], dX[well_conditioned].transpose(1, 2)
    ).transpose(1, 2)
    W[ill_conditioned] = torch.einsum(
        "mij,mnj->mni",
        torch.linalg.pinv(G[ill_conditioned]),
        dX[ill_conditioned],
    )
    return W


def compute_sdf_gradients_sites_tets(
    sites: torch.Tensor, sdf: torch.Tensor, tets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate SDF gradients at tet level and aggregate to per-site gradients.

    The SDF gradient within each tetrahedron is treated as constant, computed from a
    least-squares fit to the SDF values at the tet's vertices. Per-site gradients are
    a volume-weighted average of adjacent tet gradients.

    Args:
        sites: (N, 3) vertex coordinates.
        sdf: (N,) SDF values per site.
        tets: (M, 4) tetra indices into `sites`.

    Returns:
        grad_sdf: (N, 3) per-site gradient estimates.
        grad_sdf_tet: (M, 3) per-tet gradient estimates.
        W: (M, 4, 3) least-squares weights used to fit gradients.
    """
    tet_ids = tets
    X = sites[tet_ids]  # (M, 4, 3)
    sdf_stack = sdf[tet_ids]  # (M, 4)
    dX = X - X.mean(dim=1, keepdim=True)  # (M, 4, 3)
    dX_T = dX.transpose(1, 2)  # (M, 3, 4)

    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    W = _least_squares_weights(G, dX)  # (M, 4, 3)

    sdf_diff = sdf_stack - sdf_stack.mean(dim=1, keepdim=True)  # (M, 4)

    grad_sdf_tet = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)
    volume = volume_tetrahedron(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    grad_contrib = grad_sdf_tet * volume[:, None]

    for i in range(4):
        ids = tet_ids[:, i]  # (M,)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    grad_sdf /= weights_tot.clamp(min=1e-8).unsqueeze(1)

    return grad_sdf, grad_sdf_tet, W


def volume_tetrahedron(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Compute the absolute volume for each tetrahedron defined by (a, b, c, d)."""
    ad = a - d
    bd = b - d
    cd = c - d
    n = torch.linalg.cross(bd, cd, dim=-1)
    return torch.abs((ad * n).sum(dim=-1)) / 6.0


def smoothed_heaviside(phi: torch.Tensor, eps_H: torch.Tensor) -> torch.Tensor:
    """Smooth Heaviside step function used by curvature regularization."""
    H = torch.zeros_like(phi)
    mask1 = phi < -eps_H
    mask2 = phi > eps_H
    mask3 = (~mask1) & (~mask2)
    phi_clip = phi[mask3]
    H[mask1] = 0
    H[mask2] = 1
    H[mask3] = 0.5 + phi_clip / (2 * eps_H) + (1 / (2 * torch.pi)) * torch.sin(torch.pi * phi_clip / eps_H)
    return H


def tet_sdf_motion_mean_curvature_loss(
    sites: torch.Tensor, sites_sdf: torch.Tensor, W: torch.Tensor, tets: torch.Tensor, eps_H: torch.Tensor
) -> torch.Tensor:
    """Approximate motion-mean-curvature loss using tet-level gradients of a smoothed SDF."""
    if eps_H is None:
        eps_H = estimate_eps_H(sites, tets)  # adaptive bandwidth
    sdf_H = smoothed_heaviside(sites_sdf, eps_H)  # (M,)
    sdf_H_stack = sdf_H[tets]  # (M, 4)

    sdf_H_center = sdf_H_stack.mean(dim=1, keepdim=True)  # (M, 1)
    sdf_H_diff = sdf_H_stack - sdf_H_center  # (M, 4)

    grad_H_tet = torch.einsum("mi,mij->mj", sdf_H_diff, W)  # (M, 3)
    grad_norm = grad_H_tet.norm(dim=1)  # (M,)

    tet_sites = sites[tets]
    volume = volume_tetrahedron(tet_sites[:, 0], tet_sites[:, 1], tet_sites[:, 2], tet_sites[:, 3])
    volume = _zero_top_quantile(volume, 0.95)
    return torch.mean(volume * grad_norm)


def discrete_tet_volume_eikonal_loss(
    sites: torch.Tensor, sites_sdf_grad: torch.Tensor, tets: torch.Tensor
) -> torch.Tensor:
    """
    Eikonal regularization loss weighted by tet volumes.

    Args:
        sites_sdf_grad: Tensor of shape (N, 3) containing ∇φ at each site.
    Returns:
        A scalar tensor containing the eikonal loss.
    """
    tet_grads = sites_sdf_grad[tets]  # (M, 4, 3)
    grad_error = ((tet_grads.square().sum(dim=-1) - 1).square()).sum(dim=1)
    tet_sites = sites[tets]
    volume = volume_tetrahedron(tet_sites[:, 0], tet_sites[:, 1], tet_sites[:, 2], tet_sites[:, 3])

    loss = 0.5 * torch.mean(volume * grad_error)

    return loss


def estimate_eps_H(sites: torch.Tensor, tets: torch.Tensor, multiplier: float = 1.5) -> torch.Tensor:
    """Estimate a smoothing bandwidth from average tet edge length."""
    # Get all unique edges
    comb = torch.combinations(torch.arange(4, device=tets.device), r=2)  # (6,2)
    edges = tets[:, comb]  # (M, 6, 2)
    edges = edges.reshape(-1, 2)  # (6M, 2)

    v0 = sites[edges[:, 0]]
    v1 = sites[edges[:, 1]]
    edge_lengths = torch.norm(v0 - v1, dim=1)

    edge_lengths = _zero_top_quantile(edge_lengths, 0.95)
    avg_len = edge_lengths.mean()
    return multiplier * avg_len


def _zero_top_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """Zero out values above the quantile threshold to reduce outlier influence."""
    threshold = torch.quantile(values, q)
    return torch.where(values > threshold, torch.zeros((), device=values.device), values)
