"""Adaptive upsampling utilities for site refinement."""

import math
from typing import Tuple

import torch


def _get_sdf_values(sites: torch.Tensor, model, device: torch.device) -> torch.Tensor:
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        sdf_values = model(sites).detach()
    return sdf_values.squeeze()


def _build_neighbors_from_simplices(simplices, device: torch.device) -> torch.Tensor:
    all_tets = torch.as_tensor(simplices, device=device).long()
    edges = torch.cat(
        [
            all_tets[:, [0, 1]],
            all_tets[:, [1, 2]],
            all_tets[:, [2, 3]],
            all_tets[:, [3, 0]],
            all_tets[:, [0, 2]],
            all_tets[:, [1, 3]],
        ],
        dim=0,
    )
    neighbors, _ = torch.sort(edges, dim=1)
    return torch.unique(neighbors, dim=0)


def _min_neighbor_distances(sites: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]
    edge_len = torch.norm(edge_vec, dim=1)

    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((sites.shape[0],), float("inf"), device=sites.device)
    return min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")


def _neighbor_counts(neighbors: torch.Tensor, num_sites: int, device: torch.device) -> torch.Tensor:
    ones = torch.ones((neighbors.shape[0],), device=device)
    counts = torch.zeros((num_sites,), device=device)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    return counts


def _curvature_score(
    neighbors: torch.Tensor,
    grad_est: torch.Tensor,
    score_mode: str,
    num_sites: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)

    if score_mode == "density":
        return torch.ones(num_sites, device=device)

    curv_score = torch.zeros(num_sites, device=device)
    counts = _neighbor_counts(neighbors, num_sites, device)

    if score_mode != "cosine":
        if score_mode != "conservative":
            dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1)
        else:
            dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1) * 0.8 + 0.2
    else:
        dn2 = (1.0 - (unit_n[neighbors[:, 0]] * unit_n[neighbors[:, 1]]).sum(1)) * 0.8 + 0.2

    curv_score = curv_score.index_add(0, neighbors[:, 0], dn2)
    curv_score = curv_score.index_add(0, neighbors[:, 1], dn2)
    curv_score /= counts
    return curv_score


def _zero_crossing_sites(neighbors: torch.Tensor, sdf_values: torch.Tensor) -> torch.Tensor:
    sdf_i, sdf_j = sdf_values[neighbors[:, 0]], sdf_values[neighbors[:, 1]]
    mask_zc = sdf_i * sdf_j <= 0
    return torch.unique(neighbors[mask_zc].reshape(-1))


def _candidate_scores(
    min_dists: torch.Tensor,
    curv_score: torch.Tensor,
    zc_sites: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (min_dists[zc_sites] / torch.median(min_dists[zc_sites])) * (
        curv_score[zc_sites] / (torch.median(curv_score[zc_sites]) + eps)
    )


def _sample_candidate_sites(
    zc_sites: torch.Tensor,
    score_values: torch.Tensor,
    growth_cap: float,
    num_sites: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    M = int(min(max(1, growth_cap * num_sites), score_values.numel()))

    cumsum_scores = torch.cumsum(score_values, dim=0)
    total_score = cumsum_scores[-1].item()
    if total_score <= eps:
        return torch.empty((0,), dtype=torch.long, device=device)
    cumsum_scores /= total_score

    random_indices = torch.rand(M, device=device)
    sampled_indices = torch.searchsorted(cumsum_scores, random_indices)
    sampled_indices = torch.unique(sampled_indices)
    sampled_indices = sampled_indices[sampled_indices < score_values.numel()]

    print(f"Sampled indices: {sampled_indices.numel()} out of {score_values.numel()} candidates (M={M})")
    return zc_sites[sampled_indices]


def _tetrahedral_dirs(device: torch.device, normalize: bool) -> torch.Tensor:
    tetr_dirs = torch.as_tensor(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype=torch.float32,
        device=device,
    )
    if normalize:
        tetr_dirs = torch.nn.functional.normalize(tetr_dirs, dim=1)
    return tetr_dirs


def _tet_frame_offspring(
    sites: torch.Tensor,
    sdf_values: torch.Tensor,
    grad_est: torch.Tensor,
    cand: torch.Tensor,
    min_dists: torch.Tensor,
    eps: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tetr_dirs = _tetrahedral_dirs(device, normalize=True)

    cent_grad = grad_est[cand]
    unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)
    unit_grad = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)

    frame = _build_tangent_frame(unit_grad)

    anisotropy = torch.tensor([1.0, 1.0, 0.5], device=device)
    frame = frame * anisotropy.view(1, 1, 3)

    local_dirs = tetr_dirs.T.unsqueeze(0)
    offs = torch.matmul(frame, local_dirs).permute(0, 2, 1)
    scale = (min_dists[cand] / 2).unsqueeze(1).unsqueeze(2)
    offs = offs * scale

    centroids = sites[cand].unsqueeze(1)
    new_sites = (centroids + offs).reshape(-1, 3)

    delta = new_sites.reshape(-1, 4, 3) - centroids
    new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)
    return new_sites, new_sdf


def _parent_mask(num_sites: int, cand: torch.Tensor, device: torch.device) -> torch.Tensor:
    if cand.dtype == torch.bool:
        return ~cand
    parent_mask = torch.ones(num_sites, dtype=torch.bool, device=device)
    parent_mask[cand] = False
    return parent_mask


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def _upsample_tet_random(
    sites: torch.Tensor,
    sdf_values: torch.Tensor,
    grad_est: torch.Tensor,
    cand: torch.Tensor,
    min_dists: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tetr_dirs = _tetrahedral_dirs(device, normalize=False)

    centroids = sites[cand]
    scale = (min_dists[cand] / 4).unsqueeze(1)

    K = cand.shape[0]
    q = torch.randn(K, 4, device=device, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    R = _quat_to_rotmat(q)

    rotated_dirs = tetr_dirs.unsqueeze(0) @ R.transpose(-1, -2)
    new_sites = (centroids.unsqueeze(1) + rotated_dirs * scale.unsqueeze(1)).reshape(-1, 3)

    print("Before tet random upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

    cent_grad = grad_est[cand]
    delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)
    new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)

    updated_sites = torch.cat([sites, new_sites], dim=0)
    updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)
    return updated_sites, updated_sites_sdf


def _upsample_random_hemisphere(
    sites: torch.Tensor,
    sdf_values: torch.Tensor,
    grad_est: torch.Tensor,
    cand: torch.Tensor,
    min_dists: torch.Tensor,
    eps: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    centroids = sites[cand]
    cent_grad = grad_est[cand]
    unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)

    axis = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)

    helper = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(axis).clone()
    near_pole = axis[:, 2].abs() > 0.99
    helper[near_pole] = torch.tensor([0.0, 1.0, 0.0], device=device)

    v1 = torch.cross(helper, axis, dim=1)
    v1 = v1 / (v1.norm(dim=1, keepdim=True) + eps)
    v2 = torch.cross(axis, v1, dim=1)
    v2 = v2 / (v2.norm(dim=1, keepdim=True) + eps)

    K = centroids.shape[0]
    u = torch.rand(K, 1, device=device)
    phi = 2.0 * math.pi * torch.rand(K, 1, device=device)

    sin_theta = torch.sqrt((1.0 - u**2).clamp_min(0.0))
    dir_hemi = (torch.cos(phi) * sin_theta) * v1 + (torch.sin(phi) * sin_theta) * v2 + u * axis

    step_size = (min_dists[cand] / 4.0).unsqueeze(1)
    new_sites = centroids + step_size * dir_hemi

    print("Before upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

    delta = new_sites - centroids
    new_sdf = sdf_values[cand] + (cent_grad * delta).sum(dim=1)

    updated_sites = torch.cat([sites, new_sites], dim=0)
    updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)
    return updated_sites, updated_sites_sdf


def _build_tangent_frame(normals):  # normals: (B, 3)
    """Build a local tangent frame given normal vectors."""
    B = normals.shape[0]
    device = normals.device

    # Choose a global "up" vector that is not parallel to most normals
    up = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3)
    alt = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3)

    dot = (normals * up).sum(dim=1, keepdim=True).abs()  # (B,1)
    fallback = dot > 0.9  # (B,1) --> mask to avoid colinearity
    base = torch.where(fallback, alt, up)  # (B,3)

    tangent1 = torch.nn.functional.normalize(torch.cross(normals, base, dim=1), dim=1)
    tangent2 = torch.nn.functional.normalize(torch.cross(normals, tangent1, dim=1), dim=1)

    return torch.stack([tangent1, tangent2, normals], dim=-1)  # (B, 3, 3)


def upsample_sites_adaptive(
    sites: torch.Tensor,  # (N,3)
    simplices=None,  # np.ndarray (M,4) if tri is None
    model=None,  # SDFGrid | nn.Module | Tensor (N,)
    sites_sdf_grads=None,  # (N,3) spatial gradients ∇φ at each site
    spacing_target: float = None,  # desired final spacing  (same units as sites)
    alpha_high: float = 1.5,  # regime switches   (α_high > α_low ≥ 1)
    alpha_low: float = 1.1,
    curv_pct: float = 0.75,  # percentile threshold for curvature pass
    growth_cap: float = 0.10,  # ≤ fraction of current sites allowed per iter
    eps: float = 1e-12,
    ups_method: str = "tet_frame",  # | "tet_random" | "random",
    score: str = "legacy",  # | "density" | "cosine" | "conservative"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    # ------------------------------------------------------------------------------
    # Adaptive upsample: balances uniform coverage and high-curvature refinement
    # ------------------------------------------------------------------------------
    Returns:
        updated_sites      -- (N+4K,3)
        updated_sites_sdf  -- (N+4K,)
    """
    device = sites.device
    num_sites = sites.shape[0]

    sdf_values = _get_sdf_values(sites, model, device)

    neighbors = _build_neighbors_from_simplices(simplices, device)
    min_dists = _min_neighbor_distances(sites, neighbors)

    grad_est = sites_sdf_grads
    score_mode = score
    curv_score = _curvature_score(neighbors, grad_est, score_mode, num_sites, device, eps)

    zc_sites = _zero_crossing_sites(neighbors, sdf_values)

    median_min_dists = torch.median(min_dists)
    if spacing_target is None:
        spacing_target = median_min_dists * 0.8

    score_values = _candidate_scores(min_dists, curv_score, zc_sites, eps)
    cand = _sample_candidate_sites(zc_sites, score_values, growth_cap, num_sites, device, eps)

    if cand.numel() == 0:
        return sites, sdf_values

    if ups_method == "tet_frame":
        new_sites, new_sdf = _tet_frame_offspring(
            sites, sdf_values, grad_est, cand, min_dists, eps, device
        )
        updated_sites = torch.cat([sites, new_sites], dim=0)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)
        return updated_sites, updated_sites_sdf

    if ups_method == "tet_frame_remove_parent":
        new_sites, new_sdf = _tet_frame_offspring(
            sites, sdf_values, grad_est, cand, min_dists, eps, device
        )
        parent_mask = _parent_mask(num_sites, cand, device)
        updated_sites = torch.cat([sites[parent_mask], new_sites], dim=0)
        updated_sites_sdf = torch.cat([sdf_values[parent_mask], new_sdf], dim=0)
        return updated_sites, updated_sites_sdf

    if ups_method == "tet_random":
        return _upsample_tet_random(sites, sdf_values, grad_est, cand, min_dists, device)

    if ups_method == "random":
        return _upsample_random_hemisphere(sites, sdf_values, grad_est, cand, min_dists, eps, device)

    raise ValueError(f"Unknown upsampling method: {ups_method}")

