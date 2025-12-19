"""Adaptive upsampling utilities for site refinement."""

import math
from typing import Tuple

import torch
def upsampling_adaptive_vectorized_sites_sites_sdf(
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
    N = sites.shape[0]

    # SDF at original sites
    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    elif isinstance(model, torch.Tensor):
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        sdf_values = model(sites).detach()
    sdf_values = sdf_values.squeeze()  # (N,)

    # Build edge list (ridge points)

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
    neighbors = torch.unique(neighbors, dim=0)  # (E,2)

    # Local spacing ρᵢ  (shortest incident edge)
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]  # (E,3)
    edge_len = torch.norm(edge_vec, dim=1)  # (E,)

    # Compute minimum distance to neighbors
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((N,), float("inf"), device=device)
    min_dists = min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")  # (N,)

    # # Gradient ∇φ and curvature proxy κᵢ  (1-ring normal variation)
    # # ∇φ estimate (scatter-add of finite-difference contributions)
    sdf_diff = sdf_values[neighbors[:, 1]] - sdf_values[neighbors[:, 0]]
    sdf_diff = sdf_diff.unsqueeze(1)  # (E,1)

    counts = torch.zeros((N, 1), device=device)
    ones = torch.ones_like(sdf_diff)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    # grad_est /= counts.clamp(min=1.0)

    grad_est = sites_sdf_grads

    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)

    if score == "density":
        # Disable curvature score, use uniform density
        curv_score = torch.ones(N, device=device)  # (N,)
    else:
        curv_score = torch.zeros(N, device=device)  # (N,)
        if score != "cosine":
            if score != "conservative":
                # Legacy curvature score: squared distance between normals
                dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1)
            else:
                dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1) * 0.8 + 0.2  # (E,)
        else:
            # Cosine similarity as curvature score from dot product
            # Make it conservartive
            dn2 = (1.0 - (unit_n[neighbors[:, 0]] * unit_n[neighbors[:, 1]]).sum(1)) * 0.8 + 0.2

        curv_score = curv_score.index_add(0, neighbors[:, 0], dn2)
        curv_score = curv_score.index_add(0, neighbors[:, 1], dn2)
        curv_score /= counts.squeeze()  # mean over 1-ring

    # Zero-crossing sites
    sdf_i, sdf_j = sdf_values[neighbors[:, 0]], sdf_values[neighbors[:, 1]]
    mask_zc = sdf_i * sdf_j <= 0
    zc_sites = torch.unique(neighbors[mask_zc].reshape(-1))

    # Decide regime (uniform / hybrid / curvature)
    median_min_dists = torch.median(min_dists)  # global spacing
    if spacing_target is None:
        spacing_target = median_min_dists * 0.8  # heuristic default

    score = (min_dists[zc_sites] / torch.median(min_dists[zc_sites])) * (
        curv_score[zc_sites] / (torch.median(curv_score[zc_sites]) + eps)
    )

    M = int(min(max(1, growth_cap * N), score.numel()))

    # # Construct cumsum of the scores WITHOUT sorting
    cumsum_scores = torch.cumsum(score, dim=0)
    total_score = cumsum_scores[-1].item()  # Last element is the total sum
    cumsum_scores /= total_score  # Normalize to [0, 1]

    # Sample M indices based on the cumulative distribution
    random_indices = torch.rand(M, device=device)  # (M,)
    sampled_indices = torch.searchsorted(cumsum_scores, random_indices)  # (M,

    # Remove duplicates
    sampled_indices = torch.unique(sampled_indices)
    sampled_indices = sampled_indices[sampled_indices < score.numel()]  # Ensure valid indices

    # Show the candidates percentage
    print(f"Sampled indices: {sampled_indices.numel()} out of {score.numel()} candidates (M={M})")

    cand = zc_sites[sampled_indices]  # (K,)

    K = cand.numel()
    if K == 0:
        return sites, sdf_values  # nothing selected

    if ups_method == "tet_frame":
        # -------------------------------------------------- #
        # Insert 4 off-spring per selected site (regular tetrahedron)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
            dtype=torch.float32,
            device=device,
        )  # (4,3)
        tetr_dirs = torch.nn.functional.normalize(tetr_dirs, dim=1)  # Normalize directions

        # Build local frame from ∇ϕ (surface normal)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)
        unit_grad = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # point outward
        frame = build_tangent_frame(unit_grad)  # (K,3,3)

        # Optional anisotropic weights for axes: (t1, t2, normal)
        anisotropy = torch.tensor([1.0, 1.0, 0.5], device=device)  # shrink in normal
        frame = frame * anisotropy.view(1, 1, 3)  # (K,3,3)

        # Rotate the 4 tetrahedral directions into local frame
        local_dirs = tetr_dirs.T.unsqueeze(0)  # (1,3,4)
        offs = torch.matmul(frame, local_dirs).permute(0, 2, 1)  # (K,4,3)
        # Scale by local spacing
        # used to be /4 but because anisotropy 0.5 its now /2
        scale = (min_dists[cand] / 2).unsqueeze(1).unsqueeze(2)  # (K,1,1)
        offs = offs * scale  # (K,4,3)

        # Translate from centroid
        centroids = sites[cand].unsqueeze(1)  # (K,1,3)
        new_sites = (centroids + offs).reshape(-1, 3)  # (4K,3)

        delta = new_sites.reshape(-1, 4, 3) - centroids  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    elif ups_method == "tet_frame_remove_parent":
        # -------------------------------------------------- #
        # Insert 4 off-spring per selected site (regular tetrahedron)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
            dtype=torch.float32,
            device=device,
        )  # (4,3)
        tetr_dirs = torch.nn.functional.normalize(tetr_dirs, dim=1)  # Normalize directions

        # Build local frame from ∇ϕ (surface normal)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)
        unit_grad = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # point outward
        frame = build_tangent_frame(unit_grad)  # (K,3,3)

        # Optional anisotropic weights for axes: (t1, t2, normal)
        anisotropy = torch.tensor([1.0, 1.0, 0.5], device=device)  # shrink in normal
        frame = frame * anisotropy.view(1, 1, 3)  # (K,3,3)

        # Rotate the 4 tetrahedral directions into local frame
        local_dirs = tetr_dirs.T.unsqueeze(0)  # (1,3,4)
        offs = torch.matmul(frame, local_dirs).permute(0, 2, 1)  # (K,4,3)
        # Scale by local spacing
        # used to be /4 but because anisotropy 0.5 its now /2
        scale = (min_dists[cand] / 2).unsqueeze(1).unsqueeze(2)  # (K,1,1)
        offs = offs * scale  # (K,4,3)

        # Translate from centroid
        centroids = sites[cand].unsqueeze(1)  # (K,1,3)
        new_sites = (centroids + offs).reshape(-1, 3)  # (4K,3)

        N = sites.shape[0]
        if cand.dtype == torch.bool:
            parent_mask = ~cand
        else:
            parent_mask = torch.ones(N, dtype=torch.bool, device=sites.device)
            parent_mask[cand] = False  # drop parents

        delta = new_sites.reshape(-1, 4, 3) - centroids  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)
        updated_sites = torch.cat([sites[parent_mask], new_sites], dim=0)
        updated_sites_sdf = torch.cat([sdf_values[parent_mask], new_sdf], dim=0)

    elif ups_method == "tet_random":
        # ---------------------------------------------------------------- #
        # Canonical regular-tetrahedron vertex directions (centered at origin)
        tetr_dirs = torch.as_tensor(
            [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=torch.float32, device=device
        )  # (4,3)

        K = cand.shape[0]
        centroids = sites[cand]  # (K,3)
        scale = (min_dists[cand] / 4).unsqueeze(1)  # (K,1)

        # --- Make a random rotation per centroid via random unit quaternions ---
        def quat_to_rotmat(q):  # q: (...,4) normalized
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

        # Random unit quaternions (K,4)
        q = torch.randn(K, 4, device=device, dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        R = quat_to_rotmat(q)  # (K,3,3)

        # Rotate the canonical tetra directions per centroid: (K,4,3)
        rotated_dirs = tetr_dirs.unsqueeze(0) @ R.transpose(-1, -2)

        # Build offspring: keep proportions via `scale`
        new_sites = (centroids.unsqueeze(1) + rotated_dirs * scale.unsqueeze(1)).reshape(-1, 3)  # (4K,3)

        print("Before tet random upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

        # First-order SDF interpolation φ(new) = φ(old) + ∇φ·δ
        cent_grad = grad_est[cand]  # (K,3)
        delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)  # (K,4,3)
        new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)

        # Concatenate & return
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    elif "random":
        eps = 1e-12
        # Inputs
        centroids = sites[cand]  # (K,3)
        cent_grad = grad_est[cand]  # (K,3)
        unit_grad = cent_grad / (cent_grad.norm(dim=1, keepdim=True) + eps)  # (K,3)

        # Hemisphere axis = -∇φ direction (to match your previous "minus grad" step)
        axis = unit_grad * torch.sign(sdf_values[cand]).unsqueeze(1)  # (K,3)

        # Build an orthonormal basis {v1, v2, axis} per centroid
        helper = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(axis).clone()
        near_pole = axis[:, 2].abs() > 0.99  # if axis ~ ±z, switch helper to avoid degeneracy
        helper[near_pole] = torch.tensor([0.0, 1.0, 0.0], device=device)

        v1 = torch.cross(helper, axis, dim=1)
        v1 = v1 / (v1.norm(dim=1, keepdim=True) + eps)  # (K,3)
        v2 = torch.cross(axis, v1, dim=1)
        v2 = v2 / (v2.norm(dim=1, keepdim=True) + eps)  # (K,3)

        # Uniform random direction on hemisphere around `axis`
        # cos(theta) ~ U[0,1], phi ~ U[0, 2π)
        K = centroids.shape[0]
        u = torch.rand(K, 1, device=device)  # cos(theta)
        phi = 2.0 * math.pi * torch.rand(K, 1, device=device)  # azimuth

        sin_theta = torch.sqrt((1.0 - u**2).clamp_min(0.0))  # (K,1)
        dir_hemi = (torch.cos(phi) * sin_theta) * v1 + (torch.sin(phi) * sin_theta) * v2 + u * axis  # (K,3)
        # dir_hemi is unit-length (up to numerical eps)

        # Step radius: keep your existing spacing-based step size
        step_size = (min_dists[cand] / 4.0).unsqueeze(1)  # (K,1)
        new_sites = centroids + step_size * dir_hemi  # (K,3)

        print("Before upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])

        # First-order SDF interpolation: φ(new) = φ(old) + ∇φ · δ
        delta = new_sites - centroids  # (K,3)
        new_sdf = sdf_values[cand] + (cent_grad * delta).sum(dim=1)  # (K,)

        # Concatenate and return
        updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+K, 3)
        updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+K,)
    else:
        raise ValueError(f"Unknown upsampling method: {ups_method}")
    return updated_sites, updated_sites_sdf


def build_tangent_frame(normals):  # normals: (B, 3)
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

