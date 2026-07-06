"""Delaunay graph features and procedural parent selection."""

from __future__ import annotations

import numpy as np
import torch

GRAPH_EDGE_FEATURE_DIM = 8

def _build_neighbors_from_simplices(simplices: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    tets = torch.as_tensor(simplices, device=device).long()
    if tets.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    edges = torch.cat(
        [
            tets[:, [0, 1]],
            tets[:, [1, 2]],
            tets[:, [2, 3]],
            tets[:, [3, 0]],
            tets[:, [0, 2]],
            tets[:, [1, 3]],
        ],
        dim=0,
    )
    neighbors, _ = torch.sort(edges, dim=1)
    return torch.unique(neighbors, dim=0)


def build_directed_edges_from_simplices(
    simplices: np.ndarray | torch.Tensor,
    *,
    num_sites: int,
    device: torch.device,
) -> torch.Tensor:
    """Return unique bidirectional Delaunay graph edges from tetrahedra."""
    neighbors = _build_neighbors_from_simplices(simplices, device)
    if neighbors.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    if int(neighbors.min().item()) < 0 or int(neighbors.max().item()) >= int(num_sites):
        raise ValueError("Delaunay simplex indices are outside the site range")
    directed = torch.cat([neighbors, neighbors[:, [1, 0]]], dim=0)
    return torch.unique(directed, dim=0)


def _neighbor_counts(neighbors: torch.Tensor, num_sites: int, device: torch.device) -> torch.Tensor:
    ones = torch.ones((neighbors.shape[0],), device=device)
    counts = torch.zeros((num_sites,), device=device)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    return counts


def _min_neighbor_distances(sites: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]
    edge_len = torch.norm(edge_vec, dim=1)
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((sites.shape[0],), float("inf"), device=sites.device)
    return min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")


def _curvature_score(
    neighbors: torch.Tensor,
    grad_est: torch.Tensor,
    num_sites: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)
    counts = _neighbor_counts(neighbors, num_sites, device).clamp(min=1.0)
    dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1) * 0.8 + 0.2
    scores = torch.zeros(num_sites, device=device)
    scores = scores.index_add(0, neighbors[:, 0], dn2)
    scores = scores.index_add(0, neighbors[:, 1], dn2)
    return scores / counts


def _zero_crossing_sites(neighbors: torch.Tensor, sdf_values: torch.Tensor) -> torch.Tensor:
    sdf_i = sdf_values[neighbors[:, 0]]
    sdf_j = sdf_values[neighbors[:, 1]]
    mask = sdf_i * sdf_j <= 0
    return torch.unique(neighbors[mask].reshape(-1))


def _select_unique_to_budget(
    indices: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices.numel() == 0 or budget <= 0:
        return indices.new_empty((0,)), scores.new_empty((0,))
    count = min(int(budget), int(indices.numel()))
    order = torch.topk(scores, k=count, largest=True, sorted=True).indices
    return indices[order], scores[order]


def fourier_site_position_encoding(sites: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    """Encode normalized site coordinates with low-frequency Fourier features."""
    sites = sites.reshape(-1, 3)
    parts = [sites]
    for exponent in range(int(num_frequencies)):
        frequency = float(2**exponent)
        phase = torch.pi * frequency * sites
        parts.extend([torch.sin(phase), torch.cos(phase)])
    return torch.cat(parts, dim=1)


def delaunay_edge_features(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    directed_edges: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return relative geometry and SDF-delta features for directed graph edges."""
    if directed_edges.numel() == 0:
        return sites.new_empty((0, GRAPH_EDGE_FEATURE_DIM))
    src = directed_edges[:, 0]
    dst = directed_edges[:, 1]
    delta = sites[dst] - sites[src]
    distance = delta.norm(dim=1, keepdim=True).clamp_min(float(eps))
    direction = delta / distance
    sdf_delta = sites_sdf.reshape(-1)[dst, None] - sites_sdf.reshape(-1)[src, None]
    features = torch.cat([delta, distance, direction, sdf_delta], dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)


def local_knn_parent_features(
    parent_sites: torch.Tensor,
    target_points: torch.Tensor,
    *,
    k: int,
    radius: float,
) -> torch.Tensor:
    """Return local input-point statistics for each refinement parent."""
    parent_sites = parent_sites.reshape(-1, 3)
    target_points = target_points.reshape(-1, 3).to(device=parent_sites.device, dtype=parent_sites.dtype)
    if parent_sites.numel() == 0:
        return parent_sites.new_empty((0, 7))
    if target_points.numel() == 0:
        return parent_sites.new_zeros((parent_sites.shape[0], 7))

    k = min(int(k), int(target_points.shape[0]))
    radius = float(radius)
    distances = torch.cdist(parent_sites.unsqueeze(0), target_points.unsqueeze(0), p=2).squeeze(0)
    knn_dist, knn_idx = torch.topk(distances, k=k, largest=False, sorted=True)
    knn_points = target_points[knn_idx]
    offsets = knn_points - parent_sites[:, None, :]
    nearest_distance = knn_dist[:, :1] / radius
    mean_offset = offsets.mean(dim=1) / radius
    mean_distance = knn_dist.mean(dim=1, keepdim=True) / radius
    radius_density = (distances <= radius).sum(dim=1, keepdim=True).to(parent_sites.dtype) / max(int(k), 1)

    centered = offsets - offsets.mean(dim=1, keepdim=True)
    if k > 1:
        covariance = centered.transpose(1, 2) @ centered / float(k - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        anisotropy = (eigenvalues[:, -1:] - eigenvalues[:, :1]) / eigenvalues[:, -1:].clamp_min(1e-12)
    else:
        anisotropy = parent_sites.new_zeros((parent_sites.shape[0], 1))

    features = torch.cat(
        [
            nearest_distance.clamp(max=8.0),
            mean_offset.clamp(min=-8.0, max=8.0),
            mean_distance.clamp(max=8.0),
            radius_density.clamp(max=8.0),
            anisotropy.clamp(min=0.0, max=1.0),
        ],
        dim=1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0)


def select_procedural_refinement_parents(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    *,
    max_parents: int,
    simplices: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Select up to a fixed budget of unique zero-crossing Delaunay sites."""
    if max_parents <= 0 or sites.shape[0] < 5:
        empty = torch.empty((0,), dtype=torch.long, device=sites.device)
        return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": np.empty((0, 4))}
    if not ((sites_sdf.min() < 0) and (sites_sdf.max() > 0)):
        empty = torch.empty((0,), dtype=torch.long, device=sites.device)
        return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": np.empty((0, 4))}

    from dccvt.geometry import compute_delaunay_simplices
    from dccvt.sdf_gradients import compute_sdf_gradients_sites_tets

    with torch.no_grad():
        if simplices is None:
            simplices = compute_delaunay_simplices(sites.detach())
        else:
            simplices = np.asarray(simplices)
        if simplices.size == 0:
            empty = torch.empty((0,), dtype=torch.long, device=sites.device)
            return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": simplices}
        neighbors = _build_neighbors_from_simplices(simplices, sites.device)
        zc_sites = _zero_crossing_sites(neighbors, sites_sdf.detach().reshape(-1))
        if zc_sites.numel() == 0:
            empty = torch.empty((0,), dtype=torch.long, device=sites.device)
            return {"parent_indices": empty, "parent_scores": sites.new_empty((0,)), "simplices": simplices}

        tets = torch.as_tensor(simplices, device=sites.device).long()
        grad_est, _, _ = compute_sdf_gradients_sites_tets(sites.detach(), sites_sdf.detach().reshape(-1), tets)
        min_dists = _min_neighbor_distances(sites.detach(), neighbors)
        curv = _curvature_score(neighbors, grad_est, sites.shape[0], sites.device, eps)
        dist_scale = torch.median(min_dists[zc_sites]).clamp(min=eps)
        curv_scale = torch.median(curv[zc_sites]).clamp(min=eps)
        scores = (min_dists[zc_sites] / dist_scale) * (curv[zc_sites] / curv_scale)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        parent_indices, parent_scores = _select_unique_to_budget(zc_sites, scores, int(max_parents))
    return {
        "parent_indices": parent_indices.long(),
        "parent_scores": parent_scores,
        "simplices": simplices,
    }

