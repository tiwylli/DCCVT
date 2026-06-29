"""Losses for the site-only PoNQ-style DCCVT network."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch.nn import functional as F

from dccvt.neural.grid import trilinear_interpolate_sdf


def chamfer_distance_points(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
    """Symmetric squared Chamfer distance for two point sets."""
    if points_a.numel() == 0 or points_b.numel() == 0:
        return points_a.new_tensor(0.0)
    try:
        from pytorch3d.ops import knn_points

        d_ab = knn_points(points_a.unsqueeze(0), points_b.unsqueeze(0), K=1).dists.mean()
        d_ba = knn_points(points_b.unsqueeze(0), points_a.unsqueeze(0), K=1).dists.mean()
        return d_ab + d_ba
    except ImportError:
        return _chunked_nearest_squared_mean(points_a, points_b) + _chunked_nearest_squared_mean(points_b, points_a)


def _chunked_nearest_squared_mean(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    mins = []
    for chunk in query.split(chunk_size, dim=0):
        distances = torch.cdist(chunk.unsqueeze(0), reference.unsqueeze(0), p=2).pow(2).squeeze(0)
        mins.append(distances.min(dim=1).values)
    return torch.cat(mins, dim=0).mean()


def _loss_zero(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["sites"].sum() * 0.0


def _finite_projected_surface_points(points: torch.Tensor, *, domain_bound: float = 2.0) -> torch.Tensor:
    """Keep projected surface points usable for normalized-domain mesh losses."""
    if points.numel() == 0:
        return points.reshape(0, 3)
    finite = torch.isfinite(points).all(dim=1)
    in_domain = points.abs().amax(dim=1) <= float(domain_bound)
    return points[finite & in_domain]


def _training_cell_mask(gt_activity: torch.Tensor, near_surface: torch.Tensor) -> torch.Tensor:
    mask = gt_activity.bool()
    if mask.any():
        return mask
    near = near_surface.bool()
    if near.any():
        return near
    return torch.ones_like(mask, dtype=torch.bool)


def stage1_site_loss(
    outputs: Dict[str, torch.Tensor],
    target_points: torch.Tensor,
    gt_activity_mask: torch.Tensor,
    near_surface_mask: torch.Tensor,
    *,
    chamfer_weight: float = 100.0,
    occupancy_weight: float = 1.0,
    offset_weight: float = 0.1,
    domain_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """PoNQ-style warm-start loss for DCCVT site prediction."""
    sites = outputs["sites"]
    logits = outputs["activity_logits"]
    offset_fraction = outputs["offset_fraction"]

    batch = sites.shape[0]
    chamfer = _loss_zero(outputs)
    for b in range(batch):
        cell_mask = _training_cell_mask(gt_activity_mask[b], near_surface_mask[b])
        pred_sites = sites[b, cell_mask].reshape(-1, 3)
        chamfer = chamfer + chamfer_distance_points(pred_sites, target_points[b])
    chamfer = chamfer / max(batch, 1)

    eligible = (gt_activity_mask.bool() | near_surface_mask.bool())
    if not eligible.any():
        eligible = torch.ones_like(gt_activity_mask, dtype=torch.bool)
    occupancy = F.binary_cross_entropy_with_logits(
        logits[eligible],
        gt_activity_mask.to(dtype=logits.dtype)[eligible],
    )

    offset_reg = (offset_fraction - 0.5).pow(2).mean()
    domain = F.relu(sites.abs() - 1.0).pow(2).mean()
    loss = (
        chamfer_weight * chamfer
        + occupancy_weight * occupancy
        + offset_weight * offset_reg
        + domain_weight * domain
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "chamfer": float(chamfer.detach().cpu()),
        "occupancy": float(occupancy.detach().cpu()),
        "offset": float(offset_reg.detach().cpu()),
        "domain": float(domain.detach().cpu()),
    }
    return loss, stats


def select_cells_for_dccvt(
    outputs: Dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    batch_index: int,
    *,
    max_sites: Optional[int] = None,
) -> torch.Tensor:
    """Select a bounded number of active cells for DCCVT fine-tuning/extraction."""
    mask = candidate_mask[batch_index].bool()
    if not mask.any():
        activity = outputs["activity"][batch_index]
        top_cell = torch.argmax(activity)
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[top_cell] = True

    if max_sites is not None:
        k = outputs["sites"].shape[2]
        max_cells = max(1, int(math.ceil(max_sites / float(k))))
        active = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if active.numel() > max_cells:
            scores = outputs["activity"][batch_index, active].detach()
            keep = active[torch.topk(scores, k=max_cells, largest=True).indices]
            reduced = torch.zeros_like(mask, dtype=torch.bool)
            reduced[keep] = True
            mask = reduced
    return mask


def dccvt_finetune_loss(
    outputs: Dict[str, torch.Tensor],
    sdf_grid: torch.Tensor,
    target_points: torch.Tensor,
    active_cell_mask: torch.Tensor,
    *,
    chamfer_weight: float = 1000.0,
    cvt_weight: float = 100.0,
    max_sites_per_shape: Optional[int] = 4096,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Fine-tune predicted sites through the existing DCCVT differentiable losses."""
    from dccvt.geometry import compute_clipped_mesh, compute_cvt_loss_from_clipped_vertices, compute_delaunay_simplices

    total = _loss_zero(outputs)
    used_shapes = 0
    skipped_shapes = 0
    chamfer_value = 0.0
    cvt_value = 0.0

    for b in range(outputs["sites"].shape[0]):
        mask = select_cells_for_dccvt(outputs, active_cell_mask, b, max_sites=max_sites_per_shape)
        sites = outputs["sites"][b, mask].reshape(-1, 3)
        if sites.shape[0] < 5:
            skipped_shapes += 1
            continue

        sites_sdf = trilinear_interpolate_sdf(sdf_grid[b], sites).reshape(-1)
        if not ((sites_sdf.min() < 0) and (sites_sdf.max() > 0)):
            skipped_shapes += 1
            continue

        try:
            d3d = compute_delaunay_simplices(sites)
            projected_points, clipped_vertices, _, _ = compute_clipped_mesh(sites, d3d, sites_sdf)
            projected_points = _finite_projected_surface_points(projected_points)
            if projected_points.numel() == 0:
                skipped_shapes += 1
                continue
            chamfer = chamfer_distance_points(projected_points, target_points[b])
            cvt = compute_cvt_loss_from_clipped_vertices(sites, d3d, clipped_vertices)
        except Exception:
            skipped_shapes += 1
            continue

        if chamfer_weight != 0.0:
            total = total + float(chamfer_weight) * chamfer
        if cvt_weight != 0.0:
            total = total + float(cvt_weight) * cvt
        used_shapes += 1
        chamfer_value += float(chamfer.detach().cpu())
        cvt_value += float(cvt.detach().cpu())

    if used_shapes > 0:
        total = total / used_shapes
        chamfer_value /= used_shapes
        cvt_value /= used_shapes
    stats = {
        "dccvt_loss": float(total.detach().cpu()),
        "dccvt_chamfer": chamfer_value,
        "dccvt_cvt": cvt_value,
        "dccvt_used_shapes": float(used_shapes),
        "dccvt_skipped_shapes": float(skipped_shapes),
    }
    return total, stats


def hybrid_direct_supervised_loss(
    outputs: Dict[str, torch.Tensor],
    label_sites: torch.Tensor,
    label_sites_sdf: torch.Tensor,
    *,
    site_weight: float = 1.0,
    sdf_weight: float = 1.0,
    sign_weight: float = 0.1,
    residual_weight: float = 0.01,
    sdf_near_weight: float = 4.0,
    sdf_near_tau: float = 0.1,
    sign_temperature: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Supervise full-field hybrid direct predictions against DCCVT labels."""
    pred_sites = outputs["sites"]
    pred_sdf = outputs["sites_sdf"]
    label_sites_sdf = label_sites_sdf.to(dtype=pred_sdf.dtype)
    label_sites = label_sites.to(dtype=pred_sites.dtype)

    site_loss = F.smooth_l1_loss(pred_sites, label_sites, reduction="mean")

    sdf_per_site = F.smooth_l1_loss(pred_sdf, label_sites_sdf, reduction="none")
    near_weight = 1.0 + float(sdf_near_weight) * torch.exp(-label_sites_sdf.abs() / float(sdf_near_tau))
    sdf_loss = (sdf_per_site * near_weight).mean()

    inside_target = (label_sites_sdf < 0).to(dtype=pred_sdf.dtype)
    positive_count = inside_target.sum().clamp(min=1.0)
    negative_count = (1.0 - inside_target).sum().clamp(min=1.0)
    pos_weight = negative_count / positive_count
    sign_logits = -pred_sdf / float(sign_temperature)
    sign_loss = F.binary_cross_entropy_with_logits(sign_logits, inside_target, pos_weight=pos_weight.detach())

    residual_loss = outputs["sdf_residual"].pow(2).mean()
    total = (
        float(site_weight) * site_loss
        + float(sdf_weight) * sdf_loss
        + float(sign_weight) * sign_loss
        + float(residual_weight) * residual_loss
    )
    sign_accuracy = ((pred_sdf < 0) == (label_sites_sdf < 0)).to(torch.float32).mean()
    stats = {
        "loss": float(total.detach().cpu()),
        "site": float(site_loss.detach().cpu()),
        "sdf": float(sdf_loss.detach().cpu()),
        "sign": float(sign_loss.detach().cpu()),
        "residual": float(residual_loss.detach().cpu()),
        "sign_accuracy": float(sign_accuracy.detach().cpu()),
        "negative_fraction": float((label_sites_sdf < 0).to(torch.float32).mean().detach().cpu()),
    }
    return total, stats


def hybrid_direct_mesh_loss(
    outputs: Dict[str, torch.Tensor],
    target_points: torch.Tensor,
    *,
    chamfer_weight: float = 1000.0,
    cvt_weight: float = 100.0,
    sdfsmooth_weight: float = 100.0,
    strict: bool = False,
    delaunay_simplices=None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Fine-tune direct predictions through DCCVT clipped-mesh losses."""
    from dccvt.geometry import compute_clipped_mesh, compute_cvt_loss_from_clipped_vertices, compute_delaunay_simplices
    from dccvt.sdf_gradients import discrete_tet_volume_eikonal_loss, estimate_eps_H, tet_sdf_motion_mean_curvature_loss

    total = outputs["sites"].sum() * 0.0
    used_shapes = 0
    skipped_shapes = 0
    chamfer_value = 0.0
    cvt_value = 0.0
    smooth_value = 0.0

    for b in range(outputs["sites"].shape[0]):
        sites = outputs["sites"][b]
        sites_sdf = outputs["sites_sdf"][b]
        if sites.shape[0] < 5:
            if strict:
                raise RuntimeError(f"Mesh loss requires at least 5 sites for batch item {b}, got {sites.shape[0]}")
            skipped_shapes += 1
            continue
        if not ((sites_sdf.min() < 0) and (sites_sdf.max() > 0)):
            if strict:
                raise RuntimeError(f"Mesh loss requires positive and negative SDF values for batch item {b}")
            skipped_shapes += 1
            continue
        try:
            d3d = (
                compute_delaunay_simplices(sites)
                if delaunay_simplices is None
                else delaunay_simplices
            )
            projected_points, clipped_vertices, sites_sdf_grads, W = compute_clipped_mesh(sites, d3d, sites_sdf)
            projected_points = _finite_projected_surface_points(projected_points)
            if projected_points.numel() == 0:
                if strict:
                    raise RuntimeError(f"Mesh loss extracted no projected surface points for batch item {b}")
                skipped_shapes += 1
                continue
            chamfer = chamfer_distance_points(projected_points, target_points[b])
            cvt = sites.sum() * 0.0
            if cvt_weight != 0.0:
                cvt = compute_cvt_loss_from_clipped_vertices(sites, d3d, clipped_vertices)

            smooth = sites_sdf.sum() * 0.0
            if sdfsmooth_weight != 0.0:
                d3d_tensor = torch.as_tensor(d3d, device=sites.device).detach()
                eps_H = estimate_eps_H(sites, d3d, multiplier=1.5 * 2).detach()
                eikonal = discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3d_tensor)
                curvature = tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, d3d, eps_H)
                smooth = eikonal / 10.0 + curvature
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Mesh loss geometry failed for batch item {b}") from exc
            skipped_shapes += 1
            continue

        if chamfer_weight != 0.0:
            total = total + float(chamfer_weight) * chamfer
        if cvt_weight != 0.0:
            total = total + float(cvt_weight) * cvt
        if sdfsmooth_weight != 0.0:
            total = total + float(sdfsmooth_weight) * smooth
        used_shapes += 1
        chamfer_value += float(chamfer.detach().cpu())
        cvt_value += float(cvt.detach().cpu())
        smooth_value += float(smooth.detach().cpu())

    if used_shapes > 0:
        total = total / used_shapes
        chamfer_value /= used_shapes
        cvt_value /= used_shapes
        smooth_value /= used_shapes
    stats = {
        "mesh_loss": float(total.detach().cpu()),
        "mesh_chamfer": chamfer_value,
        "mesh_cvt": cvt_value,
        "mesh_sdfsmooth": smooth_value,
        "mesh_used_shapes": float(used_shapes),
        "mesh_skipped_shapes": float(skipped_shapes),
    }
    return total, stats
