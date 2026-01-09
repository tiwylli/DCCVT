"""Optimization loops and loss definitions for DCCVT."""

from typing import Any, Tuple

import torch
import tqdm as tqdm
from pytorch3d.loss import chamfer_distance

from dccvt.geometry import (
    compute_clipped_mesh,
    compute_cvt_loss_from_clipped_vertices,
    compute_delaunay_simplices,
)
from dccvt.mesh_ops import extract_mesh
from dccvt.device import device
from dccvt.sdf_gradients import (
    compute_sdf_gradients_sites_tets,
    discrete_tet_volume_eikonal_loss,
    estimate_eps_H,
    tet_sdf_motion_mean_curvature_loss,
)
from dccvt.upsampling import upsample_sites_adaptive


def _setup_optimizer(
    sites: torch.Tensor,
    sites_sdf: Any,
    lr_sites: float,
) -> Tuple[torch.optim.Optimizer, torch.Tensor]:
    sites_sdf = sites_sdf.detach().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [sites], "lr": lr_sites},
            {"params": [sites_sdf], "lr": lr_sites},
        ],
        betas=(0.8, 0.95),
    )
    return optimizer, sites_sdf


def _as_tet_tensor(d3dsimplices: Any, *, clone: bool = False) -> torch.Tensor:
    tets = torch.as_tensor(d3dsimplices, device=device).detach()
    return tets.clone() if clone else tets


def _compute_sdfsmooth_loss(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    d3dsimplices: Any,
    sites_sdf_grads: Any,
    W: Any,
    eps_H: Any,
    epoch: int,
    args: Any,
):
    if sites_sdf_grads is None:
        sites_sdf_grads, _, W = compute_sdf_gradients_sites_tets(sites, sites_sdf, _as_tet_tensor(d3dsimplices))
    if epoch % 100 == 0 and epoch <= 500:
        eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
        print("Estimated eps_H: ", eps_H)
    elif epoch % 100 == 0 and epoch <= 800:
        eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 2).detach()
        print("Estimated eps_H: ", eps_H)
    elif eps_H is None:
        eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 2).detach()

    eik_loss = args.w_sdfsmooth / 10 * discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
    shl = args.w_sdfsmooth * tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, d3dsimplices, eps_H)
    sdf_loss = eik_loss + shl
    return sdf_loss, sites_sdf_grads, W, eps_H


def _should_upsample_epoch(epoch: int, upsampled: float, args: Any) -> bool:
    if args.upsampling <= 0:
        return False
    return upsampled < args.upsampling and epoch / (args.num_iterations * 0.80) > upsampled / args.upsampling


def _too_many_sites(sites: torch.Tensor, args: Any) -> bool:
    return len(sites) * 1.08 > args.max_amount_sites**3


def _rebuild_optimizer(
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    *,
    lr_sites: float,
) -> Tuple[torch.optim.Optimizer, torch.Tensor]:
    sites_sdf = sites_sdf.detach().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [sites], "lr": lr_sites},
            {"params": [sites_sdf], "lr": lr_sites},
        ]
    )
    return optimizer, sites_sdf


def _prepare_upsample_inputs(
    *,
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    sites_sdf_grads: Any,
    W: Any,
    d3dsimplices: Any,
) -> Tuple[Any, torch.Tensor, Any, Any]:
    if d3dsimplices is None:
        d3dsimplices = compute_delaunay_simplices(sites)

    if sites_sdf_grads is None or sites_sdf_grads.shape[0] != sites_sdf.shape[0]:
        sites_sdf_grads, _, W = compute_sdf_gradients_sites_tets(
            sites, sites_sdf, _as_tet_tensor(d3dsimplices, clone=True)
        )
    return d3dsimplices, sites_sdf, sites_sdf_grads, W


def _maybe_upsample(
    *,
    epoch: int,
    upsampled: float,
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    sites_sdf_grads: Any,
    W: Any,
    d3dsimplices: Any,
    optimizer: torch.optim.Optimizer,
    args: Any,
    eps_H: Any,
):
    if not _should_upsample_epoch(epoch, upsampled, args):
        return False, upsampled, sites, sites_sdf, optimizer, d3dsimplices, sites_sdf_grads, W, eps_H

    print("sites length BEFORE UPSAMPLING: ", len(sites))
    if _too_many_sites(sites, args):
        print(
            "Skipping upsampling, too many sites, sites length: ",
            len(sites),
            "target size: ",
            args.max_amount_sites**3,
        )
        upsampled = args.upsampling
        sites = sites.detach().requires_grad_(True)
        optimizer, sites_sdf = _rebuild_optimizer(sites, sites_sdf, lr_sites=args.lr_sites)
        if d3dsimplices is None:
            d3dsimplices = compute_delaunay_simplices(sites)
        eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 3).detach()
        print("Estimated eps_H: ", eps_H)
        return True, upsampled, sites, sites_sdf, optimizer, d3dsimplices, sites_sdf_grads, W, eps_H

    d3dsimplices, sites_sdf, sites_sdf_grads, W = _prepare_upsample_inputs(
        sites=sites,
        sites_sdf=sites_sdf,
        sites_sdf_grads=sites_sdf_grads,
        W=W,
        d3dsimplices=d3dsimplices,
    )
    sites, sites_sdf = upsample_sites_adaptive(
        sites,
        d3dsimplices,
        sites_sdf,
        sites_sdf_grads,
    )
    sites = sites.detach().requires_grad_(True)

    d3dsimplices = compute_delaunay_simplices(sites)
    optimizer, sites_sdf = _rebuild_optimizer(sites, sites_sdf, lr_sites=args.lr_sites)
    eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
    print("Estimated eps_H: ", eps_H)
    sites_sdf_grads = None
    W = None

    upsampled += 1.0
    print("sites length AFTER: ", len(sites))

    return False, upsampled, sites, sites_sdf, optimizer, d3dsimplices, sites_sdf_grads, W, eps_H


def run_dccvt_training(
    sites: torch.Tensor,
    sites_sdf: Any,
    mnfld_points: torch.Tensor,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the DCCVT optimization loop and return updated sites and SDF values."""
    manifold_points = mnfld_points

    optimizer, sites_sdf = _setup_optimizer(sites, sites_sdf, args.lr_sites)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    #
    upsampled = 0.0
    cvt_loss = 0
    chamfer_loss_mesh = 0
    sdf_loss = 0
    d3dsimplices = None
    sites_sdf_grads = None
    eps_H = None
    W = None
    f_or_clipped_v = None

    for epoch in tqdm.tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        d3dsimplices = compute_delaunay_simplices(sites)

        v_vect, f_or_clipped_v, sites_sdf_grads, W = compute_clipped_mesh(
            sites,
            d3dsimplices,
            sites_sdf,
        )
        chamfer_loss_mesh, _ = chamfer_distance(manifold_points.detach(), v_vect.unsqueeze(0))

        cvt_loss = compute_cvt_loss_from_clipped_vertices(sites, d3dsimplices, f_or_clipped_v)

        sites_loss = args.w_cvt * cvt_loss + args.w_chamfer * chamfer_loss_mesh

        sdf_loss, sites_sdf_grads, W, eps_H = _compute_sdfsmooth_loss(
            sites, sites_sdf, d3dsimplices, sites_sdf_grads, W, eps_H, epoch, args
        )

        loss = sites_loss + sdf_loss
        # print(f"Epoch {epoch}: loss = {loss.item()}")
        loss.backward()
        # print("-----------------")

        optimizer.step()
        # scheduler.step()

        did_continue, upsampled, sites, sites_sdf, optimizer, d3dsimplices, sites_sdf_grads, W, eps_H = _maybe_upsample(
            epoch=epoch,
            upsampled=upsampled,
            sites=sites,
            sites_sdf=sites_sdf,
            sites_sdf_grads=sites_sdf_grads,
            W=W,
            d3dsimplices=d3dsimplices,
            optimizer=optimizer,
            args=args,
            eps_H=eps_H,
        )

        if args.video:
            extract_mesh(sites, sites_sdf, manifold_points, 0, args, state=f"{int(epoch)}")

        if did_continue:
            continue

    return sites, sites_sdf
