"""Optimization loops and loss definitions for DCCVT."""

from time import time
from typing import Any, Tuple

import kaolin
import torch
import tqdm as tqdm
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
from torch import nn

from dccvt.geometry import (
    compute_cvt_loss_CLIPPED_vertices,
    compute_cvt_loss_true,
    compute_cvt_loss_vectorized_delaunay,
    compute_d3d_simplices,
    get_clipped_mesh_numba,
)
from dccvt.mesh_ops import cvt_extraction, extract_mesh, sample_mesh_points_heitz
from dccvt.model_utils import resolve_sdf_values
from dccvt.runtime import device
from dccvt.sdf_gradients import (
    discrete_tet_volume_eikonal_loss,
    estimate_eps_H,
    sdf_space_grad_pytorch_diego_sites_tets,
    tet_sdf_motion_mean_curvature_loss,
)
from dccvt.upsampling import upsampling_adaptive_vectorized_sites_sites_sdf
class Voroloss_opt(nn.Module):
    def __init__(self):
        super(Voroloss_opt, self).__init__()
        self.knn = 16

    def __call__(self, points, spoints):
        """points, self.points"""
        # WARNING: fecthing for knn
        with torch.no_grad():
            indices = knn_points(points[None, :], spoints[None, :], K=self.knn).idx[0]
        point_to_voronoi_center = points - spoints[indices[:, 0]]
        voronoi_edge = spoints[indices[:, 1:]] - spoints[indices[:, 0, None]]
        voronoi_edge_l = torch.sqrt(((voronoi_edge**2).sum(-1)))
        vector_length = (point_to_voronoi_center[:, None, :] * voronoi_edge).sum(-1) / voronoi_edge_l
        sq_dist = (vector_length - voronoi_edge_l / 2) ** 2
        return sq_dist.min(1)[0]
def train_DCCVT(
    sites: torch.Tensor,
    sites_sdf: Any,
    mnfld_points: torch.Tensor,
    hotspot_model: Any,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the DCCVT optimization loop and return updated sites and SDF values."""
    use_chamfer = args.w_chamfer > 0
    use_cvt = args.w_cvt > 0
    use_voroloss = args.w_voroloss > 0
    use_sdfsmooth = args.w_sdfsmooth > 0
    use_vertex_interp = args.w_vertex_sdf_interpolation > 0

    if use_chamfer:
        optimizer = torch.optim.Adam(
            [
                {"params": [sites], "lr": args.lr_sites},
                {"params": [sites_sdf], "lr": args.lr_sites},
            ],
            betas=(0.8, 0.95),
        )
    else:
        optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])
        # SDF at original sites
        sdf_values = resolve_sdf_values(sites_sdf, sites, verbose=False)
        sites_sdf = sdf_values.requires_grad_()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    #
    upsampled = 0.0
    epoch = 0
    t0 = time()
    cvt_loss = 0
    chamfer_loss_mesh = 0
    voroloss_loss = 0
    sdf_loss = 0
    d3dsimplices = None
    sites_sdf_grads = None
    voroloss = Voroloss_opt().to(device)

    for epoch in tqdm.tqdm(range(args.num_iterations)):
        optimizer.zero_grad()

        if use_cvt or use_chamfer:
            d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

        if use_chamfer:
            if args.marching_tetrahedra:
                d3dsimplices = torch.tensor(d3dsimplices, device=device)
                marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
                    sites.unsqueeze(0), d3dsimplices, sites_sdf.unsqueeze(0), return_tet_idx=False
                )
                vertices_list, faces_list = marching_tetrehedra_mesh
                v_vect = vertices_list[0]
                # f_or_clipped_v = faces_list[0]
                _, f_or_clipped_v, _, _, _ = get_clipped_mesh_numba(
                    sites,
                    None,
                    d3dsimplices.detach().cpu().numpy(),
                    args.clip,
                    sites_sdf,
                    args.build_mesh,
                    False,
                    args.grad_interpol,
                    args.no_mp,
                )
            else:
                if args.extract_optim:
                    v_vect, f_or_clipped_v = cvt_extraction(sites, sites_sdf, d3dsimplices, False)
                    sites_sdf_grads = None
                else:
                    v_vect, f_or_clipped_v, sites_sdf_grads, tet_probs, W = get_clipped_mesh_numba(
                        sites,
                        None,
                        d3dsimplices,
                        args.clip,
                        sites_sdf,
                        args.build_mesh,
                        False,
                        args.grad_interpol,
                        args.no_mp,
                    )

            if args.build_mesh:
                triangle_faces = [[f[0], f[i], f[i + 1]] for f in f_or_clipped_v for i in range(1, len(f) - 1)]
                triangle_faces = torch.tensor(triangle_faces, device=device)
                hs_p = sample_mesh_points_heitz(v_vect, triangle_faces, num_samples=mnfld_points.shape[0])
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), hs_p.unsqueeze(0))
            else:
                chamfer_loss_mesh, _ = chamfer_distance(mnfld_points.detach(), v_vect.unsqueeze(0))

        if use_voroloss:
            voroloss_loss = voroloss(mnfld_points.squeeze(0), sites).mean()

        if use_cvt:
            if use_voroloss:
                cvt_loss = compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
            else:
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay(sites, None, d3dsimplices)
                # cvt_loss = lf.compute_cvt_loss_vectorized_delaunay_volume(sites, None, d3dsimplices)
                if args.true_cvt:
                    cvt_loss = compute_cvt_loss_true(sites, d3dsimplices, f_or_clipped_v)
                else:
                    cvt_loss = compute_cvt_loss_CLIPPED_vertices(sites, d3dsimplices, f_or_clipped_v)

        sites_loss = args.w_cvt / 1 * cvt_loss + args.w_chamfer * chamfer_loss_mesh + args.w_voroloss * voroloss_loss

        if use_sdfsmooth:
            if sites_sdf_grads is None:
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach()
                )
            if epoch % 100 == 0 and epoch <= 500:
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            elif epoch % 100 == 0 and epoch <= 800:
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 2).detach()
                print("Estimated eps_H: ", eps_H)

            # eik_loss = args.w_sdfsmooth / 1000 * lf.tet_sdf_grad_eikonal_loss(sites, tets_sdf_grads, d3dsimplices)
            eik_loss = args.w_sdfsmooth / 10 * discrete_tet_volume_eikonal_loss(sites, sites_sdf_grads, d3dsimplices)
            shl = args.w_sdfsmooth * tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, d3dsimplices, eps_H)
            sdf_loss = eik_loss + shl

        if use_vertex_interp:
            steps_verts = tet_probs[1]
            # all_vor_vertices = compute_vertices_3d_vectorized(sites, d3dsimplices)  # (M,3)
            # vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3dsimplices, sites, sites_sdf)
            # _, _, used_tet = compute_zero_crossing_vertices_3d(sites, None, None, d3dsimplices, sites_sdf)

            # projected_verts_sdf = vertices_sdf[used_tet] - steps_verts.norm(dim=1)

            step_len = (steps_verts**2).sum(dim=1).clamp_min(1e-12).sqrt()  # (M,)
            vertex_sdf_loss = args.w_vertex_sdf_interpolation * (step_len).mean()
            sdf_loss = sdf_loss + vertex_sdf_loss

        loss = sites_loss + sdf_loss
        # print(f"Epoch {epoch}: loss = {loss.item()}")
        loss.backward()
        # print("-----------------")

        optimizer.step()
        # scheduler.step()

        if upsampled < args.upsampling and epoch / (args.num_iterations * 0.80) > upsampled / args.upsampling:
            print("sites length BEFORE UPSAMPLING: ", len(sites))
            if len(sites) * 1.08 > args.target_size**3:
                print(
                    "Skipping upsampling, too many sites, sites length: ",
                    len(sites),
                    "target size: ",
                    args.target_size**3,
                )
                upsampled = args.upsampling
                sites = sites.detach().requires_grad_(True)

                if use_chamfer:
                    sites_sdf = sites_sdf.detach().requires_grad_(True)
                    optimizer = torch.optim.Adam(
                        [
                            {"params": [sites], "lr": args.lr_sites},
                            {"params": [sites_sdf], "lr": args.lr_sites},
                        ]
                    )
                    eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 3).detach()
                    print("Estimated eps_H: ", eps_H)
                else:
                    optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])

                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
                continue

            if d3dsimplices is None:
                d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

            if sites_sdf_grads is None or sites_sdf_grads.shape[0] != sites_sdf.shape[0]:
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )

            if use_chamfer:
                sites, sites_sdf = upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads, ups_method=args.ups_method, score=args.score
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = sites_sdf.detach().requires_grad_(True)

                d3dsimplices = compute_d3d_simplices(sites, args.marching_tetrahedra)

                optimizer = torch.optim.Adam(
                    [
                        {"params": [sites], "lr": args.lr_sites},
                        {"params": [sites_sdf], "lr": args.lr_sites},
                    ]
                )
                eps_H = estimate_eps_H(sites, d3dsimplices, multiplier=1.5 * 5).detach()
                print("Estimated eps_H: ", eps_H)
            else:
                sites_sdf = hotspot_model(sites).squeeze(-1)
                sites_sdf_grads, tets_sdf_grads, W = sdf_space_grad_pytorch_diego_sites_tets(
                    sites, sites_sdf, torch.tensor(d3dsimplices).to(device).detach().clone()
                )
                sites, sites_sdf = upsampling_adaptive_vectorized_sites_sites_sdf(
                    sites, d3dsimplices, sites_sdf, sites_sdf_grads, ups_method=args.ups_method, score=args.score
                )
                sites = sites.detach().requires_grad_(True)
                sites_sdf = hotspot_model(sites)
                sites_sdf = sites_sdf.detach().squeeze(-1).requires_grad_()
                optimizer = torch.optim.Adam([{"params": [sites], "lr": args.lr_sites}])

            if args.ups_extraction:
                with torch.no_grad():
                    extract_mesh(sites, sites_sdf, mnfld_points, 0, args, state=f"{int(upsampled)}ups")

            upsampled += 1.0
            print("sites length AFTER: ", len(sites))

        if args.video:
            extract_mesh(sites, sites_sdf, mnfld_points, 0, args, state=f"{int(epoch)}")

    return sites, sites_sdf
