"""Mesh extraction and sampling utilities."""

from typing import Any

import kaolin
import torch
from scipy.spatial import Delaunay
import time

from dccvt.geometry import (
    compute_circumcenters,
    compute_clipped_mesh,
    find_zero_crossing_vertices_3d,
    get_faces,
    interpolate_vertex_sdf_values,
)
from dccvt.io_utils import save_npz_bundle, save_obj_mesh, save_point_cloud_ply
from dccvt.model_utils import resolve_sdf_values
from dccvt.paths import make_dccvt_obj_path, make_voromesh_obj_path
from dccvt.runtime import device


def sample_mesh_points_heitz(vertices: torch.Tensor, faces: torch.LongTensor, num_samples: int) -> torch.Tensor:
    """
    Uniformly (area weighted) sample points on a triangular mesh
    using Heitz s low distortion square→triangle mapping.

    Args:
        vertices:    (V,3) float tensor of vertex positions.
        faces:       (F,3) long tensor of indices into `vertices`.
        num_samples: int, number of points to sample.

    Returns:
        samples: (num_samples, 3) float tensor of sampled points.
    """
    # 1) Gather triangle vertices
    v0 = vertices[faces[:, 0]]  # (F,3)
    v1 = vertices[faces[:, 1]]  # (F,3)
    v2 = vertices[faces[:, 2]]  # (F,3)

    # 2) Compute triangle areas for weighting
    e0 = v1 - v0  # (F,3)
    e1 = v2 - v0  # (F,3)
    cross = torch.cross(e0, e1, dim=1)  # (F,3)
    areas = 0.5 * cross.norm(dim=1)  # (F,)

    # 3) Sample faces proportional to area
    probs = areas / areas.sum()
    idx = torch.multinomial(probs, num_samples, replacement=True)  # (num_samples,)

    # 4) Draw uniform samples u,v in [0,1]^2
    u = torch.rand(num_samples, device=vertices.device)
    v = torch.rand(num_samples, device=vertices.device)

    # 5) Heitz–Talbot mapping to barycentric (b0,b1,b2):
    #    b0 = u/2, b1 = v/2, then shift one cell to compress diagonals
    b0 = 0.5 * u
    b1 = 0.5 * v
    offset = b1 - b0
    mask = offset > 0
    # if offset>0, push b1 out; else pull b0 back
    b1 = torch.where(mask, b1 + offset, b1)
    b0 = torch.where(mask, b0, b0 - offset)
    b2 = 1.0 - b0 - b1

    # 6) Assemble final sample positions
    #    pick the triangle vertices for each sample
    v0s = v0[idx]  # (num_samples,3)
    v1s = v1[idx]
    v2s = v2[idx]
    # reshape barycentric coords for broadcast
    b0 = b0.unsqueeze(1)  # (num_samples,1)
    b1 = b1.unsqueeze(1)
    b2 = b2.unsqueeze(1)
    samples = b0 * v0s + b1 * v1s + b2 * v2s  # (num_samples,3)

    return samples


def extract_cvt_mesh(sites, sites_sdf, d3dsimplices, build_faces: bool = False):
    """
    Extracts a mesh from the given sites and their SDF values.
    """
    d3d = torch.tensor(d3dsimplices, device=device)  # (M,4)
    all_vor_vertices = compute_circumcenters(sites, d3d)  # (M,3)

    all_vertices_sdf = interpolate_vertex_sdf_values(all_vor_vertices, d3d, sites, sites_sdf)

    # print("Voronoi vertices shape:", all_vor_vertices.shape, "SDF values shape:", all_vertices_sdf.shape)

    vertices_to_compute, zero_crossing_sites_pairs, used_tet = find_zero_crossing_vertices_3d(
        sites, None, None, d3dsimplices, sites_sdf
    )
    vertices = compute_circumcenters(sites, vertices_to_compute)

    sdf_verts = interpolate_vertex_sdf_values(vertices, d3d[used_tet], sites, sites_sdf)

    # print("Vertices to compute:", vertices.shape, "SDF values shape:", sdf_verts.shape)

    tet_sites = sites[d3d[used_tet]]  # (M,4,3)
    tet_sdf = sites_sdf[d3d[used_tet]]  # (M,4)
    # signs = torch.sign(tet_sdf)  # (M,4)
    # sign_sum = torch.abs(signs.sum(dim=1))  # (M,)
    # valid_mask = sign_sum < 4  # (M,) True if there's a sign change

    # cross_mask = signs.unsqueeze(2) * signs.unsqueeze(1) < 0  # (M,4,4)
    # site_mask = cross_mask.any(dim=2)  # (M,4) True if site has at least one sign change edge

    # -------------
    # Broadcast vertex to shape (M, 4, 3)
    v = vertices.unsqueeze(1)  # (M, 1, 3)
    phi_v = sdf_verts.unsqueeze(1)  # (M, 1)

    # Compute displacement vectors to sites
    delta = tet_sites - v  # (M, 4, 3)
    phi_i = tet_sdf  # (M, 4)

    # Compute interpolation weights (M, 4, 1)
    denom = (phi_v - phi_i).unsqueeze(-1)  # (M, 4, 1)
    numer = phi_v.unsqueeze(-1)  # (M, 1, 1)

    # Avoid division by zero
    eps = 1e-8
    denom = denom.clamp(min=-1e6, max=1e6)  # optional clamp for safety

    t = numer / denom  # (M, 4, 1)

    # Only keep valid projections: site must have opposite sign from vertex
    signs_diff = (phi_v * phi_i) < 0  # (M, 4)
    t[~signs_diff.unsqueeze(-1)] = 0.0  # zero out invalid

    # Interpolated positions per site
    p_i = v + t * delta  # (M, 4, 3)
    valid_mask = signs_diff.unsqueeze(-1)  # (M, 4, 1)

    # Average all valid interpolated positions
    num_valid = valid_mask.sum(dim=1).clamp(min=1)  # (M, 1, 1)
    projected = (p_i * valid_mask).sum(dim=1) / num_valid  # (M, 3)
    # ------------
    if build_faces:
        faces = get_faces(d3dsimplices, sites, all_vor_vertices, None, sites_sdf)  # (R0, List of simplices)
        # Compact the vertex list
        used = {idx for face in faces for idx in face}
        old2new = {old: new for new, old in enumerate(sorted(used))}
        new_vertices = all_vor_vertices[sorted(used)]
        new_faces = [[old2new[i] for i in face] for face in faces]
        return projected, new_faces

    vert_for_clipped_cvt = all_vor_vertices
    vert_for_clipped_cvt[used_tet] = projected
    return projected, vert_for_clipped_cvt


def extract_mesh(
    sites: torch.Tensor,
    model: Any,
    target_pc: torch.Tensor,
    time: float,
    args: Any,
    state: str = "",
    d3dsimplices: Any = None,
    t=time.time(),
) -> None:
    """Extract mesh artifacts for the current state and persist them to disk."""
    print(f"Extracting mesh at state: {state} with upsampling: {args.upsampling}")
    sdf_values = resolve_sdf_values(model, sites, verbose=True)  # (N,)

    sites_np = sites.detach().cpu().numpy()
    d3dsimplices = Delaunay(sites_np).simplices

    if args.w_chamfer > 0:
        v_vect, f_vect = extract_cvt_mesh(sites, sdf_values, d3dsimplices, True)
        output_obj_file = make_dccvt_obj_path(args, state, "intDCCVT")
        save_npz_bundle(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        save_point_cloud_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())

        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = compute_clipped_mesh(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        output_obj_file = make_dccvt_obj_path(args, state, "projDCCVT")
        save_npz_bundle(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        save_point_cloud_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())

    if args.w_voroloss > 0:
        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = compute_clipped_mesh(
            sites, None, d3dsimplices, args.clip, sdf_values, True, False, args.grad_interpol, args.no_mp
        )
        output_obj_file = make_voromesh_obj_path(args, state)
        save_npz_bundle(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
    if args.w_mc > 0:
        # TODO:
        print("todo: implement MC loss extraction")
    if args.w_mt > 0:
        d3dsimplices = torch.tensor(d3dsimplices, device=device)
        marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
            sites.unsqueeze(0), d3dsimplices, sdf_values.unsqueeze(0), return_tet_idx=False
        )
        vertices_list, faces_list = marching_tetrehedra_mesh
        vertices = vertices_list[0]
        faces = faces_list[0]
        vertices_np = vertices.detach().cpu().numpy()  # Shape [N, 3]
        faces_np = faces.detach().cpu().numpy()  # Shape [M, 3] (triangles)
        output_obj_file = make_dccvt_obj_path(args, state, "MT")
        save_npz_bundle(sites, sdf_values, time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, vertices_np, faces_np)
