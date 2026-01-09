"""Mesh extraction and sampling utilities."""

from typing import Any

import torch
from dccvt.geometry import (
    compute_circumcenters,
    compute_clipped_mesh_faces,
    compute_delaunay_simplices,
    find_zero_crossing_vertices_3d,
    get_faces,
    interpolate_vertex_sdf_values,
)
from dccvt.io_utils import save_npz_bundle, save_obj_mesh, save_point_cloud_ply
from dccvt.model_utils import resolve_sdf_values
from dccvt.paths import make_dccvt_obj_path
from dccvt.device import device


def extract_cvt_mesh(sites, sites_sdf, d3dsimplices, build_faces: bool = False):
    """
    Extracts a mesh from the given sites and their SDF values.
    """
    d3d = torch.as_tensor(d3dsimplices, device=device)  # (M,4)
    all_vor_vertices = compute_circumcenters(sites, d3d)  # (M,3)

    vertices_to_compute, _, used_tet = find_zero_crossing_vertices_3d(
        sites, None, None, d3dsimplices, sites_sdf
    )
    vertices = compute_circumcenters(sites, vertices_to_compute)

    sdf_verts = interpolate_vertex_sdf_values(vertices, d3d[used_tet], sites, sites_sdf)

    tet_sites = sites[d3d[used_tet]]  # (M,4,3)
    tet_sdf = sites_sdf[d3d[used_tet]]  # (M,4)

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
    elapsed_time: float,
    args: Any,
    state: str = "",
) -> None:
    """Extract mesh artifacts for the current state and persist them to disk."""
    print(f"Extracting mesh at state: {state} with upsampling: {args.upsampling}")
    sdf_values = resolve_sdf_values(model, sites, verbose=True)  # (N,)
    d3dsimplices = compute_delaunay_simplices(sites)

    clipped_cache = None

    if args.w_chamfer > 0:
        v_vect, f_vect = extract_cvt_mesh(sites, sdf_values, d3dsimplices, True)
        output_obj_file = make_dccvt_obj_path(args, state, "intDCCVT")
        save_npz_bundle(sites, sdf_values, elapsed_time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)

        clipped_cache = compute_clipped_mesh_faces(sites, None, d3dsimplices, sdf_values)
        v_vect, f_vect, sites_sdf_grads, tets_sdf_grads, W = clipped_cache
        output_obj_file = make_dccvt_obj_path(args, state, "projDCCVT")
        save_npz_bundle(sites, sdf_values, elapsed_time, args, output_obj_file.replace(".obj", ".npz"))
        save_obj_mesh(output_obj_file, v_vect.detach().cpu().numpy(), f_vect)
        save_point_cloud_ply(f"{args.save_path}/target.ply", target_pc.squeeze(0).detach().cpu().numpy())
