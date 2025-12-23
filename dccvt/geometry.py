"""Geometry utilities for Voronoi/Delaunay operations and clipping."""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pygdel3d
import torch
from numba import njit, prange
from pytorch3d.transforms import quaternion_to_matrix
from scipy.spatial import Delaunay

from dccvt.device import device
from dccvt.model_utils import resolve_sdf_values_or_fallback
from dccvt.sdf_gradients import compute_sdf_gradients_sites_tets, volume_tetrahedron


def _tetra_edges(tetrahedra: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            tetrahedra[:, [0, 1]],
            tetrahedra[:, [1, 2]],
            tetrahedra[:, [2, 3]],
            tetrahedra[:, [3, 0]],
            tetrahedra[:, [0, 2]],
            tetrahedra[:, [1, 3]],
        ],
        dim=0,
    ).to(device)


def _as_tet_tensor(d3dsimplices: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(d3dsimplices, device=device).detach()


def _barycentric_weights(
    vertices: torch.Tensor,
    v_pos: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    x0 = v_pos[:, 0]
    x1 = v_pos[:, 1]
    x2 = v_pos[:, 2]
    x3 = v_pos[:, 3]

    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0

    c1 = torch.cross(e2, e3, dim=1)
    c2 = torch.cross(e3, e1, dim=1)
    c3 = torch.cross(e1, e2, dim=1)

    adj_D = torch.stack([c1, c2, c3], dim=2)
    det_D = (e1 * c1).sum(dim=1, keepdim=True)

    rhs = vertices - x0
    w123 = torch.bmm(adj_D.transpose(1, 2), rhs.unsqueeze(-1)).squeeze(-1) / (det_D + eps)
    w0 = 1.0 - w123.sum(dim=1, keepdim=True)
    return torch.cat([w0, w123], dim=1)


def _project_vertices_by_method(
    grad_interpol: str,
    *,
    all_vor_vertices: torch.Tensor,
    d3d: torch.Tensor,
    sites: torch.Tensor,
    sites_sdf: torch.Tensor,
    sites_sdf_grad: torch.Tensor,
    vertices_sdf: torch.Tensor,
    vertices: torch.Tensor,
    tet_indices,
    quaternion_slerp: bool,
):
    tet_probs = None
    if grad_interpol == "barycentric":
        vertices_sdf_grad, bary_w = interpolate_vertex_sdf_gradients(
            all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
        )
        sdf_verts = vertices_sdf[tet_indices]
        grads = vertices_sdf_grad[tet_indices]
        proj_vertices = project_vertices_newton(grads, sdf_verts, vertices)
    elif grad_interpol == "robust":
        proj_vertices, tet_probs = project_vertices_to_tet_plane(
            d3d[tet_indices], sites, sites_sdf, sites_sdf_grad, vertices
        )
    elif grad_interpol == "hybrid":
        vertices_sdf_grad, bary_w = interpolate_vertex_sdf_gradients(
            all_vor_vertices, d3d, sites, sites_sdf_grad, quaternion_slerp=quaternion_slerp
        )
        sdf_verts = vertices_sdf[tet_indices]
        grads = vertices_sdf_grad[tet_indices]
        proj_vertices = project_vertices_newton(grads, sdf_verts, vertices)

        tpc_proj_v, tet_probs = project_vertices_to_tet_plane(
            d3d[tet_indices], sites, sites_sdf, sites_sdf_grad, vertices
        )
        neg_row_mask = (bary_w[tet_indices] < 0).any(dim=1)
        proj_vertices[neg_row_mask] = tpc_proj_v[neg_row_mask]
    else:
        raise ValueError(f"Unknown grad_interpol: {grad_interpol}")
    return proj_vertices, tet_probs


def _accumulate_centroids(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_sites: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = torch.zeros(num_sites, values.shape[1], dtype=values.dtype, device=device)
    counts = torch.zeros(num_sites, device=device, dtype=values.dtype)
    centroids.index_add_(0, indices, values)
    counts.index_add_(0, indices, torch.ones(values.shape[0], device=device, dtype=values.dtype))
    centroids /= counts.clamp(min=1).unsqueeze(1)
    return centroids, counts


def compute_delaunay_simplices(sites: torch.Tensor, marching_tetrahedra: bool) -> np.ndarray:
    """Compute Delaunay simplices for sites (scipy or pygdel3d depending on mode)."""
    sites_np = sites.detach().cpu().numpy()
    if marching_tetrahedra:
        return Delaunay(sites_np).simplices
    d3dsimplices, _ = pygdel3d.triangulate(sites_np)
    return np.array(d3dsimplices)


def compute_clipped_mesh(
    sites: torch.Tensor,
    model: Any,
    d3dsimplices: Any,
    clip: bool = True,
    sites_sdf: Optional[torch.Tensor] = None,
    build_mesh: bool = False,
    quaternion_slerp: bool = False,
    grad_interpol: str = "robust",
    no_mp: bool = False,
):
    """
    sites:           (N,3) torch tensor (requires_grad)
    model:           SDF model: sites -> (N,1) tensor of signed distances
    d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
    """
    device = sites.device
    if d3dsimplices is None:
        print("Computing Delaunay simplices...")
        sites_np = sites.detach().cpu().numpy()
        d3dsimplices, _ = pygdel3d.triangulate(sites_np)
        print("Number of Delaunay simplices:", len(d3dsimplices))
        print("Delaunay simplices shape:", d3dsimplices)
        print("Max vertex index in simplices:", d3dsimplices.max())
        print("Min vertex index in simplices:", d3dsimplices.min())
        print("Site index range:", sites_np.shape[0])

    d3d = _as_tet_tensor(d3dsimplices, device)  # (M,4)

    if build_mesh:
        # print("-> tracing mesh")
        all_vor_vertices = compute_circumcenters(sites, d3d)  # (M,3)
        faces = get_faces(d3dsimplices, sites, all_vor_vertices, model, sites_sdf)  # (R0, List of simplices)
        # Compact the vertex list
        used = {idx for face in faces for idx in face}
        old2new = {old: new for new, old in enumerate(sorted(used))}
        new_vertices = all_vor_vertices[sorted(used)]
        new_faces = [[old2new[i] for i in face] for face in faces]
        if not clip:
            # print("-> not clipping")
            return new_vertices, new_faces, None, None, None
        else:
            # print("-> clipping")
            vertices_sdf = interpolate_vertex_sdf_values(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad, tets_sdf_grads, W = compute_sdf_gradients_sites_tets(sites, sites_sdf, d3d)  # (M,3)
            proj_vertices, _ = _project_vertices_by_method(
                grad_interpol,
                all_vor_vertices=all_vor_vertices,
                d3d=d3d,
                sites=sites,
                sites_sdf=sites_sdf,
                sites_sdf_grad=sites_sdf_grad,
                vertices_sdf=vertices_sdf,
                vertices=new_vertices,
                tet_indices=sorted(used),
                quaternion_slerp=quaternion_slerp,
            )
            return proj_vertices, new_faces, sites_sdf_grad, tets_sdf_grads, W
    else:
        # print("-> not tracing mesh")
        all_vor_vertices = compute_circumcenters(sites, d3d)  # (M,3)
        vertices_to_compute, bisectors_to_compute, used_tet = find_zero_crossing_vertices_3d(
            sites, None, None, d3dsimplices, sites_sdf
        )
        vertices = compute_circumcenters(sites, vertices_to_compute)
        bisectors = compute_bisector_midpoints(sites, bisectors_to_compute)
        # points = torch.cat((vertices, bisectors), 0)
        if not clip:
            # print("-> not clipping")
            return vertices, None, None, None, None
        else:
            # print("-> clipping")
            vertices_sdf = interpolate_vertex_sdf_values(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad, tets_sdf_grads, W = compute_sdf_gradients_sites_tets(sites, sites_sdf, d3d)
            proj_vertices, tet_probs = _project_vertices_by_method(
                grad_interpol,
                all_vor_vertices=all_vor_vertices,
                d3d=d3d,
                sites=sites,
                sites_sdf=sites_sdf,
                sites_sdf_grad=sites_sdf_grad,
                vertices_sdf=vertices_sdf,
                vertices=vertices,
                tet_indices=used_tet,
                quaternion_slerp=quaternion_slerp,
            )

            # in paper this will be considered a regularisation
            if not no_mp:
                bisectors_sdf = (sites_sdf[bisectors_to_compute[:, 0]] + sites_sdf[bisectors_to_compute[:, 1]]) / 2
                bisectors_sdf_grad = (
                    sites_sdf_grad[bisectors_to_compute[:, 0]] + sites_sdf_grad[bisectors_to_compute[:, 1]]
                ) / 2

                proj_bisectors = project_vertices_newton(bisectors_sdf_grad, bisectors_sdf, bisectors)  # (M,3)

                proj_points = torch.cat((proj_vertices, proj_bisectors), 0)
            else:
                proj_points = proj_vertices

            vert_for_clipped_cvt = all_vor_vertices
            vert_for_clipped_cvt[used_tet] = proj_vertices
            # proj_points = proj_vertices
            return proj_points, vert_for_clipped_cvt, sites_sdf_grad, tet_probs, W


def find_zero_crossing_vertices_3d(sites, vor=None, tri=None, simplices=None, model=None):
    """
    Computes the indices of the sites composing vertices where neighboring sites have opposite or zero SDF values.

    Args:
        sites (torch.Tensor): (N, D) tensor of site positions.
        model (callable): Function or neural network that computes SDF values.

    Returns:
        zero_crossing_vertices_index (list of triplets): List of sites indices (si, sj, sk) where atleast 2 sites have opposing SDF signs.
    """
    sdf_values = resolve_sdf_values_or_fallback(sites, model)

    if tri is not None:
        all_tetrahedra = torch.as_tensor(np.array(tri.simplices), device=device)
    else:
        all_tetrahedra = torch.as_tensor(np.array(simplices), device=device)

    if vor is not None:
        neighbors = torch.as_tensor(np.array(vor.ridge_points), device=device)
        zero_crossing_pairs = neighbors
    else:
        zero_crossing_pairs = find_zero_crossing_site_pairs(all_tetrahedra, sdf_values)

    # Check if vertices has a pair of zero crossing sites
    sdf_0 = sdf_values[all_tetrahedra[:, 0]]  # First site in each pair
    sdf_1 = sdf_values[all_tetrahedra[:, 1]]  # Second site in each pair
    sdf_2 = sdf_values[all_tetrahedra[:, 2]]  # Third site in each pair
    sdf_3 = sdf_values[all_tetrahedra[:, 3]]  # Fourth site in each pair
    mask_zero_crossing_faces = (
        (sdf_0 * sdf_1 <= 0).squeeze()
        | (sdf_0 * sdf_2 <= 0).squeeze()
        | (sdf_0 * sdf_3 <= 0).squeeze()
        | (sdf_1 * sdf_2 <= 0).squeeze()
        | (sdf_1 * sdf_3 <= 0).squeeze()
        | (sdf_2 * sdf_3 <= 0).squeeze()
    )
    zero_crossing_sites_making_verts = all_tetrahedra[mask_zero_crossing_faces]

    return (
        zero_crossing_sites_making_verts,
        zero_crossing_pairs,
        mask_zero_crossing_faces,
    )


def find_zero_crossing_site_pairs(all_tetrahedra, sdf_values):
    tetra_edges = _tetra_edges(all_tetrahedra, device)
    # Sort each edge to ensure uniqueness (because (a, b) and (b, a) are the same)
    tetra_edges, _ = torch.sort(tetra_edges, dim=1)
    # neighbors = torch.unique(tetra_edges, dim=0)
    neighbors = tetra_edges

    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
    zero_crossing_pairs = neighbors[mask_zero_crossing_sites]

    return zero_crossing_pairs


def compute_bisector_midpoints(sites, bisectors_to_compute):
    """
    Computes the bisector points for given pairs of sites in 3D.

    Args:
        sites (torch.Tensor): (N, 3) tensor of site positions.
        bisectors_to_compute (torch.Tensor): (M, 2) tensor of index pairs.

    Returns:
        torch.Tensor: (M, 3) tensor of computed bisector points.
    """
    # Extract all site pairs at once
    si = sites[bisectors_to_compute[:, 0]]  # Shape: (M, N)
    sj = sites[bisectors_to_compute[:, 1]]  # Shape: (M, N)

    # Compute bisectors in a single vectorized operation
    bisectors = (si + sj) / 2  # Shape: (M, N)

    return bisectors


def interpolate_vertex_sdf_values(
    vertices: torch.Tensor,  # (M, 3)  positions of Voronoi vertices (e.g. circumcenters)
    tets: torch.LongTensor,  # (M, 4)  indices of the 4 sites per tetrahedron
    sites: torch.Tensor,  # (N, 3)  coordinates of the sites
    sdf: torch.Tensor,  # (N,)    scalar field value at each site
) -> torch.Tensor:
    """
    Interpolates the SDF at Voronoi vertices (e.g., circumcenters) using barycentric coordinates,
    without calling torch.linalg.solve.

    Returns
    -------
    phi_v : (M,) tensor of interpolated SDF values at Voronoi vertices
    """

    v_pos = sites[tets]  # (M, 4, 3)
    v_phi = sdf[tets]  # (M, 4)

    W = _barycentric_weights(vertices, v_pos)  # (M, 4)

    # Interpolate SDF
    phi_v = (W * v_phi).sum(dim=1)  # (M,)

    return phi_v


def get_faces(d3dsimplices, sites, vor_vertices, model=None, sites_sdf=None):
    with torch.no_grad():
        d3d = _as_tet_tensor(d3dsimplices, device)  # (M,4)
        # Generate all edges of each simplex
        #    torch.combinations gives the 6 index‐pairs within a 4‐long row
        comb = torch.combinations(torch.arange(d3d.shape[1], device=device), r=2)  # (6,2)
        # print("comb", comb.shape)
        edges = d3d[:, comb]  # (M,6,2)
        edges = edges.reshape(-1, 2)  # (M*6,2)
        edges, _ = torch.sort(edges, dim=1)  # sort each row so (a,b) == (b,a)

        # Unique ridges across all simplices
        # ridges, inverse = torch.unique(edges, dim=0, return_inverse=True) # (R,2)

        ridges = edges  # torch.unique(edges, dim=0, return_inverse=False) # (R,2)

        del comb, edges
        torch.cuda.empty_cache()

        # Evaluate SDF at each site
        sdf = resolve_sdf_values_or_fallback(sites, model, fallback=sites_sdf, flatten=True)  # (N,)

        sdf_i = sdf[ridges[:, 0]]
        sdf_j = sdf[ridges[:, 1]]
        zero_cross = sdf_i * sdf_j <= 0  # (R,)
        # Keep only the zero-crossing ridges
        ridges = ridges[zero_cross]  # (R0,2)
        faces = faces_via_dict(d3dsimplices, ridges.detach().cpu().numpy())  # (R0, List of simplices)

        # Sort faces
        torch.cuda.empty_cache()
        R = len(faces)
        counts = np.array([len(face) for face in faces], dtype=np.int64)
        Kmax = counts.max()
        faces_np = np.full((R, Kmax), -1, dtype=np.int64)

        for i, face in enumerate(faces):
            faces_np[i, : len(face)] = face

        sorted_faces_np = np.full((R, Kmax), -1, dtype=np.int64)

        # print("-> sorting faces")
        batch_sort_numba(vor_vertices.detach().cpu().numpy(), faces_np, counts, sorted_faces_np)
        faces_sorted = [sorted_faces_np[i, : counts[i]].tolist() for i in range(R)]
        return faces_sorted


def faces_via_dict(d3dsimplices, ridges):
    # build dict of (a,b) → list of simplex-indices
    face_dict = defaultdict(list)
    for si, simplex in enumerate(d3dsimplices):
        # all 6 edges of a 4-vertex simplex
        a, b, c, d = simplex
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            key = (u, v) if u < v else (v, u)
            face_dict[key].append(si)

    # face dict creates a dictionnary of all the voronoi vertex that form voronoi faces

    # now for each ridge (a,b) grab its list
    out = []
    for a, b in ridges:
        key = (a, b) if a < b else (b, a)
        lst = face_dict.get(key, [])
        out.append(np.array(lst, dtype=np.int32))

    return np.array(out, dtype=object)


@njit(parallel=True)
def batch_sort_numba(vertices, faces_list, counts, output):
    R, Kmax = faces_list.shape
    for i in prange(R):
        length = counts[i]
        sorted_i = sort_face_loop_numba(vertices, faces_list[i, :length])
        for j in range(length):
            output[i, j] = sorted_i[j]


@njit
def sort_face_loop_numba(vertices, face):
    # face: 1D np.array of ints
    n = face.shape[0]
    # gather points and centroid
    ctr = np.zeros(3, dtype=np.float64)
    for i in range(n):
        ctr += vertices[face[i]]
    ctr /= n

    # make a normal from the first 3 points
    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    normal = _normalize(_compute_normal(a, b, c))

    # reference axis
    ref = vertices[face[0]] - ctr
    dot_nr = normal[0] * ref[0] + normal[1] * ref[1] + normal[2] * ref[2]
    ref = ref - normal * dot_nr
    ref = _normalize(ref)

    # compute all angles
    angs = np.empty(n, dtype=np.float64)
    for i in range(n):
        angs[i] = _angle(face[i], vertices, ctr, normal, ref)

    # now do an insertion‐sort by angle, carrying indices
    sorted_idxs = np.empty(n, dtype=face.dtype)
    sorted_angs = np.empty(n, dtype=np.float64)
    length = 0
    for i in range(n):
        a_i = angs[i]
        idx_i = face[i]
        # find insert position
        j = length
        while j > 0 and sorted_angs[j - 1] > a_i:
            sorted_angs[j] = sorted_angs[j - 1]
            sorted_idxs[j] = sorted_idxs[j - 1]
            j -= 1
        sorted_angs[j] = a_i
        sorted_idxs[j] = idx_i
        length += 1

    return sorted_idxs


@njit
def _compute_normal(a, b, c):
    # cross( b−a, c−a )
    ab = b - a
    ac = c - a
    # cross product
    return np.array(
        (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ),
        dtype=np.float64,
    )


@njit
def _normalize(v):
    norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / (norm + 1e-12)


@njit
def _angle(idx, vertices, ctr, normal, ref):
    p = vertices[idx]
    v = p - ctr
    # project into plane
    dot_nv = normal[0] * v[0] + normal[1] * v[1] + normal[2] * v[2]
    v = v - normal * dot_nv
    # compute angle = atan2(||ref×v||, ref·v)
    cr = np.empty(3, dtype=np.float64)
    cr[0] = ref[1] * v[2] - ref[2] * v[1]
    cr[1] = ref[2] * v[0] - ref[0] * v[2]
    cr[2] = ref[0] * v[1] - ref[1] * v[0]
    num = np.sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2])
    den = ref[0] * v[0] + ref[1] * v[1] + ref[2] * v[2]
    ang = np.arctan2(num, den)
    # sign correction
    sign = (normal[0] * cr[0] + normal[1] * cr[1] + normal[2] * cr[2]) < 0
    return 2 * np.pi - ang if sign else ang


def compute_circumcenters(sites, vertices_to_compute):
    """
    Computes the circumcenters of multiple tetrahedra in a vectorized manner.

    Args:
        sites (torch.Tensor): (N, 3) tensor of site positions.
        vertices_to_compute (torch.Tensor): (M, 4) tensor of indices forming tetrahedra.

    Returns:
        torch.Tensor: (M, 3) tensor of computed Voronoi vertices.
    """
    # Extract tetrahedra site coordinates in a batched manner
    tetrahedra = sites[vertices_to_compute]  # Shape: (M, 4, 3)

    # Compute squared norms of each point
    squared_norms = (tetrahedra**2).sum(dim=2, keepdim=True)  # Shape: (M, 4, 1)

    # Construct the 4x4 matrices in batch
    ones_col = torch.ones_like(squared_norms)  # Column of ones for homogeneous coordinates

    A = torch.cat([tetrahedra, ones_col], dim=2)  # Shape: (M, 4, 4)
    Dx = torch.cat([squared_norms, tetrahedra[:, :, 1:], ones_col], dim=2)
    Dy = torch.cat([tetrahedra[:, :, :1], squared_norms, tetrahedra[:, :, 2:], ones_col], dim=2)
    Dz = torch.cat([tetrahedra[:, :, :2], squared_norms, ones_col], dim=2)

    # Compute determinants in batch
    detA = torch.linalg.det(A)  # Shape: (M,)
    detDx = torch.linalg.det(Dx)
    detDy = torch.linalg.det(Dy)  # todo, removed Negative due to orientation
    detDz = torch.linalg.det(Dz)

    # Compute circumcenters
    circumcenters = 0.5 * torch.stack([detDx / detA, detDy / detA, detDz / detA], dim=1)

    return circumcenters  # Shape: (M, 3)


def interpolate_vertex_sdf_gradients(
    vertices: torch.Tensor,  # (M, 3) positions of Voronoi vertices
    tets: torch.LongTensor,  # (M, 4) indices of sites per tetrahedron
    sites: torch.Tensor,  # (N, 3) coordinates of the sites
    site_grads: torch.Tensor,  # (N, 3) spatial gradients ∇φ at each site
    quaternion_slerp: bool = False,  # use quaternion SLERP for interpolation
) -> torch.Tensor:
    """
    Interpolates the SDF gradient at Voronoi vertices using barycentric coordinates,
    without using torch.linalg.solve.

    Returns
    -------
    grad_v : (M, 3) tensor of interpolated SDF gradients at Voronoi vertices
    """

    v_pos = sites[tets]  # (M, 4, 3)
    v_grad = site_grads[tets]  # (M, 4, 3)

    W = _barycentric_weights(vertices, v_pos)  # (M, 4)

    if quaternion_slerp:
        # Use quaternion SLERP for interpolation
        grad_v = quaternion_slerp_barycentric(v_grad, W)
    else:
        # Weighted sum of gradients
        grad_v = (W.unsqueeze(-1) * v_grad).sum(dim=1)  # (M, 3)

    return grad_v, W


def quaternion_slerp_barycentric(
    v_grad: torch.Tensor,  # (M, 4, 3), SDF gradients at the tet corners
    weights: torch.Tensor,  # (M, 4), barycentric weights
) -> torch.Tensor:
    """
    Perform quaternion-based interpolation of gradients using SLERP.
    Args:
        v_grad: (M, 4, 3) per-tet gradients (assumed unit vectors)
        weights: (M, 4) barycentric weights (sum to 1)
    Returns:
        (M, 3) interpolated unit gradients
    """

    # Normalize gradients (quaternions must be unit length vectors)
    v_grad = torch.nn.functional.normalize(v_grad, dim=-1)  # (M, 4, 3)

    # Convert each gradient to quaternion representation using axis-angle [θ * n] → quaternion
    # We'll assume each 3D unit vector lies on the sphere and can be interpreted as a rotation from a canonical vector
    # We'll pick [1,0,0] as canonical; rotation from it to each gradient gives the rotation quaternion

    # Canonical vector
    canonical = torch.tensor([1.0, 0.0, 0.0], device=v_grad.device).expand(v_grad.shape[0], 1, 3)  # (M,1,3)
    q_rots = []

    for i in range(4):
        v_i = v_grad[:, i]  # (M, 3)
        axis = torch.cross(canonical.squeeze(1), v_i, dim=1)  # (M,3)
        axis = torch.nn.functional.normalize(axis, dim=1)
        dot = (canonical.squeeze(1) * v_i).sum(dim=1, keepdim=True).clamp(-1, 1)  # (M,1)
        angle = torch.acos(dot)  # (M,1)

        half_angle = angle / 2
        q = torch.cat(
            [
                torch.cos(half_angle),  # real part
                axis * torch.sin(half_angle),  # imag part
            ],
            dim=1,
        )  # (M,4)
        q_rots.append(q)

    # Slerp pairwise and combine
    q01 = quaternion_slerp(q_rots[0], q_rots[1], weights[:, 1:2] / (weights[:, 0:1] + weights[:, 1:2] + 1e-12))
    q012 = quaternion_slerp(q01, q_rots[2], weights[:, 2:3] / (weights[:, :3].sum(dim=1, keepdim=True) + 1e-12))
    q_final = quaternion_slerp(q012, q_rots[3], weights[:, 3:4] / (weights.sum(dim=1, keepdim=True) + 1e-12))

    # Convert quaternion to rotation matrix and rotate canonical vector
    R = quaternion_to_matrix(q_final)  # (M, 3, 3)
    grad_interp = torch.matmul(R, canonical.transpose(1, 2)).squeeze(-1)  # (M, 3)

    return grad_interp


def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.
    q1, q2: (..., 4) quaternions (w, x, y, z)
    t: (..., 1) interpolation factor in [0, 1]
    Returns:
        (..., 4) interpolated quaternion
    """
    # Normalize to ensure unit quaternions
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)  # (..., 1)

    # Ensure shortest path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta_0 = torch.acos(dot)  # angle between q1 and q2
    sin_theta_0 = torch.sin(theta_0)

    # Avoid division by 0
    small_angle = sin_theta_0 < 1e-6

    s1 = torch.where(small_angle, 1.0 - t, torch.sin((1.0 - t) * theta_0) / (sin_theta_0 + 1e-12))
    s2 = torch.where(small_angle, t, torch.sin(t * theta_0) / (sin_theta_0 + 1e-12))

    return s1 * q1 + s2 * q2  # (..., 4)


def project_vertices_to_tet_plane(
    tets: torch.Tensor,  # (M, 4)
    sites: torch.Tensor,  # (N, 3)
    sdf_values: torch.Tensor,  # (N,)
    sdf_grads: torch.Tensor,  # (N, 3)
    voronoi_vertices: torch.Tensor,  # (M, 3)
) -> torch.Tensor:
    """Project Voronoi vertices onto the fitted tet plane."""
    eps = 1e-8
    # Gather tet-specific data
    tet_sites = sites[tets]  # (M, 4, 3)
    tet_sdf = sdf_values[tets]  # (M, 4)
    tet_grads = sdf_grads[tets]  # (M, 4, 3)
    # print(f"tet_sites shape: {tet_sites.shape}, tet_sdf shape: {tet_sdf.shape}, tet_grads shape: {tet_grads.shape}")

    # Project each site to its local zero level-set via Newton step
    grad_norm2 = torch.sqrt((tet_grads**2).sum(dim=-1, keepdim=True) + eps)  # (M, 4, 1)
    site_step_dir = tet_grads / grad_norm2
    steps = tet_sdf.unsqueeze(-1) * site_step_dir  # (M, 4, 3)
    projected_pts = tet_sites - steps  # (M, 4, 3)

    # Fit plane: subtract mean
    centroid = projected_pts.mean(dim=1, keepdim=True)  # (M, 1, 3)
    centered = projected_pts - centroid  # (M, 4, 3)

    # Compute covariance matrix
    cov = torch.einsum("mni,mnj->mij", centered, centered) / 4  # (M, 3, 3)
    # Compute eigenvectors — last one is normal
    _, eigvecs = torch.linalg.eigh(cov)  # (M, 3), (M, 3, 3)
    normal = eigvecs[:, :, 0]  # Smallest eigenvalue → normal direction
    normal = normal / (normal.norm(dim=1, keepdim=True) + eps)  # Normalize

    # Normalize the normal vector
    normal_norm2 = (normal**2).sum(dim=1, keepdim=True) + eps
    vert_step_dir = normal / torch.sqrt(normal_norm2)  # (M, 3)

    # Project voronoi vertices to plane
    v_to_c = voronoi_vertices - centroid.squeeze(1)  # (M, 3)
    normal_dot = (v_to_c * vert_step_dir).sum(dim=1, keepdim=True)  # (M, 1)

    steps_verts = normal_dot * vert_step_dir  # (M, 3)
    projected_verts = voronoi_vertices - steps_verts  # (M, 3)

    return projected_verts, (site_step_dir, steps_verts, tet_sites)


def project_vertices_newton(grads, sdf_verts, new_vertices):
    """
    Perform a single Newton step to clip vertices based on their SDF values and gradients.
    This function is used to refine the positions of Voronoi vertices after computing their SDFs.
    """
    # one Newton step https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    epsilon = 1e-12

    # grad_norm2 = torch.sqrt(((grads + epsilon)**2).sum(dim=1, keepdim=True))    # (M,1)
    grad_norm2 = torch.sqrt((grads**2).sum(dim=1, keepdim=True) + epsilon)  # (M,1)

    step = sdf_verts.unsqueeze(1) * grads / (grad_norm2)  # (M,3)
    proj_vertices = new_vertices - step

    return proj_vertices


def compute_cvt_loss_delaunay(sites, delaunay, simplices=None):
    """Compute CVT loss from Delaunay simplices."""
    centroids, _ = compute_voronoi_cell_centers(sites, delaunay, simplices)
    centroids = centroids.to(device)
    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(diff.abs() < 0.1, diff, torch.zeros_like(diff))
    # cvt_loss = torch.mean(penalties**2)
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


def compute_voronoi_cell_centers(sites, delau, simplices=None):
    """Compute Voronoi cell centers (circumcenters) for 2D or 3D Delaunay triangulation in PyTorch."""
    # simplices = torch.tensor(delaunay.simplices, dtype=torch.long)
    if simplices is None:
        simplices = delau.simplices

    # points = torch.tensor(delaunay.points, dtype=torch.float32)
    points = sites.detach().cpu().numpy()

    # Compute all circumcenters at once (supports both 2D & 3D)
    circumcenters_arr = circumcenter_torch(points, simplices)
    # Flatten simplices and repeat circumcenters to map them to the points
    indices = simplices.flatten()  # Flatten simplex indices
    indices = torch.tensor(indices, dtype=torch.int64, device=sites.device)  # Convert to tensor

    centers = circumcenters_arr.repeat_interleave(simplices.shape[1], dim=0).to(
        sites.device
    )  # Repeat for each vertex in simplex

    # Group circumcenters per point
    M = len(points)
    # Compute the sum of centers for each index
    centroids, _ = _accumulate_centroids(indices, centers, M, sites.device)

    distances = torch.norm(centroids[indices] - centers, dim=1)
    num_sites = centroids.shape[0]
    max_dist_per_site = torch.full((num_sites,), float("-inf"), device=sites.device)
    radius = max_dist_per_site.scatter_reduce(0, indices, distances, reduce="amax", include_self=True)

    return centroids, radius


def circumcenter_torch(points, simplices):
    """Compute the circumcenters for 2D triangles or 3D tetrahedra in a vectorized manner using PyTorch."""
    points = torch.tensor(points, dtype=torch.float32)
    simplices = torch.tensor(simplices, dtype=torch.long)

    if points.shape[1] == 2:  # **2D Case (Triangles)**
        p1, p2, p3 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]]

        # Compute determinant (D)
        D = 2 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1]))

        # Compute circumcenter coordinates
        ux = (
            (p1[:, 0] ** 2 + p1[:, 1] ** 2) * (p2[:, 1] - p3[:, 1])
            + (p2[:, 0] ** 2 + p2[:, 1] ** 2) * (p3[:, 1] - p1[:, 1])
            + (p3[:, 0] ** 2 + p3[:, 1] ** 2) * (p1[:, 1] - p2[:, 1])
        ) / D

        uy = (
            (p1[:, 0] ** 2 + p1[:, 1] ** 2) * (p3[:, 0] - p2[:, 0])
            + (p2[:, 0] ** 2 + p2[:, 1] ** 2) * (p1[:, 0] - p3[:, 0])
            + (p3[:, 0] ** 2 + p3[:, 1] ** 2) * (p2[:, 0] - p1[:, 0])
        ) / D

        return torch.stack((ux, uy), dim=1)

    elif points.shape[1] == 3:  # **3D Case (Tetrahedra)**
        """
        Compute the circumcenters of multiple tetrahedra in a 3D Delaunay triangulation.

        Parameters:
        points : tensor of shape (N, 3)
            The 3D coordinates of all input points.
        simplices : tensor of shape (M, 4)
            Indices of tetrahedron vertices in `points`.

        Returns:
        circumcenters : tensor of shape (M, 3)
            The circumcenters of all tetrahedra.
        """
        # Extract tetrahedral vertices using broadcasting
        A = points[simplices[:, 0]]  # Shape: (M, 3)
        B = points[simplices[:, 1]]
        C = points[simplices[:, 2]]
        D = points[simplices[:, 3]]

        # Compute edge vectors relative to A
        BA = B - A  # Shape: (M, 3)
        CA = C - A
        DA = D - A

        # Compute squared edge lengths
        len_BA = torch.sum(BA**2, axis=1, keepdims=True)  # Shape: (M, 1)
        len_CA = torch.sum(CA**2, axis=1, keepdims=True)
        len_DA = torch.sum(DA**2, axis=1, keepdims=True)

        # Compute cross products
        cross_CD = torch.linalg.cross(CA, DA)  # Shape: (M, 3)
        cross_DB = torch.linalg.cross(DA, BA)
        cross_BC = torch.linalg.cross(BA, CA)

        # Compute denominator (scalar for each tetrahedron)
        denominator = 0.5 / torch.sum(BA * cross_CD, axis=1, keepdims=True)  # Shape: (M, 1)

        # Compute circumcenter offsets
        circ_offset = (len_BA * cross_CD + len_CA * cross_DB + len_DA * cross_BC) * denominator  # Shape: (M, 3)

        # Compute circumcenters
        circumcenters = A + circ_offset  # Shape: (M, 3)

        return circumcenters
    else:
        raise ValueError("Only 2D (triangles) and 3D (tetrahedra) are supported.")


def compute_cvt_loss_from_clipped_vertices(sites, d3dsimplices, all_vor_vertices):
    d3dsimplices = torch.as_tensor(d3dsimplices, device=sites.device).detach()
    # compute centroids
    indices = d3dsimplices.flatten()  # Flatten simplex indices
    centers = all_vor_vertices.repeat_interleave(d3dsimplices.shape[1], dim=0).to(sites.device)
    M = len(sites)
    centroids, _ = _accumulate_centroids(indices, centers, M, sites.device)

    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(diff.abs() < 0.5, diff, torch.zeros_like(diff))
    # print number of zero in penalties
    # print("Number of zero in penalties: ", torch.sum(penalties == 0.0).item())
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


def compute_cvt_loss_true(sites, d3d, vertices=None):
    if vertices is None:
        vertices = compute_circumcenters(sites, d3d)

    # Concat sites and vertices to compute the Voronoi diagram
    points = torch.cat((sites, vertices), dim=0)
    # Avoid to get coplanar tet which create issue if the current algorithm
    points += (torch.rand_like(points) - 0.5) * 0.00001  # 0.001 % of the space ish
    d3dsimplices, _ = pygdel3d.triangulate(points.detach().cpu().numpy())
    # d3dsimplices = Delaunay(points.detach().cpu().numpy()).simplices
    d3dsimplices = torch.as_tensor(d3dsimplices, dtype=torch.int64, device=sites.device)

    ############ 2D Case (Triangles) ############
    # Compute the areas of all simplices (in 2D triangles)
    # a = points[d3dsimplices[:, 0]]
    # b = points[d3dsimplices[:, 1]]
    # c = points[d3dsimplices[:, 2]]
    # # areas_simplices = torch.linalg.norm(torch.cross(b - a, c - a), dim=1) / 2.0
    # triangle_areas = torch.linalg.norm(b - a, dim=1) * torch.linalg.norm(c - a, dim=1) / 2.0
    # triangle_center = (a + b + c) / 3.0
    # # print(triangle_areas.shape, triangle_center.shape)
    ############ 3D Case (Tetrahedra) ############
    a = points[d3dsimplices[:, 0]]
    b = points[d3dsimplices[:, 1]]
    c = points[d3dsimplices[:, 2]]
    d = points[d3dsimplices[:, 3]]

    tetrahedra_volume = volume_tetrahedron(a, b, c, d)
    tetrahedra_center = (a + b + c + d) / 4.0  # Shape: (M, 3)

    # Create a centroid for each sites
    centroids = torch.zeros_like(sites)
    volumes = torch.ones(sites.shape[0], dtype=torch.float32, device=sites.device) * 1e-8  # Avoid division by zero
    for i in range(4):
        # Filter simplices that are valid (i.e., not out of bounds)
        # We assume that the first N points are the sites
        mask = d3dsimplices[:, i] < sites.shape[0]
        # Uses index_add for atomic addition
        centroids.index_add_(0, d3dsimplices[mask, i], tetrahedra_center[mask] * tetrahedra_volume[mask].unsqueeze(1))
        volumes.index_add_(0, d3dsimplices[mask, i], tetrahedra_volume[mask])
    centroids /= volumes.unsqueeze(1)

    cvt_loss = torch.mean(torch.norm(sites - centroids, dim=1))
    # cvt_loss = torch.mean(torch.abs(sites - centroids))

    # print("Centroids shape:", centroids.shape)
    # print("Sites shape:", sites.shape)
    # return centroids, vertices
    return cvt_loss
