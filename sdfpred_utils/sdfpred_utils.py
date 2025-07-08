from scipy.spatial import Delaunay, Voronoi
import numpy as np
from sklearn.cluster import KMeans
import torch
import diffvoronoi  # delaunay3d bindings
import math
from numba import njit, prange
from collections import defaultdict

device = torch.device("cuda:0")


# Python code for creating a CVT
# Vassilis Vassiliades - Inria, Nancy - April 2018
def createCVTgrid(
    num_centroids=128,
    dimensionality=2,
    num_samples=100000,
    num_replicates=1,
    max_iterations=100000,
    tolerance=0.00001,
):
    X = np.random.rand(num_samples, dimensionality)
    kmeans = KMeans(
        init="k-means++",
        n_clusters=num_centroids,
        n_init=num_replicates,
        # n_jobs=-1,
        max_iter=max_iterations,
        tol=tolerance,
        verbose=0,
    )

    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    centroids = (np.array(centroids - 0.5)) * 10.0
    # make centroids double
    centroids = centroids.astype(np.double)

    # sites = torch.from_numpy(centroids).to(device).requires_grad_(True)

    sites = torch.from_numpy(centroids).to(device, dtype=torch.double).requires_grad_(True)
    print(sites.shape, sites.dtype)
    return sites


from typing import Tuple, Union


def octahedral_grid_points(
    grid: int,
    domain: Union[
        Tuple[float, float],
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ] = (-1.0, 1.0),
) -> torch.Tensor:
    """Return the *unique* vertices of a space‑filling regular‑octahedral grid.

    The routine conceptually starts from an *nx×ny×nz* array of congruent cubes
    that exactly tiles the axis‑aligned box given by *domain*; every cube is then
    substituted by the regular octahedron whose six vertices sit at the centres
    of its faces.  Because neighbouring cubes share faces, many octahedron
    vertices coincide—these duplicates are removed so each vertex appears once.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cubes (hence octahedra) along the x, y, and z directions.
    domain : tuple
        • ``(lo, hi)`` – same bounds for all three axes, or
        • ``((x₀, x₁), (y₀, y₁), (z₀, z₁))`` – independent per‑axis bounds.
        The cubes exactly fill this volume.
    device : str, default = "cpu"
        Device on which the returned tensor is allocated.

    Returns
    -------
    vertices : (V, 3) ``torch.Tensor``
        Deduplicated coordinates of all octahedral vertices.
    """

    # ------------------------------------------------------------
    # 1. Normalise *domain* to per‑axis (lo, hi) tuples
    # ------------------------------------------------------------
    if isinstance(domain[0], (float, int)):
        (x0, x1), (y0, y1), (z0, z1) = (domain, domain, domain)  # type: ignore
    else:
        (x0, x1), (y0, y1), (z0, z1) = domain  # type: ignore

    # ------------------------------------------------------------
    # 2. Cubic voxel edge length → take the *smallest* axis pitch so that
    #    cubes stay inside the box even when extents differ slightly.
    # ------------------------------------------------------------
    spacing_x = (x1 - x0) / grid
    spacing_y = (y1 - y0) / grid
    spacing_z = (z1 - z0) / grid
    spacing = min(spacing_x, spacing_y, spacing_z)

    # Centres at half‑spacing from each low bound
    cx = torch.arange(grid) * spacing + (x0 + spacing * 0.5)
    cy = torch.arange(grid) * spacing + (y0 + spacing * 0.5)
    cz = torch.arange(grid) * spacing + (z0 + spacing * 0.5)

    ix, iy, iz = torch.meshgrid(cx, cy, cz, indexing="ij")  # (nx,ny,nz)
    centres = torch.stack([ix, iy, iz], dim=-1).reshape(-1, 3)

    # ------------------------------------------------------------
    # 3. Six offsets from centre → face centres (the octahedron vertices)
    # ------------------------------------------------------------
    half = spacing * 0.5
    offsets = torch.tensor(
        [
            [half, 0.0, 0.0],
            [-half, 0.0, 0.0],
            [0.0, half, 0.0],
            [0.0, -half, 0.0],
            [0.0, 0.0, half],
            [0.0, 0.0, -half],
        ]
    )

    verts_per_voxel = centres[:, None, :] + offsets[None, :, :]  # (N_vox,6,3)

    # ------------------------------------------------------------
    # 4. Deduplicate via integer lattice: multiply by 2/spacing so
    #    vertices land on integer coordinates, then call torch.unique.
    # ------------------------------------------------------------
    scale = int(round(2.0 / spacing))  # gcd‑like factor
    all_verts_int = (verts_per_voxel.reshape(-1, 3) * scale).round().to(torch.long)
    unique_int = torch.unique(all_verts_int, dim=0, return_inverse=False)

    vertices = unique_int.float() / scale
    return vertices


# # def get_delaunay_neighbors_list(sites):
# #     # Detach and convert to NumPy for Delaunay triangulation
# #     points_np = sites.detach().cpu().numpy()

# #     # Compute the Delaunay triangulation
# #     tri = Delaunay(points_np)

# #     # Find the neighbors of each point
# #     neighbors = {i: set() for i in range(len(points_np))}
# #     for simplex in tri.simplices:
# #         # Each simplex is a triangle of three points; each point is a neighbor of the other two
# #         for i in range(3):
# #             for j in range(i + 1, 3):
# #                 neighbors[simplex[i]].add(simplex[j])
# #                 neighbors[simplex[j]].add(simplex[i])

# #     # Convert neighbor sets to lists for easier reading
# #     neighbors = {key: list(value) for key, value in neighbors.items()}
# #     return neighbors

# def compute_vertices_index(neighbors):
#     vertices_index_to_compute = []
#     for site, adjacents in neighbors.items():
#         for i in adjacents:
#             for n in adjacents:
#                 if n != site and n != i and n in neighbors[i]:
#                     vertices_index_to_compute.append([i,site,n])

#     # Set to store the canonical (sorted) version of each triplet
#     seen_triplets = set()
#     # Filtered list to store the unique triplets
#     filtered_triplets = []
#     # Process each triplet and keep only one permutation
#     for triplet in vertices_index_to_compute:
#         # Convert the triplet to a canonical form by sorting it
#         canonical_triplet = tuple(sorted(triplet, key=str))
#         # Check if this canonical triplet has been seen before
#         if canonical_triplet not in seen_triplets:
#             # If not seen, add it to the set and keep the triplet
#             seen_triplets.add(canonical_triplet)
#             filtered_triplets.append(triplet)

#     return filtered_triplets


def compute_zero_crossing_vertices(sites, model):
    """
    Computes the indices of the sites composing vertices where neighboring sites have opposite or zero SDF values.

    Args:
        sites (torch.Tensor): (N, D) tensor of site positions.
        model (callable): Function or neural network that computes SDF values.

    Returns:
        zero_crossing_vertices_index (list of triplets): List of sites indices (si, sj, sk) where atleast 2 sites have opposing SDF signs.
    """
    # # Compute Delaunay neighbors
    # neighbors = get_delaunay_neighbors_list(sites)

    # # Compute SDF values for all sites
    # sdf_values = model(sites)  # Assuming model outputs (N, 1) or (N,) tensor

    # # Find pairs of neighbors with opposing SDF values
    # zero_crossing_pairs = set()
    # for i, adjacents in neighbors.items():
    #     for j in adjacents:
    #         if i < j:  # Avoid duplicates
    #             sdf_i, sdf_j = sdf_values[i].item(), sdf_values[j].item()
    #             if sdf_i * sdf_j <= 0:  # Opposing signs or one is zero
    #                 zero_crossing_pairs.add((i, j))

    # # Compute vertices (triplets) and filter only those involving zero-crossing pairs
    # all_vertices = compute_vertices_index(neighbors)
    # zero_crossing_vertices_index = []

    # for triplet in all_vertices:
    #     triplet_pairs = {(triplet[0], triplet[1]), (triplet[1], triplet[2]), (triplet[0], triplet[2])}
    #     if any(pair in zero_crossing_pairs for pair in triplet_pairs):
    #         zero_crossing_vertices_index.append(triplet)
    # Compute Delaunay neighbors
    # Detach and convert to NumPy for Delaunay triangulation
    points_np = sites.detach().cpu().numpy()

    # Compute the Delaunay tessellation
    tri = Delaunay(points_np)
    vor = Voronoi(points_np)

    # Compute SDF values for all sites
    sdf_values = model(sites)  # Assuming model outputs (N, 1) or (N,) tensor

    neighbors = torch.tensor(np.array(vor.ridge_points), device=device)
    all_tetrahedra = torch.tensor(np.array(tri.simplices), device=device)

    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
    zero_crossing_pairs = neighbors[mask_zero_crossing_sites]

    sdf_0 = sdf_values[all_tetrahedra[:, 0]]  # First site in each pair
    sdf_1 = sdf_values[all_tetrahedra[:, 1]]  # Second site in each pair
    sdf_2 = sdf_values[all_tetrahedra[:, 2]]  # Third site in each pair
    mask_zero_crossing_faces = (
        (sdf_0 * sdf_1 <= 0).squeeze() | (sdf_0 * sdf_2 <= 0).squeeze() | (sdf_1 * sdf_2 <= 0).squeeze()
    )

    zero_crossing_vertices_index = all_tetrahedra[mask_zero_crossing_faces]

    return zero_crossing_vertices_index, zero_crossing_pairs


def compute_zero_crossing_vertices_3d(sites, vor=None, tri=None, simplices=None, model=None):
    """
    Computes the indices of the sites composing vertices where neighboring sites have opposite or zero SDF values.

    Args:
        sites (torch.Tensor): (N, D) tensor of site positions.
        model (callable): Function or neural network that computes SDF values.

    Returns:
        zero_crossing_vertices_index (list of triplets): List of sites indices (si, sj, sk) where atleast 2 sites have opposing SDF signs.
    """
    # # # Compute Delaunay neighbors
    # # # Detach and convert to NumPy for Delaunay triangulation
    # points_np = sites.detach().cpu().numpy()

    # # # Compute the Delaunay tessellation
    # tri = Delaunay(points_np)
    # vor = Voronoi(points_np)

    # Compute SDF values for all sites
    # model might be a true sdf grid of class SDFGrid
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    # model might be a [sites, 1] tensor
    elif isinstance(model, torch.Tensor):
        sdf_values = model
    else:
        sdf_values = model(sites).detach()  # Assuming model outputs (N, 1) or (N,) tensor

    if tri is not None:
        all_tetrahedra = torch.tensor(np.array(tri.simplices), device=device)
    else:
        all_tetrahedra = torch.tensor(np.array(simplices), device=device)

    if vor is not None:
        neighbors = torch.tensor(np.array(vor.ridge_points), device=device)
    # could compute neighbors without the voronoi diagram
    else:
        #     #neighbors = torch.tensor(np.vstack(list({tuple(sorted(edge)) for tetra in tri.simplices for edge in zip(tetra, np.roll(tetra, -1))})), device=device)
        #     tetra_edges = torch.cat([
        #     all_tetrahedra[:, [0, 1]],
        #     all_tetrahedra[:, [1, 2]],
        #     all_tetrahedra[:, [2, 3]],
        #     all_tetrahedra[:, [3, 0]],
        #     all_tetrahedra[:, [0, 2]],
        #     all_tetrahedra[:, [1, 3]]
        #                             ], dim=0).to(device)
        #     # Sort each edge to ensure uniqueness (because (a, b) and (b, a) are the same)
        #     tetra_edges, _ = torch.sort(tetra_edges, dim=1)
        #     # Get unique edges
        #     neighbors = torch.unique(tetra_edges, dim=0)

        # # Extract the SDF values for each site in the pair
        # sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
        # sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
        # # Find the indices where SDF values have opposing signs or one is zero
        # mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
        # zero_crossing_pairs = neighbors[mask_zero_crossing_sites]

        zero_crossing_pairs = compute_zero_crossing_sites_pairs(all_tetrahedra, sdf_values)

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


def compute_zero_crossing_sites_pairs(all_tetrahedra, sdf_values):
    tetra_edges = torch.cat(
        [
            all_tetrahedra[:, [0, 1]],
            all_tetrahedra[:, [1, 2]],
            all_tetrahedra[:, [2, 3]],
            all_tetrahedra[:, [3, 0]],
            all_tetrahedra[:, [0, 2]],
            all_tetrahedra[:, [1, 3]],
        ],
        dim=0,
    ).to(device)
    # Sort each edge to ensure uniqueness (because (a, b) and (b, a) are the same)
    tetra_edges, _ = torch.sort(tetra_edges, dim=1)
    # Get unique edges
    neighbors = torch.unique(tetra_edges, dim=0)

    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
    zero_crossing_pairs = neighbors[mask_zero_crossing_sites]

    return zero_crossing_pairs


def compute_vertex(s_i, s_j, s_k):
    # Unpack coordinates for each site
    x_i, y_i = s_i[0], s_i[1]
    x_j, y_j = s_j[0], s_j[1]
    x_k, y_k = s_k[0], s_k[1]

    # Calculate numerator and  for x coordinate
    n_x = x_i**2 * (y_j - y_k) - x_j**2 * (y_i - y_k) + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)

    # Calculate numerator for y coordinate
    n_y = -(
        x_i**2 * (x_j - x_k)
        - x_i * (x_j**2 - x_k**2 + y_j**2 - y_k**2)
        + x_j**2 * x_k
        - x_j * (x_k**2 - y_i**2 + y_k**2)
        - x_k * (y_i**2 - y_j**2)
    )

    # Calculate denominator
    d = 2 * (x_i * (y_j - y_k) - x_j * (y_i - y_k) + x_k * (y_i - y_j))

    # Calculate x and y coordinates
    x = n_x / d
    y = n_y / d

    # Return x, y as a tensor to maintain the computational graph
    return torch.stack([x, y])


def compute_all_vertices(sites, vertices_to_compute):
    # Initialize an empty tensor for storing vertices
    vertices = []

    for triplet in vertices_to_compute:
        si = sites[triplet[0]]
        sj = sites[triplet[1]]
        sk = sites[triplet[2]]

        # Compute vertex for the triplet (si, sj, sk)
        v = compute_vertex(si, sj, sk)

        # Append to the list
        vertices.append(v)

    # Stack the list of vertices into a single tensor for easier gradient tracking
    vertices = torch.stack(vertices)
    return vertices


def compute_vertices_2d_vectorized(sites, vertices_to_compute):
    # Extract coordinates using advanced indexing
    s_i, s_j, s_k = (
        sites[vertices_to_compute[:, 0]],
        sites[vertices_to_compute[:, 1]],
        sites[vertices_to_compute[:, 2]],
    )

    x_i, y_i = s_i[:, 0], s_i[:, 1]
    x_j, y_j = s_j[:, 0], s_j[:, 1]
    x_k, y_k = s_k[:, 0], s_k[:, 1]

    # Compute numerators and denominator
    n_x = x_i**2 * (y_j - y_k) - x_j**2 * (y_i - y_k) + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)

    n_y = -(
        x_i**2 * (x_j - x_k)
        - x_i * (x_j**2 - x_k**2 + y_j**2 - y_k**2)
        + x_j**2 * x_k
        - x_j * (x_k**2 - y_i**2 + y_k**2)
        - x_k * (y_i**2 - y_j**2)
    )

    d = 2 * (x_i * (y_j - y_k) - x_j * (y_i - y_k) + x_k * (y_i - y_j))

    # Compute x and y coordinates
    x, y = n_x / d, n_y / d

    # Stack results into a tensor
    return torch.stack([x, y], dim=1)


def compute_vertices_3d_vectorized(sites, vertices_to_compute):
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


def compute_all_bisectors_vectorized(sites, bisectors_to_compute):
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


def upsampling_inside(sites, model):
    # Compute SDF values for each site
    sdf_values = model(sites)
    sdf_values = sdf_values.detach().cpu().numpy()

    sites_np = sites.detach().cpu().numpy()
    # Compute Voronoi diagram
    vor = Voronoi(sites_np)

    # edges = []
    negative_sites = np.array([]).reshape(0, 2)

    for (point1, point2), (vertex1, vertex2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Check if vertices are valid (not -1, which indicates infinity)
        if vertex1 == -1 or vertex2 == -1:
            continue

        # Check if the sites have different signs
        if np.sign(sdf_values[point1]) != np.sign(sdf_values[point2]):
            # edges.append((vertex1, vertex2))

            # Append the site with a negative SDF value
            # if sdf_values[point1] < 0 and not np.any([np.array_equal(sites_np[point1], site) for site in negative_sites]):
            #     negative_sites = np.concatenate((negative_sites, [sites_np[point1]]), axis=0)
            # if sdf_values[point2] < 0 and not np.any([np.array_equal(sites_np[point2], site) for site in negative_sites]):
            #     negative_sites = np.concatenate((negative_sites, [sites_np[point2]]), axis=0)

            # Append the site
            # todo rename negative_sites
            if not np.any([np.array_equal(sites_np[point1], site) for site in negative_sites]):
                negative_sites = np.concatenate((negative_sites, [sites_np[point1]]), axis=0)
            if not np.any([np.array_equal(sites_np[point2], site) for site in negative_sites]):
                negative_sites = np.concatenate((negative_sites, [sites_np[point2]]), axis=0)

    new_sites = []

    for i, region_index in enumerate(vor.point_region):
        if sites_np[i] in negative_sites:
            current_site = sites[i]
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:  # Skip infinite or empty regions
                continue

            # Get vertices of the cell
            vertices = vor.vertices[region]
            vertices = torch.from_numpy(vor.vertices[region]).to(device)

            smallest_edge_length = float("inf")
            for j in range(len(vertices)):
                v1, v2 = vertices[j], vertices[(j + 1) % len(vertices)]
                # Compute edge length
                edge_length = torch.norm(v2 - v1)
                smallest_edge_length = min(smallest_edge_length, edge_length)

            # Calculate the distance from the barycenter to the vertices (circumradius)
            rad = smallest_edge_length / np.sqrt(3)  # For equilateral triangle, circumradius = edge / sqrt(3)

            # Define 3 angles for the triangle vertices, spaced 120 degrees apart
            angles = torch.tensor([0, 2 * np.pi / 3, 4 * np.pi / 3], device=device)

            # Calculate the offset vectors for each vertex around the barycenter
            offsets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * rad

            # Generate new sites by adding the offsets to the current site
            for offset in offsets:
                new_sites.append(current_site + offset)

    return new_sites


def add_upsampled_sites(sites, new_sites, min_distance=0.01):
    """
    Add new sites while ensuring a minimum distance between points.
    Parameters:
        sites (torch.Tensor): Existing sites, shape (N, 2).
        new_sites (torch.Tensor): New candidate sites, shape (M, 2).
        min_distance (float): Minimum allowable distance between sites.
    Returns:
        torch.Tensor: Updated set of sites.
    """
    # updated_sites = sites.clone()  # Clone the existing tensor to modify later
    updated_sites = sites

    for new_site in new_sites:
        # Compute distances between the new site and all existing sites
        distances = torch.norm(updated_sites - new_site, dim=1)
        # Check if the new site is far enough from all existing sites
        if torch.all(distances > min_distance):
            # print(f"Adding new site at {new_site}")
            updated_sites = torch.cat((updated_sites, new_site.unsqueeze(0)), dim=0)

    return updated_sites


def get_sites_zero_crossing_edges(sites, model):
    sites_np = sites.detach().cpu().numpy()
    # Compute Voronoi diagram
    vor = Voronoi(sites_np)

    # Compute SDF values for each site
    sdf_values = model(sites)[:, 0]
    sdf_values = sdf_values.detach().cpu().numpy()

    edges = []
    for (point1, point2), (vertex1, vertex2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Check if vertices are valid (not -1, which indicates infinity)
        if vertex1 == -1 or vertex2 == -1:
            continue

        # Check if the sites have different signs
        if np.sign(sdf_values[point1]) != np.sign(sdf_values[point2]):
            edges.append((vertex1, vertex2))

    return edges


def adaptive_density_upsampling(sites, model, num_points_per_site=5, max_distance=1.0, sigma=1.0):
    """
    Upsample sites based on the SDF gradient and density map, placing new sites along the gradient direction.
    """

    # neighbors = get_delaunay_neighbors_list(sites)
    vor = Voronoi(sites.detach().cpu().numpy())
    neighbors = torch.tensor(np.array(vor.ridge_points), device=device)

    sdf_values = model(sites)

    # # Find pairs of neighbors with opposing SDF values
    # for i, adjacents in neighbors.items():
    #     for j in adjacents:
    #         if i < j and i not in sites_to_upsample:  # Avoid duplicates
    #             sdf_i, sdf_j = sdf_values[i].item(), sdf_values[j].item()
    #             if sdf_i * sdf_j <= 0:  # Opposing signs or one is zero
    #                 sites_to_upsample.append(i)
    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask = (sdf_i * sdf_j <= 0).squeeze()
    sites_to_upsample = neighbors[mask]

    # Compute the gradient of the SDF at each site
    grad_sdf = torch.autograd.grad(
        sdf_values[:, 0],
        sites,
        torch.ones_like(sdf_values[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the density map based on the gradient magnitude
    grad_mag = grad_sdf.norm(dim=1)  # Compute the magnitude of the gradient
    density_map = 1 / (1 + grad_mag**sigma)  # Inverse relation, higher gradients = higher density

    # Generate new sites along the gradient direction
    # new_sites = []
    # for i in range(len(sites_to_upsample)):
    #     site = sites[sites_to_upsample[i]]
    #     gradient = grad_sdf[sites_to_upsample[i]]
    #     density = density_map[sites_to_upsample[i]]

    #     # Normalize the gradient direction
    #     grad_norm = gradient / (gradient.norm() + 1e-6)  # Avoid division by zero

    #     # Generate points along the gradient direction, scaled by density
    #     num_new_points = int(density * num_points_per_site)  # Number of new points based on density

    #     for j in range(num_new_points):
    #         displacement = grad_norm * (max_distance * (j + 1) / num_new_points)
    #         new_site = site + displacement
    #         new_sites.append(new_site)

    ############################## todo test this
    # Vectorized computation of new sites
    sites_to_upsample_tensor = sites[sites_to_upsample]
    gradients = grad_sdf[sites_to_upsample]
    densities = density_map[sites_to_upsample]

    # Normalize the gradient directions
    grad_norms = gradients / (gradients.norm(dim=1, keepdim=True) + 1e-6)  # Avoid division by zero

    # Compute the number of new points for each site based on density
    num_new_points = (densities * num_points_per_site).int()

    # Generate displacements for each new point
    displacements = torch.cat(
        [
            grad_norms[i].unsqueeze(0)
            * (max_distance * (torch.arange(1, num_new_points[i] + 1, device=device).unsqueeze(1)) / num_new_points[i])
            for i in range(len(num_new_points))
        ]
    )

    # Generate new sites by adding displacements to the original sites
    new_sites = sites_to_upsample_tensor.repeat_interleave(num_new_points, dim=0) + displacements
    #########################################

    # Return the new sites as a tensor
    return new_sites


def upsampling_vectorized(sites, tri=None, vor=None, simplices=None, model=None):
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    # model might be a [sites, 1] tensor
    elif isinstance(model, torch.Tensor):
        sdf_values = model
    else:
        sdf_values = model(sites).detach()  # Assuming model outputs (N, 1) or (N,) tensor

    # sites_np = sites.detach().cpu().numpy()

    if tri is not None:
        all_tetrahedra = torch.tensor(np.array(tri.simplices), device=device)
    else:
        all_tetrahedra = torch.tensor(np.array(simplices), device=device)

    if vor is not None:
        neighbors = torch.tensor(np.array(vor.ridge_points), device=device)
    # could compute neighbors without the voronoi diagram
    else:
        # neighbors = torch.tensor(np.vstack(list({tuple(sorted(edge)) for tetra in tri.simplices for edge in zip(tetra, np.roll(tetra, -1))})), device=device)
        tetra_edges = torch.cat(
            [
                all_tetrahedra[:, [0, 1]],
                all_tetrahedra[:, [1, 2]],
                all_tetrahedra[:, [2, 3]],
                all_tetrahedra[:, [3, 0]],
                all_tetrahedra[:, [0, 2]],
                all_tetrahedra[:, [1, 3]],
            ],
            dim=0,
        ).to(device)
        # Sort each edge to ensure uniqueness (because (a, b) and (b, a) are the same)
        tetra_edges, _ = torch.sort(tetra_edges, dim=1)
        # Get unique edges
        neighbors = torch.unique(tetra_edges, dim=0)

    # Extract the SDF values for each site in the pair
    sdf_i = sdf_values[neighbors[:, 0]]  # First site in each pair
    sdf_j = sdf_values[neighbors[:, 1]]  # Second site in each pair
    # Find the indices where SDF values have opposing signs or one is zero
    mask_zero_crossing_sites = (sdf_i * sdf_j <= 0).squeeze()
    sites_to_upsample = torch.unique(neighbors[mask_zero_crossing_sites].view(-1))

    print("Sites to upsample ", sites_to_upsample.shape)

    tet_centroids = sites[sites_to_upsample]

    # Tetrahedron relative positions (unit tetrahedron)
    basic_tet_1 = torch.tensor([[1, 1, 1]], device=device, dtype=torch.float32)
    basic_tet_1 = basic_tet_1.repeat(len(tet_centroids), 1)
    basic_tet_2 = torch.tensor([-1, -1, 1], device=device, dtype=torch.float32)
    basic_tet_2 = basic_tet_2.repeat(len(tet_centroids), 1)
    basic_tet_3 = torch.tensor([-1, 1, -1], device=device, dtype=torch.float32)
    basic_tet_3 = basic_tet_3.repeat(len(tet_centroids), 1)
    basic_tet_4 = torch.tensor([1, -1, -1], device=device, dtype=torch.float32)
    basic_tet_4 = basic_tet_4.repeat(len(tet_centroids), 1)

    # Compute distances for each neighbor pair in a vectorized way
    pair_dists = torch.norm(sites[neighbors[:, 0]] - sites[neighbors[:, 1]], dim=1)
    # Create a combined index tensor for both endpoints of each pair
    all_indices = torch.cat([neighbors[:, 0], neighbors[:, 1]]).long()
    all_dists = torch.cat([pair_dists, pair_dists])

    # Initialize a tensor for each site with a large value (infinity)
    min_dists = torch.full((sites.shape[0],), float("inf"), device=device, dtype=sites.dtype)

    min_dists = min_dists.scatter_reduce(0, all_indices, all_dists, reduce="amin")

    # Extract the minimal distances for the sites that need upsampling, and use one-quarter as scale.
    scale = (min_dists[sites_to_upsample] / 4).unsqueeze(1)  # shape: (num_upsampled, 1)

    new_sites = torch.cat(
        (
            tet_centroids + basic_tet_1 * scale,
            tet_centroids + basic_tet_2 * scale,
            tet_centroids + basic_tet_3 * scale,
            tet_centroids + basic_tet_4 * scale,
        ),
        dim=0,
    )
    updated_sites = torch.cat((sites, new_sites), dim=0)

    return updated_sites


def upsampling_vectorized_sites_sites_sdf(
    sites: torch.Tensor,  # (N,3)
    tri=None,  # scipy.spatial.Delaunay object            (optional)
    vor=None,  # scipy.spatial.Voronoi  object            (optional)
    simplices=None,  # np.ndarray shape (M,4) if tri is None    (optional)
    model=None,  # SDFGrid | nn.Module | Tensor of shape (N,)
    eps: float = 1e-12,
):
    """
    • Detect zero-crossing edges, pick their incident sites for up-sampling.
    • Insert four points around every selected site (regular tetrahedron, scaled to ¼
      of its shortest incident edge).
    • Estimate ∇φ at the original sites from finite differences on all edges.
    • Assign φ(new) ≈ φ(old) + ∇φ(old)·δ  to every new point.
    • Return the concatenated point cloud and the concatenated SDF vector.
    """

    if model is None:
        raise ValueError("model must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)  # (N,)
    elif isinstance(model, torch.Tensor):
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        sdf_values = model(sites).detach()
    sdf_values = sdf_values.squeeze()  # ensure shape (N,)

    if tri is not None:
        all_tetrahedra = torch.as_tensor(tri.simplices, device=device)
    else:
        all_tetrahedra = torch.as_tensor(simplices, device=device)

    if vor is not None:
        neighbors = torch.as_tensor(vor.ridge_points, device=device)
    else:
        # six edges per tet
        tetra_edges = torch.cat(
            [
                all_tetrahedra[:, [0, 1]],
                all_tetrahedra[:, [1, 2]],
                all_tetrahedra[:, [2, 3]],
                all_tetrahedra[:, [3, 0]],
                all_tetrahedra[:, [0, 2]],
                all_tetrahedra[:, [1, 3]],
            ],
            dim=0,
        )
        neighbors, _ = torch.sort(tetra_edges, dim=1)  # canonical ordering
        neighbors = torch.unique(neighbors, dim=0)

    sdf_i, sdf_j = sdf_values[neighbors[:, 0]], sdf_values[neighbors[:, 1]]
    mask_zc = sdf_i * sdf_j <= 0  # zero-crossing edge
    sites_to_up = torch.unique(neighbors[mask_zc].reshape(-1))
    print("Sites to upsample :", sites_to_up.numel())

    centroids = sites[sites_to_up]  # (K,3)

    # shortest incident edge length per site
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]
    edge_len = torch.norm(edge_vec, dim=1)  # (E,)
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((sites.shape[0],), float("inf"), device=device)
    min_dists = min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")
    scale = (min_dists[sites_to_up] / 4).unsqueeze(1)  # (K,1)

    tetr_dirs = torch.as_tensor(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ],
        dtype=torch.float32,
        device=device,
    )  # (4,3)

    new_sites = (centroids.unsqueeze(1) + tetr_dirs.unsqueeze(0) * scale.unsqueeze(1)).reshape(-1, 3)  # (4K,3)

    # Estimate ∇φ at every original site (scatter-add, O(E))
    # finite-difference contribution from each edge endpoint
    sdf_diff = (sdf_values[neighbors[:, 1]] - sdf_values[neighbors[:, 0]]).unsqueeze(1)  # (E,1)
    edge_norm2 = (edge_vec**2).sum(1, keepdim=True) + eps  # (E,1)
    contrib = sdf_diff * edge_vec / edge_norm2  # (E,3)

    grad_est = torch.zeros_like(sites)  # (N,3)
    grad_est = grad_est.index_add(0, neighbors[:, 0], contrib)
    grad_est = grad_est.index_add(0, neighbors[:, 1], contrib)

    counts = torch.zeros((sites.shape[0], 1), device=device)
    ones = torch.ones_like(sdf_diff)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    grad_est /= counts.clamp(min=1.0)  # mean

    # First-order interpolation  φ(new) = φ(old) + ∇φ·δ
    cent_grad = grad_est[sites_to_up]  # (K,3)
    delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)  # (K,4,3)
    new_sdf = sdf_values[sites_to_up].unsqueeze(1) + (  # (K,1)
        cent_grad.unsqueeze(1) * delta
    ).sum(dim=2)  # (K,4)
    new_sdf = new_sdf.reshape(-1)  # (4K,)

    updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
    updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    return updated_sites, updated_sites_sdf


def upsampling_curvature_vectorized_sites_sites_sdf(
    sites: torch.Tensor,  # (N,3)
    tri=None,  # scipy.spatial.Delaunay object            (optional)
    vor=None,  # scipy.spatial.Voronoi  object            (optional)
    simplices=None,  # np.ndarray shape (M,4) if tri is None    (optional)
    model=None,  # SDFGrid | nn.Module | Tensor of shape (N,)
    eps: float = 1e-12,
):
    """
    • Detect zero-crossing edges, pick their incident sites for up-sampling.
    • Insert four points around every selected site (regular tetrahedron, scaled to ¼
      of its shortest incident edge).
    • Estimate ∇φ at the original sites from finite differences on all edges.
    • Assign φ(new) ≈ φ(old) + ∇φ(old)·δ  to every new point.
    • Return the concatenated point cloud and the concatenated SDF vector.
    """

    if model is None:
        raise ValueError("`model` must be an SDFGrid, nn.Module or a Tensor")
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)  # (N,)
    elif isinstance(model, torch.Tensor):
        sdf_values = model.to(device)
    else:  # nn.Module / callable
        sdf_values = model(sites).detach()
    sdf_values = sdf_values.squeeze()  # ensure shape (N,)

    if tri is not None:
        all_tetrahedra = torch.as_tensor(tri.simplices, device=device)
    else:
        all_tetrahedra = torch.as_tensor(simplices, device=device)

    if vor is not None:
        neighbors = torch.as_tensor(vor.ridge_points, device=device)
    else:
        # six edges per tet
        tetra_edges = torch.cat(
            [
                all_tetrahedra[:, [0, 1]],
                all_tetrahedra[:, [1, 2]],
                all_tetrahedra[:, [2, 3]],
                all_tetrahedra[:, [3, 0]],
                all_tetrahedra[:, [0, 2]],
                all_tetrahedra[:, [1, 3]],
            ],
            dim=0,
        )
        neighbors, _ = torch.sort(tetra_edges, dim=1)  # canonical ordering
        neighbors = torch.unique(neighbors, dim=0)

    # shortest incident edge length per site
    edge_vec = sites[neighbors[:, 1]] - sites[neighbors[:, 0]]
    edge_len = torch.norm(edge_vec, dim=1)  # (E,)
    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((sites.shape[0],), float("inf"), device=device)
    min_dists = min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")

    # Estimate ∇φ at every original site (scatter-add, O(E))
    # finite-difference contribution from each edge endpoint
    sdf_diff = sdf_values[neighbors[:, 1]] - sdf_values[neighbors[:, 0]]
    sdf_diff = sdf_diff.unsqueeze(1)  # (E,1)
    edge_norm2 = (edge_vec**2).sum(1, keepdim=True) + eps  # (E,1)
    contrib = sdf_diff * edge_vec / edge_norm2  # (E,3)

    grad_est = torch.zeros_like(sites)  # (N,3)
    grad_est = grad_est.index_add(0, neighbors[:, 0], contrib)
    grad_est = grad_est.index_add(0, neighbors[:, 1], contrib)

    counts = torch.zeros((sites.shape[0], 1), device=device)
    ones = torch.ones_like(sdf_diff)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    grad_est /= counts.clamp(min=1.0)

    # --- 1-ring normal-variation curvature filter -------------------------------
    # unit normals at the sites you already have
    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)  # (N,3)

    # squared change of normals across every edge
    dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1)  # (E,)

    # accumulate the edge values back to each endpoint
    curv_score = torch.zeros(sites.shape[0], device=device)  # (N,)
    curv_score = curv_score.index_add(0, neighbors[:, 0], dn2)
    curv_score = curv_score.index_add(0, neighbors[:, 1], dn2)
    curv_score /= counts.squeeze()  # mean over 1-ring

    sdf_i, sdf_j = sdf_values[neighbors[:, 0]], sdf_values[neighbors[:, 1]]
    mask_zc = sdf_i * sdf_j <= 0  # zero-crossing edge
    sites_to_up = torch.unique(neighbors[mask_zc].reshape(-1))
    print("Sites to upsample :", sites_to_up.numel())

    # NEW: keep only those whose curvature is above a percentile
    pct = 0.75  # 75-th percentile; tweak as you like
    thresh = torch.quantile(curv_score[sites_to_up], pct)
    sites_to_up = sites_to_up[curv_score[sites_to_up] > thresh]

    print("Sites to upsample (after curvature filter):", sites_to_up.numel())

    centroids = sites[sites_to_up]  # (K,3)

    tetr_dirs = torch.as_tensor(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
        ],
        dtype=torch.float32,
        device=device,
    )  # (4,3)

    scale = (min_dists[sites_to_up] / 4).unsqueeze(1)  # (K,1)
    new_sites = (centroids.unsqueeze(1) + tetr_dirs.unsqueeze(0) * scale.unsqueeze(1)).reshape(-1, 3)  # (4K,3)

    # mean

    # First-order interpolation  φ(new) = φ(old) + ∇φ·δ
    cent_grad = grad_est[sites_to_up]  # (K,3)
    delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)  # (K,4,3)
    new_sdf = sdf_values[sites_to_up].unsqueeze(1) + (  # (K,1)
        cent_grad.unsqueeze(1) * delta
    ).sum(dim=2)  # (K,4)
    new_sdf = new_sdf.reshape(-1)  # (4K,)

    updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
    updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    return updated_sites, updated_sites_sdf


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


@njit(parallel=True)
def batch_sort_numba(vertices, faces_list, counts, output):
    R, Kmax = faces_list.shape
    for i in prange(R):
        length = counts[i]
        sorted_i = sort_face_loop_numba(vertices, faces_list[i, :length])
        for j in range(length):
            output[i, j] = sorted_i[j]


def faces_via_dict(d3dsimplices, ridges):
    # 1) build dict of (a,b) → list of simplex-indices
    face_dict = defaultdict(list)
    for si, simplex in enumerate(d3dsimplices):
        # all 6 edges of a 4-vertex simplex
        a, b, c, d = simplex
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            key = (u, v) if u < v else (v, u)
            face_dict[key].append(si)

    # face dict creates a dictionnary of all the voronoi vertex that form voronoi faces

    # 2) now for each ridge (a,b) grab its list
    out = []
    for a, b in ridges:
        key = (a, b) if a < b else (b, a)
        lst = face_dict.get(key, [])
        out.append(np.array(lst, dtype=np.int32))

    return np.array(out, dtype=object)


def interpolate_sdf_of_vertices(
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

    x0 = v_pos[:, 0]  # (M, 3)
    x1 = v_pos[:, 1]
    x2 = v_pos[:, 2]
    x3 = v_pos[:, 3]

    # Build D = [x1 - x0 | x2 - x0 | x3 - x0]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0

    D = torch.stack([e1, e2, e3], dim=2)  # (M,3,3)

    c1 = torch.cross(e2, e3, dim=1)  # cofactor for col 0
    c2 = torch.cross(e3, e1, dim=1)  # cofactor for col 1
    c3 = torch.cross(e1, e2, dim=1)  # cofactor for col 2

    adj_D = torch.stack([c1, c2, c3], dim=2)  # (M, 3, 3)

    # Determinant of D
    det_D = (e1 * c1).sum(dim=1, keepdim=True)  # (M, 1)

    # Right-hand side: x - x0
    rhs = vertices - x0  # (M, 3)

    # Inverse: D⁻¹ @ rhs = adj(D)^T @ rhs / det(D)
    w123 = torch.bmm(adj_D.transpose(1, 2), rhs.unsqueeze(-1)).squeeze(-1) / (det_D + 1e-12)  # (M, 3)
    w0 = 1.0 - w123.sum(dim=1, keepdim=True)  # (M, 1)
    W = torch.cat([w0, w123], dim=1)  # (M, 4)

    # Interpolate SDF
    phi_v = (W * v_phi).sum(dim=1)  # (M,)

    # true_vertices_sdf, true_vertices_sdf_grad = sphere_sdf_and_grad(vertices)
    # threshold = 10000
    # sdf_mask = abs(phi_v - true_vertices_sdf) > threshold
    # for i in range(sdf_mask.sum().item()):
    #     print(f"Vertex {tets[sdf_mask][i]} ,SDF mismatch: {phi_v[sdf_mask][i]} vs {true_vertices_sdf[sdf_mask][i]}, W : {W[sdf_mask][i]}")
    #     print(f"Sites positions: {sites[tets[sdf_mask][i]]}")
    #     print(f"Vertices positions: {vertices[sdf_mask][i]}")
    return phi_v


def interpolate_sdf_grad_of_vertices(
    vertices: torch.Tensor,  # (M, 3) positions of Voronoi vertices
    tets: torch.LongTensor,  # (M, 4) indices of sites per tetrahedron
    sites: torch.Tensor,  # (N, 3) coordinates of the sites
    site_grads: torch.Tensor,  # (N, 3) spatial gradients ∇φ at each site
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

    x0, x1, x2, x3 = v_pos[:, 0], v_pos[:, 1], v_pos[:, 2], v_pos[:, 3]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0

    D = torch.stack([e1, e2, e3], dim=2)  # (M, 3, 3)

    # Cofactors of D
    c1 = torch.cross(e2, e3, dim=1)
    c2 = torch.cross(e3, e1, dim=1)
    c3 = torch.cross(e1, e2, dim=1)
    adj_D = torch.stack([c1, c2, c3], dim=2)  # (M, 3, 3)

    # Determinant
    det_D = (e1 * c1).sum(dim=1, keepdim=True)  # (M, 1)

    # Vector from x0 to each vertex
    rhs = vertices - x0  # (M, 3)

    # Solve D⁻¹ (x - x0)
    w123 = torch.bmm(adj_D.transpose(1, 2), rhs.unsqueeze(-1)).squeeze(-1) / (det_D + 1e-12)  # (M, 3)
    w0 = 1.0 - w123.sum(dim=1, keepdim=True)
    W = torch.cat([w0, w123], dim=1)  # (M, 4)

    we = torch.abs(W).max(dim=1, keepdim=True)[0]  # (M, 1)

    # Weighted sum of gradients
    grad_v = (W.unsqueeze(-1) * v_grad).sum(dim=1)  # (M, 3)

    return grad_v


def volume_tetrahedron(a, b, c, d):
    ad = a - d
    bd = b - d
    cd = c - d
    n = torch.cross(bd, cd)
    return torch.abs((ad * n).sum(dim=-1)) / 6.0


def sdf_space_grad_pytorch_diego(sites, sdf, tets):
    # sites: (N, 3)
    # sdf: (N,)
    # tets: (M, 4)

    M = tets.shape[0]
    tet_ids = tets
    a, b, c, d = (
        sites[tet_ids[:, 0]],
        sites[tet_ids[:, 1]],
        sites[tet_ids[:, 2]],
        sites[tet_ids[:, 3]],
    )
    sdf_a, sdf_b, sdf_c, sdf_d = (
        sdf[tet_ids[:, 0]],
        sdf[tet_ids[:, 1]],
        sdf[tet_ids[:, 2]],
        sdf[tet_ids[:, 3]],
    )

    center = (a + b + c + d) / 4
    sdf_center = (sdf_a + sdf_b + sdf_c + sdf_d) / 4

    volume = volume_tetrahedron(a, b, c, d)

    # Build dX: (M, 4, 3)
    X = torch.stack([a, b, c, d], dim=1)  # (M, 4, 3)
    dX = X - center[:, None, :]
    # print("dX shape:", dX.shape)

    dX_T = dX.transpose(1, 2)  # (M, 3, 4)
    # print("dX_T shape:", dX_T.shape)

    # G = dX^T @ dX: (M, 3, 3)
    # G = torch.einsum('mic,mjc->mij', dX, dX)
    G = torch.bmm(dX_T, dX)  # (M, 3, 3)
    # print("G shape:", G.shape)
    # Inverse G: (M, 3, 3)
    Ginv = torch.linalg.pinv(G)  # stable pseudo-inverse for singular cases
    # print("Ginv shape:", Ginv.shape)

    # Weights: Ginv @ dX^T -> (M, 3, 4)
    # W = torch.einsum('mij,mkj->mki', Ginv, dX)  # (M, 4, 3)
    W = torch.einsum("mij,mnj->mni", Ginv, dX)

    sdf_stack = torch.stack([sdf_a, sdf_b, sdf_c, sdf_d], dim=1)  # (M, 4)
    sdf_diff = sdf_stack - sdf_center[:, None]

    elem = torch.einsum("mi,mij->mj", sdf_diff, W)  # (M, 3)

    grad_sdf = torch.zeros_like(sites)  # (N, 3)
    weights_tot = torch.zeros_like(sdf)  # (N,)

    for i in range(4):
        ids = tet_ids[:, i]
        grad_contrib = elem * volume[:, None]  # (M, 3)
        grad_sdf.index_add_(0, ids, grad_contrib)
        weights_tot.index_add_(0, ids, volume)

    # Avoid division by zero (e.g., isolated vertices)
    weights_tot_clamped = weights_tot.clamp(min=1e-8).unsqueeze(1)  # (N, 1)
    grad_sdf /= weights_tot_clamped

    return grad_sdf


def sphere_sdf_and_grad(
    points: torch.Tensor,
    center: torch.Tensor = torch.zeros(3).to(device),
    radius: float = 0.5,
):
    """
    Compute the signed distance and gradient of a sphere at given 3D points.

    Args:
        points: (N, 3) tensor of 3D query points
        center: (3,) tensor for the sphere center
        radius: float, sphere radius

    Returns:
        sdf: (N,) signed distance to sphere
        grad: (N, 3) normalized gradient vectors
    """
    vec = points - center  # (N, 3)
    dist = torch.norm(vec, dim=-1, keepdim=True)  # (N, 1)
    sdf = dist.squeeze(-1) - radius  # (N,)
    grad = vec / (dist + 1e-8)  # (N, 3) unit vectors
    return sdf, grad


# def get_clipped_mesh_numba(sites, model, d3dsimplices, clip=True, sites_sdf=None, build_mesh=False):
#     """
#     sites:           (N,3) torch tensor (requires_grad)
#     model:           SDF model: sites -> (N,1) tensor of signed distances
#     d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
#     """
#     device = sites.device
#     vertices_sdf = None
#     vertices_sdf_grad = None
#     if d3dsimplices is None:
#         print("Computing Delaunay simplices...")
#         sites_np = sites.detach().cpu().numpy()
#         d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(sites_np.shape[1]*sites_np.shape[0]))
#         d3dsimplices = np.array(d3dsimplices)

#     d3d = torch.tensor(d3dsimplices).to(device).detach()            # (M,4)

#     vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)

#     if sites_sdf is not None:
#         vertices_sdf = interpolate_sdf_of_vertices(vor_vertices, d3d, sites, sites_sdf)
#         # Tets spatial gradient of the SDF
#         sites_sdf_grad = sdf_space_grad_pytorch_diego(sites, sites_sdf, d3d)  # (M,3)
#         vertices_sdf_grad, we = interpolate_sdf_grad_of_vertices(vor_vertices, d3d, sites, sites_sdf_grad)

#     faces = get_faces(d3dsimplices, sites, vor_vertices, model, sites_sdf)  # (R0, List of simplices)

#     # Compact the vertex list
#     used = {idx for face in faces for idx in face}
#     old2new = {old: new for new, old in enumerate(sorted(used))}
#     new_vertices = vor_vertices[sorted(used)]
#     new_faces = [[old2new[i] for i in face] for face in faces]

#     if vertices_sdf is not None and vertices_sdf_grad is not None:
#         sdf_verts = vertices_sdf[sorted(used)]
#         grads = vertices_sdf_grad[sorted(used)]  # (M,3)

#     elif model is not None:
#         # clip the vertices of the faces to the zero-crossing of the sdf
#         sdf_verts = model(new_vertices).view(-1)           # (M,)
#         # compute gradients ∇f(v)  — note create_graph=True if you
#         #    want second-order gradients to flow back into the model
#         grads = torch.autograd.grad(
#             outputs=sdf_verts,
#             inputs=new_vertices,
#             grad_outputs=torch.ones_like(sdf_verts),
#             create_graph=True,
#         )[0]                                               # (M,3)

#     if not clip:
#         print("-> not clipping")
#         return new_vertices, new_faces, sdf_verts, grads
#     else:
#         print("-> clipping")
#         #proj_vertices = newton_step_clipping(grads, sdf_verts, new_vertices)  # (M,3)
#         #tet_probs = sites_sdf_grad  # Placeholder for tet probabilities, if needed later

#         proj_vertices, tet_probs = tet_plane_clipping(d3d[sorted(used)], sites, sites_sdf, sites_sdf_grad, new_vertices)  # (M,3)

#     return proj_vertices, new_faces, sdf_verts, grads, tet_probs


def get_clipped_mesh_numba(sites, model, d3dsimplices, clip=True, sites_sdf=None, build_mesh=False):
    """
    sites:           (N,3) torch tensor (requires_grad)
    model:           SDF model: sites -> (N,1) tensor of signed distances
    d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
    """
    device = sites.device
    vertices_sdf = None
    vertices_sdf_grad = None
    if d3dsimplices is None:
        print("Computing Delaunay simplices...")
        sites_np = sites.detach().cpu().numpy()
        d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(sites_np.shape[1] * sites_np.shape[0]))
        d3dsimplices = np.array(d3dsimplices)

    d3d = torch.tensor(d3dsimplices).to(device).detach()  # (M,4)

    if build_mesh:
        print("-> tracing mesh")
        all_vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)
        faces = get_faces(d3dsimplices, sites, all_vor_vertices, model, sites_sdf)  # (R0, List of simplices)
        # Compact the vertex list
        used = {idx for face in faces for idx in face}
        old2new = {old: new for new, old in enumerate(sorted(used))}
        new_vertices = all_vor_vertices[sorted(used)]
        new_faces = [[old2new[i] for i in face] for face in faces]
        if not clip:
            print("-> not clipping")
            return new_vertices, new_faces, None, None
        else:
            print("-> clipping")
            vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad = sdf_space_grad_pytorch_diego(sites, sites_sdf, d3d)  # (M,3)
            vertices_sdf_grad = interpolate_sdf_grad_of_vertices(all_vor_vertices, d3d, sites, sites_sdf_grad)

            sdf_verts = vertices_sdf[sorted(used)]
            grads = vertices_sdf_grad[sorted(used)]  # (M,3)

            proj_vertices, tet_probs = tet_plane_clipping(
                d3d[sorted(used)], sites, sites_sdf, sites_sdf_grad, new_vertices
            )  # (M,3)
            return proj_vertices, new_faces, sdf_verts, grads, tet_probs
    else:
        print("-> not tracing mesh")
        all_vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)
        vertices_to_compute, bisectors_to_compute, used_tet = compute_zero_crossing_vertices_3d(
            sites, None, None, d3dsimplices, sites_sdf
        )
        vertices = compute_vertices_3d_vectorized(sites, vertices_to_compute)
        bisectors = compute_all_bisectors_vectorized(sites, bisectors_to_compute)
        # TODO:idea use bisectors as center of face and use knn to do the face connectivity to the vertices
        # points = torch.cat((vertices, bisectors), 0)
        if not clip:
            print("-> not clipping")
            return vertices, None, None, None
        else:
            print("-> clipping")
            vertices_sdf = interpolate_sdf_of_vertices(all_vor_vertices, d3d, sites, sites_sdf)
            sites_sdf_grad = sdf_space_grad_pytorch_diego(sites, sites_sdf, d3d)
            vertices_sdf_grad = interpolate_sdf_grad_of_vertices(all_vor_vertices, d3d, sites, sites_sdf_grad)

            sdf_verts = vertices_sdf[used_tet]
            grads = vertices_sdf_grad[used_tet]
            proj_vertices, tet_probs = tet_plane_clipping(d3d[used_tet], sites, sites_sdf, sites_sdf_grad, vertices)

            # print("-> computing bisectors")
            # print(sites.shape, sites_sdf.shape, "bisectors shape", bisectors.shape, "bisectors_to_compute shape", bisectors_to_compute.shape)

            bisectors_sdf = (sites_sdf[bisectors_to_compute[:, 0]] + sites_sdf[bisectors_to_compute[:, 1]]) / 2
            bisectors_sdf_grad = (
                sites_sdf_grad[bisectors_to_compute[:, 0]] + sites_sdf_grad[bisectors_to_compute[:, 1]]
            ) / 2

            proj_bisectors = newton_step_clipping(bisectors_sdf_grad, bisectors_sdf, bisectors)  # (M,3)

            proj_points = torch.cat((proj_vertices, proj_bisectors), 0)

            return proj_points, None, sdf_verts, grads, tet_probs


def get_faces(d3dsimplices, sites, vor_vertices, model=None, sites_sdf=None):
    with torch.no_grad():
        d3d = torch.tensor(d3dsimplices).to(device).detach()  # (M,4)
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
        if model is not None:
            sdf = model(sites).detach().view(-1)  # (N,)
        else:
            sdf = sites_sdf  # (N,)

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


def newton_step_clipping(grads, sdf_verts, new_vertices):
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


def tet_plane_clipping(
    tets: torch.Tensor,  # (M, 4)
    sites: torch.Tensor,  # (N, 3)
    sdf_values: torch.Tensor,  # (N,)
    sdf_grads: torch.Tensor,  # (N, 3)
    voronoi_vertices: torch.Tensor,  # (M, 3)
) -> torch.Tensor:
    eps = 1e-12
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

    return projected_verts, (site_step_dir, vert_step_dir, tet_sites)


def upsampling_adaptive_vectorized_sites_sites_sdf(
    sites: torch.Tensor,  # (N,3)
    simplices=None,  # np.ndarray (M,4) if tri is None
    model=None,  # SDFGrid | nn.Module | Tensor (N,)
    spacing_target: float = None,  # desired final spacing  (same units as sites)
    alpha_high: float = 1.5,  # regime switches   (α_high > α_low ≥ 1)
    alpha_low: float = 1.1,
    curv_pct: float = 0.75,  # percentile threshold for curvature pass
    growth_cap: float = 0.10,  # ≤ fraction of current sites allowed per iter
    eps: float = 1e-12,
):
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

    all_tets = torch.as_tensor(simplices, device=device)

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

    idx_all = torch.cat([neighbors[:, 0], neighbors[:, 1]])
    dists_all = torch.cat([edge_len, edge_len])
    min_dists = torch.full((N,), float("inf"), device=device)
    min_dists = min_dists.scatter_reduce(0, idx_all, dists_all, reduce="amin")  # (N,)

    # Gradient ∇φ and curvature proxy κᵢ  (1-ring normal variation)
    # ∇φ estimate (scatter-add of finite-difference contributions)
    sdf_diff = sdf_values[neighbors[:, 1]] - sdf_values[neighbors[:, 0]]
    sdf_diff = sdf_diff.unsqueeze(1)  # (E,1)
    edge_norm2 = (edge_vec**2).sum(1, keepdim=True) + eps
    contrib = sdf_diff * edge_vec / edge_norm2  # (E,3)

    grad_est = torch.zeros_like(sites)  # (N,3)
    grad_est = grad_est.index_add(0, neighbors[:, 0], contrib)
    grad_est = grad_est.index_add(0, neighbors[:, 1], contrib)

    counts = torch.zeros((N, 1), device=device)
    ones = torch.ones_like(sdf_diff)
    counts = counts.index_add(0, neighbors[:, 0], ones)
    counts = counts.index_add(0, neighbors[:, 1], ones)
    grad_est /= counts.clamp(min=1.0)

    unit_n = grad_est / (grad_est.norm(dim=1, keepdim=True) + eps)
    dn2 = ((unit_n[neighbors[:, 0]] - unit_n[neighbors[:, 1]]) ** 2).sum(1)  # (E,)
    curv_score = torch.zeros(N, device=device)  # (N,)
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

    # --- UNIFORM ---------------------------------------------------- #
    print(median_min_dists, spacing_target * alpha_high, spacing_target * alpha_low)
    if median_min_dists > spacing_target * alpha_high:
        print("Uniform upsampling regime")
        cand = zc_sites[min_dists[zc_sites] > spacing_target]
        print(f"Number of candidates in uniform regime: {cand.numel()}")

    # --- HYBRID ----------------------------------------------------- #
    elif median_min_dists > spacing_target * alpha_low:
        print("Hybrid upsampling regime")
        score = (min_dists[zc_sites] / median_min_dists) * (curv_score[zc_sites] / (torch.median(curv_score) + eps))
        M = int(min(max(1, growth_cap * N), score.numel()))
        topk = torch.topk(score, k=M, largest=True).indices
        cand = zc_sites[topk]
        print(f"Number of candidates in hybrid regime: {cand.numel()}")

    # --- CURVATURE -------------------------------------------------- #
    else:
        print("Curvature upsampling regime")
        thresh = torch.quantile(curv_score[zc_sites], curv_pct)
        mask = (curv_score[zc_sites] > thresh) & (min_dists[zc_sites] > spacing_target * 0.5)
        cand = zc_sites[mask]
        print(f"Number of candidates in curvature regime: {cand.numel()}")

    K = cand.numel()
    if K == 0:
        return sites, sdf_values  # nothing selected

    # Insert 4 off-spring per selected site (regular tetrahedron)
    tetr_dirs = torch.as_tensor(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype=torch.float32,
        device=device,
    )  # (4,3)

    centroids = sites[cand]  # (K,3)
    scale = (min_dists[cand] / 4).unsqueeze(1)  # (K,1)
    new_sites = (centroids.unsqueeze(1) + tetr_dirs.unsqueeze(0) * scale.unsqueeze(1)).reshape(-1, 3)  # (4K,3)
    print("Before upsampling, number of sites:", sites.shape[0], "amount added:", new_sites.shape[0])
    # First-order SDF interpolation φ(new) = φ(old) + ∇φ·δ
    cent_grad = grad_est[cand]  # (K,3)
    delta = new_sites.reshape(-1, 4, 3) - centroids.unsqueeze(1)  # (K,4,3)
    new_sdf = (sdf_values[cand].unsqueeze(1) + (cent_grad.unsqueeze(1) * delta).sum(2)).reshape(-1)  # (4K,)

    # Concatenate & return
    updated_sites = torch.cat([sites, new_sites], dim=0)  # (N+4K,3)
    updated_sites_sdf = torch.cat([sdf_values, new_sdf], dim=0)  # (N+4K,)

    return updated_sites, updated_sites_sdf
