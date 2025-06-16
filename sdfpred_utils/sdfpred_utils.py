from scipy.spatial import Delaunay, Voronoi
import numpy as np
from sklearn.cluster import KMeans
import torch
import diffvoronoi #delaunay3d bindings
import math
from numba import njit, prange
from collections import defaultdict

device = torch.device("cuda:0")

# Python code for creating a CVT
# Vassilis Vassiliades - Inria, Nancy - April 2018
def createCVTgrid(num_centroids = 128, dimensionality = 2, num_samples = 100000, num_replicates = 1, max_iterations = 100000, tolerance = 0.00001):
    X = np.random.rand(num_samples,dimensionality)
    kmeans = KMeans(
        init='k-means++', 
        n_clusters=num_centroids, 
        n_init=num_replicates, 
        #n_jobs=-1, 
        max_iter=max_iterations, 
        tol=tolerance,
        verbose=0)

    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    centroids = (np.array(centroids- 0.5))* 10.0
    #make centroids double
    centroids = centroids.astype(np.double)

    #sites = torch.from_numpy(centroids).to(device).requires_grad_(True)

    sites = torch.from_numpy(centroids).to(device, dtype=torch.double).requires_grad_(True)
    print(sites.shape, sites.dtype)
    return sites

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
    mask_zero_crossing_faces = (sdf_0 * sdf_1 <= 0).squeeze() | (sdf_0 * sdf_2 <= 0).squeeze() | (sdf_1 * sdf_2 <= 0).squeeze()

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
    #model might be a true sdf grid of class SDFGrid
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
    mask_zero_crossing_faces = (sdf_0*sdf_1<=0).squeeze() | (sdf_0*sdf_2<=0).squeeze() | (sdf_0*sdf_3<=0).squeeze() | (sdf_1*sdf_2<=0).squeeze() | (sdf_1*sdf_3<=0).squeeze() | (sdf_2*sdf_3<=0).squeeze()
    zero_crossing_vertices_index = all_tetrahedra[mask_zero_crossing_faces]
    
    return zero_crossing_vertices_index, zero_crossing_pairs


def compute_zero_crossing_sites_pairs(all_tetrahedra, sdf_values):
    tetra_edges = torch.cat([
        all_tetrahedra[:, [0, 1]],
        all_tetrahedra[:, [1, 2]],
        all_tetrahedra[:, [2, 3]],
        all_tetrahedra[:, [3, 0]],
        all_tetrahedra[:, [0, 2]],
        all_tetrahedra[:, [1, 3]]
                                ], dim=0).to(device)
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
    n_x = (
        x_i**2 * (y_j - y_k)
        - x_j**2 * (y_i - y_k)
        + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)
    )

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
    s_i, s_j, s_k = sites[vertices_to_compute[:, 0]], sites[vertices_to_compute[:, 1]], sites[vertices_to_compute[:, 2]]
    
    x_i, y_i = s_i[:, 0], s_i[:, 1]
    x_j, y_j = s_j[:, 0], s_j[:, 1]
    x_k, y_k = s_k[:, 0], s_k[:, 1]
    
    # Compute numerators and denominator
    n_x = (
        x_i**2 * (y_j - y_k)
        - x_j**2 * (y_i - y_k)
        + (x_k**2 + (y_i - y_k) * (y_j - y_k)) * (y_i - y_j)
    )
    
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
    squared_norms = (tetrahedra ** 2).sum(dim=2, keepdim=True)  # Shape: (M, 4, 1)

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

def compute_vertices_3d_vectorized_w_sdf_grads(
    sites: torch.Tensor,                # (N, 3)
    vertices_to_compute: torch.Tensor,  # (M, 4)
    sdf_values: torch.Tensor,           # (N,) or (N,1)
    sdf_gradients: torch.Tensor         # (N, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the circumcenters of multiple tetrahedra in a vectorized manner,
    then interpolates per-vertex SDF values and SDF gradients at those circumcenters.

    Args:
        sites:               (N, 3) tensor of site positions.
        vertices_to_compute: (M, 4) tensor of indices forming tetrahedra.
        sdf_values:          (N,) tensor of SDF values at each site.
        sdf_gradients:       (N, 3) tensor of SDF gradients at each site.

    Returns:
        circumcenters:      (M, 3) tensor of computed Voronoi vertices.
        sdf_at_centers:     (M,) tensor of interpolated SDF values.
        grad_at_centers:    (M, 3) tensor of interpolated SDF gradients.
    """
    # --- 1) Circumcenters ---
    tetra = sites[vertices_to_compute]                    # (M,4,3)
    norms = (tetra**2).sum(dim=2, keepdim=True)           # (M,4,1)
    ones = torch.ones_like(norms)                         # (M,4,1)

    A  = torch.cat([tetra,       ones], dim=2)            # (M,4,4)
    Dx = torch.cat([norms,       tetra[:,:,1:], ones], dim=2)
    Dy = torch.cat([tetra[:,:,:1], norms,        tetra[:,:,2:], ones], dim=2)
    Dz = torch.cat([tetra[:,:,:2], norms,        ones], dim=2)

    detA  = torch.linalg.det(A)   # (M,)
    detDx = torch.linalg.det(Dx)
    detDy = torch.linalg.det(Dy)
    detDz = torch.linalg.det(Dz)

    circumcenters = 0.5 * torch.stack([detDx, detDy, detDz], dim=1) / detA.unsqueeze(1)  # (M,3)

    # --- 2) Barycentric weights ---
    v0 = tetra[:,0]  # (M,3)
    v1 = tetra[:,1]
    v2 = tetra[:,2]
    v3 = tetra[:,3]

    T = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=2)         # (M,3,3)
    b = (circumcenters - v0).unsqueeze(2)                       # (M,3,1)
    lambda123 = torch.linalg.solve(T, b).squeeze(2)            # (M,3)
    lambda0 = 1.0 - lambda123.sum(dim=1, keepdim=True)         # (M,1)
    weights = torch.cat([lambda0, lambda123], dim=1)           # (M,4)

    # --- 3) Interpolate SDF values ---
    sdf4 = sdf_values[vertices_to_compute].squeeze(-1)                     # (M,4)
    sdf_at_centers = (weights * sdf4).sum(dim=1)               # (M,)

    # --- 4) Interpolate gradients ---
    grad4 = sdf_gradients[vertices_to_compute]                 # (M,4,3)
    grad_at_centers = (weights.unsqueeze(2) * grad4).sum(dim=1)  # (M,3)
    return circumcenters, sdf_at_centers, grad_at_centers




# def compute_all_bisectors(sites, bisectors_to_compute):
#     # Initialize an empty tensor for storing bisectors
#     bisectors = []
    
#     for pairs in bisectors_to_compute:
#         si = sites[pairs[0]]
#         sj = sites[pairs[1]]
#         b = (si + sj) / 2
#         bisectors.append(b)

#     # Stack the list of bisectors into a single tensor for easier gradient tracking
#     bisectors = torch.stack(bisectors)
#     return bisectors
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
    
    #edges = []
    negative_sites = np.array([]).reshape(0,2)

    for (point1, point2), (vertex1, vertex2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Check if vertices are valid (not -1, which indicates infinity)
        if vertex1 == -1 or vertex2 == -1:
            continue
        
        # Check if the sites have different signs
        if np.sign(sdf_values[point1]) != np.sign(sdf_values[point2]):
            #edges.append((vertex1, vertex2))
            
            # Append the site with a negative SDF value
            # if sdf_values[point1] < 0 and not np.any([np.array_equal(sites_np[point1], site) for site in negative_sites]):
            #     negative_sites = np.concatenate((negative_sites, [sites_np[point1]]), axis=0)
            # if sdf_values[point2] < 0 and not np.any([np.array_equal(sites_np[point2], site) for site in negative_sites]):
            #     negative_sites = np.concatenate((negative_sites, [sites_np[point2]]), axis=0)
            
            # Append the site 
            #todo rename negative_sites
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
    #updated_sites = sites.clone()  # Clone the existing tensor to modify later
    updated_sites = sites
    
    for new_site in new_sites:
        # Compute distances between the new site and all existing sites
        distances = torch.norm(updated_sites - new_site, dim=1)
        # Check if the new site is far enough from all existing sites
        if torch.all(distances > min_distance):
            #print(f"Adding new site at {new_site}")
            updated_sites = torch.cat((updated_sites, new_site.unsqueeze(0)), dim=0)

    return updated_sites

def get_sites_zero_crossing_edges(sites, model):
    sites_np = sites.detach().cpu().numpy()
    # Compute Voronoi diagram
    vor = Voronoi(sites_np)

    # Compute SDF values for each site
    sdf_values = model(sites)[:,0]
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
    
    #neighbors = get_delaunay_neighbors_list(sites)
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
    grad_sdf = torch.autograd.grad(sdf_values[:,0], sites, torch.ones_like(sdf_values[:,0]), create_graph=True, retain_graph=True)[0]

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
    displacements = torch.cat([
        grad_norms[i].unsqueeze(0) * (max_distance * (torch.arange(1, num_new_points[i] + 1, device=device).unsqueeze(1)) / num_new_points[i])
        for i in range(len(num_new_points))
    ])

    # Generate new sites by adding displacements to the original sites
    new_sites = sites_to_upsample_tensor.repeat_interleave(num_new_points, dim=0) + displacements
    #########################################
    
    # Return the new sites as a tensor
    return new_sites


def get_zero_crossing_mesh_3d(sites, model):
    sites_np = sites.detach().cpu().numpy()
    vor = Voronoi(sites_np)  # Compute 3D Voronoi diagram

    sdf_values = model(sites)[:, 0].detach().cpu().numpy()  # Compute SDF values

    valid_faces = []  # List of polygonal faces
    used_vertices = set()  # Set of indices for valid vertices

    for (point1, point2), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices:
            continue  # Skip infinite ridges

        # Check if SDF changes sign across this ridge
        if np.sign(sdf_values[point1]) != np.sign(sdf_values[point2]):
            valid_faces.append(ridge_vertices)
            used_vertices.update(ridge_vertices)

    # **Filter Voronoi vertices**
    used_vertices = sorted(used_vertices)  # Keep unique, sorted indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
    filtered_vertices = vor.vertices[used_vertices]

    # **Re-index faces to match the new filtered vertex list**
    filtered_faces = [[vertex_map[v] for v in face] for face in valid_faces]

    return filtered_vertices, filtered_faces

def upsampling_vectorized(sites, tri=None, vor=None, simplices=None, model=None):
    if model.__class__.__name__ == "SDFGrid":
        sdf_values = model.sdf(sites)
    else:
        sdf_values = model(sites).detach()
        
    #sites_np = sites.detach().cpu().numpy()
    
    if tri is not None:
        all_tetrahedra = torch.tensor(np.array(tri.simplices), device=device)
    else:
        all_tetrahedra = torch.tensor(np.array(simplices), device=device)

    
    if vor is not None:
        neighbors = torch.tensor(np.array(vor.ridge_points), device=device)
    # could compute neighbors without the voronoi diagram
    else:
        #neighbors = torch.tensor(np.vstack(list({tuple(sorted(edge)) for tetra in tri.simplices for edge in zip(tetra, np.roll(tetra, -1))})), device=device)
        tetra_edges = torch.cat([
        all_tetrahedra[:, [0, 1]],
        all_tetrahedra[:, [1, 2]],
        all_tetrahedra[:, [2, 3]],
        all_tetrahedra[:, [3, 0]],
        all_tetrahedra[:, [0, 2]],
        all_tetrahedra[:, [1, 3]]
                                ], dim=0).to(device)
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
    
    print("Sites to upsample ",sites_to_upsample.shape)
    
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
    min_dists = torch.full((sites.shape[0],), float('inf'), device=device, dtype=sites.dtype)

    min_dists = min_dists.scatter_reduce(0, all_indices, all_dists, reduce='amin')
    
    # Extract the minimal distances for the sites that need upsampling, and use one-quarter as scale.
    scale = (min_dists[sites_to_upsample] / 4).unsqueeze(1)  # shape: (num_upsampled, 1)
    
    new_sites = torch.cat((tet_centroids + basic_tet_1 * scale, tet_centroids + basic_tet_2 * scale, tet_centroids + basic_tet_3 * scale, tet_centroids + basic_tet_4 * scale), dim=0)

    updated_sites = torch.cat((sites, new_sites), dim=0)
    
    return updated_sites

def batch_sort_face_loops_from_mask(vertices, face_mask):
    """
    vertices:  (M,3)  circumcenters
    face_mask: (R,N)  bool tensor where face_mask[r,n]=True iff simplex n contributed to face r
    returns: Python list of R sorted faces (each a list of vertex-indices into `vertices`)
    """
    R, N = face_mask.shape
    device = vertices.device

    # how many verts per face, and pad to the max
    counts = face_mask.sum(dim=1)            # (R,)
    #print("counts", counts.shape)
    Kmax   = int(counts.max().item())
    #print("Kmax", Kmax)

    # flatten mask → get (ridge_idx, simplex_idx)
    ridge_idx, simp_idx = face_mask.nonzero(as_tuple=True)  # both (S,)

    # offsets to compute “position within each group”:
    offsets = torch.cat((torch.tensor([0], device=device),
                         counts.cumsum(0)[:-1]))               # (R,)
    # expand offsets to length S by repeating each offset counts[r] times
    offs_exp = torch.repeat_interleave(offsets, counts)    # (S,)
    group_pos = torch.arange(ridge_idx.size(0), device=device) - offs_exp  # (S,)

    # build a padded index tensor and mask
    idxs = torch.full((R, Kmax), -1, dtype=torch.long, device=device)
    idxs[ridge_idx, group_pos] = simp_idx
    valid = idxs >= 0   # (R, Kmax)

    # gather all points, compute per-face centroids
    pts = vertices[idxs.clamp(min=0)]                   # (R, Kmax, 3)
    ctr = (pts * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / \
          counts.unsqueeze(-1).unsqueeze(-1)            # (R,1,3)

    # batched SVD → normals
    U, S, Vh = torch.linalg.svd(pts - ctr)              # Vh: (R,3,3)
    normals  = Vh[:, -1, :]                             # (R,3)

    # reference axis in each plane
    ref = pts[:, 0, :] - ctr.squeeze(1)                 # (R,3)
    ref = ref - normals * (normals * ref).sum(dim=1, keepdim=True)

    # project into plane
    vecs = pts - ctr                                    # (R,Kmax,3)
    dotn = (vecs * normals.unsqueeze(1)).sum(dim=2, keepdim=True)
    vecs = vecs - normals.unsqueeze(1)*dotn

    # angles = atan2(||ref×v||, ref·v), with sign
    cross   = torch.cross(ref.unsqueeze(1).expand_as(vecs), vecs, dim=2)
    norms   = torch.linalg.norm(cross, dim=2)
    dots    = (vecs * ref.unsqueeze(1)).sum(dim=2)
    ang     = torch.atan2(norms, dots)
    sign_m  = ((normals.unsqueeze(1)*cross).sum(dim=2) < 0)
    ang[sign_m] = 2*math.pi - ang[sign_m]

    # force padded slots to sort last
    ang[~valid] = float('inf')

    # per-face sort
    order = torch.argsort(ang, dim=1)                   # (R,Kmax)

    # unpack back into Python lists
    faces = []
    for r in range(R):
        k = counts[r].item()
        picks = order[r, :k]
        faces.append( idxs[r, picks].tolist() )
    return faces


def batch_face_loops_from_mask(face_mask):
    """
    face_mask: (C, M) bool tensor where
               face_mask[r, s] == True if simplex s contributes to face r
    returns: Python list of C lists, each containing the simplex-indices
    """
    C, M = face_mask.shape
    device = face_mask.device

    # how many verts per face, and pad to the max
    counts = face_mask.sum(dim=1)            # (C,)
    Kmax   = int(counts.max().item())

    # flatten mask → (ridge_idx, simplex_idx)
    ridge_idx, simp_idx = face_mask.nonzero(as_tuple=True)  # both (S,)

    # offsets to compute position within each face
    offsets = torch.cat((counts.new_zeros(1),
                         counts.cumsum(0)[:-1]))            # (C,)
    offs_exp = torch.repeat_interleave(offsets, counts)     # (S,)
    pos     = torch.arange(ridge_idx.size(0), device=device) - offs_exp  # (S,)

    # build a padded index tensor and fill it
    idxs = torch.full((C, Kmax), -1, dtype=torch.long, device=device)
    idxs[ridge_idx, pos] = simp_idx

    # unpack into Python lists
    faces = []
    for r in range(C):
        k = counts[r].item()
        if k > 0:
            faces.append(idxs[r, :k].tolist())
        else:
            faces.append([])
    return faces

def get_clipped_mesh_torch(sites, model, d3dsimplices, batch_size=1024):
    """
    sites:           (N,3) torch tensor (requires_grad)
    model:           SDF model: sites -> (N,1) tensor of signed distances
    d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
    """
    device = sites.device
    if d3dsimplices is None:
        sites_np = sites.detach().cpu().numpy()
        d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(sites_np.shape[1]*sites_np.shape[0]))
        d3dsimplices = np.array(d3dsimplices)
    d3d = torch.tensor(d3dsimplices).to(device)              # (M,4)

    # Compute per‐simplex circumcenters (Voronoi vertices)
    vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)

    # Generate all edges of each simplex
    #    torch.combinations gives the 6 index‐pairs within a 4‐long row
    comb = torch.combinations(torch.arange(d3d.shape[1], device=device), r=2)  # (6,2)
    #print("comb", comb.shape)
    edges = d3d[:, comb]                    # (M,6,2)
    edges = edges.reshape(-1,2)             # (M*6,2)
    edges, _ = torch.sort(edges, dim=1)     # sort each row so (a,b) == (b,a)

    # Unique ridges across all simplices
    ridges, inverse = torch.unique(edges, dim=0, return_inverse=True)  # (R,2)

    # Evaluate SDF at each site
    sdf = model(sites).view(-1)             # (N,)
    sdf_i = sdf[ridges[:,0]]
    sdf_j = sdf[ridges[:,1]]
    zero_cross = (sdf_i * sdf_j <= 0)       # (R,)

    # Keep only the zero-crossing ridges
    ridges = ridges[zero_cross]             # (R0,2)

    # # For each kept ridge, find the simplices that share both its sites
    # #    Build a (R0, M) mask of membership
    # #    mask_a[r,s] = True if simplex s contains ridges[r,0]
    # mask_a = (d3d.unsqueeze(0) == ridges[:,0].unsqueeze(1).unsqueeze(2)).any(dim=2)
    # print("mask_a", mask_a.shape)
    # print("mask_a", mask_a[0])
    # mask_b = (d3d.unsqueeze(0) == ridges[:,1].unsqueeze(1).unsqueeze(2)).any(dim=2)
    # face_mask = mask_a & mask_b             # (R0, M)
    # print("face_mask", face_mask.shape)
    # print("face_mask", face_mask[0])

    # # Extract and sort each face’s loop of vertices
    # # faces = []
    # # for r in range(ridges.shape[0]):
    # #     simplex_idxs = torch.nonzero(face_mask[r], as_tuple=False).squeeze(1)
    # #     faces.append(sort_face_loop_torch(vor_vertices, simplex_idxs))

    # faces = batch_sort_face_loops_from_mask(vor_vertices, face_mask)
    
    faces = []
    R0 = ridges.shape[0]
    for start in range(0, R0, batch_size):
        end = min(start + batch_size, R0)
        ridges_chunk = ridges[start:end]
        a0 = ridges_chunk[:,0].view(-1,1,1)
        b0 = ridges_chunk[:,1].view(-1,1,1)
        mask_a = (d3d.unsqueeze(0) == a0).any(dim=2)
        mask_b = (d3d.unsqueeze(0) == b0).any(dim=2)
        face_mask = mask_a & mask_b             # (R0, M)
        #print("face_mask", face_mask.shape)
        faces_chunk = batch_sort_face_loops_from_mask(vor_vertices, face_mask)
        #faces_chunk = batch_face_loops_from_mask(face_mask)
        faces.extend(faces_chunk)

    #print("faces", len(faces))
    
    # Compact the vertex list
    used = {idx for face in faces for idx in face}
    old2new = {old: new for new, old in enumerate(sorted(used))}
    new_vertices = vor_vertices[sorted(used)]
    new_faces = [[old2new[i] for i in face] for face in faces]


    # clip the vertices of the faces to the zero-crossing of the sdf
    sdf_verts = model(new_vertices).view(-1)           # (M,)

    # compute gradients ∇f(v)  — note create_graph=True if you
    #    want second-order gradients to flow back into the model
    grads = torch.autograd.grad(
        outputs=sdf_verts,
        inputs=new_vertices,
        grad_outputs=torch.ones_like(sdf_verts),
        create_graph=True,
    )[0]                                               # (M,3)

    # one Newton step https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    epsilon = 1e-6
    grad_norm2 = torch.sqrt(((grads + epsilon)**2).sum(dim=1, keepdim=True))    # (M,1)
    step = sdf_verts.unsqueeze(1) * grads / (grad_norm2 + epsilon)
    proj_vertices = new_vertices - step    

    #print("-> vertices:", new_vertices.shape)
    #print("-> projected vertices:", proj_vertices.shape)
    #print("-> #faces:", len(new_faces))
    return proj_vertices, new_faces

def sample_mesh_points(vertices: torch.Tensor,
                       faces: torch.LongTensor,
                       num_samples: int) -> torch.Tensor:
    """
    Uniformly (area-weighted) sample points on a triangular mesh.

    Args:
        vertices: (V,3) float tensor of vertex positions.
        faces:    (F,3) long tensor of indices into `vertices`.
        num_samples: int, number of points to sample.

    Returns:
        samples: (num_samples, 3) float tensor of sampled points.
    """
    # 1) Gather triangle vertices
    v0 = vertices[faces[:, 0]]  # (F,3)
    v1 = vertices[faces[:, 1]]  # (F,3)
    v2 = vertices[faces[:, 2]]  # (F,3)

    # 2) Compute each triangle’s area
    #    area = 0.5 * ||(v1 - v0) × (v2 - v0)||_2
    tri_edges0 = v1 - v0        # (F,3)
    tri_edges1 = v2 - v0        # (F,3)
    cross_prod = torch.cross(tri_edges0, tri_edges1, dim=1)  # (F,3)
    tri_areas = 0.5 * cross_prod.norm(dim=1)                  # (F,)

    # 3) Sample faces proportional to area
    face_probs = tri_areas / tri_areas.sum()
    #    draw `num_samples` face indices with replacement
    idx = torch.multinomial(face_probs, num_samples, replacement=True)  # (num_samples,)

    # 4) For each sampled face, sample a point via barycentric coords
    u = torch.rand(num_samples, device=vertices.device)
    v = torch.rand(num_samples, device=vertices.device)

    # warp to ensure uniformity on triangle
    sqrt_u = torch.sqrt(u)
    b0 = 1 - sqrt_u
    b1 = sqrt_u * (1 - v)
    b2 = sqrt_u * v
    # reshape for broadcasting
    b0 = b0.unsqueeze(1)  # (num_samples,1)
    b1 = b1.unsqueeze(1)
    b2 = b2.unsqueeze(1)

    # select the triangle’s vertices
    v0_sel = v0[idx]  # (num_samples,3)
    v1_sel = v1[idx]
    v2_sel = v2[idx]

    # 5) form the sampled points
    samples = b0 * v0_sel + b1 * v1_sel + b2 * v2_sel  # (num_samples,3)

    return samples

def sample_mesh_points_heitz(vertices: torch.Tensor,
                             faces: torch.LongTensor,
                             num_samples: int) -> torch.Tensor:
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
    e0 = v1 - v0               # (F,3)
    e1 = v2 - v0               # (F,3)
    cross = torch.cross(e0, e1, dim=1)  # (F,3)
    areas = 0.5 * cross.norm(dim=1)     # (F,)

    # 3) Sample faces proportional to area
    probs = areas / areas.sum()
    idx   = torch.multinomial(probs, num_samples, replacement=True)  # (num_samples,)

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
    samples = b0*v0s + b1*v1s + b2*v2s  # (num_samples,3)

    return samples

@njit
def _compute_normal(a, b, c):
    # cross( b−a, c−a )  
    ab = b - a
    ac = c - a
    # cross product
    return np.array((
        ab[1]*ac[2] - ab[2]*ac[1],
        ab[2]*ac[0] - ab[0]*ac[2],
        ab[0]*ac[1] - ab[1]*ac[0],
    ), dtype=np.float64)

@njit
def _normalize(v):
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return v / (norm + 1e-12)

@njit
def _angle(idx, vertices, ctr, normal, ref):
    p = vertices[idx]
    v = p - ctr
    # project into plane
    dot_nv = normal[0]*v[0] + normal[1]*v[1] + normal[2]*v[2]
    v = v - normal * dot_nv
    # compute angle = atan2(||ref×v||, ref·v)
    cr = np.empty(3, dtype=np.float64)
    cr[0] = ref[1]*v[2] - ref[2]*v[1]
    cr[1] = ref[2]*v[0] - ref[0]*v[2]
    cr[2] = ref[0]*v[1] - ref[1]*v[0]
    num = np.sqrt(cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2])
    den = ref[0]*v[0] + ref[1]*v[1] + ref[2]*v[2]
    ang = np.arctan2(num, den)
    # sign correction
    sign = (normal[0]*cr[0] + normal[1]*cr[1] + normal[2]*cr[2]) < 0
    return 2*np.pi - ang if sign else ang

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
    dot_nr = normal[0]*ref[0] + normal[1]*ref[1] + normal[2]*ref[2]
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
        while j > 0 and sorted_angs[j-1] > a_i:
            sorted_angs[j] = sorted_angs[j-1]
            sorted_idxs[j] = sorted_idxs[j-1]
            j -= 1
        sorted_angs[j]   = a_i
        sorted_idxs[j]   = idx_i
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
        a,b,c,d = simplex
        for u,v in ((a,b),(a,c),(a,d),(b,c),(b,d),(c,d)):
            key = (u,v) if u < v else (v,u)
            face_dict[key].append(si)

    # 2) now for each ridge (a,b) grab its list
    out = []
    for (a,b) in ridges:
        key = (a,b) if a < b else (b,a)
        lst = face_dict.get(key, [])
        out.append(np.array(lst, dtype=np.int32))
    return np.array(out, dtype=object)





# import torch
# from pytorch3d.ops import knn_points

# def approximate_sdf_and_gradient(
#     query_points: torch.Tensor,
#     target_points: torch.Tensor,
#     K: int = 6
# ):
#     """
#     Approximate an unsigned SDF and its gradient via k-nearest neighbors.

#     Args:
#         query_points:  Tensor of shape (N, 3) the points at which to approximate the SDF.
#         target_points: Tensor of shape (M, 3) the point cloud defining the surface.
#         K:             Number of nearest neighbours to use (default: 6).

#     Returns:
#         d_mean:  Tensor of shape (N,)    the mean distance to the K nearest neighbors.
#         grad:    Tensor of shape (N, 3)  the (unit averaged) gradient vector.
#     """
#     # Add batch dimension
#     P = query_points.unsqueeze(0)   # (1, N, 3)
#     Q = target_points#.unsqueeze(0)  # (1, M, 3)
    
#     # KNN search: returns squared distances, indices, and the actual neighbor coords
#     knn = knn_points(P, Q, K=K, return_nn=True)
#     # knn.dists: (1, N, K) squared distances
#     # knn.knn:   (1, N, K, 3) coords of the K neighbors

#     # Compute Euclidean distances: (1, N, K)
#     dists = torch.sqrt(knn.dists + 1e-12)

#     # 1) Mean distance per query point: (N,)
#     d_mean = dists.mean(dim=2).squeeze(0)

#     # 2) Compute per-neighbor unit vectors: (1, N, K, 3)
#     #    vector from query → neighbor = (knn - P[...,None,:])
#     vecs = knn.knn - P.unsqueeze(2)
#     #    normalize along last dim:
#     unit_vecs = vecs / (dists.unsqueeze(-1) + 1e-12)

#     # 3) Average those unit vectors: (1, N, 3), then normalize final gradient
#     g_raw = unit_vecs.mean(dim=2)        # (1, N, 3)
#     g_norm = torch.norm(g_raw, dim=2, keepdim=True)  # (1, N, 1)
#     grad  = (g_raw / (g_norm + 1e-12)).squeeze(0)     # (N, 3)

#     return d_mean, grad



def interpolate_sdf_of_vertices(
    vertices: torch.Tensor,        # (M, 3)  positions of Voronoi vertices
    tets:     torch.LongTensor,    # (M, 4)  indices of the 4 sites for each tet
    sites:    torch.Tensor,        # (N, 3)  xyz of all sites
    sdf:      torch.Tensor,        # (N,)    φ at each site
) -> torch.Tensor:
    # ------------------------------------------------------------
    # barycentric interpolation to Voronoi vertices (= tetra circum-centres)
    # ------------------------------------------------------------
    """
    Returns
    -------
    phi_v   : (M,)   SDF value at each Voronoi vertex
    """
    # --------------------------------------------------------
    # 1. Gather positions / field values for the tetra vertices
    # --------------------------------------------------------
    v_pos  = sites[tets]       # (M,4,3)
    v_phi  = sdf[tets]         # (M,4)

    # --------------------------------------------------------
    # 2. Barycentric coords of the vertex inside its tet
    #    Solve  (x - x0) = (x1-x0, x2-x0, x3-x0) · [w1 w2 w3]^T
    # --------------------------------------------------------
    x0   = v_pos[:, 0]                   # (M,3)
    D    = torch.stack(                 # (M,3,3)
        (v_pos[:, 1] - x0,
         v_pos[:, 2] - x0,
         v_pos[:, 3] - x0), dim=2)

    rhs  = (vertices - x0)              # (M,3)

    # batched solve: D · w123 = rhs   →  w123 = D⁻¹ rhs
    w123 = torch.linalg.solve(D, rhs.unsqueeze(-1)).squeeze(-1)   # (M,3)

    w0   = 1.0 - w123.sum(dim=1, keepdim=True)                    # (M,1)
    W    = torch.cat((w0, w123), dim=1)                           # (M,4)
    
    # Make sure weights are non-negative and not too large (> 1.0)
    W = torch.clamp(W, min=0.0, max=1.0)                          # (M,4)
    # Normalize weights to sum to 1.0
    W = W / (W.sum(dim=1, keepdim=True) + 1e-12)                  # (M,4)
    
    # --------------------------------------------------------
    # 3. Interpolate φ
    # --------------------------------------------------------
    phi_v  = (W * v_phi).sum(dim=1)                               # (M,)

    return phi_v

def analytic_grad_per_tet(sites, tets, sdf):
    """
    Compute the spatial gradient of a linearly interpolated scalar field in a tetrahedron.

    Args:
        sites: (N,3) Tensor of site positions
        tets:  (M,4) LongTensor of indices into `sites`
        sdf:   (N,)  Scalar values at sites

    Returns:
        grad_phi: (M,3) Spatial gradient of the scalar field per tetrahedron
    """
    v0 = sites[tets[:, 0]]  # (M,3)
    v1 = sites[tets[:, 1]]
    v2 = sites[tets[:, 2]]
    v3 = sites[tets[:, 3]]

    D = torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1)  # (M,3,3)
    
    # Compute inverse transpose of D manually using adjugate/det
    D_invT = torch.inverse(D).transpose(1, 2)  # (M,3,3) -- can replace with closed-form for perf

    phi0 = sdf[tets[:, 0]]
    phi1 = sdf[tets[:, 1]]
    phi2 = sdf[tets[:, 2]]
    phi3 = sdf[tets[:, 3]]

    dphi = torch.stack([phi1 - phi0, phi2 - phi0, phi3 - phi0], dim=-1).unsqueeze(-1)  # (M,3,1)

    grad_phi = torch.bmm(D_invT, dphi).squeeze(-1)  # (M,3)
    return grad_phi


def get_clipped_mesh_numba(sites, model, d3dsimplices, clip=True, sites_sdf=None, offset=None):
    """
    sites:           (N,3) torch tensor (requires_grad)
    model:           SDF model: sites -> (N,1) tensor of signed distances
    d3dsimplices:    torch.LongTensor of shape (M,4) from Delaunay
    """
    device = sites.device
    vertices_sdf = None
    vertices_sdf_grad = None
    if d3dsimplices is None:
        sites_np = sites.detach().cpu().numpy()
        d3dsimplices = diffvoronoi.get_delaunay_simplices(sites_np.reshape(sites_np.shape[1]*sites_np.shape[0]))
        d3dsimplices = np.array(d3dsimplices)
    
    d3d = torch.tensor(d3dsimplices).to(device).detach()            # (M,4)
    #print(f"Before compute vertices 3d vectorized: Allocated: {torch.cuda.memory_allocated() / 1e6} MB, Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

    # Compute per‐simplex circumcenters (Voronoi vertices)
    # if sdf_v is not None:
    #     vor_vertices, sdf_vertices, sdf_vertices_grads= compute_vertices_3d_vectorized_w_sdf_grads(sites, d3d, sdf_v, sdf_v_grads)
    
    vor_vertices = compute_vertices_3d_vectorized(sites, d3d)  # (M,3)
    if sites_sdf is not None:
        vertices_sdf = interpolate_sdf_of_vertices(vor_vertices, d3d, sites, sites_sdf)
        if offset is not None:
            offset = interpolate_sdf_of_vertices(vor_vertices, d3d, sites, offset)
        vertices_sdf_grad = analytic_grad_per_tet(sites, d3d, sites_sdf)

    with torch.no_grad():
        # Generate all edges of each simplex
        #    torch.combinations gives the 6 index‐pairs within a 4‐long row
        comb = torch.combinations(torch.arange(d3d.shape[1], device=device), r=2)  # (6,2)
        #print("comb", comb.shape)
        edges = d3d[:, comb]                    # (M,6,2)
        edges = edges.reshape(-1,2)             # (M*6,2)
        edges, _ = torch.sort(edges, dim=1)     # sort each row so (a,b) == (b,a)

        # Unique ridges across all simplices
        #ridges, inverse = torch.unique(edges, dim=0, return_inverse=True) # (R,2)
        ridges = torch.unique(edges, dim=0, return_inverse=False) # (R,2)
        
        del comb, edges
        torch.cuda.empty_cache()
        
        #print(ridges.dtype)
        #print(f"After ridge: Allocated: {torch.cuda.memory_allocated() / 1e6} MB, Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

        # Evaluate SDF at each site
        #print(sites.dtype)
        if model is not None:
            sdf = model(sites).detach().view(-1)             # (N,)
        else:
            sdf = sites_sdf        # (N,)
        #print(f"After sdf: Allocated: {torch.cuda.memory_allocated() / 1e6} MB, Reserved: {torch.cuda.memory_reserved() / 1e6} MB")
        
        sdf_i = sdf[ridges[:,0]]
        sdf_j = sdf[ridges[:,1]]
        zero_cross = (sdf_i * sdf_j <= 0)       # (R,)
        # Keep only the zero-crossing ridges
        ridges = ridges[zero_cross]             # (R0,2)
        filtered_ridges = ridges.detach().cpu().numpy()

        faces = faces_via_dict(d3dsimplices, filtered_ridges)  # (R0, List of simplices)

        # Sort faces
        torch.cuda.empty_cache()
        R = len(faces)
        counts = np.array([len(face) for face in faces], dtype=np.int64)
        Kmax = counts.max()
        faces_np = np.full((R, Kmax), -1, dtype=np.int64)
        
        for i, face in enumerate(faces):
            faces_np[i, :len(face)] = face

        sorted_faces_np = np.full((R, Kmax), -1, dtype=np.int64)

        #print("-> sorting faces")
        batch_sort_numba(vor_vertices.detach().cpu().numpy(), faces_np, counts, sorted_faces_np)
        faces_sorted = [sorted_faces_np[i, :counts[i]].tolist() for i in range(R)]
        faces = faces_sorted

    # Compact the vertex list
    #print("-> compacting vertices")
    used = {idx for face in faces for idx in face}
    old2new = {old: new for new, old in enumerate(sorted(used))}
    new_vertices = vor_vertices[sorted(used)]
    new_faces = [[old2new[i] for i in face] for face in faces]

    del counts, Kmax, faces_np, sorted_faces_np, faces_sorted, faces, old2new #used, 
    torch.cuda.empty_cache()



    if vertices_sdf is not None and vertices_sdf_grad is not None:
        sdf_verts = vertices_sdf[sorted(used)]
        grads = vertices_sdf_grad[sorted(used)]  # (M,3)
        if offset is not None:
            offset = offset[sorted(used)]
        else:
            offset = 0.0
    elif model is not None:
        # clip the vertices of the faces to the zero-crossing of the sdf
        sdf_verts = model(new_vertices).view(-1)           # (M,)
        # compute gradients ∇f(v)  — note create_graph=True if you
        #    want second-order gradients to flow back into the model
        grads = torch.autograd.grad(
            outputs=sdf_verts,
            inputs=new_vertices,
            grad_outputs=torch.ones_like(sdf_verts),
            create_graph=True,
        )[0]                                               # (M,3)
    else:
        return new_vertices, new_faces

    if not clip:
        #print("-> not clipping")
        return new_vertices, new_faces, sdf_verts, grads

    # one Newton step https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    epsilon = 1e-12
    grad_norm2 = torch.sqrt(((grads + epsilon)**2).sum(dim=1, keepdim=True))    # (M,1)
    step = (sdf_verts + offset).unsqueeze(1) * grads / (grad_norm2 + epsilon)
    
    proj_vertices = new_vertices - step    

    #print("-> vertices:", new_vertices.shape)
    #print("-> projected vertices:", proj_vertices.shape)
    #print("-> #faces:", len(new_faces))
    return proj_vertices, new_faces, sdf_verts, grads