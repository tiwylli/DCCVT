import torch
from scipy.spatial import Voronoi, Delaunay
import numpy as np
import math
import sdfpred_utils.sdfpred_utils as su

from pytorch3d.ops import knn_points, knn_gather
import torch
from torch import nn
import pygdel3d

device = torch.device("cuda:0")


# def compute_edge_smoothing_loss(edges, sites, model):
#     """
#     Computes the loss to smooth edges by minimizing the dot product between the
#     edge orientation and the gradient of the SDF at the midpoint of the edge, without a loop.

#     Args:
#         edges: Tensor of edges, each defined as [v1_idx, v2_idx, site1_idx, site2_idx].
#         sites: Tensor of site positions.
#         model

#     Returns:
#         smoothing_loss: The computed edge smoothing loss.
#     """
#     # Extract indices for vertices and sites
#     v1_idx, v2_idx, site1_idx, site2_idx = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]

#     # Extract positions of site1 and site2
#     site1 = sites[site1_idx]  # Shape: (M, D)
#     site2 = sites[site2_idx]  # Shape: (M, D)

#     # Compute site direction and edge orientation
#     site_direction = site2 - site1  # Shape: (M, D)
#     site_direction = site_direction / torch.norm(site_direction, dim=1, keepdim=True)  # Normalize

#     # Perpendicular orientation (2D case)
#     edge_orientation = torch.stack([-site_direction[:, 1], site_direction[:, 0]], dim=1)  # Shape: (M, 2)

#     # Compute midpoints of edges
#     midpoints = (site1 + site2) / 2.0  # Shape: (M, D)
#     midpoints.requires_grad_(True)  # Enable gradient tracking for midpoints

#     # Compute SDF values at midpoints
#     sdf_values = model(midpoints)[:,0]

#     # Compute SDF gradients at midpoints
#     torch.autograd.set_detect_anomaly(True)
#     gradients_sdf = torch.autograd.grad(sdf_values, midpoints, grad_outputs=torch.ones_like(sdf_values), create_graph=True)[0]  # Shape: (M, D)

#     # Dot product between edge orientation and SDF gradient
#     dot_products = torch.sum(edge_orientation * gradients_sdf, dim=1)  # Shape: (M,)

#     # Compute smoothing loss
#     smoothing_loss = torch.mean(dot_products**2)  # Scalar

#     return smoothing_loss


def compute_edge_smoothing_loss(bisectors_to_compute, sites, model):
    """
    Computes the loss to smooth edges by minimizing the dot product between the
    edge orientation and the gradient of the SDF at the midpoint of the edge, without a loop.

    Args:
        bisectors_to_compute: Tensor of sites index pairs, each defined as [site1_idx, site2_idx].
        sites: Tensor of site positions.
        model

    Returns:
        smoothing_loss: The computed edge smoothing loss.
    """
    # Extract positions of site1 and site2
    site1 = sites[bisectors_to_compute[:, 0]]  # Shape: (M, D)
    site2 = sites[bisectors_to_compute[:, 1]]  # Shape: (M, D)

    # Compute site direction and edge orientation
    site_direction = site2 - site1  # Shape: (M, D)
    site_direction = site_direction / torch.norm(site_direction, dim=1, keepdim=True)  # Normalize

    # Perpendicular orientation (2D case)
    edge_orientation = torch.stack([-site_direction[:, 1], site_direction[:, 0]], dim=1)  # Shape: (M, 2)

    # Compute midpoints of edges
    midpoints = (site1 + site2) / 2.0  # Shape: (M, D)
    midpoints.requires_grad_(True)  # Enable gradient tracking for midpoints

    # Compute SDF values at midpoints
    sdf_values = model(midpoints)[:, 0]

    # Compute SDF gradients at midpoints
    torch.autograd.set_detect_anomaly(True)
    gradients_sdf = torch.autograd.grad(
        sdf_values, midpoints, grad_outputs=torch.ones_like(sdf_values), create_graph=True
    )[0]  # Shape: (M, D)

    # Dot product between edge orientation and SDF gradient
    dot_products = torch.sum(edge_orientation * gradients_sdf, dim=1)  # Shape: (M,)

    # Compute smoothing loss
    smoothing_loss = torch.mean(dot_products**2)  # Scalar

    return smoothing_loss


def compute_ridge_smoothing_loss(ridge_points, sites, model):
    """
    Computes a smoothing loss for Voronoi ridges by minimizing the dot product between
    the ridge normal and the gradient of the SDF at the midpoint of the ridge.

    Args:
        ridge_points: Tensor of site index pairs, each defined as [site1_idx, site2_idx].
        sites: Tensor of site positions.
        model: SDF model.

    Returns:
        smoothing_loss: The computed ridge smoothing loss.
    """
    site1 = sites[ridge_points[:, 0]]  # Shape: (M, 3)
    site2 = sites[ridge_points[:, 1]]  # Shape: (M, 3)

    # ridge_direction = site2 - site1  # Shape: (M, 3)
    # ridge_direction = ridge_direction / torch.norm(ridge_direction, dim=1, keepdim=True)  # Normalize

    midpoints = (site1 + site2) / 2.0  # Shape: (M, 3)
    midpoints.requires_grad_(True)  # Enable gradient tracking

    sdf1 = model.sdf(site1)  # Shape: (M,)
    sdf2 = model.sdf(site2)  # Shape: (M,)

    estimated_normals = (sdf2 - sdf1).unsqueeze(1) * (site2 - site1)  # Shape: (M, 3)
    estimated_normals = estimated_normals / torch.norm(estimated_normals, dim=1, keepdim=True)  # Normalize

    sdf_values = model.sdf(midpoints)

    gradients_sdf = torch.autograd.grad(
        sdf_values, midpoints, grad_outputs=torch.ones_like(sdf_values), create_graph=True
    )[0]  # Shape: (M, 3)

    dot_products = torch.sum(estimated_normals * gradients_sdf, dim=1)  # Shape: (M,)

    smoothing_loss = torch.mean(dot_products**2)  # Scalar

    return smoothing_loss


# Todo vectorize
def compute_cvt_loss(sites):
    # Convert sites to NumPy for Voronoi computation
    sites_np = sites.detach().cpu().numpy()
    vor = Voronoi(sites_np)

    centroids = []
    valid_indices = []

    for i in range(len(sites_np)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]

        # Ensure the region is valid (finite and non-empty)
        if region and -1 not in region:
            vertices = vor.vertices[region]
            centroid = vertices.mean(axis=0)  # Compute centroid
            centroids.append(centroid)
            valid_indices.append(i)  # Store indices of valid centroids

    if len(centroids) == 0:
        return torch.tensor(0.0, device=sites.device)  # Return zero loss if no valid centroids

    # Convert centroids to a PyTorch tensor
    centroids = torch.tensor(np.array(centroids), device=sites.device, dtype=sites.dtype)

    # Select only valid sites for loss computation
    valid_sites = sites[valid_indices]

    # Compute Mean Squared Error (MSE) loss
    # cvt_loss = torch.mean(torch.norm(valid_sites - centroids, p=2, dim=1) ** 2)
    penalties = torch.where(
        abs(valid_sites - centroids) < 10, valid_sites - centroids, torch.tensor(0.0, device=sites.device)
    )
    cvt_loss = torch.mean(penalties**2)

    return cvt_loss


def compute_cvt_loss_vectorized_voronoi(sites, vor=None, model=None):
    # Convert sites to NumPy for Voronoi computation
    # sdf_values = model(sites)
    sites_np = sites.detach().cpu().numpy()
    if vor is None:
        vor = Voronoi(sites_np)

    # Todo C++ loop for this
    # create a nested list of vertices for each site
    centroids = [
        vor.vertices[vor.regions[vor.point_region[i]]].mean(axis=0)
        for i in range(len(sites_np))
        if vor.regions[vor.point_region[i]] and -1 not in vor.regions[vor.point_region[i]]
    ]
    centroids = torch.tensor(np.array(centroids), device=sites.device, dtype=sites.dtype)
    valid_indices = torch.tensor(
        [
            i
            for i in range(len(sites_np))
            if vor.regions[vor.point_region[i]] and -1 not in vor.regions[vor.point_region[i]]
        ],
        device=sites.device,
    )
    valid_sites = sites[valid_indices]
    # sdf_weights = 1 / (1 + torch.abs(sdf_values[valid_indices]))
    penalties = torch.where(
        abs(valid_sites - centroids) < 10, valid_sites - centroids, torch.tensor(0.0, device=sites.device)
    )
    # cvt_loss = torch.mean(((penalties)*sdf_weights)**2)
    cvt_loss = torch.mean(penalties**2)
    return cvt_loss


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

# WRONG: Need to compute the tetrahedra based on the vertices, not the sites
# def compute_voronoi_cell_centers_index_based_torch_volume(points, delau, simplices=None):
#     """Compute Voronoi cell centers (circumcenters) for 2D or 3D Delaunay triangulation in PyTorch."""
#     # simplices = torch.tensor(delaunay.simplices, dtype=torch.long)
#     if simplices is None:
#         simplices = delau.simplices

#     # Compute the volume of each simplex (triangle or tetrahedron)
#     simplices = torch.tensor(simplices, dtype=torch.long, device=points.device)
#     tetrahedra_points = points[simplices]  # Shape: (M, 4, 3) for tetrahedra
#     if tetrahedra_points.shape[1] == 3:  # 2D case (triangles)
#         a = tetrahedra_points[:, 0]
#         b = tetrahedra_points[:, 1]
#         c = tetrahedra_points[:, 2]
#         area = 0.5 * torch.abs(
#             a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1])
#         )
#         volumes = area  # Shape: (M,)
#         centroid = (a + b + c) / 3.0  # Shape: (M, 3)
#     elif tetrahedra_points.shape[1] == 4:  # 3D case (tetrahedra)
#         a = tetrahedra_points[:, 0]
#         b = tetrahedra_points[:, 1]
#         c = tetrahedra_points[:, 2]
#         d = tetrahedra_points[:, 3]
#         volumes = su.volume_tetrahedron(a, b, c, d)  # Shape: (M,)
#         centroid = (a + b + c + d) / 4.0  # Shape: (M, 3)

#     M = len(points)
#     site_centroids = torch.zeros(M, points.shape[1], dtype=torch.float32, device=points.device)
#     site_volume = torch.zeros(M, device=points.device)

#     for i in range(simplices.shape[1]):
#         indices = simplices[:, i]  # Indices of the i-th vertex in each simplex
#         site_centroids.index_add_(0, indices, centroid * volumes.unsqueeze(1))
#         site_volume.index_add_(0, indices, volumes)

#     centroids = site_centroids / site_volume.clamp(min=1).unsqueeze(1)  # Avoid division by zero

#     return centroids


def compute_voronoi_cell_centers_index_based_torch(sites, delau, simplices=None):
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
    centroids = torch.zeros(M, points.shape[1], dtype=torch.float32, device=sites.device)
    counts = torch.zeros(M, device=centers.device)

    centroids.index_add_(0, indices, centers)  # Sum centers per unique index
    counts.index_add_(0, indices, torch.ones(centers.shape[0], device=centers.device))  # Count occurrences
    centroids /= counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    distances = torch.norm(centroids[indices] - centers, dim=1)
    num_sites = centroids.shape[0]
    max_dist_per_site = torch.full((num_sites,), float("-inf"), device=sites.device)
    radius = max_dist_per_site.scatter_reduce(0, indices, distances, reduce="amax", include_self=True)

    return centroids, radius

def intersection_plane_line(plane_ori, plane_dir, sites, directions, dimension):
    num_sites = sites.shape[0]
    num_neighbors = plane_ori.shape[1]
    num_dirs = directions.shape[0]

    # Here move to the same direction for all the planes and sites
    # (num_sites, num_neighbors, num_dirs, dimension)
    directions_repeat = directions.view(1, 1, num_dirs, dimension).expand(num_sites, num_neighbors, num_dirs, dimension)
    plane_dir_repeat = plane_dir.unsqueeze(2).expand(-1, -1, num_dirs, -1)
    plane_ori_repeat = plane_ori.unsqueeze(2).expand(-1, -1, num_dirs, -1)
    # All rays are originating from the sites (batched) and for all directions
    ray_origin_repeat = sites.view(num_sites, 1, 1, dimension).expand(-1, num_neighbors, num_dirs, -1)

    #### Plane intersection with rays ####
    # (n . d)
    denom = torch.sum(plane_dir_repeat * directions_repeat, dim=-1)
    parallel_mask = denom.abs() < 1e-8 # avoid parallel rays (unstable)
    # (n . (p - o))
    numer = torch.sum(plane_dir_repeat * (plane_ori_repeat - ray_origin_repeat), dim=-1)
    # Compute the intersection distance
    t = torch.full_like(numer, float('inf'))
    t[~parallel_mask] = numer[~parallel_mask] / (denom[~parallel_mask] + 1e-8)
    valid = (t >= 0) & (~parallel_mask) # Filter valid intersections
    t[~valid] = float('inf')

    # Compute intersection points
    valid_mask = t.isfinite() # will be false for all invalid intersection (marked as infinit)
    valid_t = torch.where(valid_mask, t, torch.zeros_like(t)) # replace invalid t with 0 to avoid NaN in autodiff
    intersection_points = ray_origin_repeat + valid_t.unsqueeze(-1) * directions_repeat

    # intersection points (ray starts from site) -- only retain valid intersections
    distances = torch.full_like(t, float('inf'))
    distances[valid_mask] = torch.norm(intersection_points[valid_mask] - ray_origin_repeat[valid_mask], dim=-1)

    # Find the minimum distance for each site to all directions
    # here 1 is because the distance vector is (num_sites, num_neighbors, num_dirs)
    min_distances, _ = torch.min(distances, dim=1) 
    return min_distances

def compute_cvt_dist(sites, N=12, M=16, random=True, max_distance=0.1, dimension=2):
    device = sites.device

    # Get the N closest sites to each site (excluding self)
    knn_indices = knn_points(sites.unsqueeze(0), sites.unsqueeze(0), K=N+1).idx.squeeze(0)[:, 1:] 
    knn_sites = knn_gather(sites.unsqueeze(0), knn_indices.unsqueeze(0)).squeeze(0)

    # Prepare planes (midpoints between site and neighbors)
    # It will be N planes for each site
    # (num_sites, N, 2)
    sites_repeat = sites.unsqueeze(1).expand(-1, N, -1) 
    plane_ori = (knn_sites + sites_repeat) * 0.5       
    plane_dir = knn_sites - plane_ori
    plane_dir = plane_dir / (torch.norm(plane_dir, dim=-1, keepdim=True) + 1e-8)

    # Generate M directions (half circle + opposite)
    if dimension == 2:
        angles = torch.linspace(0, math.pi, M, device=device) # half-circle
        if random:
            angles += torch.rand(1, device=device) * (math.pi / M)
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)  # (M, 2)
        directions_opp = -directions.clone()  # (M, 2)
    elif dimension == 3:
        # Fibonnaci sphere sampling for 3D directions
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        indices = torch.arange(M, device=device)
        theta = 2 * math.pi * indices / phi  # Azimuthal angle
        y = 1 - (indices / (M - 1)) * 2 # y goes from 1 to -1
        radius = torch.sqrt(1 - y**2)  # Radius in the xy-plane
        directions = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), y), dim=1)

        # Randomly pertub the directions with random rotation XYX
        if random:
            angles = torch.rand(3, device=device) * (math.pi / M)
            # Compute rotation matrix for random angles XYX
            rot_X = torch.tensor([[1, 0, 0],
                                  [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                                  [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device)
            rot_Y = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                                  [0, 1, 0],
                                  [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device)
            rot_X2 = torch.tensor([[1, 0, 0],
                                   [0, torch.cos(angles[2]), -torch.sin(angles[2])],
                                   [0, torch.sin(angles[2]), torch.cos(angles[2])]], device=device)
            rotation_matrix = torch.matmul(rot_X, torch.matmul(rot_Y, rot_X2))
            directions = torch.matmul(directions.unsqueeze(1), rotation_matrix).squeeze(1)
        directions_opp = -directions.clone()  # (M, 3)
    else:
        raise ValueError("Only 2D and 3D dimensions are supported.")

    # Forward and backward intersection distances
    min_dir = intersection_plane_line(plane_ori, plane_dir, sites, directions, dimension)       # (num_sites, M)
    min_dir_opp = intersection_plane_line(plane_ori, plane_dir, sites, directions_opp, dimension)

    # Compute sample positions (intersection points from forward rays)
    ray_points = sites.unsqueeze(1) + min_dir.unsqueeze(-1) * directions.unsqueeze(0)  # (num_sites, M, 2)
    ray_points_opp = sites.unsqueeze(1) + min_dir_opp.unsqueeze(-1) * directions_opp.unsqueeze(0)  # (num_sites, M, 2)
    
    # Keep only directions that intersect in both directions
    mask = (min_dir < max_distance) & (min_dir_opp < max_distance)

    # Return MSE between opposite directions and points
    mse = torch.mean(torch.abs(min_dir[mask] - min_dir_opp[mask]))
    return mse, ray_points[mask], ray_points_opp[mask]

# Compute CVT loss
def compute_cvt_loss_true(sites, d3d):
    vertices = su.compute_vertices_3d_vectorized(sites, d3d)
    
    # Concat sites and vertices to compute the Voronoi diagram
    points = torch.concatenate((sites, vertices), axis=0)
    # Avoid to get coplanar tet which create issue if the current algorithm
    points += (torch.rand_like(points) - 0.5) * 0.00001 # 0.001 % of the space ish 
    d3dsimplices, _ = pygdel3d.triangulate(points.detach().cpu().numpy())
    # d3dsimplices = Delaunay(points.detach().cpu().numpy()).simplices
    d3dsimplices = torch.tensor(d3dsimplices, dtype=torch.int64, device=sites.device)
   
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
    
    tetrahedra_volume = su.volume_tetrahedron(a, b, c, d)
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
    

def compute_cvt_loss_vectorized_delaunay(sites, delaunay, simplices=None):
    #TODO: Not a L1 but norm here
    centroids, _ = compute_voronoi_cell_centers_index_based_torch(sites, delaunay, simplices)
    centroids = centroids.to(device)
    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(abs(diff) < 0.1, diff, torch.tensor(0.0, device=sites.device))
    # cvt_loss = torch.mean(penalties**2)
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss

def compute_cvt_loss_vectorized_delaunay_tetrahedra(sites, delaunay, simplices=None,max_distance=0.1):
    # Compute all tetrahedra centroid 
    simplices = torch.tensor(simplices, dtype=torch.long, device=sites.device)
    tetrahedra_points = sites[simplices]  # Shape: (M, 4, 3) for tetrahedra
    if tetrahedra_points.shape[1] == 3:  # 2D case (triangles)
        a = tetrahedra_points[:, 0]
        b = tetrahedra_points[:, 1]
        c = tetrahedra_points[:, 2]
        area = 0.5 * torch.abs(
            a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1])
        )
        centroid = (a + b + c) / 3.0  # Shape: (M, 3)
    elif tetrahedra_points.shape[1] == 4:  # 3D case (tetrahedra)
        a = tetrahedra_points[:, 0]
        b = tetrahedra_points[:, 1]
        c = tetrahedra_points[:, 2]
        d = tetrahedra_points[:, 3]
        centroid = (a + b + c + d) / 4.0  # Shape: (M, 3)

    # Compute squared norms of each point
    squared_norms = (tetrahedra_points**2).sum(dim=2, keepdim=True)  # Shape: (M, 4, 1)

    # Construct the 4x4 matrices in batch
    ones_col = torch.ones_like(squared_norms)  # Column of ones for homogeneous coordinates

    A = torch.cat([tetrahedra_points, ones_col], dim=2)  # Shape: (M, 4, 4)
    Dx = torch.cat([squared_norms, tetrahedra_points[:, :, 1:], ones_col], dim=2)
    Dy = torch.cat([tetrahedra_points[:, :, :1], squared_norms, tetrahedra_points[:, :, 2:], ones_col], dim=2)
    Dz = torch.cat([tetrahedra_points[:, :, :2], squared_norms, ones_col], dim=2)

    # Compute determinants in batch
    detA = torch.linalg.det(A)  # Shape: (M,)
    detDx = torch.linalg.det(Dx)
    detDy = torch.linalg.det(Dy)  # todo, removed Negative due to orientation
    # detDz = torch.linalg.det(Dz)

    # Compute circumcenters
    circumcenters = 0.5 * torch.stack([detDx / detA, detDy / detA], dim=1)

    penalties = torch.where(abs(centroid - circumcenters) < max_distance, centroid - circumcenters, torch.tensor(0.0, device=sites.device))
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


def compute_cvt_loss_CLIPPED_vertices(sites, sites_sdf, sites_sdf_grad, d3dsimplices, all_vor_vertices):
    d3dsimplices = torch.tensor(d3dsimplices, device=sites.device).detach()
    # all_vor_vertices = su.compute_vertices_3d_vectorized(sites, d3dsimplices)  # (M,3)
    # vertices_to_compute, _, used_tet = su.compute_zero_crossing_vertices_3d(
    #     sites, None, None, d3dsimplices.cpu().numpy(), sites_sdf
    # )
    # vertices = su.compute_vertices_3d_vectorized(sites, vertices_to_compute)
    # clipped, _ = su.tet_plane_clipping(d3dsimplices[used_tet], sites, sites_sdf, sites_sdf_grad, vertices)

    # # replace at used_tet index the vertices with the clipped ones
    # all_vor_vertices[used_tet] = clipped

    # compute centroids
    indices = d3dsimplices.flatten()  # Flatten simplex indices
    centers = all_vor_vertices.repeat_interleave(d3dsimplices.shape[1], dim=0).to(sites.device)
    M = len(sites)
    centroids = torch.zeros(M, 3, dtype=torch.float32, device=sites.device)
    counts = torch.zeros(M, device=sites.device)

    centroids.index_add_(0, indices, centers)  # Sum centers per unique index
    counts.index_add_(0, indices, torch.ones(centers.shape[0], device=centers.device))  # Count occurrences
    centroids /= counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    diff = torch.linalg.norm(sites - centroids, dim=1)
    penalties = torch.where(abs(diff) < 0.5, diff, torch.tensor(0.0, device=sites.device))
    # print number of zero in penalties
    # print("Number of zero in penalties: ", torch.sum(penalties == 0.0).item())
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


# def compute_cvt_loss_vectorized_delaunay_volume(sites, delaunay, simplices=None):
#     centroids, radius = compute_voronoi_cell_centers_index_based_torch(sites, delaunay, simplices)
#     centroids = centroids.to(device)
#     radius = radius.to(device)

#     cell_v_approx = ((4.0 / 3.0) * math.pi * radius**3).to(device)
#     diff = torch.linalg.norm(sites - centroids, dim=1)
#     penalties = torch.where(abs(diff) < 0.1, diff * cell_v_approx, torch.tensor(0.0, device=sites.device))
#     # cvt_loss = torch.mean(torch.abs(penalties))
#     cvt_loss = torch.mean(torch.abs(penalties))
#     return cvt_loss


def sdf_weighted_min_distance_loss(model, sites):
    """
    Computes a minimum distance regularization loss for sites, weighted by the absolute SDF values.

    Args:
        model: A neural network that predicts SDF values.
        sites (torch.Tensor): Tensor of shape (N, D), where N is the number of sites and D is the spatial dimension.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    # Get SDF predictions and take absolute values for weighting
    sdf_values = model(sites).squeeze()  # Assuming model outputs shape (N, 1) or (N,)
    sdf_weights = 1 / (1 + torch.abs(sdf_values))  # Inverse weighting (closer sites affect more)

    # Compute pairwise distances between all sites
    distances = torch.cdist(sites, sites, p=2)  # Shape: (N, N)

    # Mask diagonal to ignore self-distances
    mask = torch.eye(distances.size(0), device=distances.device, dtype=torch.bool)
    distances = distances.masked_fill(mask, float("inf"))

    # Find the minimum distance for each site to another site
    min_distances = distances.min(dim=1).values  # Shape: (N,)

    # Compute the average minimum distance
    avg_min_distance = min_distances.mean()

    # Compute penalties based on deviation from the average
    penalties = min_distances - avg_min_distance

    # Apply absolute SDF values as weighting factors
    weighted_penalties = sdf_weights * (penalties**2)

    # Compute final loss
    regularization_loss = weighted_penalties.mean()

    return regularization_loss


def chamfer_distance(true_point_cloud, vertices):
    true_point_cloud = true_point_cloud.half().detach()
    vertices = vertices.half()
    # Compute pairwise distances
    # From point cloud to mesh edge points
    dist1 = torch.cdist(true_point_cloud, vertices).min(dim=1)[0]
    # From mesh edge points to point cloud
    dist2 = torch.cdist(vertices, true_point_cloud).min(dim=1)[0]
    # Chamfer distance is the sum (or average) of these distances
    chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
    return chamfer_dist


def eikonal(model, input_dimensions, p=[]):
    if len(p) == 0:
        # Generate random points in the 2D plane (x, y)
        # p = torch.rand((128**input_dimensions, input_dimensions), device=device, requires_grad=True) - 0.5
        p = torch.rand((100000, input_dimensions), device=device, requires_grad=True) - 0.5
        p = p * 20

        # Todo: experimental
        # instead of p, i want a uniform grid of points in 3d
        if input_dimensions == 3:
            p = torch.linspace(-10, 10, 50)
            p = torch.meshgrid(p, p, p)
            p = torch.stack((p[0].flatten(), p[1].flatten(), p[2].flatten()), dim=1)
            p = p.to(device)
            p.requires_grad = True

    # Compute gradients for Eikonal loss
    grads = torch.autograd.grad(
        outputs=model(p)[:, 0],  # Network output
        inputs=p,  # Input coordinates
        grad_outputs=torch.ones_like(model(p)[:, 0]),  # Gradient w.r.t. output
        create_graph=True,
        retain_graph=True,
    )[0]

    # Eikonal loss: Enforce gradient norm to be 1
    eikonal_loss = ((grads.norm(2, dim=1) - 1).abs()).mean()

    return eikonal_loss


def domain_restriction_box(target_point_cloud, model, num_points=500, buffer_scale=0.2):
    min_x, min_y = target_point_cloud[:, 0].min().item(), target_point_cloud[:, 1].min().item()
    max_x, max_y = target_point_cloud[:, 0].max().item(), target_point_cloud[:, 1].max().item()

    # Calculate the width and height of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # Define buffer zones based on the scale
    buffer_x = buffer_scale * bbox_width
    buffer_y = buffer_scale * bbox_height

    num_per_region = num_points // 4

    # Left of the bounding box
    left_x = torch.empty(num_per_region).uniform_(min_x - 2 * buffer_x, min_x)
    left_y = torch.empty(num_per_region).uniform_(min_y - buffer_y, max_y + buffer_y)

    # Right of the bounding box
    right_x = torch.empty(num_per_region).uniform_(max_x, max_x + 2 * buffer_x)
    right_y = torch.empty(num_per_region).uniform_(min_y - buffer_y, max_y + buffer_y)

    # Above the bounding box
    top_x = torch.empty(num_per_region).uniform_(min_x - buffer_x, max_x + buffer_x)
    top_y = torch.empty(num_per_region).uniform_(max_y, max_y + 2 * buffer_y)

    # Below the bounding box
    bottom_x = torch.empty(num_per_region).uniform_(min_x - buffer_x, max_x + buffer_x)
    bottom_y = torch.empty(num_per_region).uniform_(min_y - 2 * buffer_y, min_y)

    # Combine all points
    points_x = torch.cat([left_x, right_x, top_x, bottom_x])
    points_y = torch.cat([left_y, right_y, top_y, bottom_y])

    # Stack into (num_points, 2)
    points = torch.stack([points_x, points_y], dim=1)

    # Compute the SDF values and loss
    sdf_values = model(points)[:, 0]

    domain_loss = torch.relu(-sdf_values).mean()

    return domain_loss


def domain_restriction_sphere(target_point_cloud, model, buffer_scale=0.2, input_dim=2, num_shells=10):
    assert input_dim in [2, 3], "input_dim must be either 2 or 3"

    # Compute the centroid of the point cloud
    centroid = target_point_cloud.mean(dim=0)

    # Compute the maximum distance from the centroid (bounding sphere radius)
    distances = torch.norm(target_point_cloud - centroid, dim=1)
    max_radius = distances.max().item()

    # Expand the sphere slightly with a buffer
    sphere_radius = max_radius * (1 + buffer_scale)

    # Define multiple shells with increasing radii
    shell_radii = torch.linspace(sphere_radius, sphere_radius * 10, num_shells)

    # Generate random points on each shell
    points = []
    num_per_shell = 100 * input_dim

    for radius in shell_radii:
        # Sample random directions
        rand_dirs = torch.randn((num_per_shell, input_dim), device=device)
        rand_dirs = rand_dirs / torch.norm(rand_dirs, dim=1, keepdim=True)  # Normalize to unit vectors

        # Scale by shell radius
        shell_points = centroid[:input_dim] + rand_dirs * radius
        points.append(shell_points)

    # Combine all sampled points
    points = torch.cat(points, dim=0)

    # Compute the SDF values and apply domain loss
    sdf_values = model(points)[:, 0]
    domain_loss = torch.relu(-sdf_values).mean()

    return domain_loss


def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = 0.5 * torch.autograd.grad(dot_grad, points, hvp, retain_graph=True, create_graph=True)[0]
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads**2, dim=-1) + 1e-5)
    return div


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type="abs"):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == "abs":
        if len(all_grads.shape) == 3:
            eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
        else:
            eikonal_term = ((all_grads.norm(2, dim=1) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term


# suggestion christian doesnt work with current implementation
def point_cloud_loss(points, model):
    batch_size = 2**15

    # Compute the minimal distance between batch and the mesh sampled points
    x_rand = torch.rand([batch_size, 3], device=device, dtype=torch.float64)
    x_rand_batch = torch.zeros((batch_size), device=device, dtype=torch.int64)
    points_batch = torch.zeros((points.shape[0]), device=device, dtype=torch.int64)
    indices_points = pc.nearest(x_rand, points, x_rand_batch, points_batch)
    target = torch.sqrt((x_rand - points[indices_points]) ** 2).sum(dim=1, keepdim=True)
    output = model(x_rand.detach())
    relative_l2_error = (output - target.to(output.dtype)) ** 2  # / (output.detach()**2 + 0.01)
    loss = relative_l2_error.mean()
    return loss


def update_div_weight(current_iteration, n_iterations, lambda_div, divdecay="linear", params=[]):
    # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
    # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
    #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
    #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

    # self.weights = weights #sdf, intern, normal, eikonal, div

    assert len(params) >= 2, params
    assert len(params[1:-1]) % 2 == 0
    decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

    curr = current_iteration / n_iterations
    we, e = min([tup for tup in decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
    w0, s = max([tup for tup in decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

    # Divergence term anealing functions
    if divdecay == "linear":  # linearly decrease weight from iter s to iter e
        if current_iteration < s * n_iterations:
            lambda_div = w0
        elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
            lambda_div = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
        else:
            lambda_div = we
    elif divdecay == "quintic":  # linearly decrease weight from iter s to iter e
        if current_iteration < s * n_iterations:
            lambda_div = w0
        elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
            lambda_div = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
        else:
            lambda_div = we
    elif divdecay == "step":  # change weight at s
        if current_iteration < s * n_iterations:
            lambda_div = w0
        else:
            lambda_div = we
    elif divdecay == "none":
        pass
    else:
        raise Warning("unsupported div decay value")
    return lambda_div


def discrete_tet_volume_eikonal_loss(sites, sites_sdf_grad, tets: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularization loss.

    Args:
        sites_sdf_grad: Tensor of shape (N, 3) containing ∇φ at each site.
        variant: 'a' for E1a: ½ mean((||∇φ|| - 1)²)
    Returns:
        A scalar tensor containing the eikonal loss.
    """
    grad_a = sites_sdf_grad[tets[:, 0]]  # (M,3)
    grad_b = sites_sdf_grad[tets[:, 1]]  # (M,3)
    grad_c = sites_sdf_grad[tets[:, 2]]  # (M,3)
    grad_d = sites_sdf_grad[tets[:, 3]]  # (M,3)

    grad_a_error = ((grad_a**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_b_error = ((grad_b**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_c_error = ((grad_c**2).sum(dim=-1) - 1) ** 2  # (M,)
    grad_d_error = ((grad_d**2).sum(dim=-1) - 1) ** 2  # (M,)

    # grad_avg = (grad_a + grad_b + grad_c + grad_d) / 4.0
    # grad_norm2 = (grad_avg**2).sum(dim=-1)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]

    volume = su.volume_tetrahedron(a, b, c, d)

    # loss = 0.5 * torch.mean(volume * (grad_norm2 - 1) ** 2)
    loss = 0.5 * torch.mean(volume * (grad_a_error + grad_b_error + grad_c_error + grad_d_error))  # (M,)

    return loss


def tet_sdf_grad_eikonal_loss(sites, tet_sdf_grad, tets: torch.Tensor) -> torch.Tensor:
    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]

    volume = su.volume_tetrahedron(a, b, c, d)
    # trim 5% biggest volumes
    volume = torch.where(volume > torch.quantile(volume, 0.95), torch.tensor(0.0, device=sites.device), volume)

    grad_norm2 = (tet_sdf_grad**2).sum(dim=1)  # (M,)
    # loss = 0.5 * torch.mean(volume * (grad_norm2 - 1) ** 2)  # (M,)
    loss = 0.5 * torch.mean(volume * (grad_norm2 - 1) ** 2)  # (M,)
    # loss = 0.5 * torch.mean((grad_norm2 - 1) ** 2)  # (M,)

    return loss


def smoothed_heaviside(phi, eps_H):
    H = torch.zeros_like(phi)
    mask1 = phi < -eps_H
    mask2 = phi > eps_H
    mask3 = (~mask1) & (~mask2)
    phi_clip = phi[mask3]
    H[mask1] = 0
    H[mask2] = 1
    H[mask3] = 0.5 + phi_clip / (2 * eps_H) + (1 / (2 * np.pi)) * torch.sin(np.pi * phi_clip / eps_H)
    return H


def estimate_eps_H(sites, tets, multiplier=1.5):
    # Get all unique edges
    comb = torch.combinations(torch.arange(4), r=2)  # (6,2)
    edges = tets[:, comb]  # (M, 6, 2)
    edges = edges.reshape(-1, 2)  # (6M, 2)

    v0 = sites[edges[:, 0]]
    v1 = sites[edges[:, 1]]
    edge_lengths = torch.norm(v0 - v1, dim=1)

    # remove top 5% longest edges
    edge_lengths = torch.where(
        edge_lengths > torch.quantile(edge_lengths, 0.95), torch.tensor(0.0, device=sites.device), edge_lengths
    )

    avg_len = edge_lengths.mean()
    return multiplier * avg_len


def tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, tets, eps_H) -> torch.Tensor:
    if eps_H is None:
        eps_H = estimate_eps_H(sites, tets)  # adaptive bandwidth
    sdf_H = smoothed_heaviside(sites_sdf, eps_H)  # (M,)
    sdf_H_a = sdf_H[tets[:, 0]]
    sdf_H_b = sdf_H[tets[:, 1]]
    sdf_H_c = sdf_H[tets[:, 2]]
    sdf_H_d = sdf_H[tets[:, 3]]
    sdf_H_stack = torch.stack([sdf_H_a, sdf_H_b, sdf_H_c, sdf_H_d], dim=1)  # (M, 4)

    sdf_H_center = sdf_H_stack.mean(dim=1, keepdim=True)  # (M, 1)
    sdf_H_diff = sdf_H_stack - sdf_H_center  # (M, 4)

    grad_H_tet = torch.einsum("mi,mij->mj", sdf_H_diff, W)  # (M, 3)
    grad_norm = grad_H_tet.norm(dim=1)  # (M, 1)

    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]
    volume = su.volume_tetrahedron(a, b, c, d)  # (M,)
    # trim 5% biggest volumes
    volume = torch.where(volume > torch.quantile(volume, 0.95), torch.tensor(0.0, device=sites.device), volume)
    penalties = torch.mean(volume * grad_norm)
    # penalties = torch.mean(grad_norm)

    # return torch.mean(volume * grad_norm)
    return penalties


def heaviside_derivative(phi: torch.Tensor, eps_H: float) -> torch.Tensor:
    """
    Derivative H'(φ̂) of the smoothed Heaviside function.
    """
    H_prime = torch.zeros_like(phi)

    inside = (phi >= -eps_H) & (phi <= eps_H)
    H_prime[inside] = 1 / (2 * eps_H) + (1 / (2 * math.pi * eps_H)) * torch.cos(math.pi * phi[inside] / eps_H)

    return H_prime


def smoothed_heaviside_loss(sites, sdf, grad_sdf, tets, eps_H=0.07, eps_grad=1e-8):
    """
    Fast approximation of E2 = ∑_t |∇H(φ̂)| * Volume(t)
    using average of chain-rule gradients.
    """
    # 1. Compute ∇φ at each site
    grad_phi = grad_sdf  # su.sdf_space_grad_pytorch_diego(sites, sdf, tets)  # (N,3)

    # 2. Compute H'(φ̂) at each site
    H_prime = heaviside_derivative(sdf, eps_H)  # (N,)

    # 3. Multiply: ∇H = H'(φ̂) · ∇φ
    grad_H = grad_phi * H_prime[:, None]  # (N,3)

    # 4. Gather per tet: average over 4 sites
    g0 = grad_H[tets[:, 0]].norm(dim=1)
    g1 = grad_H[tets[:, 1]].norm(dim=1)
    g2 = grad_H[tets[:, 2]].norm(dim=1)
    g3 = grad_H[tets[:, 3]].norm(dim=1)
    grad_H_avg = (g0 + g1 + g2 + g3) / 4  # (M,3)

    # 5. Norm of gradient
    # grad_norm = grad_H_avg.norm(dim=1)  # (M,)

    # 6. Compute volume
    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]
    volume = su.volume_tetrahedron(a, b, c, d)  # (M,)

    # 7. Mask small gradients
    mask = grad_H_avg >= eps_grad

    return torch.mean(volume[mask] * grad_H_avg[mask])


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
