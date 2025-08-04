import torch
from scipy.spatial import Voronoi
import numpy as np
import math
import sdfpred_utils.sdfpred_utils as su

from pytorch3d.ops import knn_points, knn_gather
import torch
from torch import nn

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


def compute_voronoi_cell_centers_index_based_torch(points, delau, simplices=None):
    """Compute Voronoi cell centers (circumcenters) for 2D or 3D Delaunay triangulation in PyTorch."""
    # simplices = torch.tensor(delaunay.simplices, dtype=torch.long)
    if simplices is None:
        simplices = delau.simplices

    # points = torch.tensor(delaunay.points, dtype=torch.float32)
    points = points.detach().cpu().numpy()

    # Compute all circumcenters at once (supports both 2D & 3D)
    circumcenters_arr = circumcenter_torch(points, simplices)
    # Flatten simplices and repeat circumcenters to map them to the points
    indices = simplices.flatten()  # Flatten simplex indices
    centers = circumcenters_arr.repeat_interleave(simplices.shape[1], dim=0)  # Repeat for each vertex in simplex

    # Group circumcenters per point
    # result = [centers[indices == i] for i in range(len(points))]
    # centroids = torch.stack([torch.mean(c, dim=0) for c in result])
    M = len(points)
    indices = torch.tensor(indices)
    # Compute the sum of centers for each index
    centroids = torch.zeros(M, 3, dtype=torch.float32)
    counts = torch.zeros(M, device=centers.device)

    centroids.index_add_(0, indices, centers)  # Sum centers per unique index
    counts.index_add_(0, indices, torch.ones(centers.shape[0], device=centers.device))  # Count occurrences

    centroids /= counts.clamp(min=1).unsqueeze(1)  # Avoid division by zero

    return centroids


def compute_cvt_loss_vectorized_delaunay(sites, delaunay, simplices=None):
    centroids = compute_voronoi_cell_centers_index_based_torch(sites, delaunay, simplices).to(device)
    penalties = torch.where(abs(sites - centroids) < 0.1, sites - centroids, torch.tensor(0.0, device=sites.device))
    # cvt_loss = torch.mean(penalties**2)
    cvt_loss = torch.mean(torch.abs(penalties))
    return cvt_loss


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


# def discrete_tet_volume_eikonal_loss(sites, grad_sdf, tets: torch.Tensor) -> torch.Tensor:
#     """
#     Eikonal regularization loss.

#     Args:
#         grad_sdf: Tensor of shape (N, 3) containing ∇φ at each site.
#         variant: 'a' for E1a: ½ mean((||∇φ|| - 1)²)
#     Returns:
#         A scalar tensor containing the eikonal loss.
#     """
#     grad_a = grad_sdf[tets[:, 0]]  # (M,3)
#     grad_b = grad_sdf[tets[:, 1]]  # (M,3)
#     grad_c = grad_sdf[tets[:, 2]]  # (M,3)
#     grad_d = grad_sdf[tets[:, 3]]  # (M,3

#     grad_avg = (grad_a + grad_b + grad_c + grad_d) / 4.0
#     grad_norm2 = (grad_avg**2).sum(dim=-1)

#     a = sites[tets[:, 0]]
#     b = sites[tets[:, 1]]
#     c = sites[tets[:, 2]]
#     d = sites[tets[:, 3]]

#     volume = su.volume_tetrahedron(a, b, c, d)

#     loss = 0.5 * torch.mean(volume * (grad_norm2 - 1) ** 2)

#     return loss


def tet_sdf_grad_eikonal_loss(sites, tet_sdf_grad, tets: torch.Tensor) -> torch.Tensor:
    a = sites[tets[:, 0]]
    b = sites[tets[:, 1]]
    c = sites[tets[:, 2]]
    d = sites[tets[:, 3]]

    volume = su.volume_tetrahedron(a, b, c, d)
    grad_norm2 = (tet_sdf_grad**2).sum(dim=1)  # (M,)
    loss = 0.5 * torch.mean(volume * (grad_norm2 - 1) ** 2)  # (M,)

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


def tet_sdf_motion_mean_curvature_loss(sites, sites_sdf, W, tets) -> torch.Tensor:
    sdf_H = smoothed_heaviside(sites_sdf, eps_H=0.07)  # (M,)
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

    return torch.mean(volume * grad_norm)


# def heaviside_derivative(phi: torch.Tensor, eps_H: float) -> torch.Tensor:
#     """
#     Derivative H'(φ̂) of the smoothed Heaviside function.
#     """
#     H_prime = torch.zeros_like(phi)

#     inside = (phi >= -eps_H) & (phi <= eps_H)
#     H_prime[inside] = 1 / (2 * eps_H) + (1 / (2 * math.pi * eps_H)) * torch.cos(math.pi * phi[inside] / eps_H)

#     return H_prime


# def smoothed_heaviside_loss(sites, sdf, grad_sdf, tets, eps_H=0.07, eps_grad=1e-8):
#     """
#     Fast approximation of E2 = ∑_t |∇H(φ̂)| * Volume(t)
#     using average of chain-rule gradients.
#     """
#     # 1. Compute ∇φ at each site
#     grad_phi = grad_sdf  # su.sdf_space_grad_pytorch_diego(sites, sdf, tets)  # (N,3)

#     # 2. Compute H'(φ̂) at each site
#     H_prime = heaviside_derivative(sdf, eps_H)  # (N,)

#     # 3. Multiply: ∇H = H'(φ̂) · ∇φ
#     grad_H = grad_phi * H_prime[:, None]  # (N,3)

#     # 4. Gather per tet: average over 4 sites
#     g0 = grad_H[tets[:, 0]].norm(dim=1)
#     g1 = grad_H[tets[:, 1]].norm(dim=1)
#     g2 = grad_H[tets[:, 2]].norm(dim=1)
#     g3 = grad_H[tets[:, 3]].norm(dim=1)
#     grad_H_avg = (g0 + g1 + g2 + g3) / 4  # (M,3)

#     # 5. Norm of gradient
#     # grad_norm = grad_H_avg.norm(dim=1)  # (M,)

#     # 6. Compute volume
#     a = sites[tets[:, 0]]
#     b = sites[tets[:, 1]]
#     c = sites[tets[:, 2]]
#     d = sites[tets[:, 3]]
#     volume = su.volume_tetrahedron(a, b, c, d)  # (M,)

#     # 7. Mask small gradients
#     mask = grad_H_avg >= eps_grad

#     return torch.mean(volume[mask] * grad_H_avg[mask])


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
