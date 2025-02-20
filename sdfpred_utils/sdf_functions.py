import torch
import numpy as np
import os

device = torch.device("cuda:0")

def plot_sdf(ax, model):
    # Generate a grid of points
    grid_size = 100
    x = np.linspace(-10.0, 10.0, grid_size)
    y = np.linspace(-10.0, 10.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Convert grid points to a PyTorch tensor
    grid_points_tensor = torch.tensor(grid_points, device=device, dtype=torch.double)
    
    # Compute the SDF values
    sdf_values = model(grid_points_tensor)[:,0]
    sdf_values = sdf_values.reshape(grid_size, grid_size)
    #print(sdf_values)
    # Plot the SDF contour
    ax.contour(xx, yy, sdf_values.cpu().detach().numpy(), levels=[0] ,colors='black')
    return xx, yy, sdf_values

def generate_rectangle_points(num_points: int, width: float, height: float, scale: float = 1.0):
    """
    Generate uniformly distributed points on the edges of a rectangle.
    
    Parameters:
        num_points (int): Total number of points.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        scale (float, optional): Scaling factor for the points. Default is 1.0.
        device (str, optional): Device to store the tensor ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns:
        torch.Tensor: Tensor of shape (num_points, 2) containing the generated points.
    """
    num_edge_points = num_points // 4  # Distribute points equally on four edges
    
    # Top edge
    top_edge = torch.stack([
        torch.linspace(-width / 2, width / 2, num_edge_points, device=device),
        torch.full((num_edge_points,), height / 2, device=device)
    ], dim=1)
    
    # Bottom edge
    bottom_edge = torch.stack([
        torch.linspace(-width / 2, width / 2, num_edge_points, device=device),
        torch.full((num_edge_points,), -height / 2, device=device)
    ], dim=1)
    
    # Right edge
    right_edge = torch.stack([
        torch.full((num_edge_points,), width / 2, device=device),
        torch.linspace(-height / 2, height / 2, num_edge_points, device=device)
    ], dim=1)
    
    # Left edge
    left_edge = torch.stack([
        torch.full((num_edge_points,), -width / 2, device=device),
        torch.linspace(-height / 2, height / 2, num_edge_points, device=device)
    ], dim=1)
    
    # Combine all edges
    points = torch.cat([top_edge, bottom_edge, right_edge, left_edge], dim=0)
    
    # Apply scaling
    points *= scale
    
    return points

def star_sdf(vertices, r=3.0, rf=0.5, origin=torch.tensor([0, 0],device=device)):
    """
    Compute the SDF of a five-pointed star with a custom origin.

    Args:
        vertices (torch.Tensor): Input points, shape (N, 2).
        r (float): Outer radius of the star.
        rf (float): Ratio between the inner and outer radii of the star.
        origin (tuple): The origin of the star (x, y).

    Returns:
        torch.Tensor: SDF values for each point, shape (N,).
    """
    # Shift points relative to the origin
    p = vertices - origin

    # Define constants
    k1 = torch.tensor([0.809016994375, -0.587785252292], device=device, dtype=p.dtype)  # Cosine and sine for 72 degrees
    k2 = torch.tensor([-0.809016994375, -0.587785252292], device=device, dtype=p.dtype)  # Cosine and sine for -72 degrees

    # Reflect and transform the points without in-place modification
    p_reflected_x = torch.abs(p[:, 0]).unsqueeze(1)  # Reflect across the y-axis
    p_reflected = torch.cat((p_reflected_x, p[:, 1].unsqueeze(1)), dim=1)

    p_reflected -= 2.0 * torch.clamp(torch.sum(p_reflected * k1, dim=1, keepdim=True), min=0.0) * k1
    p_reflected -= 2.0 * torch.clamp(torch.sum(p_reflected * k2, dim=1, keepdim=True), min=0.0) * k2

    # Reflect again across the y-axis
    p_reflected_x2 = torch.abs(p_reflected[:, 0]).unsqueeze(1)
    p_reflected = torch.cat((p_reflected_x2, p_reflected[:, 1].unsqueeze(1)), dim=1)

    # Shift downwards
    p_reflected_y_shift = (p_reflected[:, 1] - r).unsqueeze(1)
    p_reflected = torch.cat((p_reflected[:, 0].unsqueeze(1), p_reflected_y_shift), dim=1)

    # Vector to the inner vertices
    ba = rf * torch.tensor([-k1[1], k1[0]], device=device, dtype=p.dtype) - torch.tensor([0, 1], device=device, dtype=p.dtype)
    
    # Projection along ba
    h = torch.clamp(torch.sum(p_reflected * ba, dim=1) / torch.sum(ba * ba), min=0.0, max=r)
    
    # Compute distance and sign
    dist = torch.norm(p_reflected - h.unsqueeze(1) * ba, dim=1)
    sign = torch.sign(p_reflected[:, 1] * ba[0] - p_reflected[:, 0] * ba[1])
    
    return dist * sign

def moon_sdf(vertices, d=0.5*3.0, ra=1.0*3.0, rb=0.8*3.0, origin=torch.tensor([0, 0], device=device)):
    """
    Compute the SDF of a croissant-like shape with a custom origin.

    Args:
        vertices (torch.Tensor): Input points, shape (N, 2).
        d (float): Distance between the centers of the circles.
        ra (float): Radius of the larger circle.
        rb (float): Radius of the smaller circle.
        origin (tuple): The origin of the croissant (x, y).

    Returns:
        torch.Tensor: SDF values for each point, shape (N,).
    """
    # Shift points relative to the origin
    p = vertices - origin
    
    # Reflect across the x-axis (croissant symmetry) without in-place modification
    p_reflected = torch.clone(p)
    p_reflected[:, 1] = torch.abs(p[:, 1])
    
    # Compute parameters of the crescent
    a = (ra**2 - rb**2 + d**2) / (2.0 * d)
    b = torch.sqrt(torch.clamp(torch.tensor(ra**2 - a**2, device=p.device, dtype=p.dtype), min=0.0))
    
    # Vector to the offset circle
    offset = torch.tensor([a, b], device=p.device, dtype=p.dtype)
    
    # Check point location relative to crescent
    condition = d * (p_reflected[:, 0] * b - p_reflected[:, 1] * a) > d**2 * torch.clamp(b - p_reflected[:, 1], min=0.0)
    
    # SDF computation
    dist_to_outer_circle = torch.norm(p_reflected - offset, dim=1)
    dist_to_inner_circle = torch.norm(p_reflected - torch.tensor([d, 0.0], device=p.device, dtype=p.dtype), dim=1)
    dist_to_main_circle = torch.norm(p_reflected, dim=1)
    
    sdf_values = torch.where(
        condition,
        dist_to_outer_circle,
        torch.max(
            dist_to_main_circle - ra,
            -(dist_to_inner_circle - rb)
        )
    )
    
    return sdf_values

def torus_sdf(vertices, r_inner=3.0/2, r_outer=3.0, origin=torch.tensor([0, 0], device=device)):
    """
    Compute the SDF loss for a 2D torus (ring) given a set of vertices.

    Parameters:
        vertices (torch.Tensor): Tensor of shape (N, 2), where each row is a 2D vertex.
        r_inner (float): Inner radius of the torus (ring).
        r_outer (float): Outer radius of the torus (ring).
        origin (torch.Tensor): Origin of the torus (center), a tensor of shape (2,).

    Returns:
        torch.Tensor: Signed distance values for each vertex, shape (N,).
    """
    # Calculate the Euclidean distance of each vertex to the torus origin
    distances_to_center = torch.norm(vertices - origin, dim=1)  # L2 norm along x and y for each vertex

    # Compute the unsigned distance to the ring
    unsigned_distance = torch.abs(distances_to_center - r_outer)
    
    # Compute the SDF
    sdf_values = unsigned_distance - (r_outer - r_inner)
    
    return sdf_values

def bunny_sdf(points):
    gridsize = 1024
    sdf_grid_path = os.path.join(os.path.dirname(__file__), f"../models_resources/sdf_grid_{gridsize}_centered.pt")
    grid = torch.load(sdf_grid_path).to(device)
    
    x_min, y_min, x_max, y_max = 0.0, 0.0, 7.78495, 7.7166999999999994
    x_min = x_max*-1.5
    y_min = y_max*-1.5
    x_max = x_max*2
    y_max = y_max*2

    # Normalize points to [0, 1] range
    points_normalized = (points - torch.tensor([x_min, y_min], device=device)) / torch.tensor([x_max - x_min, y_max - y_min], device=device)

    # Scale to grid coordinates
    gridsize = grid.shape[0]
    points_grid = points_normalized * (gridsize - 1)

    # Separate grid coordinates into integer and fractional parts
    x = points_grid[:, 0]
    y = points_grid[:, 1]
    x0 = x.floor().long().clamp(0, gridsize - 1)
    y0 = y.floor().long().clamp(0, gridsize - 1)
    x1 = (x0 + 1).clamp(0, gridsize - 1)
    y1 = (y0 + 1).clamp(0, gridsize - 1)
    dx = x - x0
    dy = y - y0

    # Perform bilinear interpolation
    values = (
        (1 - dx) * (1 - dy) * grid[x0, y0] +
        dx * (1 - dy) * grid[x1, y0] +
        (1 - dx) * dy * grid[x0, y1] +
        dx * dy * grid[x1, y1]
    )
    return values

def generate_sdf_points(grid_size: int, width: float, height: float, sdf_function, threshold=1e-3):
    """
    Generate points where the SDF of a star is approximately zero.
    
    Parameters:
        grid_size (int): Number of points along one axis for the grid.
        width (float): Width of the grid space.
        height (float): Height of the grid space.
        sdf_function (callable): Function that computes the SDF.
        threshold (float, optional): Numerical threshold to consider a point as zero. Default is 1e-3.
        device (str, optional): Device to store the tensor ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns:
        torch.Tensor: Tensor of shape (N, 2) containing the selected points.
    """
    x = torch.linspace(-width / 2, width / 2, grid_size, device=device)
    y = torch.linspace(-height / 2, height / 2, grid_size, device=device)
    
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    sdf_values = sdf_function(grid_points)
    zero_points = grid_points[torch.abs(sdf_values) < threshold]
    
    return zero_points
