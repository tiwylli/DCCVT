from scipy.spatial import Delaunay, Voronoi
import numpy as np
from sklearn.cluster import KMeans
import torch

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

def get_delaunay_neighbors_list(sites):
    # Detach and convert to NumPy for Delaunay triangulation
    points_np = sites.detach().cpu().numpy()
    
    # Compute the Delaunay triangulation
    tri = Delaunay(points_np)

    # Find the neighbors of each point
    neighbors = {i: set() for i in range(len(points_np))}
    for simplex in tri.simplices:
        # Each simplex is a triangle of three points; each point is a neighbor of the other two
        for i in range(3):
            for j in range(i + 1, 3):
                neighbors[simplex[i]].add(simplex[j])
                neighbors[simplex[j]].add(simplex[i])

    # Convert neighbor sets to lists for easier reading
    neighbors = {key: list(value) for key, value in neighbors.items()}
    return neighbors

def compute_vertices_index(neighbors):
    vertices_index_to_compute = []
    for site, adjacents in neighbors.items():
        for i in adjacents:
            for n in adjacents:
                if n != site and n != i and n in neighbors[i]:              
                    vertices_index_to_compute.append([i,site,n])
            
    # Set to store the canonical (sorted) version of each triplet
    seen_triplets = set()
    # Filtered list to store the unique triplets
    filtered_triplets = []
    # Process each triplet and keep only one permutation
    for triplet in vertices_index_to_compute:
        # Convert the triplet to a canonical form by sorting it
        canonical_triplet = tuple(sorted(triplet, key=str))
        # Check if this canonical triplet has been seen before
        if canonical_triplet not in seen_triplets:
            # If not seen, add it to the set and keep the triplet
            seen_triplets.add(canonical_triplet)
            filtered_triplets.append(triplet)

    return filtered_triplets

def compute_zero_crossing_vertices(sites, model):
    """
    Computes the indices of the sites composing vertices where neighboring sites have opposite or zero SDF values.

    Args:
        sites (torch.Tensor): (N, D) tensor of site positions.
        model (callable): Function or neural network that computes SDF values.

    Returns:
        zero_crossing_vertices_index (list of triplets): List of sites indices (si, sj, sk) where atleast 2 sites have opposing SDF signs.
    """
    # Compute Delaunay neighbors
    neighbors = get_delaunay_neighbors_list(sites)

    # Compute SDF values for all sites
    sdf_values = model(sites)  # Assuming model outputs (N, 1) or (N,) tensor

    # Find pairs of neighbors with opposing SDF values
    zero_crossing_pairs = set()
    for i, adjacents in neighbors.items():
        for j in adjacents:
            if i < j:  # Avoid duplicates
                sdf_i, sdf_j = sdf_values[i].item(), sdf_values[j].item()
                if sdf_i * sdf_j <= 0:  # Opposing signs or one is zero
                    zero_crossing_pairs.add((i, j))
    

    # Compute vertices (triplets) and filter only those involving zero-crossing pairs
    all_vertices = compute_vertices_index(neighbors)
    zero_crossing_vertices_index = []
    
    for triplet in all_vertices:
        triplet_pairs = {(triplet[0], triplet[1]), (triplet[1], triplet[2]), (triplet[0], triplet[2])}
        if any(pair in zero_crossing_pairs for pair in triplet_pairs):
            zero_crossing_vertices_index.append(triplet)

    return zero_crossing_vertices_index, zero_crossing_pairs

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

def compute_all_bisectors(sites, bisectors_to_compute):
    # Initialize an empty tensor for storing bisectors
    bisectors = []
    
    for pairs in bisectors_to_compute:
        si = sites[pairs[0]]
        sj = sites[pairs[1]]
        b = (si + sj) / 2
        bisectors.append(b)

    # Stack the list of bisectors into a single tensor for easier gradient tracking
    bisectors = torch.stack(bisectors)
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
    
    neighbors = get_delaunay_neighbors_list(sites)

    sdf_values = model(sites) 

    sites_to_upsample = []
    # Find pairs of neighbors with opposing SDF values
    for i, adjacents in neighbors.items():
        for j in adjacents:
            if i < j and i not in sites_to_upsample:  # Avoid duplicates
                sdf_i, sdf_j = sdf_values[i].item(), sdf_values[j].item()
                if sdf_i * sdf_j <= 0:  # Opposing signs or one is zero
                    sites_to_upsample.append(i)
    

    # Step 1: Compute the gradient of the SDF at each site
    grad_sdf = torch.autograd.grad(sdf_values[:,0], sites, torch.ones_like(sdf_values[:,0]), create_graph=True, retain_graph=True)[0]

    # Step 2: Compute the density map based on the gradient magnitude
    grad_mag = grad_sdf.norm(dim=1)  # Compute the magnitude of the gradient
    density_map = 1 / (1 + grad_mag**sigma)  # Inverse relation, higher gradients = higher density

    # Step 3 & 4: Generate new sites along the gradient direction
    new_sites = []
    for i in range(len(sites_to_upsample)):
        site = sites[sites_to_upsample[i]]
        gradient = grad_sdf[sites_to_upsample[i]]
        density = density_map[sites_to_upsample[i]]
        
        # Normalize the gradient direction
        grad_norm = gradient / (gradient.norm() + 1e-6)  # Avoid division by zero
        
        # Generate points along the gradient direction, scaled by density
        num_new_points = int(density * num_points_per_site)  # Number of new points based on density
        
        for j in range(num_new_points):
            displacement = grad_norm * (max_distance * (j + 1) / num_new_points)
            new_site = site + displacement
            new_sites.append(new_site)

    # Return the new sites as a tensor
    return new_sites


                
