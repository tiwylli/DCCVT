import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import trimesh
from mesh_to_sdf import get_surface_point_cloud

import sdfpred_utils.loss_functions as lf
import sdfpred_utils.Steik_utils as Stu 
import sdfpred_utils.sdfpred_utils as su
import polyscope as ps

try:
    import tinycudann as tcnn
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()


from common import read_image, write_image, ROOT_DIR

DATA_DIR = "."
IMAGES_DIR = "."

def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

    parser.add_argument("obj", nargs="?", default="./Resources/chair_low.obj", help="Image to match")
    parser.add_argument("config", nargs="?", default="data/sdf_frequency.json", help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
    parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

    args = parser.parse_args()
    return args

def sample_directions(n_rays):
    """
    Sample n_rays directions uniformly on the unit sphere.
    This uses the Fibonacci sphere algorithm.
    """
    indices = np.arange(0, n_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_rays)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T  # shape (n_rays, 3)

def sign_distances(mesh, query_points, n_rays=32, offset=1e-6):
    """
    For each query point, cast n_rays stab rays. If any ray 
    escapes (i.e. produces no intersection) then mark that point as outside.
    
    Returns an array of signed distances based on a preliminary 
    unsigned distance computation (for example, using a nearest-point query).
    You would combine the sign with the unsigned distance to obtain the final value.
    """
    # Create an array to hold the sign for each query point:
    # +1 for outside, -1 for inside.
    signs = np.empty(len(query_points), dtype=int)
    
    # Precompute the stab directions (uniformly on the sphere)
    directions = sample_directions(n_rays)
    
    # For each query point, cast rays in the chosen directions
    for i, point in enumerate(query_points):
        outside_detected = False
        
        for d in directions:
            # Offset the ray origin a tiny bit in the direction d to avoid self-intersections:
            origin = point + offset * d
            # Cast a single ray from the origin along direction d.
            # The intersect_location call returns the intersections along the ray.
            locations = mesh.ray.intersects_location(ray_origins=[origin],
                                                      ray_directions=[d])
            # Check if this stab ray did *not* hit the object.
            if len(locations[0]) == 0:
                outside_detected = True
                break
        
        # If any ray escapes, mark the sign as +1 (outside), else -1 (inside)
        signs[i] = 1 if outside_detected else -1
    
    return signs

#import torch_cluster as pc
if __name__ == "__main__":
    ps.init()
    
    print("================================================================")
    print("This script replicates the behavior of the native CUDA example  ")
    print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
    print("================================================================")

    print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

    device = torch.device("cuda")
    args = get_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    # Only the distance
    interval = 10
    n_channels = 1
    model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    print(model)

    # Load the mesh and normalize it to fit in the unit cube
    mesh = trimesh.load_mesh(args.obj)
    mesh.vertices -= mesh.center_mass
    mesh.vertices /= mesh.scale
    mesh.apply_translation([0.5, 0.5, 0.5])
    center = mesh.vertices.mean(axis=0)
    radius = np.linalg.norm(mesh.vertices - center, axis=1).max()
    #bvh = trimesh.collision.mesh_to_BVH(mesh)
    # Create a ProximityQuery object
    pq = trimesh.proximity.ProximityQuery(mesh)
    
    # nb_points = 10000    
    # #sample 3d points uniformly in the unit 
    # ucp = np.random.uniform(0, 1, size=(int(nb_points/8), 3))
    # #sample 3d points uniformly on the mesh
    # sp = mesh.sample(int(nb_points/2))
    # #sample 3d points perturbed from the surface
    # sigma = radius/1024.0
    # logistic_scale = sigma * np.sqrt(3) / np.pi
    # pp = mesh.sample(int(3*nb_points/8))
    # noise = np.random.logistic(loc=0.0, scale=logistic_scale, size=pp.shape)
    # pp += noise
    
    # query_points = np.concatenate((ucp, sp, pp), axis=0)

    # # Compute the closest points on the surface and the distances
    # _, unsigned_distances, _ = pq.on_surface(query_points)

    # # Determine the sign for each query point using stab rays:
    # ray_signs = sign_distances(mesh, query_points, n_rays=32, offset=1e-6)

    # # Combine sign and unsigned distances:
    # signed_distances = unsigned_distances * ray_signs

    #print("Query Points:\n", query_points)
    #print("Unsigned Distances:\n", unsigned_distances)
    #print("Ray-based Signs (1: outside, -1: inside):\n", ray_signs)
    #print("Signed Distances:\n", signed_distances)
    
    #pc = ps.register_point_cloud("query_points", query_points, radius=0.01)
    #pc.add_scalar_quantity("signed_distances", signed_distances)
    #ps.show()
    
    


    print(f"Beginning optimization with {args.n_steps} training steps.")
    
    for i in range(args.n_steps):       
             
        nb_points = 1000
        #sample 3d points uniformly in the unit 1/8*nb_points
        ucp = torch.rand((int(nb_points/8), 3), device=device)
        #sample 3d points uniformly on the mesh 1/2*nb_points
        sp = torch.tensor(mesh.sample(int(nb_points/2)), device=device)
        #sample 3d points perturbed from the surface 3/8*nb_points
        sigma = radius/1024.0
        logistic_scale = sigma * torch.sqrt(torch.tensor(3.)) / torch.pi
        pp = torch.tensor(mesh.sample(int(3*nb_points/8)), device=device)
        noise = torch.tensor(np.random.logistic(loc=0.0, scale=logistic_scale, size=pp.shape), device=device)
        pp += noise
        query_points = torch.concatenate((ucp, sp, pp), axis=0).to(device)

        _, unsigned_distances, _ = pq.on_surface(query_points.cpu().numpy())

        ray_signs = sign_distances(mesh, query_points.cpu().numpy(), n_rays=32, offset=1e-6)

        signed_distances = unsigned_distances * ray_signs
        signed_distances = torch.tensor(signed_distances, device=device)

        targets = signed_distances
        output = model(query_points)
        # Compute the loss using the relative L2 error
        relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            #torch.cuda.synchronize()
            #elapsed_time = time.perf_counter() - prev_time
            #print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                # write_image(path, sdf_to_image(model(xyz).detach().cpu().numpy(), resolution))
                pc = ps.register_point_cloud("query_points", query_points.cpu().numpy(), radius=0.01)
                pc.add_scalar_quantity("signed_distances true", signed_distances.cpu().numpy())
                print(output.shape)
                print(output.reshape(-1).shape)
                
                pc.add_scalar_quantity("signed_distances model", output.reshape(-1).cpu().numpy())
                
                ps.show()
                print("done.")

            # Ignore the time spent saving the image
            #prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        with torch.no_grad():
            #write_image(args.result_filename, model(xyz).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
            print("done.")



    
    tcnn.free_temporary_memory()