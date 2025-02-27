#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# @file   mlp_learning_an_image_pytorch.py
# @author Thomas Müller, NVIDIA
# @brief  Replicates the behavior of the CUDA mlp_learning_an_image.cu sample
#         using tiny-cuda-nn's PyTorch extension. Runs ~2x slower than native.

import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import trimesh
from mesh_to_sdf import sample_sdf_near_surface


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

    parser.add_argument("obj", nargs="?", default="data/albert.jpg", help="Image to match")
    parser.add_argument("config", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn")
    parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
    parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

    args = parser.parse_args()
    return args

# Use matplotlib colormap to generate a positive/negative image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
cmap = plt.get_cmap("coolwarm")
def sdf_to_image(sdf, resolution, range=0.5):
    print(sdf.shape, np.min(sdf), np.max(sdf))    
    # norm = mcolors.Normalize(vmin=-range, vmax=range)
    # sdfnorm = norm(sdf)
    sdfnorm = sdf
    
    img = cmap(sdfnorm).reshape(resolution + (4,))
    img = img[:, :, :3] # Remove alpha channel
    return img
    
import pyexr
import polyscope as ps

def polyscope_sdf(model):
    # Render the SDF as an implicit surface (zero-level set)
    def model_sdf(pts):
        pts_tensor = torch.tensor(pts * 0.1, dtype=torch.float64, device=device)
        # print(pts_tensor.min(), pts_tensor.max())
        sdf_values = model(pts_tensor)
        sdf_values_np = sdf_values.detach().cpu().numpy().flatten()  # Convert to NumPy
        
        return sdf_values_np

    ps.render_implicit_surface("SDF Surface", model_sdf, mode="sphere_march", enabled=True)

def polyscope_sdf_ref(points):
    points = points.unsqueeze(0)
    # Render the SDF as an implicit surface (zero-level set)
    def model_sdf(pts):
        pts_tensor = torch.tensor(pts * 0.1, dtype=torch.float64, device=device)
        pts_tensor = pts_tensor.unsqueeze(0)
        dist = torch.cdist(pts_tensor, points)
        dist = dist.squeeze(0)
        sdf_values = dist.min(dim=1).values
        sdf_values_np = sdf_values.detach().cpu().numpy().flatten()  # Convert to NumPy
        
        return sdf_values_np

    ps.render_implicit_surface("SDF Surface", model_sdf, mode="sphere_march", enabled=True)

import drjit as dr
from drjit.cuda import Float, Array3f, TensorXf

# Better version: use loop obj in drjit
@dr.syntax
def trace(o: Array3f, d: Array3f, sdf) -> Array3f:
    i = 0
    o = o + d*1.5 # Advance of 1 unit (the camera is at 0.5, 0.5, -2)
    while i < 256:
        # Ray marching
        o = dr.select(Float(sdf(o.torch().permute(1, 0)).squeeze(-1)) > 0.04, o + 0.01*d, o)
        
        # SDF Sphere tracing
        # o = dr.fma(d, Float(sdf(o.torch().permute(1, 0)).squeeze(-1)), o)
        i += 1
        
    # Assuming that we are inside the [0, 0, 0] to [1, 1, 1] box
    o = dr.clip(o, 0, 1)
    return o

def render(sdf, resolution) -> TensorXf:
    x = dr.linspace(Float, 0, 1, resolution[0])
    y = dr.linspace(Float, 0, 1, resolution[1])
    x, y = dr.meshgrid(x, y)
    
    o = Array3f(0.5, 0.5, -2)
    d = dr.normalize(Array3f(x, y, 3)) # Perspective camera

    p = trace(o, d, sdf)
    dist = dr.norm(p - o) - 2
    return dist


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
    n_channels = 1

    model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
    print(model)

    # Load the mesh
    mesh = trimesh.load_mesh(args.obj)
    
    # Sample points on the mesh, this will be to approximate the SDF
    # points, face_indices = trimesh.sample.sample_surface(mesh, 100000)
    # points = points.astype(np.float32)
    # points = torch.from_numpy(points).to(device)

    # # Normalize the points to [0, 1]   
    # points -= points.min(dim=0).values
    # points /= points.max(dim=0).values
    
    # # Scale = 0.5 and translate = 0.5
    # points = points * 0.5 + 0.5
    
    # # shift points between [-0.5, 0.5]
    # points -= 0.5
    # # Scale points between [-1, 1]
    # points *= 2 
    
        
    grid_points, grid_sdf = sample_sdf_near_surface(mesh, number_of_points=30000)
    grid_points = torch.tensor(grid_points, device=device, dtype=torch.float16)
    grid_sdf = torch.tensor(grid_sdf, device=device, dtype=torch.float16)
    
    # Normalize the points to [0, 1]
    grid_points /= 2
    grid_points += 0.5
    print(grid_points.min(), grid_points.max())
    
    grid_sdf /= 2
    print(grid_sdf.min(), grid_sdf.max())   
    
    print(grid_points.shape, grid_sdf.shape)
    
    # Show point with values in polyscope
    ps_cloud = ps.register_point_cloud("points", grid_points.cpu().numpy())
    ps_cloud.add_scalar_quantity("sdf", grid_sdf.cpu().numpy())
    ps.show()
    
    # Sample a sphere centered in [0.5, 0.5, 0.5] with radius 0.3
    # points = torch.rand([100000, 3], device=device, dtype=torch.float32)
    # points -= 0.5
    # points /= torch.sum(points**2, dim=1, keepdim=True).sqrt()
    # points *= 0.3
    # points += 0.5
    
    
    #===================================================================================================
    # The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
    # tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
    #===================================================================================================
    # encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
    # network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
    # model = torch.nn.Sequential(encoding, network)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Variables for saving/displaying image results
    resolution = torch.Size([128, 128])
    img_shape = resolution + torch.Size([1]) # n_channels
    n_pixels = resolution[0] * resolution[1] 

    # This is for evaluating the model on a 3D grid
    # from [half, 1-half] -- close to [0, 1]
    half_dx =  0.5 / resolution[0]
    half_dy =  0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys])

    xy = torch.stack((yv.flatten(), xv.flatten())).t()
    xyz = torch.cat((xy, torch.ones_like(xy[:,0:1]) * 0.5), dim=1)

    # Torch no grad
    # with torch.no_grad():
    #     path = f"reference.jpg"
    #     print(f"Writing '{path}'... ", end="")
        
    #     # Compute the minimal distance between batch and the mesh sampled points
    #     xyz_batch = torch.zeros((n_pixels), device=device, dtype=torch.int64)
    #     points_batch = torch.zeros((points.shape[0]), device=device, dtype=torch.int64)
        
    #     indices_points = pc.nearest(xyz, points, xyz_batch, points_batch)
    #     target = torch.sqrt(((xyz - points[indices_points])**2).sum(dim=1, keepdim=True))
        
    #     ref_img = target.reshape(img_shape).detach().cpu().numpy()
    #     write_image(path, ref_img)
    #     pyexr.write("reference.exr", ref_img)   
        
    #     # polyscope_sdf_ref(points)
    #     # ps.show()
        
    #     print("done.")

    prev_time = time.perf_counter()

    batch_size = 2**15
    interval = 10

    print(f"Beginning optimization with {args.n_steps} training steps.")
    
    for i in range(args.n_steps):            
        
        # # Circle target centered in [0.5, 0.5, 0.5] with radius 0.3
        # # targets = torch.abs(torch.sqrt(((batch - 0.5)**2).sum(dim=1, keepdim=True)) - 0.3)
        
        # # Compute the minimal distance between batch and the mesh sampled points
        # x_rand = torch.rand([batch_size, 3], device=device, dtype=torch.float32)
        # x_rand_batch = torch.zeros((batch_size), device=device, dtype=torch.int64)
        # points_batch = torch.zeros((points.shape[0]), device=device, dtype=torch.int64)
        # indices_points = pc.nearest(x_rand, points, x_rand_batch, points_batch)
        # target = torch.sqrt((x_rand - points[indices_points])**2).sum(dim=1, keepdim=True)
        # output = model(x_rand.detach())
    
        loss = 0
        p1 = grid_points.detach()
        target = grid_sdf.detach()
        output = model(p1) # sdf 0 , deform 1-3
        #loss = loss_fn(output1[...,0], ref_value1)
        
        
        
        # print(output[0])
        relative_l2_error = (output - target.to(output.dtype))**2 #/ (output.detach()**2 + 0.01)
        loss = relative_l2_error.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0:
            loss_val = loss.item()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - prev_time
            print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

            path = f"{i}.jpg"
            print(f"Writing '{path}'... ", end="")
            with torch.no_grad():
                # write_image(path, sdf_to_image(model(xyz).detach().cpu().numpy(), resolution))
                img_out = model(xyz).reshape(img_shape).detach().cpu().numpy()
                write_image(path, img_out)
                pyexr.write(f"{i}.exr", img_out)
                
                # Compute the drjit sphere tracing
                dist = render(model, resolution).numpy()
                print(dist, dist.shape)
                write_image(f"sdf_{i}.jpg",dist.reshape(img_shape))
                pyexr.write(f"sdf_{i}.exr", dist.reshape(img_shape))
                
                output = output.squeeze(-1)
                ps_cloud.add_scalar_quantity(f"sdf_{i}", output.cpu().numpy())
                ps.show()
            
                # polyscope_sdf(model)
                # ps.show()
            print("done.")

            # Ignore the time spent saving the image
            prev_time = time.perf_counter()

            if i > 0 and interval < 1000:
                interval *= 10

    if args.result_filename:
        print(f"Writing '{args.result_filename}'... ", end="")
        with torch.no_grad():
            write_image(args.result_filename, model(xyz).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
        print("done.")

    tcnn.free_temporary_memory()