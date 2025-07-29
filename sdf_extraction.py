from skimage import measure
import numpy as np
from sdfpred_utils.Steik_utils import get_3d_grid
import argparse
import trimesh
import polyscope
import sys
import torch
import tqdm
import kaolin
import pygdel3d
import scipy.spatial

def grid_to_mesh(grid_dict, z, scale=1.0, translate=(0, 0, 0)):
    cell_width = grid_dict["xyz"][0][2] - grid_dict["xyz"][0][1]

    # Check z value shape
    if z.ndim != 3:
        raise ValueError("Input z must be a 3D array.")

    # Perform marching cubes to extract the mesh from the volume
    verts, faces, normals, values = measure.marching_cubes(
        volume=z, level=0.0, spacing=(cell_width, cell_width, cell_width)
    )

    # Adjust vertices based on the grid dictionary and scaling
    verts = verts + np.array(
        [grid_dict["xyz"][0][0], grid_dict["xyz"][1][0], grid_dict["xyz"][2][0]]
    )
    verts = verts * (1 / scale) - translate 

    return (verts, faces, normals, values)

###### SIMPLE SDF FUNCTIONS ######

def sdf_sphere(pnts):
    """
    Compute the signed distance function for a sphere centered at the origin with radius 0.5.
    :param pnts: Points (N, 3)
    :return: SDF values (N,)
    """
    return np.linalg.norm(pnts, axis=1) - 0.5

####### HOTSPOT SDF ########

sys.path.append("3rdparty/HotSpot")
import models.Net as Net
def load_hotspot(input_path):
    # From DCCVT code
    model = Net.Network(
        latent_size=0,  # args.latent_size,
        in_dim=3,
        decoder_hidden_dim=128,  # args.decoder_hidden_dim,
        nl="sine",  # args.nl,
        encoder_type="none",  # args.encoder_type,
        decoder_n_hidden_layers=5,  # args.decoder_n_hidden_layers,
        neuron_type="quadratic",  # args.neuron_type,
        init_type="mfgi",  # args.init_type,
        sphere_init_params=[1.6, 0.1],  # args.sphere_init_params,
        n_repeat_period=30,  # args.n_repeat_period,
    )
    model.to(0)
    model.load_state_dict(torch.load(input_path, weights_only=True, map_location=torch.device("cuda")))
    return model

def sdf_hotspot(pnts, input_path):
    # Load the HotSpot decoder model
    decoder = load_hotspot(input_path)
    # Perform inference on the points (batched)
    z = []
    for point in tqdm.tqdm(torch.split(pnts, 100000, dim=0)):
        # point: (100000, 3)
        point = torch.tensor(point, device=0, dtype=torch.float32)

        z.append(
            decoder(point)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
    z = (
        np.concatenate(z, axis=0)
    )
    return z

######## OUR SDF ########

def volume_tetrahedron(a, b, c, d):
    ad = a - d
    bd = b - d
    cd = c - d
    n = torch.cross(bd, cd)
    return torch.abs((ad * n).sum(dim=-1)) / 6.0

def orientation(a, b, c, d):
    return torch.sign(torch.einsum('ij,ij->i', torch.cross(b - a, c - a), d - a))

def is_inside_tetrahedron(pnts, a, b, c, d):
    # pnts: (N, 3)
    N = pnts.shape[0]

    a = a.expand(N, 3)
    b = b.expand(N, 3)
    c = c.expand(N, 3)
    d = d.expand(N, 3)

    o1 = orientation(b, c, d, a) * orientation(b, c, d, pnts) >= 0
    o2 = orientation(a, c, d, b) * orientation(a, c, d, pnts) >= 0
    o3 = orientation(a, b, d, c) * orientation(a, b, d, pnts) >= 0
    o4 = orientation(a, b, c, d) * orientation(a, b, c, pnts) >= 0

    return o1 & o2 & o3 & o4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mesh from SDF grid.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for the mesh.")
    parser.add_argument("--translate", type=float, nargs=3, default=(0, 0, 0), help="Translation vector for the mesh.")
    parser.add_argument("--output", type=str, default="", help="Output file path for the mesh.")
    parser.add_argument("--viz", action="store_true", help="Visualize the mesh using polyscope.", default=False)
    parser.add_argument("--resolution", type=int, default=64, help="Resolution of the 3D grid.")
    parser.add_argument("--method", type=str, default="marching_cubes", choices=["mc", "mt"], help="Method to use for mesh extraction.")

    # add submodule for command line arguments
    # hotspot, tetrahedron, sphere
    parser.add_argument("--shape", type=str, default="sphere", choices=["hotspot", "ours", "sphere"], help="Shape to use for SDF.")
    parser.add_argument("--input", type=str, default="", help="Input file path for the SDF in various format.")

    args = parser.parse_args()

    verts, faces, normals, values = None, None, None, None
    if args.method == "mc":
        # Example for a sphere
        grid_dict = get_3d_grid(resolution=args.resolution)

        # Points (N, 3)
        pnts = grid_dict["grid_points"]
        
        # SDF values (N,)
        if args.shape == "sphere":
            # Compute SDF for a sphere
            sdf_values = sdf_sphere(pnts)
        elif args.shape == "hotspot":
            # Placeholder for hotspot SDF computation
            sdf_values = sdf_hotspot(pnts, args.input)
        elif args.shape == "ours":
            # Load NPZ file containing the SDF points
            data = np.load(args.input)
            sites = data["sites"]
            sdf_values = data["sdf_values"]

            # Compute delaunay triangulation
            d3dsimplices, _ = pygdel3d.triangulate(np.array(sites))

            # Go through all the simplices to found where the grid points 
            # by checking each tetrahedron if it contains the grid points
            sites = torch.tensor(sites, device=0, dtype=torch.float32)
            d3dsimplices = torch.tensor(d3dsimplices, device=0)
            index_points = torch.zeros(pnts.shape[0], dtype=torch.int32, device=0)
            pnts = torch.tensor(pnts, device=0, dtype=torch.float32)

            # Use marching tetrahedra to compute the SDF values at the grid points
            # d3dsimplices = d3dsimplices[:10000]
            for index, simplex in tqdm.tqdm(enumerate(d3dsimplices), total=len(d3dsimplices)):
                tet_points = sites[simplex]
                
                # Compute the tetrahedron barycentric coordinates 
                # from tet_points using volume_thetahedron
                tet_volume = volume_tetrahedron(
                    tet_points[0], tet_points[1], tet_points[2], tet_points[3]
                )

                is_inside = is_inside_tetrahedron(pnts, tet_points[0], tet_points[1], tet_points[2], tet_points[3])
                index_points[is_inside] = index

            print(index_points)
            print(torch.max(index_points), torch.min(index_points))
            sdf_values = torch.zeros(pnts.shape[0], device=0, dtype=torch.float32)
            
            # Get all tet points vectorized based of index_points
            tet_points = sites[d3dsimplices[index_points]]
            tet_sdfs = sdf_values[d3dsimplices[index_points]]

            # For each point, get its corresponding tetrahedron vertices and SDFs
            # Compute barycentric coordinates for each point in its tetrahedron
            def barycentric_coords(p, tet):
                # p: (N, 3), tet: (N, 4, 3)
                v0 = tet[:, 0]
                v1 = tet[:, 1]
                v2 = tet[:, 2]
                v3 = tet[:, 3]
                # Compute volumes for barycentric coordinates
                def vol(a, b, c, d):
                    return torch.abs(torch.einsum(
                        'ij,ij->i',
                        torch.cross(b - a, c - a),
                        d - a
                    )) / 6.0
                v = vol(v0, v1, v2, v3)
                w0 = vol(p, v1, v2, v3) / v
                w1 = vol(v0, p, v2, v3) / v
                w2 = vol(v0, v1, p, v3) / v
                w3 = vol(v0, v1, v2, p) / v
                return torch.stack([w0, w1, w2, w3], dim=1)

            bary_coords = barycentric_coords(pnts, tet_points)
            # Interpolate SDF values using barycentric coordinates
            tet_sdf_values = data["sdf_values"][d3dsimplices[index_points].cpu().numpy()]  # (N, 4)
            tet_sdf_values = torch.tensor(tet_sdf_values, device=0, dtype=torch.float32)
            sdf_values = (bary_coords * tet_sdf_values).sum(dim=1)
            sdf_values = sdf_values.cpu().numpy()

        else:
            raise ValueError(f"Unknown shape: {args.shape}")
        
        # Reshape SDF values to 3D grid
        # TODO: Weird reshape, should be fixed
        z = sdf_values.reshape(
            grid_dict["xyz"][1].shape[0], grid_dict["xyz"][0].shape[0], grid_dict["xyz"][2].shape[0]
            ).transpose(1, 0, 2).astype(np.float32)
        
        # Extract mesh with marching cubes
        verts, faces, normals, values = grid_to_mesh(grid_dict, z, scale=args.scale, translate=args.translate)
    elif args.method == "mt":
        sites = None
        sdf_values = None
        if args.shape == "ours":
            # Load NPZ file containing the SDF points
            # np.savez(
            #     s,
            #     sites=sites.detach().cpu().numpy(),
            #     sdf_values=sdf_values.detach().cpu().numpy(),
            #     # sdf_gradients=sdf_gradients.detach().cpu().numpy(),
            #     # sdf_hessians=hess_sdf.detach().cpu().numpy(),
            #     v_vect=v_vect.detach().cpu().numpy(),
            #     f_vect=f_vect,
            #     train_time=t,
            #     # grads_mesh_extraction_time=time() - t0 - train_time,
            #     accuracy=accuracy,
            #     completeness=completeness,
            #     chamfer=chamfer,
            #     precision=precision,
            #     recall=recall,
            #     f1=f1,
            # )
            # from this load sites and sdf_values
            data = np.load(args.input)
            sites = data["sites"]
            sdf_values = data["sdf_values"]

            # Convert to torch tensors
            pnts = torch.tensor(sites, device=0, dtype=torch.float32).cpu().numpy() # Depending on the delaunay implementation
            sdf_values = torch.tensor(sdf_values, device=0, dtype=torch.float32)
            
        else:
            # Create a regular grid for the mesh extraction
            grid_dict = get_3d_grid(resolution=args.resolution)
            pnts = grid_dict["grid_points"] 
            
            # Add small random noise to the points
            noise = np.random.normal(scale=1e-4, size=pnts.shape)
            pnts += noise # TODO: This is a hack to avoid numerical issues with the SDF computation

            # Evaluate the SDF at the grid points
            sdf_values = None
            if args.shape == "sphere":
                # Compute SDF for a sphere
                sdf_values = sdf_sphere(pnts)
            elif args.shape == "hotspot":
                # Placeholder for hotspot SDF computation
                sdf_values = sdf_hotspot(pnts, args.input)
            else:
                raise ValueError(f"Unknown shape: {args.shape}")
            sdf_values = torch.tensor(sdf_values, device=0, dtype=torch.float32)

        # Use gDel3D to build the tetrahedral mesh
        print("Building tetrahedral mesh using pygdel3d...")
        
        # d3dsimplices, _ = pygdel3d.triangulate(np.array(pnts))
        d3dsimplices = scipy.spatial.Delaunay(pnts).simplices

        print("Performing marching tetrahedra...")
        d3dsimplices = torch.tensor(d3dsimplices, device=0)
        pnts = torch.tensor(pnts, device=0, dtype=torch.float32)
        marching_tetrehedra_mesh = kaolin.ops.conversions.marching_tetrahedra(
            pnts.unsqueeze(0), d3dsimplices, sdf_values.unsqueeze(0), return_tet_idx=False
        )
        vertices_list, faces_list = marching_tetrehedra_mesh
        verts = vertices_list[0].detach().cpu().numpy()
        faces = faces_list[0].detach().cpu().numpy()
        normals = None # Assuming normals are not computed in this case (TODO: Implement if needed)

    if args.output != "":
        # Save mesh to file
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        mesh.export(args.output)

    if args.viz:
        # Visualize the mesh using polyscope
        polyscope.init()
        polyscope_mesh = polyscope.register_surface_mesh("mesh", verts, faces)
        polyscope.show()





