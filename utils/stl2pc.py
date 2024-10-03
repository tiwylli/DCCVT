import numpy as np
import trimesh
from pxr import Usd, UsdGeom, Gf

# Function to sample points from an STL file
def sample_stl_to_pointcloud(stl_file, num_samples=10000):
    # Load the STL file using trimesh
    mesh = trimesh.load_mesh(stl_file)
    
    # Sample points on the surface
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    
    return points

# Function to save the point cloud as a USD file
def save_pointcloud_to_usd(points, usd_file):
    # Create a new USD stage (file)
    stage = Usd.Stage.CreateNew(usd_file)
    
    # Define the root Xform (transform node)
    root = UsdGeom.Xform.Define(stage, '/pointcloud')

    # Create a Points geometry under the Xform
    points_geom = UsdGeom.Points.Define(stage, root.GetPath().AppendChild('points'))

    # Set the points' positions
    points_geom.GetPointsAttr().Set([Gf.Vec3f(p[0], p[1], p[2]) for p in points])
    
    # Save and close the stage
    stage.GetRootLayer().Save()

if __name__ == "__main__":
    # Input STL file
    stl_file = "path/to/input/file.stl"
    
    # Output USD file
    usd_file = "path/to/output/file.usd"
    
    # Number of points to sample
    num_samples = 10000
    
    # Step 1: Sample points from the STL file
    points = sample_stl_to_pointcloud(stl_file, num_samples)
    
    # Step 2: Save the point cloud to a USD file
    save_pointcloud_to_usd(points, usd_file)

    print(f"Point cloud saved to {usd_file}")
