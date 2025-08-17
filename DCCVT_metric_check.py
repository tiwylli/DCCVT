import polyscope as ps
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import argparse
import sdfpred_utils.sdfpred_utils as su
import fcpw
import voronoiaccel
import sys

N_POINTS = 10000000 // 100
ERROR_SCALE = 1e5
GT_DIR = "/home/wylliam/dev/Kyushu_experiments/mesh/thingi32/"
# Beltegeuse default
if os.environ.get("USER", "") == "beltegeuse":
    GT_DIR = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments/mesh/thingi32/"


def rescale_hotspot(mesh_path):
    # Function to rescale exactly how we optimize
    mesh = trimesh.load(mesh_path)

    points_gt, _ = trimesh.sample.sample_surface(mesh, 9600) # previous: (32**2)*150)
    # center and scale point cloud
    cp = points_gt.mean(axis=0)
    points = points_gt - cp[None, :]
    scale = np.percentile(np.linalg.norm(points, axis=-1), 70) / 0.45
    scale = max(scale, np.abs(points).max())
    return scale, points_gt / scale

import re
DIGIT_RUN = re.compile(r"(\d+)")
def extract_key_from_dir(dirname: str) -> str:
    """Extract numeric key (e.g., '64764') from folder name."""
    m = DIGIT_RUN.search(dirname)
    return m.group(1) if m else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render OBJ to image using Polyscope.")
    parser.add_argument("obj_file", type=str, help="Path to the OBJ file.")
    args = parser.parse_args()
    
    # ps.set_autocenter_structures(False)
    # ps.set_autoscale_structures(False)
    # ps.set_automatically_compute_scene_extents(False)
    # ps.init()

    obj_path = args.obj_file
    # Get last dirname
    dirname = os.path.dirname(obj_path)
    technique_name = dirname.split(os.sep)[-1]


    key = extract_key_from_dir(technique_name)
    if key is None:
        print(f"[WARN] Could not extract key from {obj_path}, skipping.")
        sys.exit(1)

    gt_obj = os.path.join(GT_DIR, f"{key}.obj")
    if not os.path.exists(gt_obj):
        print(f"[WARN] GT mesh not found for key {key}, skipping {obj_path}")
        sys.exit(1)


    ps.init()
    print(f"Generate points... {key}")
    gt_pts, gt_normals, gt_mesh = su.sample_points_on_mesh(gt_obj, n_points=N_POINTS, GT=True)
    obj_pts, obj_normals, obj_mesh = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)
    
    print("Compute scenes ... ")
    gt_scene = fcpw.scene_3D()
    gt_scene.set_object_count(1)
    gt_scene.set_object_vertices(np.array(gt_mesh.vertices), 0)
    gt_scene.set_object_triangles(np.array(gt_mesh.faces), 0)
    aggregate_type = fcpw.aggregate_type.bvh_surface_area
    build_vectorized_bvh = True
    gt_scene.build(aggregate_type, build_vectorized_bvh)
    
    scene_obj = fcpw.scene_3D()
    scene_obj.set_object_count(1)
    scene_obj.set_object_vertices(np.array(obj_mesh.vertices), 0)
    scene_obj.set_object_triangles(np.array(obj_mesh.faces), 0)
    aggregate_type = fcpw.aggregate_type.bvh_surface_area
    build_vectorized_bvh = True
    scene_obj.build(aggregate_type, build_vectorized_bvh)

    print("Compute metrics ... ")
    # cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 =  voronoiaccel.compute_error_fcpw(
    #     np.array(gt_mesh.vertices), np.array(gt_mesh.faces).astype(np.int32), np.array(gt_pts), np.array(gt_normals),
    #     np.array(obj_mesh.vertices), np.array(obj_mesh.faces).astype(np.int32), np.array(obj_pts), np.array(obj_normals),
    #     0.003, 0.45)
    cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2, cp_obj, cp_pts = (
        su.chamfer_accuracy_completeness_f1_accel(obj_pts, obj_normals, gt_pts, gt_normals, scenes=(gt_scene, scene_obj))
    )
    cd2 = cd2 * ERROR_SCALE  # Scale the Chamfer distance
    cd1 = cd1 * ERROR_SCALE  # Scale the Chamfer distance   
    completeness1 = completeness1 * ERROR_SCALE  # Scale the completeness
    completeness2 = completeness2 * ERROR_SCALE  # Scale the completeness
    accuracy1 = accuracy1 * ERROR_SCALE  # Scale the accuracy
    accuracy2 = accuracy2 * ERROR_SCALE  # Scale the accuracy
    print(f"  CD: {cd2:.4f},\t F1: {f1:.4f},\t NC: {nc:.4f},\t Recall: {recall:.4f},\t Precision: {precision:.4f},\t Completeness2: {completeness2:.4f},\t Accuracy2: {accuracy2:.4f}")

    ps.init()
    ps.register_surface_mesh("GT Mesh", gt_mesh.vertices, gt_mesh.faces, enabled=True)
    ps.register_surface_mesh("OBJ Mesh", obj_mesh.vertices, obj_mesh.faces, enabled=True)
    ps.register_point_cloud("GT Points", gt_pts, enabled=True, radius=0.0005)
    ps.register_point_cloud("OBJ Points", obj_pts, enabled=True, radius=0.0005)
    ps.register_point_cloud("GT Closest Points", cp_pts, enabled=True, radius=0.0005)
    ps.register_point_cloud("OBJ Closest Points", cp_obj, enabled=True, radius=0.0005)
    ps.show()
