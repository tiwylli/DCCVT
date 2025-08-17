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

N_POINTS = 10000000 // 10
ERROR_SCALE = 1e5
COLOR_REF = (0.6, 0.6, 0.6)
COLOR_OTHER = (0.7, 0.5, 0.2)
COLOR_OURS = (0.2, 0.5, 0.7)
COLOR_POINT = (0.7, 0.2, 0.2)
CAMERA_CONFIG = {
    "fov": 26,
    "cam_position": np.array([-0.6, -2.2, 0.1]),
    "target": np.array([0, 0, -0.05])
}
GT_DIR = "/home/wylliam/dev/Kyushu_experiments/mesh/thingi32/"
# Beltegeuse default
if os.environ.get("USER", "") == "beltegeuse":
    GT_DIR = "/home/beltegeuse/projects/Voronoi/Kyushu_experiments/mesh/thingi32/"

def load_obj_vertices_faces(path):
    vertices = []
    faces = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # vertex
                parts = line.strip().split()
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith('f '):  # face
                parts = line.strip().split()[1:]
                # Keep original face structure (no triangulation)
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    return vertices, faces


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

def obj2image(path, cam_position=np.array([1, -2, 0]), target=np.array([0, 0, 0]), fov=60, aspect=1.0, rescale=False, color=(0.2, 0.5, 0.7), edge_width=1):
    """Render an OBJ file to an image using Polyscope."""
    
    # Trimesh without triangulate
    vertices, faces = load_obj_vertices_faces(path)
    if rescale:
        scale, pc_target = rescale_hotspot(path)
        vertices /= scale
    else:
        pc_target = None

    # Add [0, 0, 0.5] to vertices
    vertices += np.array([0, 0, -0.1])
    if pc_target is not None:
        pc_target += np.array([0, 0, -0.1])

    # TODO: See how to add shadows
    ps.set_ground_plane_mode("none")  # set +Z as up direction
    # ps.set_shadow_darkness(0.1)              # lighter shadows
    # ps.set_ground_height(0.) # in world coordinates

    ps.set_length_scale(1.)
    low = np.array((-1, -1, -1)) 
    high = np.array((1., 1., 1.)) 
    ps.set_bounding_box(low, high)


    # Make Z- orientation
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    # Change camera to orto
    ps.set_view_projection_mode("perspective")
    ps.set_length_scale(0.7)

    look_dir = target - cam_position
    look_dir /= np.linalg.norm(look_dir)  # Normalize the look direction

    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=fov, aspect=aspect)
    extrinsics = ps.CameraExtrinsics(root=cam_position, look_dir=look_dir, up_dir=(0., 0., 1.))
    params = ps.CameraParameters(intrinsics, extrinsics)

    ps.set_view_camera_parameters(params)

    ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, edge_width=edge_width, back_face_policy="identical", smooth_shade=False, material='clay', color=color)

    if pc_target is not None:
        # Draw point cloud
        ps.register_point_cloud("point_cloud", pc_target, color=COLOR_POINT, radius=0.007)
    img = ps.screenshot_to_buffer()

    ps.remove_all_structures()
    
    return img

import re
DIGIT_RUN = re.compile(r"(\d+)")
def extract_key_from_dir(dirname: str) -> str:
    """Extract numeric key (e.g., '64764') from folder name."""
    m = DIGIT_RUN.search(dirname)
    return m.group(1) if m else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render OBJ to image using Polyscope.")
    parser.add_argument("obj_directory", type=str, help="Path to the OBJ file.")
    parser.add_argument("--cam_position", type=float, nargs=3, default=[1, -2, 0], help="Camera position in world coordinates.")
    parser.add_argument("--target", type=float, nargs=3, default=[0, 0, -0.05], help="Target point in world coordinates.")
    parser.add_argument("--fov", type=float, default=30, help="Field of view in degrees.")
    parser.add_argument("--resolution", type=float, nargs=2, default=[512, 512], help="Resolution of the output image (width, height).")
    parser.add_argument("--rescale", action='store_true', help="Rescale the mesh based on its bounding box.")
    parser.add_argument("--color", type=float, nargs=3, default=[0.2, 0.5, 0.7], help="Color of the mesh in RGB format.")
    parser.add_argument("--edge_width", type=float, default=1.0, help="Width of the edges in the mesh.")
    
    parser.add_argument("--filter", type=str, default=None, help="Filter for OBJ files (e.g., 'final', 'init'). If None, all OBJ files will be processed.")
    parser.add_argument("--recursive", action='store_true', help="Recursively search for OBJ files in subdirectories.")
    parser.add_argument("--name", type=str, default=None, help="Name of the output image file. If not provided, the image will be saved in the same directory as the OBJ file with '_rendered' suffix.")
    parser.add_argument("--metrics", action='store_true', help="If set, compute metrics for the rendered images.")

    args = parser.parse_args()
    
    ps.set_allow_headless_backends(True)  
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    ps.set_automatically_compute_scene_extents(False)
    ps.set_window_size(args.resolution[0], args.resolution[1])
    ps.init()

    aspect = args.resolution[0] / args.resolution[1]
    
    gt_cache = {}
    
    if args.recursive:
        obj_files = glob.glob(os.path.join(args.obj_directory, "**", "*.obj"), recursive=True)
        if args.filter:
            obj_files = [f for f in obj_files if args.filter in f]
            
        # Sort files by their directory name to ensure consistent order
        obj_files.sort(key=lambda x: os.path.dirname(x))
        
        errors = {}
        for obj_path in obj_files:
            # Get last dirname
            dirname = os.path.dirname(obj_path)
            technique_name = dirname.split(os.sep)[-1]
            
            if args.metrics:
                key = extract_key_from_dir(technique_name)
                if key is None:
                    print(f"[WARN] Could not extract key from {obj_path}, skipping.")
                    continue
                
                gt_obj = os.path.join(GT_DIR, f"{key}.obj")
                if not os.path.exists(gt_obj):
                    print(f"[WARN] GT mesh not found for key {key}, skipping {obj_path}")
                    continue
                
                if key in gt_cache:
                    # gt_pts, gt_normals, gt_mesh, gt_scene = gt_cache[key]
                    gt_pts, gt_normals, gt_mesh = gt_cache[key]
                    # print("Resample GT points...")
                    # gt_pts, gt_normals, gt_mesh = su.sample_points_on_mesh(gt_obj, n_points=N_POINTS, GT=True)
                else:
                    try:
                        ps.init()
                        print(f"Compute GT points... {key}")
                        gt_pts, gt_normals, gt_mesh = su.sample_points_on_mesh(gt_obj, n_points=N_POINTS, GT=True)
                        # gt_scene = fcpw.scene_3D()
                        # gt_scene.set_object_count(1)
                        # gt_scene.set_object_vertices(np.array(gt_mesh.vertices), 0)
                        # gt_scene.set_object_triangles(np.array(gt_mesh.faces), 0)
                        # aggregate_type = fcpw.aggregate_type.bvh_surface_area
                        # build_vectorized_bvh = True
                        # gt_scene.build(aggregate_type, build_vectorized_bvh)
                        gt_cache[key] = (gt_pts, gt_normals, gt_mesh) #, gt_scene)
                    except Exception as e:
                        print(f"[ERROR] sampling GT for {gt_obj}: {e}")
                        continue
            
            
            print(f"Rendering {obj_path}...")
            # Render the OBJ file to an image
            img = obj2image(obj_path, np.array(args.cam_position), np.array(args.target), args.fov, aspect, args.rescale, tuple(args.color), args.edge_width)
            
            # Make the image inside the directory of the OBJ file

            if args.name:
                output = os.path.join(args.obj_directory, f"{args.name}_{technique_name}.png")
            else:
                output = os.path.join(dirname, f"{os.path.basename(obj_path).replace('.obj', '.png')}")
            # Save the image
            Image.fromarray(img).save(output)
            
            
            if args.metrics:
                obj_pts, obj_normals, obj_mesh = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)
                # scene_obj = fcpw.scene_3D()
                # scene_obj.set_object_count(1)
                # scene_obj.set_object_vertices(np.array(obj_mesh.vertices), 0)
                # scene_obj.set_object_triangles(np.array(obj_mesh.faces), 0)
                # aggregate_type = fcpw.aggregate_type.bvh_surface_area
                # build_vectorized_bvh = True
                # scene_obj.build(aggregate_type, build_vectorized_bvh)


                # cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 = (
                #     su.chamfer_accuracy_completeness_f1_accel(obj_pts, obj_normals, gt_cache[key][0], gt_cache[key][1], scenes=(gt_scene, scene_obj))
                # )
                # Accel
                cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 =  voronoiaccel.compute_error_fcpw(np.array(gt_mesh.vertices), np.array(gt_mesh.faces).astype(np.int32), np.array(gt_pts), 
                                                                    np.array(obj_mesh.vertices), np.array(obj_mesh.faces).astype(np.int32), np.array(obj_pts), 0.003, 0.45)
                cd2 = cd2 * ERROR_SCALE  # Scale the Chamfer distance
                cd1 = cd1 * ERROR_SCALE  # Scale the Chamfer distance   
                completeness1 = completeness1 * ERROR_SCALE  # Scale the completeness
                completeness2 = completeness2 * ERROR_SCALE  # Scale the completeness
                accuracy1 = accuracy1 * ERROR_SCALE  # Scale the accuracy
                accuracy2 = accuracy2 * ERROR_SCALE  # Scale the accuracy

                errors[obj_path] = {
                    "cd1": cd1,
                    "cd2": cd2,
                    "f1": f1,
                    "nc": nc,
                    "recall": recall,
                    "precision": precision,
                    "completeness1": completeness1,
                    "completeness2": completeness2,
                    "accuracy1": accuracy1,
                    "accuracy2": accuracy2
                }
                print(f"  CD: {cd2:.4f},\t F1: {f1:.4f},\t NC: {nc:.4f},\t Recall: {recall:.4f},\t Precision: {precision:.4f},\t Completeness2: {completeness2:.4f},\t Accuracy2: {accuracy2:.4f}")

        # Compute average 
        if errors:
            avg_errors = {k: np.mean([e[k] for e in errors.values()]) for k in errors[next(iter(errors))].keys()}
            print("Average metrics:")
            print(f"  CD: {avg_errors['cd2']:.4f},\t F1: {avg_errors['f1']:.4f},\t NC: {avg_errors['nc']:.4f},\t Recall: {avg_errors['recall']:.4f},\t Precision: {avg_errors['precision']:.4f},\t Completeness2: {avg_errors['completeness2']:.4f},\t Accuracy2: {avg_errors['accuracy2']:.4f}")

    else:
        for obj_path in glob.glob(os.path.join(args.obj_directory, "*.obj")):
            if args.filter and args.filter not in obj_path:
                continue

            print(f"Rendering {obj_path}...")
            # Render the OBJ file to an image
            img = obj2image(obj_path, np.array(args.cam_position), np.array(args.target), args.fov, aspect, args.rescale, tuple(args.color), args.edge_width)
            
            # Make the image inside the directory of the OBJ file
            output = obj_path.replace(".obj", "_rendered.png")

            # Save the image
            Image.fromarray(img).save(output)