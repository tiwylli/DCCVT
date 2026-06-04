import numpy as np
import os
import argparse
import h5py
from multiprocessing import Process, Queue
import time
import subprocess
import trimesh
import joblib
from tqdm import tqdm
import utils

DEFAULT_INPUT_DIR = "/data/nmaruani/DATASETS/ABC/"
DEFAULT_OUTPUT_DIR = "/data/nmaruani/DATASETS/gt_Quadrics/"
DEFAULT_SDFGEN = "./SDFGen"


def get_gt_from_intersectionpn(name_list):
    cell_voxel_size = 8
    num_of_int_params = 3
    num_of_float_params = 3

    point_sample_num = int(1e6)

    grid_size_list = [32, 64, 128]
    LOD_gt_int = {}
    LOD_gt_float = {}
    LOD_input_sdf = {}
    LOD_input_voxel = {}
    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        LOD_gt_int[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1, num_of_int_params], np.uint8)
        LOD_gt_float[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1, num_of_float_params], np.float32)
        LOD_input_sdf[grid_size] = np.ones(
            [grid_size_1, grid_size_1, grid_size_1], np.float32)
        LOD_input_voxel[grid_size] = np.zeros(
            [grid_size_1, grid_size_1, grid_size_1], np.uint8)

    in_name = name_list[2]
    out_name = name_list[3]
    sdfgen = name_list[4] if len(name_list) > 4 else DEFAULT_SDFGEN

    in_obj_name = in_name + ".obj"
    in_sdf_name = in_name + ".sdf"
    out_hdf5_name = out_name + ".hdf5"
    tmp_hdf5_name = out_hdf5_name + ".tmp"
    subprocess.run([sdfgen, in_obj_name, "128", "0"], check=True)

    # read
    gt_mesh = trimesh.load(in_obj_name)
    gt_points, face_idx = trimesh.sample.sample_surface(
        gt_mesh, point_sample_num)
    gt_normals = np.array(gt_mesh.face_normals[face_idx])

    sdf_129 = utils.read_sdf_file_as_3d_array(in_sdf_name)  # 128

    # compute gt
    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        voxel_size = grid_size*cell_voxel_size
        downscale = 1024//voxel_size
        # prepare downsampled voxels and intersections
        tmp_sdf = sdf_129[0::downscale, 0::downscale, 0::downscale]
        LOD_input_sdf[grid_size][:] = tmp_sdf

    # record data
    hdf5_file = h5py.File(tmp_hdf5_name, 'w')
    hdf5_file.create_dataset(
        "pointcloud", [point_sample_num, 3], np.float32, compression=9)
    hdf5_file["pointcloud"][:] = gt_points

    hdf5_file.create_dataset(
        "normals", [point_sample_num, 3], np.float32, compression=9)
    hdf5_file["normals"][:] = gt_normals

    for grid_size in grid_size_list:
        grid_size_1 = grid_size+1
        hdf5_file.create_dataset(str(
            grid_size)+"_sdf", [grid_size_1, grid_size_1, grid_size_1], np.float32, compression=9)
        hdf5_file[str(grid_size)+"_sdf"][:] = LOD_input_sdf[grid_size]
    hdf5_file.close()
    os.replace(tmp_hdf5_name, out_hdf5_name)
    os.remove(in_sdf_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PoNQ ABC HDF5 files from normalized model.obj files.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help="Directory containing one subdirectory per model with model.obj.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where generated .hdf5 files are written.")
    parser.add_argument("--names-file", default=None,
                        help="Optional split file containing model IDs or .hdf5 names to process.")
    parser.add_argument("--sdfgen", default=DEFAULT_SDFGEN,
                        help="Path to the SDFGen executable.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs for HDF5 generation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum number of names to process, useful for smoke tests.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip names whose output .hdf5 already exists.")
    return parser.parse_args()


def get_model_names(target_dir, names_file=None, limit=None):
    if names_file is None:
        obj_names = os.listdir(target_dir)
        obj_names = sorted(obj_names)
    else:
        with open(names_file, 'r') as f:
            obj_names = [os.path.splitext(line.strip())[0]
                         for line in f if line.strip()]

    if limit is not None:
        obj_names = obj_names[:limit]
    return obj_names


if __name__ == '__main__':
    args = parse_args()

    target_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(target_dir):
        print("ERROR: this dir does not exist: "+target_dir)
        exit()

    write_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if not os.path.exists(args.sdfgen):
        print("ERROR: SDFGen does not exist: "+args.sdfgen)
        exit()

    obj_names = get_model_names(target_dir, args.names_file, args.limit)

    obj_names_len = len(obj_names)

    list_of_names = []
    for idx in range(obj_names_len):
        in_name = os.path.join(target_dir, obj_names[idx], "model")
        out_name = os.path.join(write_dir, obj_names[idx])
        if args.skip_existing and os.path.exists(out_name + ".hdf5"):
            continue
        if not os.path.exists(in_name + ".obj"):
            raise FileNotFoundError(in_name + ".obj")
        list_of_names.append(
            [0, idx, in_name, out_name, args.sdfgen])
    joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(get_gt_from_intersectionpn)
                               (name) for name in (tqdm(list_of_names)))
   
