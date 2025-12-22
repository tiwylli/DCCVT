"""Filesystem IO helpers for saving meshes, point clouds, and metadata."""

from pathlib import Path
import shutil
from typing import Iterable, List

import numpy as np


def save_npz_bundle(sites, sites_sdf, time, args, output_file: str) -> None:
    np.savez(
        output_file,
        sites=sites.detach().cpu().numpy(),
        sites_sdf=sites_sdf.detach().cpu().numpy(),
        train_time=time,
        args=str(args),
    )


def save_obj_mesh(filename: str, vertices, faces) -> None:
    """
    Save a mesh to an OBJ file.

    Args:
        filename: str, path to output .obj file
        vertices: (N, 3) array of float vertex positions
        faces: (M, 3) or (M, K) array of int indices (0-based)
    """
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Offset indices by +1 for OBJ (1-based indexing)
        for face in faces:
            # Ensure all face entries are written, even if quads or ngons
            indices = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {indices}\n")


def save_point_cloud_ply(filename: str, points) -> None:
    """
    Save a point cloud to a PLY file.

    Args:
        filename: str, path to output .ply file
        points: (N, 3) array of float point positions
    """
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def copy_experiment_script(arg_lists: Iterable[List[str]], script_path: str, output_dir: str) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    script_copy_path = output_dir_path / Path(script_path).name
    shutil.copy2(script_path, script_copy_path)

    # Copy arg_lists to a sidecar file for reproducibility
    arg_list_file = output_dir_path / "arg_lists.txt"
    with arg_list_file.open("w", encoding="utf-8") as f:
        for arg_list in arg_lists:
            f.write(" ".join(arg_list) + "\n")
    print(f"Copied script to {script_copy_path} and arg lists to {arg_list_file}")
