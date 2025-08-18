import os
import re
import json
import tqdm
import torch
import pandas as pd
import numpy as np
import sdfpred_utils.sdfpred_utils as su
import polyscope as ps
import argparse
import voronoiaccel


# ---------------------------
# Args
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Compute metrics over experiment folders.")
    parser.add_argument(
        "--root-dir",
        default="/home/wylliam/dev/Kyushu_experiments/",
        help="Project root directory (default matches current script).",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Directory with GT meshes (defaults to <root-dir>/mesh/thingi32/).",
    )
    parser.add_argument(
        "--experiments-dir",
        default=None,
        help="Directory with experiment outputs (defaults to <root-dir>/outputs/FIGURE_CASE_64764/).",
    )
    # Filters
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Include OBJ files whose name contains 'final'.",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Include OBJ files whose name contains 'init'.",
    )
    # Optional: control number of sampled points (kept default)
    parser.add_argument(
        "--n-points",
        type=int,
        default=10000000 // 10,
        help="Number of points to sample per mesh for metrics.",
    )
    args = parser.parse_args()

    # Defaults dependent on root-dir
    if args.gt_dir is None:
        args.gt_dir = os.path.join(args.root_dir, "mesh/thingi32/")
    if args.experiments_dir is None:
        # keep your current default
        args.experiments_dir = os.path.join(args.root_dir, "outputs/FIGURE_CASE_64764/")

    # If neither flag is given, default to include 'final' (old behavior)
    if not args.include_final and not args.include_init:
        args.include_final = True

    return args


# ---------------------------
# Config (device, RNG)
# ---------------------------
device = torch.device("cuda:0")
print("Using device:", torch.cuda.get_device_name(device))
torch.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)

# ---------------------------
# Helpers
# ---------------------------
DIGIT_RUN = re.compile(r"(\d+)")
ERROR_SCALE = 1e5


def extract_key_from_dir(dirname: str) -> str:
    """Extract numeric key (e.g., '64764') from folder name."""
    m = DIGIT_RUN.search(dirname)
    return m.group(1) if m else None


def list_obj_files(folder: str, include_final: bool = True, include_init: bool = False):
    """
    Return absolute paths to OBJ files filtered by name containing 'final' and/or 'init'.
    If both flags are True, returns union.
    """
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        fl = f.lower()
        if not fl.endswith(".obj"):
            continue
        is_final = "final" in fl
        is_init = "init" in fl
        if (include_final and is_final) or (include_init and is_init):
            out.append(os.path.join(folder, f))
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    ROOT_DIR = args.root_dir
    GT_DIR = args.gt_dir
    EXPERIMENTS_DIR = args.experiments_dir
    N_POINTS = args.n_points
    OUT_CSV = os.path.join(EXPERIMENTS_DIR, "metrics.csv")  # keep filename as-is

    gt_cache = {}
    records = []

    subdirs = [d for d in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    print(f"Scanning {len(subdirs)} subfolders in {EXPERIMENTS_DIR}")

    for dname in tqdm.tqdm(subdirs, desc="Folders"):
        folder_path = os.path.join(EXPERIMENTS_DIR, dname)
        key = extract_key_from_dir(dname)
        if key is None:
            continue

        gt_obj = os.path.join(GT_DIR, f"{key}.obj")
        if not os.path.exists(gt_obj):
            print(f"[WARN] GT mesh not found for key {key}, skipping {dname}")
            continue

        if key not in gt_cache:
            try:
                # ps.init()
                gt_pts, gt_normals, gt_mesh = su.sample_points_on_mesh(gt_obj, n_points=N_POINTS, GT=True)
                gt_cache[key] = (gt_pts, gt_normals, gt_mesh)
            except Exception as e:
                print(f"[ERROR] sampling GT for {gt_obj}: {e}")
                continue

        final_or_init_objs = list_obj_files(
            folder_path, include_final=args.include_final, include_init=args.include_init
        )
        if not final_or_init_objs:
            continue

        for obj_path in final_or_init_objs:
            try:
                obj_pts, obj_normals, obj_mesh = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)

                # cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 = (
                #     su.chamfer_accuracy_completeness_f1(obj_pts, obj_normals, gt_cache[key][0], gt_cache[key][1])
                # )

                # Accel
                cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 = (
                    voronoiaccel.compute_error_fcpw(
                        np.array(gt_cache[key][2].vertices),
                        np.array(gt_cache[key][2].faces).astype(np.int32),
                        np.array(gt_cache[key][0]),
                        np.array(gt_cache[key][1]),
                        np.array(obj_mesh.vertices),
                        np.array(obj_mesh.faces).astype(np.int32),
                        np.array(obj_pts),
                        np.array(obj_normals),
                        0.003,
                        0.45,
                    )
                )
                cd2 = cd2 * ERROR_SCALE  # Scale the Chamfer distance
                cd1 = cd1 * ERROR_SCALE  # Scale the Chamfer distance
                completeness1 = completeness1 * ERROR_SCALE  # Scale the completeness
                completeness2 = completeness2 * ERROR_SCALE  # Scale the completeness
                accuracy1 = accuracy1 * ERROR_SCALE  # Scale the accuracy
                accuracy2 = accuracy2 * ERROR_SCALE  # Scale the accuracy

                fname = os.path.basename(obj_path)
                label = f"{dname}_{os.path.splitext(fname)[0]}"
                records.append(
                    {
                        "object_id": key,
                        "folder": dname,
                        "filename": fname,
                        "label": label,
                        "chamfer_distance_1": float(cd1),
                        "chamfer_distance_2": float(cd2),
                        "f1_score": float(f1),
                        "normal_consistency": float(nc),
                        "recall": float(recall),
                        "precision": float(precision),
                        "completeness_1": float(completeness1),
                        "completeness_2": float(completeness2),
                        "accuracy_1": float(accuracy1),
                        "accuracy_2": float(accuracy2),
                    }
                )
            except Exception as e:
                print(f"[ERROR] metrics for {obj_path}: {e}")

    if not records:
        print("No records found; nothing to write.")
        return

    df = pd.DataFrame(records)
    df.sort_values(by=["object_id", "folder", "filename"], inplace=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
