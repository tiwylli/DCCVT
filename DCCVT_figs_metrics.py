import os
import re
import json
import tqdm
import torch
import pandas as pd
import numpy as np
import sdfpred_utils.sdfpred_utils as su
import polyscope as ps

# ---------------------------
# Config
# ---------------------------
device = torch.device("cuda:0")
print("Using device:", torch.cuda.get_device_name(device))
torch.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)

ROOT_DIR = "/home/wylliam/dev/Kyushu_experiments/"
GT_DIR = os.path.join(ROOT_DIR, "mesh/thingi32/")

# EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "outputs/FIGURE_CASE_441708/")
# EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "outputs/FIGURE_CASE_64764/")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "outputs/ALL_CASE_DCCVT/")

# EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "outputs/Ablation_64764/")

OUT_CSV = os.path.join(EXPERIMENTS_DIR, "metrics_final_obj_only.csv")
N_POINTS = 100000

# ---------------------------
# Helpers
# ---------------------------
DIGIT_RUN = re.compile(r"(\d+)")


def extract_key_from_dir(dirname: str) -> str:
    """Extract numeric key (e.g., '64764') from folder name."""
    m = DIGIT_RUN.search(dirname)
    return m.group(1) if m else None


def list_final_obj_files(folder: str):
    """Return absolute paths to all OBJ files containing 'final' in name."""
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".obj") and "final" in f.lower()]


# ---------------------------
# Main
# ---------------------------
def main():
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
                gt_pts, gt_normals, _ = su.sample_points_on_mesh(gt_obj, n_points=N_POINTS, GT=True)
                gt_cache[key] = (gt_pts, gt_normals)
            except Exception as e:
                print(f"[ERROR] sampling GT for {gt_obj}: {e}")
                continue

        final_objs = list_final_obj_files(folder_path)
        if not final_objs:
            continue

        for obj_path in final_objs:
            try:
                obj_pts, obj_normals, _ = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)

                # ps.register_point_cloud(
                #     f"GT_{key}", gt_cache[key][0], radius=0.01, color=(1, 0, 0), point_render_mode="quad"
                # )
                # ps.register_point_cloud(
                #     obj_path,
                #     obj_pts,
                #     radius=0.01,
                #     color=(0, 1, 0),
                #     point_render_mode="quad",
                #     enabled=False,
                # )

                cd1, cd2, f1, nc = su.chamfer_accuracy_completeness_f1(
                    obj_pts, obj_normals, gt_cache[key][0], gt_cache[key][1]
                )
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
