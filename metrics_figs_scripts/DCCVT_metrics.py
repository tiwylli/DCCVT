import argparse
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
ACCEL_DIR = ROOT_DIR / "accel"
if str(ACCEL_DIR) not in sys.path:
    sys.path.append(str(ACCEL_DIR))

import numpy as np
import pandas as pd
import torch
import tqdm
import trimesh

import metrics_figs_scripts.metrics_utils as su
from dccvt.argparse_utils import ROOT_DIR as DCCVT_ROOT
from dccvt.device import device

# cuda devices
if device.type == "cuda":
    print("Using device: ", torch.cuda.get_device_name(device))
else:
    print("Using device: ", device)
# Improve reproducibility
torch.manual_seed(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(69)

ROOT_DIR = DCCVT_ROOT
GT_DIR = os.path.join(ROOT_DIR, "mesh/thingi32/")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "outputs/20251222_182320/")
N_POINTS = 100000


def generate_metrics_dict():
    metrics_dict = {}
    # tqdm
    print("Generating metrics dictionary...")

    for gt_file in tqdm.tqdm(os.listdir(GT_DIR), desc="Processing GT files"):
        if not gt_file.endswith(".obj"):
            continue
        current_experiment_folder = EXPERIMENTS_DIR + gt_file.split(".")[0] + "/"
        current_unconverged_experiment_folder = EXPERIMENTS_DIR + "unconverged_" + gt_file.split(".")[0] + "/"
        gt_pts, gt_normals, _ = su.sample_points_on_mesh(GT_DIR + gt_file, n_points=N_POINTS, GT=True)

        # Check if the current experiment folder exists
        if not os.path.exists(current_experiment_folder):
            print(f"Experiment folder {current_experiment_folder} does not exist. Skipping...")
            continue

        metrics_dict[current_experiment_folder.split("/")[-2]] = {}

        for obj_file in os.listdir(current_experiment_folder):
            if not obj_file.endswith(".obj"):
                continue
            obj_path = os.path.join(current_experiment_folder, obj_file)
            obj_pts, obj_normals, _ = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)
            cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 = (
                su.chamfer_accuracy_completeness_f1(obj_pts, obj_normals, gt_pts, gt_normals)
            )
            metrics_dict[gt_file.split(".")[0]][obj_file.split(".")[0]] = {
                "chamfer_distance_1": cd1.astype(float),
                "chamfer_distance_2": cd2.astype(float),
                "f1_score": f1.astype(float),
                "normal_consistency": nc.astype(float),
                "recall": float(recall),
                "precision": float(precision),
                "completeness_1": float(completeness1),
                "completeness_2": float(completeness2),
                "accuracy_1": float(accuracy1),
                "accuracy_2": float(accuracy2),
            }

        if not os.path.exists(current_unconverged_experiment_folder):
            print(f"Experiment folder {current_unconverged_experiment_folder} does not exist. Skipping...")
            continue

        metrics_dict[current_unconverged_experiment_folder.split("/")[-2]] = {}

        for obj_file in os.listdir(current_unconverged_experiment_folder):
            if not obj_file.endswith(".obj"):
                continue
            obj_path = os.path.join(current_unconverged_experiment_folder, obj_file)
            obj_pts, obj_normals, _ = su.sample_points_on_mesh(obj_path, n_points=N_POINTS, GT=False)
            cd1, cd2, f1, nc, recall, precision, completeness1, completeness2, accuracy1, accuracy2 = (
                su.chamfer_accuracy_completeness_f1(obj_pts, obj_normals, gt_pts, gt_normals)
            )

            metrics_dict["unconverged_" + gt_file.split(".")[0]][obj_file.split(".")[0]] = {
                "chamfer_distance_1": cd1.astype(float),
                "chamfer_distance_2": cd2.astype(float),
                "f1_score": f1.astype(float),
                "normal_consistency": nc.astype(float),
                "recall": float(recall),
                "precision": float(precision),
                "completeness_1": float(completeness1),
                "completeness_2": float(completeness2),
                "accuracy_1": float(accuracy1),
                "accuracy_2": float(accuracy2),
            }

    # save dictionary to a file
    with open(os.path.join(EXPERIMENTS_DIR, "metrics_dict.json"), "w") as f:
        import json

        json.dump(metrics_dict, f)
    with open(os.path.join(EXPERIMENTS_DIR, "metrics_dict.json"), "r") as f:
        data = json.load(f)
    # Flatten to long format
    records = []
    for object_id, methods in data.items():
        for method, metrics in methods.items():
            for metric_name, value in metrics.items():
                records.append({"object_id": object_id, "method": method, "metric": metric_name, "value": value})

    # Convert to DataFrame
    df_long = pd.DataFrame(records)

    # Optionally pivot to wide format (one row per object/method)
    df_wide = df_long.pivot_table(index=["object_id", "method"], columns="metric", values="value").reset_index()

    # Save to CSV
    df_wide.to_csv(EXPERIMENTS_DIR + "metrics.csv", index=False)

    print("Metrics dictionary saved to metrics_dict.json")

    return metrics_dict


if __name__ == "__main__":
    metrics = generate_metrics_dict()
    print(metrics)
