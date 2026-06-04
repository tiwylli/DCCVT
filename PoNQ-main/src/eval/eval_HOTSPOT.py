import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import trimesh
from sklearn.neighbors import KDTree

F1_THRESHOLD = 0.003
EDGE_ANGLE_THRESHOLD = 30
EDGE_SAMPLE_NUM = int(1e5)
EF1_THRESHOLD = 0.005

MODE_NAMES = ("ponq_thingi", "raw", "bbox_aligned")


def read_model_ids(path: Path) -> list[str]:
    model_ids = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model_ids.append(Path(line).stem)
    return model_ids


def uniform_edge_sampling(mesh: trimesh.Trimesh, angle_threshold: float, sample_count: int) -> np.ndarray:
    sharp = mesh.face_adjacency_angles > np.radians(angle_threshold)
    sharp_edges = mesh.face_adjacency_edges[sharp]
    if len(sharp_edges) == 0:
        return np.array([])

    vertices = mesh.vertices
    edge_length = np.sqrt(((vertices[sharp_edges[:, 1]] - vertices[sharp_edges[:, 0]]) ** 2).sum(-1))
    selected_edges = np.random.choice(len(edge_length), sample_count, p=edge_length / edge_length.sum())
    lambdas = np.random.rand(len(selected_edges))[:, None]
    sampled_points = (
        vertices[sharp_edges[selected_edges][:, 1]] * lambdas
        + vertices[sharp_edges[selected_edges][:, 0]] * (1 - lambdas)
    )
    return sampled_points


def apply_bbox_alignment(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh) -> None:
    pred_vertices = pred_mesh.vertices
    gt_vertices = gt_mesh.vertices

    pred_min = pred_vertices.min(axis=0)
    pred_max = pred_vertices.max(axis=0)
    gt_min = gt_vertices.min(axis=0)
    gt_max = gt_vertices.max(axis=0)

    pred_center = (pred_min + pred_max) / 2.0
    gt_center = (gt_min + gt_max) / 2.0
    pred_extent = np.linalg.norm(pred_max - pred_min)
    gt_extent = np.linalg.norm(gt_max - gt_min)
    if pred_extent == 0:
        raise ValueError("Prediction bbox diagonal is zero")

    scale = gt_extent / pred_extent
    pred_mesh.vertices[:] = (pred_vertices - pred_center) * scale + gt_center


def transform_meshes(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, mode: str) -> None:
    if mode == "ponq_thingi":
        gt_mesh.vertices[:] /= 2.0
        pred_mesh.vertices[:] /= 2.0
    elif mode == "raw":
        return
    elif mode == "bbox_aligned":
        apply_bbox_alignment(pred_mesh, gt_mesh)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def sample_mesh(mesh: trimesh.Trimesh, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = mesh.sample(sample_count, return_index=True)
    normals = mesh.face_normals[face_indices]
    return points, normals


def compute_surface_metrics(
    gt_points: np.ndarray,
    gt_normals: np.ndarray,
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
) -> tuple[float, float, float, float, float]:
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < F1_THRESHOLD) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    gt2pred_mean_cd2 = np.mean(np.square(dist))
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    gt2pred_nc = np.mean(np.abs(np.sum(gt_normals * neighbor_normals, axis=1)))

    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist < F1_THRESHOLD) / float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    pred2gt_mean_cd2 = np.mean(np.square(dist))
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    pred2gt_nc = np.mean(np.abs(np.sum(pred_normals * neighbor_normals, axis=1)))

    cd1 = gt2pred_mean_cd1 + pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2 + pred2gt_mean_cd2
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    nc = (gt2pred_nc + pred2gt_nc) / 2.0
    return cd1, cd2, f1, nc, precision


def compute_edge_metrics(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh) -> tuple[float, float]:
    gt_edge_points = uniform_edge_sampling(gt_mesh, EDGE_ANGLE_THRESHOLD, EDGE_SAMPLE_NUM)
    pred_edge_points = uniform_edge_sampling(pred_mesh, EDGE_ANGLE_THRESHOLD, EDGE_SAMPLE_NUM)
    if len(pred_edge_points) == 0:
        pred_edge_points = np.zeros([486, 3], np.float32)
    if len(gt_edge_points) == 0:
        return 0.0, 1.0

    pred_tree = KDTree(pred_edge_points)
    dist, _ = pred_tree.query(gt_edge_points, k=1)
    edge_recall = np.sum(dist < EF1_THRESHOLD) / float(len(dist))
    gt2pred_mean_ecd2 = np.mean(np.square(dist))

    gt_tree = KDTree(gt_edge_points)
    dist, _ = gt_tree.query(pred_edge_points, k=1)
    edge_precision = np.sum(dist < EF1_THRESHOLD) / float(len(dist))
    pred2gt_mean_ecd2 = np.mean(np.square(dist))

    ecd2 = gt2pred_mean_ecd2 + pred2gt_mean_ecd2
    ef1 = 0.0 if edge_recall + edge_precision == 0 else (
        2 * edge_recall * edge_precision / (edge_recall + edge_precision)
    )
    return ecd2, ef1


def evaluate_one(item: tuple[int, str, Path, Path, str, int]) -> np.ndarray:
    idx, model_id, gt_path, pred_path, mode, sample_count = item
    gt_mesh = trimesh.load(gt_path, force="mesh")
    pred_mesh = trimesh.load(pred_path, force="mesh")

    try:
        transform_meshes(gt_mesh, pred_mesh, mode)
        gt_points, gt_normals = sample_mesh(gt_mesh, sample_count)
        pred_points, pred_normals = sample_mesh(pred_mesh, sample_count)
    except Exception:
        pred_points = np.zeros((1, 3))
        pred_normals = np.zeros((1, 3))
        gt_points, gt_normals = sample_mesh(gt_mesh, sample_count)
        pred_mesh = trimesh.Trimesh(pred_points, np.array([]))

    cd1, cd2, f1, nc, _ = compute_surface_metrics(gt_points, gt_normals, pred_points, pred_normals)
    ecd2, ef1 = compute_edge_metrics(gt_mesh, pred_mesh)
    return np.array([idx, cd1, cd2, f1, nc, ecd2, ef1], dtype=np.float64)


def save_name_from_pred_dir(pred_dir: Path) -> str:
    return pred_dir.name if pred_dir.name else pred_dir.parent.name


def print_metric_line(name: str, out: np.ndarray) -> dict[str, float]:
    mean_scores = out.mean(0)
    metrics = {
        "cd_x1e5": float(mean_scores[2] * 1e5),
        "f1": float(mean_scores[3]),
        "nc": float(mean_scores[4]),
        "ecd": float(mean_scores[5] * 1e2),
        "ef1": float(mean_scores[6]),
    }
    print("CD (x 1e-5), F1, NC, ECD, EF1")
    print(
        "{} & {:.3f}  &  {:.3f}  &  {:.3f} & {:.3f} & {:.3f}".format(
            name,
            metrics["cd_x1e5"],
            metrics["f1"],
            metrics["nc"],
            metrics["ecd"],
            metrics["ef1"],
        )
    )
    return metrics


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["mode", "count", "cd_x1e5", "f1", "nc", "ecd", "ef1", "npy_path"]
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_mode(
    *,
    mode: str,
    model_ids: list[str],
    pred_dir: Path,
    gt_dir: Path,
    pred_suffix: str,
    sample_count: int,
) -> np.ndarray:
    items = []
    for idx, model_id in enumerate(model_ids):
        gt_path = gt_dir / f"{model_id}.obj"
        pred_path = pred_dir / f"{model_id}{pred_suffix}"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        items.append((idx, model_id, gt_path, pred_path, mode, sample_count))

    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(evaluate_one)(item) for item in items)
    return np.array(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HotSpot PoNQ metrics with scale/alignment diagnostics.")
    parser.add_argument("pred_dir", type=str, help="Directory containing predicted <mesh_id>.obj files.")
    parser.add_argument(
        "-gt_dir",
        type=str,
        default="/export/livia/home/vision/Wcharawi/dev/DCCVT/mesh/thingi32",
        help="Ground-truth mesh root containing <mesh_id>.obj files.",
    )
    parser.add_argument(
        "-all_models",
        type=str,
        default="src/eval/hotspot_thingi32_g33_ids.txt",
        help="Text file of mesh ids.",
    )
    parser.add_argument("-pred_suffix", type=str, default=".obj")
    parser.add_argument("-mode", choices=("all",) + MODE_NAMES, default="all")
    parser.add_argument("-sample_num", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    model_ids = read_model_ids(Path(args.all_models))
    modes = MODE_NAMES if args.mode == "all" else (args.mode,)

    results_dir = Path("src/eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = save_name_from_pred_dir(pred_dir)

    summary_rows = []
    for mode in modes:
        print(f"Evaluating mode: {mode}")
        out = evaluate_mode(
            mode=mode,
            model_ids=model_ids,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            pred_suffix=args.pred_suffix,
            sample_count=args.sample_num,
        )
        npy_path = results_dir / f"results_{base_name}_{mode}.npy"
        np.save(npy_path, out)
        metrics = print_metric_line(f"{base_name}_{mode}", out)
        summary_rows.append({"mode": mode, "count": len(model_ids), "npy_path": str(npy_path), **metrics})

    summary_path = results_dir / f"results_{base_name}_hotspot_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
