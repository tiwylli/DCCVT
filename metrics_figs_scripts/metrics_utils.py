"""Compatibility helpers for metrics scripts."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
ACCEL_DIR = ROOT_DIR / "accel"
if ACCEL_DIR.exists() and str(ACCEL_DIR) not in sys.path:
    sys.path.append(str(ACCEL_DIR))


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected mesh at {mesh_path}")
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()
        if isinstance(mesh, (list, tuple)):
            mesh = trimesh.util.concatenate(mesh)
    return mesh


def sample_points_on_mesh(mesh_path: str, n_points: int, GT: bool = False):
    mesh = _load_mesh(mesh_path)
    points, face_index = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_index]
    return points.astype(np.float32), normals.astype(np.float32), mesh


def _nn_stats(src_pts, src_normals, dst_pts, dst_normals, threshold: float):
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    tree = cKDTree(dst_pts)
    dist, idx = tree.query(src_pts)
    dist2 = dist**2
    ratio = (dist < threshold).mean()
    if src_normals is not None and dst_normals is not None:
        matched_normals = np.asarray(dst_normals)[idx]
        nc = np.abs(np.einsum("ij,ij->i", src_normals, matched_normals)).mean()
    else:
        nc = 0.0
    return dist, dist2, nc, ratio, idx


def chamfer_accuracy_completeness_f1(
    obj_pts,
    obj_normals,
    gt_pts,
    gt_normals,
    threshold: float = 0.003,
):
    dist_pred, dist2_pred, nc_pred, recall, _ = _nn_stats(
        obj_pts, obj_normals, gt_pts, gt_normals, threshold
    )
    dist_tgt, dist2_tgt, nc_tgt, precision, _ = _nn_stats(
        gt_pts, gt_normals, obj_pts, obj_normals, threshold
    )
    cd1 = dist_pred.mean() + dist_tgt.mean()
    cd2 = dist2_pred.mean() + dist2_tgt.mean()
    f1 = 2 * precision * recall / (precision + recall)
    nc = 0.5 * (nc_pred + nc_tgt)
    return (
        cd1,
        cd2,
        f1,
        nc,
        recall,
        precision,
        dist_pred.mean(),
        dist2_pred.mean(),
        dist_tgt.mean(),
        dist2_tgt.mean(),
    )


def _closest_points_fcpw(scene, query_points, max_radius: float | None):
    import fcpw

    query_points = np.asarray(query_points, dtype=np.float32)
    if max_radius is None:
        squared_max_radii = np.full(len(query_points), np.inf, dtype=np.float32)
    else:
        squared_max_radii = np.full(len(query_points), max_radius * max_radius, dtype=np.float32)
    interactions = fcpw.interaction_3D_list()
    scene.find_closest_points(query_points, squared_max_radii, interactions, True)
    closest_points = np.array([i.p for i in interactions], dtype=np.float32)
    closest_normals = np.array([i.n for i in interactions], dtype=np.float32)
    return closest_points, closest_normals


def chamfer_accuracy_completeness_f1_accel(
    obj_pts,
    obj_normals,
    gt_pts,
    gt_normals,
    *,
    scenes=None,
    threshold: float = 0.003,
    max_radius: float = 0.45,
):
    if scenes is None:
        raise ValueError("scenes must be provided for accelerated metrics.")
    gt_scene, obj_scene = scenes
    cp_obj, cp_obj_normals = _closest_points_fcpw(gt_scene, obj_pts, max_radius)
    cp_pts, cp_pts_normals = _closest_points_fcpw(obj_scene, gt_pts, max_radius)

    diff_pred = cp_obj - obj_pts
    diff_tgt = cp_pts - gt_pts
    dist_pred = np.linalg.norm(diff_pred, axis=1)
    dist_tgt = np.linalg.norm(diff_tgt, axis=1)
    dist2_pred = (diff_pred**2).sum(axis=1)
    dist2_tgt = (diff_tgt**2).sum(axis=1)

    recall = (dist_pred < threshold).mean()
    precision = (dist_tgt < threshold).mean()
    f1 = 2 * precision * recall / (precision + recall)

    nc_pred = np.abs(np.einsum("ij,ij->i", cp_obj_normals, obj_normals)).mean()
    nc_tgt = np.abs(np.einsum("ij,ij->i", cp_pts_normals, gt_normals)).mean()
    nc = 0.5 * (nc_pred + nc_tgt)

    cd1 = dist_pred.mean() + dist_tgt.mean()
    cd2 = dist2_pred.mean() + dist2_tgt.mean()

    return (
        cd1,
        cd2,
        f1,
        nc,
        recall,
        precision,
        dist_pred.mean(),
        dist2_pred.mean(),
        dist_tgt.mean(),
        dist2_tgt.mean(),
        cp_obj,
        cp_pts,
    )
