#!/usr/bin/env python3
"""Visualize DCCVT `.npz` generator bundles and `.obj` meshes in Polyscope."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _discover_files(root: Path, recursive: bool) -> tuple[list[Path], list[Path]]:
    pattern = "**/*" if recursive else "*"
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    npz_files = [path for path in files if path.suffix.lower() == ".npz"]
    obj_files = [path for path in files if path.suffix.lower() == ".obj"]
    return npz_files, obj_files


def _structure_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return str(relative.with_suffix("")).replace("/", "::")


def _load_npz_sites(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "sites" not in data or "sites_sdf" not in data:
        missing = {"sites", "sites_sdf"} - set(data.files)
        raise KeyError(f"missing keys: {sorted(missing)}")
    sites = np.asarray(data["sites"], dtype=np.float32)
    sites_sdf = np.asarray(data["sites_sdf"], dtype=np.float32).reshape(-1)
    if sites.ndim != 2 or sites.shape[1] != 3:
        raise ValueError(f"`sites` must have shape (N, 3), got {sites.shape}")
    if sites_sdf.shape[0] != sites.shape[0]:
        raise ValueError(f"`sites_sdf` length {sites_sdf.shape[0]} does not match sites length {sites.shape[0]}")
    return sites, sites_sdf


def _load_obj_vertices_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(part.split("/")[0]) - 1 for part in parts]
                if len(face) >= 3:
                    faces.append(face)
    if not vertices:
        raise ValueError("OBJ contains no vertices")
    if not faces:
        raise ValueError("OBJ contains no faces")
    return np.asarray(vertices, dtype=np.float32), faces


def _sdf_summary(sdf: np.ndarray) -> str:
    neg = int(np.count_nonzero(sdf < 0))
    zero = int(np.count_nonzero(sdf == 0))
    pos = int(np.count_nonzero(sdf > 0))
    return f"min={sdf.min():.6f} max={sdf.max():.6f} neg={neg} zero={zero} pos={pos}"


def _print_discovered(root: Path, npz_files: Iterable[Path], obj_files: Iterable[Path]) -> None:
    print(f"folder: {root}")
    for path in npz_files:
        try:
            sites, sites_sdf = _load_npz_sites(path)
            print(f"[npz] {_structure_name(root, path)} sites={sites.shape} sdf={sites_sdf.shape} {_sdf_summary(sites_sdf)}")
        except Exception as exc:
            print(f"[npz:skip] {path}: {exc}")
    for path in obj_files:
        try:
            vertices, faces = _load_obj_vertices_faces(path)
            print(f"[obj] {_structure_name(root, path)} vertices={vertices.shape} faces={len(faces)}")
        except Exception as exc:
            print(f"[obj:skip] {path}: {exc}")


def _show_polyscope(
    *,
    root: Path,
    npz_files: list[Path],
    obj_files: list[Path],
    point_radius: float,
    mesh_enabled: bool,
    points_enabled: bool,
) -> None:
    import polyscope as ps

    ps.init()
    for path in npz_files:
        name = _structure_name(root, path)
        try:
            sites, sites_sdf = _load_npz_sites(path)
        except Exception as exc:
            print(f"[npz:skip] {path}: {exc}")
            continue
        cloud = ps.register_point_cloud(f"{name}::sites", sites, radius=point_radius, enabled=points_enabled)
        cloud.add_scalar_quantity("sites_sdf", sites_sdf, enabled=True)
        print(f"[npz] {name} sites={sites.shape} {_sdf_summary(sites_sdf)}")

    for path in obj_files:
        name = _structure_name(root, path)
        try:
            vertices, faces = _load_obj_vertices_faces(path)
        except Exception as exc:
            print(f"[obj:skip] {path}: {exc}")
            continue
        ps.register_surface_mesh(
            f"{name}::mesh",
            vertices,
            faces,
            enabled=mesh_enabled,
            back_face_policy="identical",
            smooth_shade=False,
        )
        print(f"[obj] {name} vertices={vertices.shape} faces={len(faces)}")

    ps.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load all DCCVT `.npz` generator bundles and `.obj` meshes in a folder into Polyscope."
    )
    parser.add_argument("folder", help="Folder containing DCCVT `.npz` and `.obj` outputs.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under the target folder.")
    parser.add_argument("--point-radius", type=float, default=0.003, help="Polyscope radius for site point clouds.")
    parser.add_argument(
        "--mesh-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OBJ mesh structures by default.",
    )
    parser.add_argument(
        "--points-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show NPZ site point-cloud structures by default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print discovered contents without opening Polyscope.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    npz_files, obj_files = _discover_files(root, args.recursive)
    if not npz_files and not obj_files:
        print(f"No `.npz` or `.obj` files found in {root}")
        return

    if args.dry_run:
        _print_discovered(root, npz_files, obj_files)
        return

    _show_polyscope(
        root=root,
        npz_files=npz_files,
        obj_files=obj_files,
        point_radius=args.point_radius,
        mesh_enabled=args.mesh_enabled,
        points_enabled=args.points_enabled,
    )


if __name__ == "__main__":
    main()
