#!/usr/bin/env python3
"""Convert DCCVT `.npz` site bundles to colored point-cloud PLY files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _discover_npz_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(path for path in root.glob(pattern) if path.is_file())


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


def _sdf_summary(sdf: np.ndarray) -> str:
    neg = int(np.count_nonzero(sdf < 0))
    zero = int(np.count_nonzero(sdf == 0))
    pos = int(np.count_nonzero(sdf > 0))
    return f"min={sdf.min():.6f} max={sdf.max():.6f} neg={neg} zero={zero} pos={pos}"


def _color_scale(values: np.ndarray, clip_percentile: float, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    clip_percentile = float(np.clip(clip_percentile, 0.0, 100.0))
    if symmetric:
        scale = np.percentile(np.abs(finite), clip_percentile)
        if scale <= 0 or not np.isfinite(scale):
            scale = float(np.max(np.abs(finite)))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        return -float(scale), float(scale)

    lo = np.percentile(finite, 100.0 - clip_percentile)
    hi = np.percentile(finite, clip_percentile)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return float(lo), float(hi)


def _sdf_to_blue_white_red(values: np.ndarray, *, clip_percentile: float, symmetric: bool) -> np.ndarray:
    lo, hi = _color_scale(values, clip_percentile, symmetric)
    clipped = np.clip(values, lo, hi)
    t = (clipped - lo) / max(hi - lo, 1e-12)
    t = np.nan_to_num(t, nan=0.5, posinf=1.0, neginf=0.0)

    colors = np.empty((values.shape[0], 3), dtype=np.float32)
    lower = t <= 0.5
    upper = ~lower

    lower_t = (t[lower] / 0.5)[:, None]
    upper_t = ((t[upper] - 0.5) / 0.5)[:, None]
    blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    colors[lower] = blue + lower_t * (white - blue)
    colors[upper] = white + upper_t * (red - white)
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def _default_output_path(source: Path, root: Path, output_dir: Path | None, recursive: bool) -> Path:
    name = source.with_suffix("").name + "_sites_sdf.ply"
    if output_dir is None:
        return source.with_name(name)
    if recursive:
        relative_parent = source.parent.relative_to(root)
        return output_dir / relative_parent / name
    return output_dir / name


def _write_ascii_ply(path: Path, sites: np.ndarray, sites_sdf: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {sites.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property float sites_sdf\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, sdf, color in zip(sites, sites_sdf, colors):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{float(sdf):.9g} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def convert_folder(
    *,
    root: Path,
    output_dir: Path | None,
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    clip_percentile: float,
    symmetric: bool,
) -> None:
    npz_files = _discover_npz_files(root, recursive)
    if not npz_files:
        print(f"No `.npz` files found in {root}")
        return

    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()

    for source in npz_files:
        try:
            sites, sites_sdf = _load_npz_sites(source)
        except Exception as exc:
            print(f"[skip] {source}: {exc}")
            continue

        output_path = _default_output_path(source, root, output_dir, recursive)
        exists_note = " exists" if output_path.exists() else ""
        print(f"[npz] {source} -> {output_path}{exists_note} sites={sites.shape} {_sdf_summary(sites_sdf)}")

        if dry_run:
            continue
        if output_path.exists() and not overwrite:
            print(f"[skip] {output_path} already exists; pass --overwrite to replace it")
            continue

        colors = _sdf_to_blue_white_red(sites_sdf, clip_percentile=clip_percentile, symmetric=symmetric)
        _write_ascii_ply(output_path, sites, sites_sdf, colors)
        print(f"[write] {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert DCCVT `.npz` site/SDF bundles to colored point-cloud PLY files.")
    parser.add_argument("folder", help="Folder containing DCCVT `.npz` bundles.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively under the target folder.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for generated PLY files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PLY files.")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Percentile used to robustly clip SDF values before color mapping.",
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center the blue-white-red color scale at SDF 0.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    convert_folder(
        root=root,
        output_dir=output_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        clip_percentile=args.clip_percentile,
        symmetric=args.symmetric,
    )


if __name__ == "__main__":
    main()
