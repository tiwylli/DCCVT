from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mesh_tools as mt
import numpy as np
import torch
import yaml
from CNN_to_PoNQ_or_lite import CNN_to_PoNQ
from SDF_CNN import CNN_3d_multiple_split
from tqdm import tqdm


def read_model_ids(names_file: Path, limit: int | None) -> list[str]:
    model_ids = []
    with names_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model_ids.append(Path(line).stem)
    if limit is not None:
        model_ids = model_ids[:limit]
    return model_ids


def make_mask_close(sdf_grid: np.ndarray, grid_n: int) -> np.ndarray:
    threshold = 2 / grid_n * np.sqrt(3)
    close = np.abs(sdf_grid) < threshold
    mask = (
        close[:-1, :-1, :-1]
        & close[1:, :-1, :-1]
        & close[:-1, 1:, :-1]
        & close[:-1, :-1, 1:]
        & close[1:, 1:, :-1]
        & close[1:, :-1, 1:]
        & close[:-1, 1:, 1:]
        & close[1:, 1:, 1:]
    )
    return mask.reshape((grid_n - 1) ** 3)


def load_cache_input(cache_path: Path, grid_n: int, use_cache_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        if "sdf_grid" not in cache:
            raise KeyError(f"Missing `sdf_grid` in {cache_path}")
        sdf_grid = np.asarray(cache["sdf_grid"], dtype=np.float32)
        if sdf_grid.shape != (grid_n, grid_n, grid_n):
            raise ValueError(
                f"Expected sdf_grid shape {(grid_n, grid_n, grid_n)} in {cache_path}, got {sdf_grid.shape}"
            )

        if use_cache_mask and "near_surface_mask" in cache:
            mask = np.asarray(cache["near_surface_mask"], dtype=bool).reshape(-1)
            expected = (grid_n - 1) ** 3
            if mask.shape[0] != expected:
                raise ValueError(
                    f"Expected near_surface_mask length {expected} in {cache_path}, got {mask.shape[0]}"
                )
        else:
            mask = make_mask_close(sdf_grid, grid_n)

    return sdf_grid, mask


def export_min_cut(pt_path: str, grid_scale: float, add_noise: bool = False) -> dict[str, Any]:
    pt_file = Path(pt_path)
    try:
        ponq = torch.load(pt_file, map_location="cpu")
        vertices, faces = ponq.min_cut_surface(grid_scale, add_noise=add_noise)
        mt.export_obj(vertices, faces, str(pt_file.with_suffix("")))
        return {
            "mesh_id": pt_file.stem,
            "obj_path": str(pt_file.with_suffix(".obj")),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "export_status": "ok",
            "export_error": "",
        }
    except Exception as exc:
        mt.export_obj(np.array([]), np.array([]), str(pt_file.with_suffix("")))
        return {
            "mesh_id": pt_file.stem,
            "obj_path": str(pt_file.with_suffix(".obj")),
            "vertex_count": 0,
            "face_count": 0,
            "export_status": "failed",
            "export_error": repr(exc),
        }


def check_inputs(model_ids: list[str], cache_root: Path, gt_mesh_root: Path) -> None:
    missing_cache = [model_id for model_id in model_ids if not (cache_root / f"{model_id}.npz").exists()]
    missing_gt = [model_id for model_id in model_ids if not (gt_mesh_root / f"{model_id}.obj").exists()]
    if missing_cache:
        raise FileNotFoundError(f"Missing cache files for {len(missing_cache)} ids: {missing_cache[:20]}")
    if missing_gt:
        raise FileNotFoundError(f"Missing GT OBJ files for {len(missing_gt)} ids: {missing_gt[:20]}")


def write_metadata(
    *,
    output_dir: Path,
    config_path: Path,
    cfg: dict[str, Any],
    model_ids: list[str],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "checkpoint": cfg["training"]["model_name"],
        "cache_root": cfg["path"]["cache_root"],
        "gt_mesh_root": cfg["path"]["gt_mesh_root"],
        "names_file": cfg["data"]["names"],
        "grid_n": int(cfg["data"]["grid_n"]),
        "use_cache_mask": bool(cfg["data"].get("use_cache_mask", True)),
        "subd": int(args.subd),
        "n_jobs": int(args.n_jobs),
        "limit": args.limit,
        "overwrite": bool(args.overwrite),
        "model_count": len(model_ids),
        "model_ids": model_ids,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)


def save_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "mesh_id",
        "cache_path",
        "pt_path",
        "obj_path",
        "mask_cells",
        "vertex_count",
        "face_count",
        "status",
        "error",
        "export_status",
        "export_error",
    ]
    with (output_dir / "generation_summary.csv").open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_dir_from_config(cfg: dict[str, Any], grid_n: int) -> Path:
    checkpoint_stem = Path(cfg["training"]["model_name"]).stem
    return Path(cfg["path"]["out_dir"]) / f"HotSpot_{checkpoint_stem}_{grid_n - 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PoNQ CNN checkpoint on HotSpot dense SDF caches.")
    parser.add_argument("config", help="Path to HotSpot PoNQ eval YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of IDs to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing .pt/.obj outputs.")
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel jobs for min-cut OBJ export.")
    parser.add_argument("--subd", type=int, default=0, help="PoNQ-lite subdivision level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as fin:
        cfg = yaml.load(fin, Loader=yaml.Loader)

    grid_n = int(cfg["data"]["grid_n"])
    cache_root = Path(cfg["path"]["cache_root"])
    gt_mesh_root = Path(cfg["path"]["gt_mesh_root"])
    use_cache_mask = bool(cfg["data"].get("use_cache_mask", True))
    model_ids = read_model_ids(Path(cfg["data"]["names"]), args.limit)
    check_inputs(model_ids, cache_root, gt_mesh_root)

    output_dir = output_dir_from_config(cfg, grid_n)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir=output_dir, config_path=config_path, cfg=cfg, model_ids=model_ids, args=args)

    device = "cuda"
    model = CNN_3d_multiple_split(grid_n=grid_n, K=4, ef_dim=128, device=device).to(device)
    model.load_state_dict(torch.load(cfg["training"]["model_name"], map_location=device))
    model.eval()

    rows = []
    pt_paths = []
    for model_id in tqdm(model_ids, desc="PoNQ HotSpot inference"):
        cache_path = cache_root / f"{model_id}.npz"
        pt_path = output_dir / f"{model_id}.pt"
        obj_path = output_dir / f"{model_id}.obj"
        row = {
            "mesh_id": model_id,
            "cache_path": str(cache_path),
            "pt_path": str(pt_path),
            "obj_path": str(obj_path),
        }

        if not args.overwrite and pt_path.exists() and obj_path.exists():
            row.update({"status": "skipped", "error": ""})
            rows.append(row)
            continue

        try:
            sdf_grid, mask = load_cache_input(cache_path, grid_n, use_cache_mask)
            row["mask_cells"] = int(mask.sum())
            sdfs = torch.from_numpy(sdf_grid[None, None, ...])
            mask_tensor = torch.from_numpy(mask[None, ...])
            ponq = CNN_to_PoNQ(model, sdfs, grid_n, mask_tensor, args.subd, device=device)
            torch.save(ponq, pt_path)
            pt_paths.append(str(pt_path))
            row.update({"status": "ok", "error": ""})
        except Exception as exc:
            row.update({"status": "failed", "error": repr(exc)})
        rows.append(row)

    export_rows_by_id = {}
    if pt_paths:
        export_rows = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(export_min_cut)(pt_path, (grid_n - 1) / 2**args.subd, add_noise=args.subd > 0)
            for pt_path in tqdm(pt_paths, desc="PoNQ min-cut export")
        )
        export_rows_by_id = {row["mesh_id"]: row for row in export_rows}

    merged_rows = []
    for row in rows:
        export_row = export_rows_by_id.get(row["mesh_id"], {})
        merged_rows.append(row | export_row)
    save_summary(output_dir, merged_rows)

    failures = [row for row in merged_rows if row.get("status") == "failed" or row.get("export_status") == "failed"]
    print(f"Saved outputs in {output_dir}")
    print(f"Processed IDs: {len(model_ids)}")
    print(f"Failures: {len(failures)}")
    if failures:
        for row in failures[:20]:
            print(f"{row['mesh_id']}: {row.get('error') or row.get('export_error')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
