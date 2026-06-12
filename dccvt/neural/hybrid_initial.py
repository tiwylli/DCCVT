"""Initial HotSpot-on-canonical-grid baseline for hybrid DCCVT experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

from dccvt.neural.grid import make_canonical_sites, trilinear_interpolate_sdf, validate_grid_n


@dataclass
class HybridInitialHotSpotConfig:
    """Resolved configuration for the initial HotSpot canonical baseline."""

    cache_root: str = "outputs/neural_hotspot_sdf/thingi32_g33"
    split_file: str = "PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt"
    output_root: str = "outputs/neural_dccvt/hybrid_initial_hotspot"
    seed: int = 69
    w_cvt: float = 100.0
    w_sdfsmooth: float = 100.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridInitialHotSpotConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_initial_hotspot_config(path: str | Path | None = None) -> HybridInitialHotSpotConfig:
    """Load an initial-baseline JSON config, or return defaults."""
    if path is None:
        return HybridInitialHotSpotConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return HybridInitialHotSpotConfig.from_dict(data)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible extraction metadata."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_mesh_ids(path: str | Path) -> list[str]:
    """Read mesh ids from a split file, ignoring blanks and comments."""
    mesh_ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                mesh_ids.append(Path(value).stem)
    return mesh_ids


def parse_mesh_ids(value: str | None) -> list[str] | None:
    """Parse comma- or whitespace-separated mesh ids from a CLI value."""
    if value is None:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def build_initial_hotspot_field(
    sdf_grid: np.ndarray | torch.Tensor,
    *,
    grid_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical DCCVT sites and HotSpot SDF sampled at those sites."""
    sdf_tensor = torch.as_tensor(sdf_grid, dtype=torch.float32)
    if sdf_tensor.dim() != 3 or len(set(sdf_tensor.shape)) != 1:
        raise ValueError(f"`sdf_grid` must have cubic shape (G,G,G), got {tuple(sdf_tensor.shape)}")
    resolved_grid_n = validate_grid_n(grid_n if grid_n is not None else int(sdf_tensor.shape[-1]))
    if int(sdf_tensor.shape[-1]) != resolved_grid_n:
        raise ValueError(f"grid_n={resolved_grid_n} does not match SDF grid shape {tuple(sdf_tensor.shape)}")

    sites = make_canonical_sites(resolved_grid_n, dtype=torch.float32)
    sites_sdf = trilinear_interpolate_sdf(sdf_tensor, sites).reshape(-1)
    return sites, sites_sdf


def save_resolved_config(
    output_root: str | Path,
    *,
    config: HybridInitialHotSpotConfig,
    args: argparse.Namespace,
) -> None:
    """Save resolved config and command-line arguments for reproducibility."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "seed": int(config.seed),
        "args": vars(args),
    }
    (output_path / "resolved_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _cache_mesh_id(cache: dict[str, np.ndarray], cache_path: str | Path) -> str:
    return str(np.asarray(cache.get("mesh_id", np.array(Path(cache_path).stem))).item())


def _expected_obj_path(output_dir: Path, state: str, variant: str, config: HybridInitialHotSpotConfig) -> Path:
    return output_dir / (
        f"DCCVT_0_{state}_{variant}_cvt{int(config.w_cvt)}_sdfsmooth{int(config.w_sdfsmooth)}.obj"
    )


def extract_initial_hotspot_cache(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    config: HybridInitialHotSpotConfig,
    overwrite: bool = False,
    state: str = "initial_hotspot",
) -> dict[str, Any]:
    """Extract initial DCCVT meshes for one HotSpot cache."""
    output_path = Path(output_dir)
    int_obj = _expected_obj_path(output_path, state, "intDCCVT", config)
    proj_obj = _expected_obj_path(output_path, state, "projDCCVT", config)
    if not overwrite and int_obj.exists() and proj_obj.exists():
        return {
            "cache_path": str(cache_path),
            "output_dir": str(output_path),
            "status": "skipped_existing",
        }

    cache = _load_cache(cache_path)
    sdf_grid_np = np.asarray(cache["sdf_grid"], dtype=np.float32)
    target_points_np = np.asarray(cache["target_points"], dtype=np.float32).reshape(-1, 3)
    grid_n = int(np.asarray(cache["grid_n"]).item())
    mesh_id = _cache_mesh_id(cache, cache_path)
    sites_cpu, sites_sdf_cpu = build_initial_hotspot_field(sdf_grid_np, grid_n=grid_n)

    positive_count = int((sites_sdf_cpu > 0).sum().item())
    negative_count = int((sites_sdf_cpu < 0).sum().item())
    diagnostics = {
        "mesh_id": mesh_id,
        "grid_n": int(grid_n),
        "site_count": int(sites_cpu.shape[0]),
        "positive_sdf_count": positive_count,
        "negative_sdf_count": negative_count,
        "seed": int(config.seed),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    field_file = output_path / f"{mesh_id}_initial_hotspot_field.npz"
    np.savez_compressed(
        field_file,
        sites=sites_cpu.numpy().astype(np.float32),
        sites_sdf=sites_sdf_cpu.numpy().astype(np.float32),
        sdf_grid=sdf_grid_np.astype(np.float32),
        target_points=target_points_np.astype(np.float32),
        diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
        resolved_config=np.array(json.dumps(config.to_dict(), sort_keys=True)),
        seed=np.array(config.seed, dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )

    if sites_cpu.shape[0] < 5 or positive_count == 0 or negative_count == 0:
        print(f"Skipping extraction for {mesh_id}: need at least 5 sites and both positive/negative SDF values.")
        return {
            "cache_path": str(cache_path),
            "output_dir": str(output_path),
            "field_file": str(field_file),
            "diagnostics": diagnostics,
            "status": "skipped_non_extractable",
        }

    from dccvt.device import device as dccvt_device
    from dccvt.device import initialize_runtime
    from dccvt.mesh_ops import extract_mesh

    initialize_runtime(config.seed)
    target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
    args = SimpleNamespace(
        save_path=str(output_path),
        upsampling=0,
        w_cvt=config.w_cvt,
        w_sdfsmooth=config.w_sdfsmooth,
    )
    extract_mesh(
        sites_cpu.to(dccvt_device),
        sites_sdf_cpu.to(dccvt_device),
        target_pc,
        0.0,
        args,
        state=state,
    )
    return {
        "cache_path": str(cache_path),
        "output_dir": str(output_path),
        "field_file": str(field_file),
        "diagnostics": diagnostics,
        "status": "extracted",
        "int_obj": str(int_obj),
        "proj_obj": str(proj_obj),
    }


def run_initial_hotspot_extraction(
    *,
    config: HybridInitialHotSpotConfig,
    mesh_ids: list[str] | None = None,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Run initial HotSpot extraction over a split or explicit mesh ids."""
    seed_everything(config.seed)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_mesh_ids = mesh_ids if mesh_ids is not None else read_mesh_ids(config.split_file)
    results: list[dict[str, Any]] = []

    for mesh_id in resolved_mesh_ids:
        cache_path = Path(config.cache_root) / f"{mesh_id}.npz"
        output_dir = output_root / mesh_id
        try:
            if not cache_path.exists():
                raise FileNotFoundError(f"HotSpot cache not found: {cache_path}")
            print(f"Extracting initial HotSpot baseline for {mesh_id}")
            result = extract_initial_hotspot_cache(
                cache_path,
                output_dir,
                config=config,
                overwrite=overwrite,
            )
        except Exception as exc:
            result = {
                "cache_path": str(cache_path),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"Failed initial HotSpot extraction for {mesh_id}: {exc}")
            if fail_fast:
                raise
        results.append(result)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved initial HotSpot summary: {summary_path}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract canonical-grid HotSpot initial DCCVT meshes.")
    parser.add_argument("--config", default="configs/neural_hybrid_initial_hotspot.json")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids. Defaults to config split.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--w-cvt", type=float, default=None)
    parser.add_argument("--w-sdfsmooth", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _apply_overrides(config: HybridInitialHotSpotConfig, args: argparse.Namespace) -> HybridInitialHotSpotConfig:
    values = config.to_dict()
    for arg_name, key in (
        ("cache_root", "cache_root"),
        ("split_file", "split_file"),
        ("output_root", "output_root"),
        ("seed", "seed"),
        ("w_cvt", "w_cvt"),
        ("w_sdfsmooth", "w_sdfsmooth"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            values[key] = value
    return HybridInitialHotSpotConfig.from_dict(values)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _apply_overrides(load_initial_hotspot_config(args.config), args)
    save_resolved_config(config.output_root, config=config, args=args)
    run_initial_hotspot_extraction(
        config=config,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
