"""Sparse canonical HotSpot field plus procedural DCCVT site refinement."""

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
class HybridSparseRefineConfig:
    """Resolved configuration for sparse-to-refined HotSpot DCCVT extraction."""

    cache_root: str = "outputs/neural_hotspot_sdf/thingi32_g33"
    split_file: str = "PoNQ-main/src/eval/hotspot_thingi32_g33_ids.txt"
    output_root: str = "outputs/neural_dccvt/hybrid_sparse_refine_v0"
    base_grid_n: int = 17
    hotspot_grid_n: int = 33
    refinement_mode: str = "procedural_upsample"
    upsampling_rounds: int = 1
    growth_cap: float = 0.10
    clamp_domain: bool = True
    seed: int = 69
    w_cvt: float = 100.0
    w_sdfsmooth: float = 100.0

    def __post_init__(self) -> None:
        self.base_grid_n = validate_grid_n(self.base_grid_n)
        self.hotspot_grid_n = validate_grid_n(self.hotspot_grid_n)
        self.refinement_mode = str(self.refinement_mode)
        self.upsampling_rounds = int(self.upsampling_rounds)
        self.growth_cap = float(self.growth_cap)
        self.clamp_domain = bool(self.clamp_domain)
        self.seed = int(self.seed)
        self.w_cvt = float(self.w_cvt)
        self.w_sdfsmooth = float(self.w_sdfsmooth)
        if self.refinement_mode != "procedural_upsample":
            raise ValueError(f"Unknown sparse refinement mode: {self.refinement_mode}")
        if self.upsampling_rounds < 0:
            raise ValueError("upsampling_rounds must be non-negative")
        if self.growth_cap <= 0.0:
            raise ValueError("growth_cap must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HybridSparseRefineConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sparse_refine_config(path: str | Path | None = None) -> HybridSparseRefineConfig:
    """Load a sparse-refinement JSON config, or return defaults."""
    if path is None:
        return HybridSparseRefineConfig()
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return HybridSparseRefineConfig.from_dict(data)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible refinement."""
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


def build_sparse_base_field(
    sdf_grid: np.ndarray | torch.Tensor,
    *,
    base_grid_n: int,
    hotspot_grid_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse canonical sites and HotSpot SDF sampled at those sites."""
    sdf_tensor = torch.as_tensor(sdf_grid, dtype=torch.float32)
    if sdf_tensor.dim() != 3 or len(set(sdf_tensor.shape)) != 1:
        raise ValueError(f"`sdf_grid` must have cubic shape (G,G,G), got {tuple(sdf_tensor.shape)}")

    resolved_hotspot_grid_n = validate_grid_n(
        hotspot_grid_n if hotspot_grid_n is not None else int(sdf_tensor.shape[-1])
    )
    if int(sdf_tensor.shape[-1]) != resolved_hotspot_grid_n:
        raise ValueError(
            f"hotspot_grid_n={resolved_hotspot_grid_n} does not match SDF grid shape {tuple(sdf_tensor.shape)}"
        )

    sites = make_canonical_sites(base_grid_n, dtype=torch.float32)
    sites_sdf = trilinear_interpolate_sdf(sdf_tensor, sites).reshape(-1)
    return sites, sites_sdf


def refine_sparse_field(
    sdf_grid: np.ndarray | torch.Tensor,
    base_sites: torch.Tensor,
    base_sites_sdf: torch.Tensor,
    *,
    upsampling_rounds: int,
    growth_cap: float,
    clamp_domain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Procedurally spawn sites with DCCVT adaptive upsampling and resample SDF."""
    from dccvt.geometry import compute_delaunay_simplices
    from dccvt.sdf_gradients import compute_sdf_gradients_sites_tets
    from dccvt.upsampling import upsample_sites_adaptive

    sdf_tensor = torch.as_tensor(sdf_grid, dtype=torch.float32, device=base_sites.device)
    sites = base_sites.detach().clone()
    sites_sdf = base_sites_sdf.detach().clone().reshape(-1)
    completed_rounds = 0

    for _ in range(int(upsampling_rounds)):
        if sites.shape[0] < 5 or not ((sites_sdf.min() < 0) and (sites_sdf.max() > 0)):
            break

        simplices = compute_delaunay_simplices(sites)
        if simplices.size == 0:
            break
        tets = torch.as_tensor(simplices, device=sites.device).long()
        sites_sdf_grads, _, _ = compute_sdf_gradients_sites_tets(sites, sites_sdf, tets)
        refined_sites, _ = upsample_sites_adaptive(
            sites,
            simplices=simplices,
            sites_sdf=sites_sdf,
            sites_sdf_grads=sites_sdf_grads,
            growth_cap=float(growth_cap),
        )
        if refined_sites.shape[0] == sites.shape[0]:
            break

        sites = refined_sites.detach()
        if clamp_domain:
            sites = sites.clamp(-1.0, 1.0)
        sites_sdf = trilinear_interpolate_sdf(sdf_tensor, sites).reshape(-1).detach()
        completed_rounds += 1

    diagnostics = {
        "completed_upsampling_rounds": completed_rounds,
        "base_site_count": int(base_sites.shape[0]),
        "refined_site_count": int(sites.shape[0]),
        "spawned_site_count": int(sites.shape[0] - base_sites.shape[0]),
    }
    return sites, sites_sdf, diagnostics


def save_resolved_config(
    output_root: str | Path,
    *,
    config: HybridSparseRefineConfig,
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


def _expected_obj_path(output_dir: Path, state: str, variant: str, config: HybridSparseRefineConfig) -> Path:
    return output_dir / (
        f"DCCVT_{config.upsampling_rounds}_{state}_{variant}_"
        f"cvt{int(config.w_cvt)}_sdfsmooth{int(config.w_sdfsmooth)}.obj"
    )


def extract_sparse_refine_cache(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    config: HybridSparseRefineConfig,
    overwrite: bool = False,
    extract: bool = True,
    state: str = "sparse_refine_v0",
) -> dict[str, Any]:
    """Build and optionally extract one sparse-to-refined HotSpot DCCVT field."""
    output_path = Path(output_dir)
    int_obj = _expected_obj_path(output_path, state, "intDCCVT", config)
    proj_obj = _expected_obj_path(output_path, state, "projDCCVT", config)
    if extract and not overwrite and int_obj.exists() and proj_obj.exists():
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
    if grid_n != config.hotspot_grid_n:
        raise ValueError(f"Cache grid_n={grid_n} does not match config hotspot_grid_n={config.hotspot_grid_n}")

    base_sites, base_sites_sdf = build_sparse_base_field(
        sdf_grid_np,
        base_grid_n=config.base_grid_n,
        hotspot_grid_n=config.hotspot_grid_n,
    )
    refined_sites, refined_sites_sdf, refine_stats = refine_sparse_field(
        sdf_grid_np,
        base_sites,
        base_sites_sdf,
        upsampling_rounds=config.upsampling_rounds,
        growth_cap=config.growth_cap,
        clamp_domain=config.clamp_domain,
    )

    positive_count = int((refined_sites_sdf > 0).sum().item())
    negative_count = int((refined_sites_sdf < 0).sum().item())
    diagnostics = {
        "mesh_id": mesh_id,
        "hotspot_grid_n": int(grid_n),
        "base_grid_n": int(config.base_grid_n),
        "refinement_mode": config.refinement_mode,
        "requested_upsampling_rounds": int(config.upsampling_rounds),
        "growth_cap": float(config.growth_cap),
        "positive_sdf_count": positive_count,
        "negative_sdf_count": negative_count,
        "max_abs_site_coord": float(refined_sites.abs().max().item()),
        "seed": int(config.seed),
        **refine_stats,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    field_file = output_path / f"{mesh_id}_sparse_refine_v0_field.npz"
    np.savez_compressed(
        field_file,
        base_sites=base_sites.numpy().astype(np.float32),
        base_sites_sdf=base_sites_sdf.numpy().astype(np.float32),
        sites=refined_sites.numpy().astype(np.float32),
        sites_sdf=refined_sites_sdf.numpy().astype(np.float32),
        sdf_grid=sdf_grid_np.astype(np.float32),
        target_points=target_points_np.astype(np.float32),
        diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
        resolved_config=np.array(json.dumps(config.to_dict(), sort_keys=True)),
        seed=np.array(config.seed, dtype=np.int64),
        mesh_id=np.array(mesh_id),
    )

    can_extract = (
        extract
        and refined_sites.shape[0] >= 5
        and positive_count > 0
        and negative_count > 0
    )
    if can_extract:
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime
        from dccvt.mesh_ops import extract_mesh

        initialize_runtime(config.seed)
        target_pc = torch.from_numpy(target_points_np[None, ...]).to(dccvt_device)
        args = SimpleNamespace(
            save_path=str(output_path),
            upsampling=config.upsampling_rounds,
            w_cvt=config.w_cvt,
            w_sdfsmooth=config.w_sdfsmooth,
        )
        extract_mesh(
            refined_sites.to(dccvt_device),
            refined_sites_sdf.to(dccvt_device),
            target_pc,
            0.0,
            args,
            state=state,
        )
        status = "extracted"
    elif extract:
        print(f"Skipping sparse-refine extraction for {mesh_id}: field is not extractable.")
        status = "skipped_non_extractable"
    else:
        status = "field_saved"

    return {
        "cache_path": str(cache_path),
        "output_dir": str(output_path),
        "field_file": str(field_file),
        "diagnostics": diagnostics,
        "status": status,
        "int_obj": str(int_obj) if can_extract else None,
        "proj_obj": str(proj_obj) if can_extract else None,
    }


def run_sparse_refine_extraction(
    *,
    config: HybridSparseRefineConfig,
    mesh_ids: list[str] | None = None,
    overwrite: bool = False,
    extract: bool = True,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Run sparse-to-refined extraction over a split or explicit mesh ids."""
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
            print(f"Extracting sparse-refine HotSpot field for {mesh_id}")
            result = extract_sparse_refine_cache(
                cache_path,
                output_dir,
                config=config,
                overwrite=overwrite,
                extract=extract,
            )
        except Exception as exc:
            result = {
                "cache_path": str(cache_path),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": repr(exc),
            }
            print(f"Failed sparse-refine extraction for {mesh_id}: {exc}")
            if fail_fast:
                raise
        results.append(result)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved sparse-refine summary: {summary_path}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract sparse-to-refined HotSpot DCCVT meshes.")
    parser.add_argument("--config", default="configs/neural_hybrid_sparse_refine_v0.json")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids. Defaults to config split.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--base-grid-n", type=int, default=None)
    parser.add_argument("--hotspot-grid-n", type=int, default=None)
    parser.add_argument("--upsampling-rounds", type=int, default=None)
    parser.add_argument("--growth-cap", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--w-cvt", type=float, default=None)
    parser.add_argument("--w-sdfsmooth", type=float, default=None)
    parser.add_argument("--no-clamp-domain", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _apply_overrides(config: HybridSparseRefineConfig, args: argparse.Namespace) -> HybridSparseRefineConfig:
    values = config.to_dict()
    for arg_name, key in (
        ("cache_root", "cache_root"),
        ("split_file", "split_file"),
        ("output_root", "output_root"),
        ("base_grid_n", "base_grid_n"),
        ("hotspot_grid_n", "hotspot_grid_n"),
        ("upsampling_rounds", "upsampling_rounds"),
        ("growth_cap", "growth_cap"),
        ("seed", "seed"),
        ("w_cvt", "w_cvt"),
        ("w_sdfsmooth", "w_sdfsmooth"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            values[key] = value
    if args.no_clamp_domain:
        values["clamp_domain"] = False
    return HybridSparseRefineConfig.from_dict(values)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _apply_overrides(load_sparse_refine_config(args.config), args)
    save_resolved_config(config.output_root, config=config, args=args)
    run_sparse_refine_extraction(
        config=config,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        overwrite=args.overwrite,
        extract=not args.no_extract,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
