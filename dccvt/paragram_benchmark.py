"""Reproducible Paragram versus gDel3D adjacency benchmarks.

This module deliberately stays separate from DCCVT's production geometry path.
Paragram exposes a Delaunay neighbor graph, whereas DCCVT currently consumes
Delaunay tetrahedra. The common representation used here is therefore an
undirected edge list derived from each backend's native output.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


_PYTHON_BIN_DIR = str(Path(sys.executable).parent.absolute())
if (_PYTHON_BIN_DIR not in os.environ.get("PATH", "").split(os.pathsep)) and (
    Path(_PYTHON_BIN_DIR) / "ninja"
).exists():
    os.environ["PATH"] = _PYTHON_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


SCHEMA_VERSION = 1
DELAUNAY_PROFILE_SHARE = 0.2385
PARAGRAM_STATUS_NAMES = (
    "success",
    "plane_overflow",
    "vertex_overflow",
    "empty_cell",
    "security_radius_not_reached",
    "inconsistent_boundary",
)
PARAGRAM_BUILD_ENV = (
    "TORCH_EXTENSIONS_DIR",
    "TORCH_CUDA_ARCH_LIST",
    "MAX_JOBS",
    "FAST_COMPILE",
    "DEBUG_CUDA",
    "NO_FAST_MATH",
    "PROFILE_CUDA",
    "USE_PRECOMPILED_HEADERS",
    "PARAGRAM_NUM_CHUNKS",
    "PARAGRAM_MEM_FRAC",
    "PYTORCH_CUDA_ALLOC_CONF",
)
DEFAULT_NPZ_PATHS = (
    "outputs/optimized_control_20260622/313444_b32_nearsample_ups/"
    "DCCVT_10_final_intDCCVT_cvt100_sdfsmooth100.npz",
    "outputs/alphashape/unconverged_100_313444_b64/"
    "voromesh_64_init_DCCVT_cvt0_sdfsmooth0.npz",
)
_TET_EDGE_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    size: int | None = None
    seed: int = 69
    path: str | None = None
    clusters: int = 10
    cluster_std: float = 0.08


def _safe_empty_cuda_cache() -> None:
    """Best-effort cleanup when a native kernel may have poisoned the context."""
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_points(spec: DatasetSpec) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or generate a contiguous float32 ``(N, 3)`` point array."""
    if spec.kind == "npz":
        if not spec.path:
            raise ValueError("NPZ datasets require a path")
        path = Path(spec.path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            if "sites" not in archive:
                raise KeyError(f"{path} does not contain a 'sites' array")
            points = np.asarray(archive["sites"], dtype=np.float32)
        metadata = {"path": str(path), "sha256": file_sha256(path)}
    else:
        if not spec.size or spec.size < 4:
            raise ValueError("Synthetic datasets require size >= 4")
        generator = np.random.default_rng(spec.seed)
        if spec.kind == "uniform":
            points = generator.uniform(-1.0, 1.0, size=(spec.size, 3)).astype(np.float32)
        elif spec.kind == "clustered":
            centers = generator.uniform(-0.75, 0.75, size=(spec.clusters, 3))
            assignments = generator.integers(0, spec.clusters, size=spec.size)
            noise = generator.normal(0.0, spec.cluster_std, size=(spec.size, 3))
            points = np.clip(centers[assignments] + noise, -1.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported dataset kind: {spec.kind}")
        metadata = {
            "seed": spec.seed,
            "clusters": spec.clusters if spec.kind == "clustered" else None,
            "cluster_std": spec.cluster_std if spec.kind == "clustered" else None,
        }

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point array contains NaN or infinity")
    points = np.ascontiguousarray(points, dtype=np.float32)
    metadata.update(
        {
            "name": spec.name,
            "kind": spec.kind,
            "num_points": int(points.shape[0]),
            "dtype": str(points.dtype),
            "bounds_min": points.min(axis=0).tolist(),
            "bounds_max": points.max(axis=0).tolist(),
        }
    )
    return points, metadata


def canonicalize_edges(edges: torch.Tensor, num_points: int) -> torch.Tensor:
    """Return unique, lexicographically sorted, undirected non-self edges."""
    edges = torch.as_tensor(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected edge shape (E, 2), got {tuple(edges.shape)}")
    edges = edges.to(dtype=torch.int64)
    if edges.numel() == 0:
        return edges.reshape(0, 2)
    if bool(((edges < 0) | (edges >= num_points)).any().item()):
        raise ValueError("Edge index is outside the point range")
    edges = torch.sort(edges, dim=1).values
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.numel() == 0:
        return edges.reshape(0, 2)
    codes = torch.unique(edges[:, 0] * num_points + edges[:, 1], sorted=True)
    return torch.stack((torch.div(codes, num_points, rounding_mode="floor"), codes % num_points), dim=1)


def tetrahedra_to_edges(tetrahedra: torch.Tensor, num_points: int) -> torch.Tensor:
    tetrahedra = torch.as_tensor(tetrahedra)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError(f"Expected tetrahedra with shape (M, 4), got {tuple(tetrahedra.shape)}")
    pairs = torch.tensor(_TET_EDGE_INDICES, device=tetrahedra.device)
    edges = tetrahedra.to(dtype=torch.int64)[:, pairs].reshape(-1, 2)
    return canonicalize_edges(edges, num_points)


def validate_csr(adjacency: torch.Tensor, offsets: torch.Tensor, num_points: int) -> None:
    if adjacency.ndim != 1:
        raise ValueError("CSR adjacency must be one-dimensional")
    if offsets.ndim != 1 or offsets.shape[0] != num_points + 1:
        raise ValueError(f"CSR offsets must have shape ({num_points + 1},)")
    offsets64 = offsets.to(dtype=torch.int64)
    if int(offsets64[0].item()) != 0 or int(offsets64[-1].item()) != adjacency.numel():
        raise ValueError("CSR offsets do not span the adjacency array")
    if bool((offsets64[1:] < offsets64[:-1]).any().item()):
        raise ValueError("CSR offsets must be nondecreasing")
    if adjacency.numel() and bool(((adjacency < 0) | (adjacency >= num_points)).any().item()):
        raise ValueError("CSR adjacency contains an out-of-range index")


def csr_to_edges(adjacency: torch.Tensor, offsets: torch.Tensor, num_points: int) -> torch.Tensor:
    validate_csr(adjacency, offsets, num_points)
    counts = (offsets[1:] - offsets[:-1]).to(dtype=torch.int64)
    owners = torch.repeat_interleave(
        torch.arange(num_points, device=offsets.device, dtype=torch.int64), counts
    )
    return canonicalize_edges(torch.stack((owners, adjacency.to(dtype=torch.int64)), dim=1), num_points)


def status_histogram(status: torch.Tensor) -> dict[str, int]:
    values, counts = torch.unique(torch.as_tensor(status).to(dtype=torch.int64).cpu(), return_counts=True)
    histogram: dict[str, int] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        name = PARAGRAM_STATUS_NAMES[value] if 0 <= value < len(PARAGRAM_STATUS_NAMES) else f"unknown_{value}"
        histogram[name] = int(count)
    return histogram


def compare_edges(reference: torch.Tensor, candidate: torch.Tensor, num_points: int) -> dict[str, Any]:
    reference = canonicalize_edges(reference, num_points)
    candidate = canonicalize_edges(candidate, num_points)
    reference_codes = reference[:, 0] * num_points + reference[:, 1]
    candidate_codes = candidate[:, 0] * num_points + candidate[:, 1]
    positions = torch.searchsorted(reference_codes, candidate_codes)
    positions = positions.clamp(max=max(reference_codes.numel() - 1, 0))
    common = 0 if reference_codes.numel() == 0 else int((reference_codes[positions] == candidate_codes).sum().item())
    precision = common / candidate.shape[0] if candidate.shape[0] else 1.0
    recall = common / reference.shape[0] if reference.shape[0] else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = reference.shape[0] + candidate.shape[0] - common

    def degrees(edges: torch.Tensor) -> torch.Tensor:
        return torch.bincount(edges.reshape(-1), minlength=num_points)

    degree_delta = (degrees(candidate) - degrees(reference)).abs()
    return {
        "reference_edges": int(reference.shape[0]),
        "candidate_edges": int(candidate.shape[0]),
        "common_edges": common,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": common / union if union else 1.0,
        "degree_mean_absolute_error": float(degree_delta.float().mean().item()),
        "degree_max_absolute_error": int(degree_delta.max().item()) if degree_delta.numel() else 0,
    }


def amdahl_speedup(backend_speedup: float, share: float = DELAUNAY_PROFILE_SHARE) -> float | None:
    if not math.isfinite(backend_speedup) or backend_speedup <= 0:
        return None
    return 1.0 / ((1.0 - share) + share / backend_speedup)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_timings(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(records)}
    for key in ("raw_wall_ms", "raw_cuda_ms", "solver_reported_ms", "end_to_end_ms"):
        values = [float(record[key]) for record in records if record.get(key) is not None]
        summary[key] = {
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "p10": _percentile(values, 10),
            "p90": _percentile(values, 90),
        }
    return summary


def _cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _paragram_once(
    points_np: np.ndarray, common_output: bool = True
) -> tuple[dict[str, Any], torch.Tensor | None]:
    paragram = importlib.import_module("paragram")
    total_start = time.perf_counter()
    points = torch.from_numpy(points_np).to(device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    start_event, end_event = _cuda_event_pair()
    raw_start = time.perf_counter()
    start_event.record()
    diagram = paragram.voronoi_diagram(points)
    end_event.record()
    end_event.synchronize()
    raw_wall_ms = (time.perf_counter() - raw_start) * 1000.0
    raw_cuda_ms = start_event.elapsed_time(end_event)
    record: dict[str, Any] = {
        "raw_wall_ms": raw_wall_ms,
        "raw_cuda_ms": raw_cuda_ms,
        "solver_reported_ms": None,
        "adjacency_entries": int(diagram.adjacency.numel()),
        "status_histogram": status_histogram(diagram.status),
    }
    if not common_output:
        record["edges"] = None
        record["end_to_end_ms"] = (time.perf_counter() - total_start) * 1000.0
        record["conversion_error"] = None
        return record, None
    try:
        edges = csr_to_edges(diagram.adjacency, diagram.offsets, points.shape[0])
        torch.cuda.synchronize()
        record["edges"] = int(edges.shape[0])
        record["end_to_end_ms"] = (time.perf_counter() - total_start) * 1000.0
        record["conversion_error"] = None
    except Exception as exc:  # Preserve raw solver data if graph conversion fails.
        edges = None
        record["edges"] = None
        record["end_to_end_ms"] = None
        record["conversion_error"] = f"{type(exc).__name__}: {exc}"
    return record, edges


def _gdel_once(
    points_np: np.ndarray, common_output: bool = True
) -> tuple[dict[str, Any], torch.Tensor | None]:
    pygdel3d = importlib.import_module("pygdel3d")
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    raw_start = time.perf_counter()
    tetrahedra_np, solver_seconds = pygdel3d.triangulate(points_np)
    torch.cuda.synchronize()
    raw_wall_ms = (time.perf_counter() - raw_start) * 1000.0
    tetrahedra_np = np.asarray(tetrahedra_np)
    record: dict[str, Any] = {
        "raw_wall_ms": raw_wall_ms,
        "raw_cuda_ms": None,
        # gDel3D's PerfTimer and Python demo both report this value in milliseconds.
        "solver_reported_ms": float(solver_seconds),
        "tetrahedra": int(tetrahedra_np.shape[0]),
    }
    if not common_output:
        record["edges"] = None
        record["end_to_end_ms"] = (time.perf_counter() - total_start) * 1000.0
        record["conversion_error"] = None
        return record, None
    try:
        tetrahedra = torch.as_tensor(tetrahedra_np, device="cuda", dtype=torch.int64)
        edges = tetrahedra_to_edges(tetrahedra, points_np.shape[0])
        torch.cuda.synchronize()
        record["edges"] = int(edges.shape[0])
        record["end_to_end_ms"] = (time.perf_counter() - total_start) * 1000.0
        record["conversion_error"] = None
    except Exception as exc:  # Preserve raw solver data if graph conversion fails.
        edges = None
        record["edges"] = None
        record["end_to_end_ms"] = None
        record["conversion_error"] = f"{type(exc).__name__}: {exc}"
    return record, edges


def benchmark_backend(
    backend: str,
    points_np: np.ndarray,
    repetitions: int,
    warmups: int,
    common_output: bool = True,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    runner = _paragram_once if backend == "paragram" else _gdel_once
    try:
        for _ in range(warmups):
            runner(points_np, common_output=common_output)
        records: list[dict[str, Any]] = []
        last_edges = None
        for _ in range(repetitions):
            record, last_edges = runner(points_np, common_output=common_output)
            records.append(record)
        return {
            "status": "ok",
            "summary": summarize_timings(records),
            "last_sample": records[-1],
            "samples": records,
        }, last_edges
    except Exception as exc:
        _safe_empty_cuda_cache()
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, None


def measure_paragram_first_call(seed: int) -> dict[str, Any]:
    try:
        import_start = time.perf_counter()
        paragram = importlib.import_module("paragram")
        import_ms = (time.perf_counter() - import_start) * 1000.0
        generator = torch.Generator(device="cuda").manual_seed(seed)
        points = torch.rand((1024, 3), generator=generator, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        first_start = time.perf_counter()
        diagram = paragram.voronoi_diagram(points)
        torch.cuda.synchronize()
        return {
            "status": "ok",
            "import_ms": import_ms,
            "first_call_ms": (time.perf_counter() - first_start) * 1000.0,
            "status_histogram": status_histogram(diagram.status),
            "note": "The first call includes extension loading and compilation when no cached build exists.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _edge_spacing(points: torch.Tensor, edges: torch.Tensor) -> float:
    if not edges.numel():
        return 0.0
    max_sample = 1_000_000
    if edges.shape[0] > max_sample:
        step = max(1, edges.shape[0] // max_sample)
        edges = edges[::step][:max_sample]
    lengths = torch.linalg.vector_norm(points[edges[:, 0]] - points[edges[:, 1]], dim=1)
    return float(lengths.median().item())


def benchmark_warm_start(
    points_np: np.ndarray,
    frames: int,
    displacement_fractions: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    try:
        paragram = importlib.import_module("paragram")
        base = torch.from_numpy(points_np).to(device="cuda", dtype=torch.float32)
        base_diagram = paragram.voronoi_diagram(base)
        base_edges = csr_to_edges(base_diagram.adjacency, base_diagram.offsets, base.shape[0])
        spacing = _edge_spacing(base, base_edges)
        results = []
        for displacement_fraction in displacement_fractions:
            current = base.clone()
            previous = base_diagram
            generator = torch.Generator(device="cuda").manual_seed(seed + round(displacement_fraction * 1e6))
            frame_records = []
            for frame in range(frames):
                current = current + torch.randn(
                    current.shape, generator=generator, device=current.device, dtype=current.dtype
                ) * (spacing * displacement_fraction)

                # Alternate execution order to reduce systematic thermal/order bias.
                order = ("warm", "cold") if frame % 2 == 0 else ("cold", "warm")
                diagrams: dict[str, Any] = {}
                timings: dict[str, float] = {}
                for mode in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if mode == "warm":
                        diagrams[mode] = paragram.voronoi_diagram(
                            current,
                            initial_guesses=previous.adjacency,
                            initial_guesses_offsets=previous.offsets,
                        )
                    else:
                        diagrams[mode] = paragram.voronoi_diagram(current)
                    torch.cuda.synchronize()
                    timings[mode] = (time.perf_counter() - start) * 1000.0

                warm_edges = csr_to_edges(
                    diagrams["warm"].adjacency, diagrams["warm"].offsets, current.shape[0]
                )
                cold_edges = csr_to_edges(
                    diagrams["cold"].adjacency, diagrams["cold"].offsets, current.shape[0]
                )
                frame_records.append(
                    {
                        "frame": frame + 1,
                        "warm_ms": timings["warm"],
                        "cold_ms": timings["cold"],
                        "speedup": timings["cold"] / timings["warm"] if timings["warm"] else None,
                        "warm_status_histogram": status_histogram(diagrams["warm"].status),
                        "cold_status_histogram": status_histogram(diagrams["cold"].status),
                        "warm_cold_agreement": compare_edges(cold_edges, warm_edges, current.shape[0]),
                    }
                )
                previous = diagrams["warm"]
            results.append(
                {
                    "displacement_fraction": displacement_fraction,
                    "displacement_absolute": spacing * displacement_fraction,
                    "median_edge_spacing": spacing,
                    "frames": frame_records,
                    "warm_ms_median": statistics.median(record["warm_ms"] for record in frame_records),
                    "cold_ms_median": statistics.median(record["cold_ms"] for record in frame_records),
                    "speedup_median": statistics.median(record["speedup"] for record in frame_records),
                }
            )
        return {"status": "ok", "controlled_proxy": True, "results": results}
    except Exception as exc:
        _safe_empty_cuda_cache()
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _sample_free_vram(stop: threading.Event, samples: list[int]) -> None:
    while not stop.is_set():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            samples.append(int(free_bytes))
        except RuntimeError:
            pass
        stop.wait(0.001)


def _memory_worker(
    connection: Connection,
    spec_dict: dict[str, Any],
    backend: str,
    common_output: bool,
) -> None:
    try:
        spec = DatasetSpec(**spec_dict)
        points_np, _ = generate_points(spec)
        torch.cuda.init()
        # Load/compile the backend before taking the baseline.
        tiny = points_np[: min(1024, points_np.shape[0])]
        (_paragram_once if backend == "paragram" else _gdel_once)(tiny)
        _safe_empty_cuda_cache()
        torch.cuda.synchronize()
        baseline_free, total = torch.cuda.mem_get_info()
        samples = [int(baseline_free)]
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_free_vram, args=(stop, samples), daemon=True)
        sampler.start()
        start = time.perf_counter()
        runner = _paragram_once if backend == "paragram" else _gdel_once
        record = None
        for _ in range(3):
            record, _ = runner(points_np, common_output=common_output)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stop.set()
        sampler.join(timeout=1.0)
        connection.send(
            {
                "status": "ok",
                "baseline_used_bytes": int(total - baseline_free),
                "peak_incremental_bytes": int(baseline_free - min(samples)),
                "total_vram_bytes": int(total),
                "samples": len(samples),
                "iterations": 3,
                "elapsed_ms": elapsed_ms,
                "host_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "backend_record": record,
            }
        )
    except Exception as exc:
        connection.send(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def measure_memory(
    spec: DatasetSpec,
    backend: str,
    timeout_seconds: float,
    common_output: bool = True,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_memory_worker,
        args=(child, asdict(spec), backend, common_output),
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5)
            return {"status": "error", "error": f"Memory worker timed out after {timeout_seconds}s"}
        try:
            result = parent.recv()
        except EOFError:
            process.join(timeout=5)
            return {
                "status": "error",
                "error": "Memory worker exited without returning a result",
                "worker_exit_code": process.exitcode,
            }
        process.join(timeout=5)
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_metadata(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    direct_url = distribution.read_text("direct_url.json")
    return {
        "installed": True,
        "version": distribution.version,
        "direct_url": json.loads(direct_url) if direct_url else None,
    }


def collect_environment(command: Sequence[str]) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvidia_smi": _command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "nvcc": _command_output(["nvcc", "--version"]),
        "dccvt_git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "dccvt_git_dirty": bool(_command_output(["git", "status", "--short"])),
        "paragram": _package_metadata("paragram"),
        "pygdel3d": _package_metadata("pygdel3d"),
        "paragram_environment": {name: os.environ.get(name) for name in PARAGRAM_BUILD_ENV},
    }


def build_dataset_specs(
    root: Path,
    sizes: Iterable[int],
    distributions: Iterable[str],
    npz_paths: Iterable[str],
    include_default_npz: bool,
    seed: int,
    clusters: int,
    cluster_std: float,
) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            name=f"{distribution}_{size}",
            kind=distribution,
            size=size,
            seed=seed,
            clusters=clusters,
            cluster_std=cluster_std,
        )
        for distribution in distributions
        for size in sizes
    ]
    paths = list(npz_paths)
    if include_default_npz:
        paths.extend(str(root / path) for path in DEFAULT_NPZ_PATHS if (root / path).exists())
    seen: set[Path] = set()
    for value in paths:
        path = Path(value).resolve()
        if path in seen:
            continue
        seen.add(path)
        specs.append(DatasetSpec(name=f"npz_{path.parent.name}_{path.stem}", kind="npz", path=str(path)))
    return specs


def run_benchmark(
    specs: Sequence[DatasetSpec],
    backends: Sequence[str],
    repetitions: int,
    warmups: int,
    warm_start: bool,
    warm_frames: int,
    warm_displacements: Sequence[float],
    memory: bool,
    memory_timeout: float,
    seed: int,
    command: Sequence[str],
    common_output: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Paragram benchmark")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "environment": collect_environment(command),
        "config": {
            "backends": list(backends),
            "repetitions": repetitions,
            "warmups": warmups,
            "warm_start": warm_start,
            "warm_frames": warm_frames,
            "warm_displacements": list(warm_displacements),
            "memory": memory,
            "memory_timeout": memory_timeout,
            "seed": seed,
            "common_output": common_output,
            "delaunay_profile_share": DELAUNAY_PROFILE_SHARE,
        },
        "paragram_first_call": measure_paragram_first_call(seed) if "paragram" in backends else None,
        "cases": [],
    }
    first_call = result["paragram_first_call"]
    any_error = bool(first_call and first_call["status"] == "error")
    for case_index, spec in enumerate(specs):
        case: dict[str, Any] = {"spec": asdict(spec), "backends": {}, "comparison": None}
        case_error = False
        try:
            points_np, dataset_metadata = generate_points(spec)
            case["dataset"] = dataset_metadata
        except Exception as exc:
            case["status"] = "error"
            case["error"] = f"{type(exc).__name__}: {exc}"
            result["cases"].append(case)
            any_error = True
            continue

        edges_by_backend: dict[str, torch.Tensor] = {}
        # Alternate backend order between cases to reduce consistent order bias.
        order = list(backends)
        if case_index % 2:
            order.reverse()
        for backend in order:
            backend_result, edges = benchmark_backend(
                backend, points_np, repetitions, warmups, common_output=common_output
            )
            if memory:
                backend_result["memory"] = measure_memory(
                    spec, backend, memory_timeout, common_output=common_output
                )
                if backend_result["memory"]["status"] == "error":
                    case_error = True
                    any_error = True
            case["backends"][backend] = backend_result
            if backend_result["status"] == "error":
                case_error = True
                any_error = True
            if edges is not None:
                edges_by_backend[backend] = edges

        if "gdel3d" in edges_by_backend and "paragram" in edges_by_backend:
            case["comparison"] = compare_edges(
                edges_by_backend["gdel3d"], edges_by_backend["paragram"], points_np.shape[0]
            )
            gdel_time = case["backends"]["gdel3d"]["summary"]["raw_wall_ms"]["median"]
            paragram_time = case["backends"]["paragram"]["summary"]["raw_wall_ms"]["median"]
            if gdel_time and paragram_time:
                solver_speedup = gdel_time / paragram_time
                case["comparison"]["paragram_raw_speedup"] = solver_speedup
                case["comparison"]["projected_training_speedup"] = amdahl_speedup(solver_speedup)

        if warm_start and "paragram" in backends:
            case["warm_start"] = benchmark_warm_start(
                points_np, warm_frames, warm_displacements, seed + case_index
            )
            if case["warm_start"]["status"] == "error":
                case_error = True
                any_error = True
        case["status"] = "partial" if case_error else "ok"
        result["cases"].append(case)
        del edges_by_backend
        _safe_empty_cuda_cache()

    result["status"] = "partial" if any_error else "complete"
    return result


def write_json(result: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

