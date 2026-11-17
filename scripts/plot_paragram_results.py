#!/usr/bin/env python3
"""Render runtime and VRAM scaling figures from a Paragram benchmark JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--large-scale-input",
        type=Path,
        help="Optional native-output capacity sweep JSON to extend the runtime plot",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def synthetic_cases(payload: dict, distribution: str) -> list[dict]:
    cases = [
        case
        for case in payload["cases"]
        if case["dataset"]["kind"] == distribution and case["dataset"]["name"].startswith(distribution)
    ]
    return sorted(cases, key=lambda case: case["dataset"]["num_points"])


def successful_scale_runs(payload: dict | None, distribution: str, backend: str) -> list[dict]:
    if payload is None:
        return []
    return [
        run
        for run in payload["runs"]
        if run["distribution"] == distribution
        and run["backend"] == backend
        and run["status"] == "ok"
        and "raw_wall_ms" in run
    ]


def plot_runtime(payload: dict, output: Path, large_scale_payload: dict | None = None) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    backend_styles = (
        ("gdel3d", "gDel3D", "tab:blue"),
        ("paragram", "Paragram", "tab:orange"),
    )
    for axis, distribution in zip(axes, ("uniform", "clustered")):
        cases = synthetic_cases(payload, distribution)
        sizes = [case["dataset"]["num_points"] for case in cases]
        for backend, label, color in backend_styles:
            raw_by_size = {
                case["dataset"]["num_points"]: case["backends"][backend]["summary"][
                    "raw_wall_ms"
                ]["median"]
                for case in cases
            }
            scale_runs = successful_scale_runs(large_scale_payload, distribution, backend)
            raw_by_size.update({run["sites"]: run["raw_wall_ms"] for run in scale_runs})
            raw_points = sorted(raw_by_size.items())
            end_to_end = [
                case["backends"][backend]["summary"]["end_to_end_ms"]["median"] for case in cases
            ]
            axis.plot(
                [size for size, _ in raw_points],
                [runtime for _, runtime in raw_points],
                color=color,
                marker="o",
                label=f"{label} native API",
            )
            axis.plot(
                sizes,
                end_to_end,
                color=color,
                marker="x",
                linestyle="--",
                label=f"{label} to edges (≤1M)",
            )
            single_call_runs = [run for run in scale_runs if run["repetitions"] == 1]
            axis.scatter(
                [run["sites"] for run in single_call_runs],
                [run["raw_wall_ms"] for run in single_call_runs],
                marker="^",
                s=55,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(distribution.capitalize())
        axis.set_xlabel("Sites")
    axes[0].set_ylabel("Synchronized wall time (ms)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Delaunay/Voronoi runtime and capacity on RTX 3080")
    if large_scale_payload is not None:
        figure.text(
            0.5,
            0.01,
            "Hollow triangles are single-call capacity probes; failed capacity points are omitted.",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_memory(payload: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, distribution in zip(axes, ("uniform", "clustered")):
        cases = synthetic_cases(payload, distribution)
        for backend, label in (("gdel3d", "gDel3D"), ("paragram", "Paragram")):
            points = [
                (case["dataset"]["num_points"], case["backends"][backend]["memory"])
                for case in cases
            ]
            sizes = [size for size, memory in points if memory["status"] == "ok"]
            peak_mib = [
                memory["peak_incremental_bytes"] / 2**20
                for _, memory in points
                if memory["status"] == "ok"
            ]
            axis.plot(sizes, peak_mib, marker="o", label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(distribution.capitalize())
        axis.set_xlabel("Sites")
    axes[0].set_ylabel("Peak incremental VRAM (MiB)")
    axes[1].legend()
    figure.suptitle("Isolated three-call memory trial on RTX 3080")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    large_scale_payload = None
    if args.large_scale_input is not None:
        with args.large_scale_input.open(encoding="utf-8") as stream:
            large_scale_payload = json.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime(
        payload,
        args.output_dir / "paragram_runtime_scaling.png",
        large_scale_payload=large_scale_payload,
    )
    plot_memory(payload, args.output_dir / "paragram_vram_scaling.png")


if __name__ == "__main__":
    main()
