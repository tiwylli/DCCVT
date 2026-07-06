"""CLI for the hybrid-direct mesh fine-tuning study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from dccvt.neural.experiments.mesh_finetune_cv.config import DEFAULT_OUTPUT_ROOT, load_experiment_config
from dccvt.neural.experiments.mesh_finetune_cv.folds import _write_ids, assign_folds, prepare_experiment, read_model_ids
from dccvt.neural.experiments.mesh_finetune_cv.jobs import (
    _select_folds,
    _select_variants,
    build_evaluation_jobs,
    build_inference_jobs,
    build_training_jobs,
    create_evaluation_views,
    run_jobs,
    selected_ids_file,
)
from dccvt.neural.experiments.mesh_finetune_cv.summary import summarize_results

def _parse_csv_ints(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hybrid direct mesh-loss adaptation study.")
    parser.add_argument("--config", default="configs/neural_hybrid_mesh_finetune_cv.json")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "infer", "evaluate", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold indices.")
    parser.add_argument("--variants", default=None, help="Optional comma-separated variant names.")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed training or inference outputs.")
    parser.add_argument("--parallel", action="store_true", help="Run training jobs across available GPUs.")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    output_root = Path(args.output_root).resolve()
    all_folds = assign_folds(read_model_ids(config.source_ids_file), config.fold_count)
    folds = _select_folds(all_folds, _parse_csv_ints(args.folds))
    variants = _select_variants(config.variants, _parse_csv_strings(args.variants))
    include_baseline = not args.no_baseline

    if args.stage in {"prepare", "all"}:
        if args.dry_run:
            print(f"[prepare] output_root={output_root}")
        else:
            prepare_experiment(config, output_root)
    elif not (output_root / "resolved_config.json").exists():
        if args.dry_run:
            print(f"[prepare-required] output_root={output_root}")
        else:
            prepare_experiment(config, output_root)

    if args.stage in {"train", "all"}:
        jobs = build_training_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            force=args.force,
        )
        if args.parallel:
            from scripts.run_hybrid_direct_channel_ablation import run_parallel

            run_parallel(
                [(job.name, list(job.command)) for job in jobs],
                output_root=output_root / "runs",
                devices=args.devices,
                min_free_gb=args.min_free_gb,
                poll_seconds=args.poll_seconds,
                max_jobs=args.max_jobs,
                dry_run=args.dry_run,
            )
        else:
            run_jobs(jobs, dry_run=args.dry_run)

    if args.stage in {"infer", "all"}:
        jobs = build_inference_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            include_baseline=include_baseline,
            force=args.force,
        )
        run_jobs(jobs, dry_run=args.dry_run)
        if not args.dry_run:
            create_evaluation_views(
                config,
                output_root,
                folds=folds,
                variants=variants,
                include_baseline=include_baseline,
            )

    if args.stage in {"evaluate", "all"}:
        ids_path = selected_ids_file(output_root, folds)
        if args.dry_run:
            print(f"[evaluation-ids] {ids_path}")
        else:
            _write_ids(ids_path, (model_id for fold in folds for model_id in fold.test_ids))
            create_evaluation_views(
                config,
                output_root,
                folds=folds,
                variants=variants,
                include_baseline=include_baseline,
            )
        jobs = build_evaluation_jobs(
            config,
            output_root,
            folds=folds,
            variants=variants,
            include_baseline=include_baseline,
        )
        run_jobs(jobs, dry_run=args.dry_run)

    if args.stage in {"summarize", "all"}:
        if not include_baseline:
            raise SystemExit("Summarization requires the baseline")
        if args.dry_run:
            print(f"[summarize] {output_root / 'summary'}")
        else:
            summary = summarize_results(config, output_root, folds=folds, variants=variants)
            print(json.dumps(summary, indent=2, sort_keys=True))
