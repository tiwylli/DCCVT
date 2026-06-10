#!/usr/bin/env python3
"""Run the five hybrid direct input-channel ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_dccvt_hybrid_direct.py"

ABLATIONS: tuple[tuple[str, str], ...] = (
    ("hotspot_sdf", "configs/neural_hybrid_direct_ablation_hotspot_sdf.json"),
    ("hotspot_point_udf", "configs/neural_hybrid_direct_ablation_hotspot_point_udf.json"),
    ("hotspot_point_udf_abs", "configs/neural_hybrid_direct_ablation_hotspot_point_udf_abs.json"),
    ("hotspot_point_udf_confidence", "configs/neural_hybrid_direct_ablation_hotspot_point_udf_confidence.json"),
    ("full", "configs/neural_hybrid_direct_ablation_full.json"),
)
CONTROLLED_TRAINING_ARGS = ("--config", "--checkpoint-dir")
NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True)
class GpuInfo:
    """Memory status for one physical GPU."""

    index: int
    memory_total_mb: int
    memory_used_mb: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


@dataclass
class RunningJob:
    """A training subprocess assigned to one GPU."""

    run_name: str
    gpu_index: int
    process: subprocess.Popen
    log_file: object
    log_path: Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train all hybrid direct channel-ablation variants. Arguments not "
            "recognized by this wrapper are forwarded to train_dccvt_hybrid_direct.py."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", default="outputs/neural_dccvt/hybrid_direct_ablation")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    parser.add_argument("--parallel", action="store_true", help="Run ablations concurrently on available GPUs.")
    parser.add_argument(
        "--devices",
        default="auto",
        help="GPU ids to use, or 'auto' to use CUDA_VISIBLE_DEVICES/all visible GPUs.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Minimum free GPU memory required before launching a job.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait before checking busy GPUs again.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum concurrent jobs. Defaults to one job per eligible GPU.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow non-empty per-ablation checkpoint directories.",
    )
    return parser


def _reject_controlled_training_args(training_args: Sequence[str]) -> None:
    for arg in training_args:
        for controlled_arg in CONTROLLED_TRAINING_ARGS:
            if arg == controlled_arg or arg.startswith(f"{controlled_arg}="):
                raise SystemExit(
                    f"{controlled_arg} is controlled by this runner and cannot be passed as a training argument"
                )


def _training_arg_value(args: Sequence[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == option and index + 1 < len(args):
            return args[index + 1]
    return None


def _prepare_parallel_training_args(training_args: Sequence[str]) -> list[str]:
    device_value = _training_arg_value(training_args, "--device")
    if device_value == "cpu":
        raise SystemExit("--device cpu is not supported in --parallel mode")
    if device_value is None:
        return [*training_args, "--device", "cuda"]
    return list(training_args)


def _validate_output_dirs(output_root: Path, *, allow_existing: bool) -> None:
    if allow_existing:
        return
    for run_name, _ in ABLATIONS:
        checkpoint_dir = output_root / run_name
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise SystemExit(f"Refusing to reuse non-empty checkpoint directory: {checkpoint_dir}")


def build_commands(
    *,
    output_root: str | Path,
    training_args: Sequence[str],
    allow_existing: bool = False,
) -> list[tuple[str, list[str]]]:
    _reject_controlled_training_args(training_args)
    output_root = Path(output_root)
    _validate_output_dirs(output_root, allow_existing=allow_existing)

    commands: list[tuple[str, list[str]]] = []
    for run_name, config_path in ABLATIONS:
        checkpoint_dir = output_root / run_name
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(ROOT / config_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            *training_args,
        ]
        commands.append((run_name, command))
    return commands


def parse_nvidia_smi_output(output: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=index,memory.total,memory.used` output."""
    gpus: list[GpuInfo] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError(f"Could not parse nvidia-smi GPU row: {raw_line!r}")
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    memory_total_mb=int(parts[-2]),
                    memory_used_mb=int(parts[-1]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"Could not parse nvidia-smi GPU row: {raw_line!r}") from exc
    return gpus


def query_gpus() -> list[GpuInfo]:
    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("nvidia-smi was not found; --parallel requires CUDA GPUs") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"nvidia-smi failed with exit code {exc.returncode}: {exc.stderr}") from exc
    return parse_nvidia_smi_output(result.stdout)


def _parse_int_list(value: str, *, source: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit():
            raise SystemExit(f"{source} must contain integer GPU ids, got {token!r}")
        ids.append(int(token))
    if not ids:
        raise SystemExit(f"{source} did not contain any GPU ids")
    return ids


def resolve_device_ids(devices: str, *, env: dict[str, str] | None = None) -> list[int] | None:
    """Return allowed physical GPU ids, or None for every detected GPU."""
    if devices != "auto":
        return _parse_int_list(devices, source="--devices")

    env = os.environ if env is None else env
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return None
    if visible.strip() in {"-1", "NoDevFiles"}:
        raise SystemExit("CUDA_VISIBLE_DEVICES does not expose any GPUs")
    return _parse_int_list(visible, source="CUDA_VISIBLE_DEVICES")


def filter_available_gpus(
    gpus: Sequence[GpuInfo],
    *,
    allowed_ids: Sequence[int] | None,
    min_free_gb: float,
    excluded_ids: Sequence[int] = (),
) -> list[GpuInfo]:
    min_free_mb = int(float(min_free_gb) * 1024)
    allowed = set(allowed_ids) if allowed_ids is not None else None
    excluded = set(excluded_ids)
    available = []
    for gpu in gpus:
        if allowed is not None and gpu.index not in allowed:
            continue
        if gpu.index in excluded:
            continue
        if gpu.memory_free_mb >= min_free_mb:
            available.append(gpu)
    return sorted(available, key=lambda gpu: (-gpu.memory_free_mb, gpu.index))


def _eligible_gpu_count(gpus: Sequence[GpuInfo], allowed_ids: Sequence[int] | None) -> int:
    if allowed_ids is None:
        return len(gpus)
    detected = {gpu.index for gpu in gpus}
    return len([gpu_id for gpu_id in allowed_ids if gpu_id in detected])


def _max_active_jobs(max_jobs: int | None, eligible_gpu_count: int, command_count: int) -> int:
    if max_jobs is not None and max_jobs < 1:
        raise SystemExit("--max-jobs must be >= 1")
    limit = eligible_gpu_count if max_jobs is None else min(max_jobs, eligible_gpu_count)
    return min(limit, command_count)


def _format_gpu_status(gpus: Sequence[GpuInfo], allowed_ids: Sequence[int] | None = None) -> str:
    allowed = set(allowed_ids) if allowed_ids is not None else None
    parts = []
    for gpu in gpus:
        if allowed is not None and gpu.index not in allowed:
            continue
        parts.append(
            f"gpu={gpu.index} free={gpu.memory_free_mb / 1024:.1f}G "
            f"used={gpu.memory_used_mb / 1024:.1f}G total={gpu.memory_total_mb / 1024:.1f}G"
        )
    return "; ".join(parts) if parts else "no matching GPUs"


def build_parallel_dry_run_assignments(
    commands: Sequence[tuple[str, list[str]]],
    *,
    gpus: Sequence[GpuInfo],
    allowed_ids: Sequence[int] | None,
    min_free_gb: float,
    max_jobs: int | None = None,
) -> list[tuple[str, int, list[str]]]:
    available = filter_available_gpus(gpus, allowed_ids=allowed_ids, min_free_gb=min_free_gb)
    max_active = _max_active_jobs(max_jobs, len(available), len(commands))
    if max_active < 1:
        return []
    slots = available[:max_active]
    assignments: list[tuple[str, int, list[str]]] = []
    for index, (run_name, command) in enumerate(commands):
        assignments.append((run_name, slots[index % len(slots)].index, command))
    return assignments


def _command_text(command: Sequence[str], *, gpu_index: int | None = None) -> str:
    prefix = []
    if gpu_index is not None:
        prefix.append(f"CUDA_VISIBLE_DEVICES={gpu_index}")
    return " ".join([*prefix, *(shlex.quote(part) for part in command)])


def _start_job(run_name: str, command: list[str], checkpoint_dir: Path, gpu: GpuInfo) -> RunningJob:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "train.log"
    log_file = log_path.open("w", encoding="utf-8")
    command_text = _command_text(command, gpu_index=gpu.index)
    log_file.write(f"{command_text}\n")
    log_file.flush()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[{run_name}] started on GPU {gpu.index}; log={log_path}", flush=True)
    return RunningJob(run_name=run_name, gpu_index=gpu.index, process=process, log_file=log_file, log_path=log_path)


def _terminate_active_jobs(active_jobs: dict[int, RunningJob]) -> None:
    for job in active_jobs.values():
        if job.process.poll() is None:
            job.process.terminate()
    for job in active_jobs.values():
        if job.process.poll() is None:
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.process.kill()
        job.log_file.close()


def run_parallel(
    commands: Sequence[tuple[str, list[str]]],
    *,
    output_root: str | Path,
    devices: str,
    min_free_gb: float,
    poll_seconds: float,
    max_jobs: int | None,
    dry_run: bool,
) -> None:
    output_root = Path(output_root)
    allowed_ids = resolve_device_ids(devices)
    gpus = query_gpus()
    print(f"Detected GPUs: {_format_gpu_status(gpus, allowed_ids)}", flush=True)
    eligible_count = _eligible_gpu_count(gpus, allowed_ids)
    if eligible_count < 1:
        raise SystemExit("No detected GPUs match the selected --devices/CUDA_VISIBLE_DEVICES")

    if dry_run:
        assignments = build_parallel_dry_run_assignments(
            commands,
            gpus=gpus,
            allowed_ids=allowed_ids,
            min_free_gb=min_free_gb,
            max_jobs=max_jobs,
        )
        if not assignments:
            print(
                f"No GPU currently has at least {min_free_gb:.1f}G free; real run would wait and poll.",
                flush=True,
            )
            return
        for run_name, gpu_index, command in assignments:
            print(f"[{run_name}] gpu={gpu_index} {_command_text(command, gpu_index=gpu_index)}", flush=True)
        return

    max_active = _max_active_jobs(max_jobs, eligible_count, len(commands))
    pending = list(commands)
    active_jobs: dict[int, RunningJob] = {}
    last_wait_message = 0.0

    while pending or active_jobs:
        failed_job: RunningJob | None = None
        for gpu_index, job in list(active_jobs.items()):
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.log_file.close()
            del active_jobs[gpu_index]
            if return_code == 0:
                print(f"[{job.run_name}] completed on GPU {job.gpu_index}", flush=True)
            else:
                print(
                    f"[{job.run_name}] failed on GPU {job.gpu_index} with exit code {return_code}; log={job.log_path}",
                    flush=True,
                )
                failed_job = job
                break
        if failed_job is not None:
            _terminate_active_jobs(active_jobs)
            raise SystemExit(f"{failed_job.run_name} failed; stopped remaining ablation jobs")

        launched = False
        if pending and len(active_jobs) < max_active:
            gpus = query_gpus()
            available = filter_available_gpus(
                gpus,
                allowed_ids=allowed_ids,
                min_free_gb=min_free_gb,
                excluded_ids=active_jobs.keys(),
            )
            for gpu in available:
                if not pending or len(active_jobs) >= max_active:
                    break
                run_name, command = pending.pop(0)
                checkpoint_dir = output_root / run_name
                active_jobs[gpu.index] = _start_job(run_name, command, checkpoint_dir, gpu)
                launched = True

        if not pending and not active_jobs:
            break
        if launched:
            continue

        if pending and len(active_jobs) < max_active:
            now = time.monotonic()
            if now - last_wait_message >= max(float(poll_seconds), 1.0):
                gpus = query_gpus()
                print(
                    f"Waiting for a GPU with >= {min_free_gb:.1f}G free. "
                    f"Current: {_format_gpu_status(gpus, allowed_ids)}",
                    flush=True,
                )
                last_wait_message = now
            if active_jobs:
                time.sleep(min(max(float(poll_seconds), 1.0), 5.0))
            else:
                time.sleep(max(float(poll_seconds), 1.0))
        elif active_jobs:
            time.sleep(min(max(float(poll_seconds), 1.0), 5.0))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args, training_args = parser.parse_known_args(argv)
    if args.parallel:
        training_args = _prepare_parallel_training_args(training_args)
    commands = build_commands(
        output_root=args.output_root,
        training_args=training_args,
        allow_existing=args.allow_existing,
    )

    if args.parallel:
        run_parallel(
            commands,
            output_root=args.output_root,
            devices=args.devices,
            min_free_gb=args.min_free_gb,
            poll_seconds=args.poll_seconds,
            max_jobs=args.max_jobs,
            dry_run=args.dry_run,
        )
        return

    for run_name, command in commands:
        command_text = _command_text(command)
        print(f"[{run_name}] {command_text}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
