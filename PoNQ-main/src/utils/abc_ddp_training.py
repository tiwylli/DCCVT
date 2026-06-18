"""Four-GPU reproduction of the original three-phase PoNQ ABC training."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import random
import subprocess

import h5py
import numpy as np
import torch
import torch.distributed as dist
from pytorch3d.ops import knn_gather, knn_points
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from SDF_CNN import CNN_3d_multiple_split


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_ids(path):
    ids = []
    with Path(path).open("r") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.extend(Path(part).stem for part in line.replace(",", " ").split())
    return ids


def _make_mask_close(sdf_input, grid_n):
    close = np.abs(sdf_input) < (2.0 / grid_n * np.sqrt(3.0))
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


def _make_gt_mask(samples, grid_n):
    cells = grid_n - 1
    indices = np.floor((samples + 1.0) * 0.5 * cells).astype(np.int64)
    indices = np.clip(indices, 0, cells - 1)
    mask = np.zeros((cells, cells, cells), dtype=np.bool_)
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return mask.reshape(cells ** 3)


class RankLocalABCDataset(Dataset):
    """Rank-local ABC loader with fixed PoNQ sample indices.

    The original PoNQ loader materializes the full training set in memory.
    That is fragile under DDP because each rank owns hundreds of shapes. By
    default this dataset caches only small SDF-derived fields and reads the
    sampled point cloud/normals from HDF5 per batch.
    """

    def __init__(self, hdf5_root, model_ids, sample_count, grid_n, seed, preload_mode="sdf"):
        if preload_mode not in {"none", "sdf", "full"}:
            raise ValueError("preload_mode must be one of: none, sdf, full")
        self.hdf5_root = Path(hdf5_root)
        self.model_ids = list(model_ids)
        self.grid_n = int(grid_n)
        self.preload_mode = preload_mode
        self.records = {}
        rng = np.random.RandomState(seed)
        self.sample_indices = rng.choice(1000000, int(sample_count))
        if self.preload_mode == "none":
            return

        unique_ids = list(dict.fromkeys(self.model_ids))
        for item_index, model_id in enumerate(unique_ids, start=1):
            path = self._path(model_id)
            with h5py.File(str(path), "r") as handle:
                sdf = np.asarray(handle["{}_sdf".format(self.grid_n - 1)][:], dtype=np.float32)
                pointcloud = np.asarray(handle["pointcloud"][:], dtype=np.float32)
                normals = (
                    np.asarray(handle["normals"][:], dtype=np.float32)
                    if self.preload_mode == "full"
                    else None
                )
            scaled_sdf = 2.0 * sdf
            scaled_points = 2.0 * pointcloud
            sdf_record = (
                scaled_sdf[None, ...],
                _make_mask_close(scaled_sdf, self.grid_n),
                _make_gt_mask(scaled_points, self.grid_n),
            )
            if self.preload_mode == "full":
                self.records[model_id] = (
                    sdf_record[0],
                    sdf_record[1],
                    scaled_points[self.sample_indices],
                    normals[self.sample_indices],
                    sdf_record[2],
                )
            else:
                self.records[model_id] = sdf_record
            print(
                "{} {}/{} {}".format(
                    "preload" if self.preload_mode == "full" else "cache",
                    item_index,
                    len(unique_ids),
                    model_id,
                ),
                flush=True,
            )

    def _path(self, model_id):
        return self.hdf5_root / "{}.hdf5".format(model_id)

    def __len__(self):
        return len(self.model_ids)

    def _load_sdf_record(self, handle):
        sdf = np.asarray(handle["{}_sdf".format(self.grid_n - 1)][:], dtype=np.float32)
        pointcloud = np.asarray(handle["pointcloud"][:], dtype=np.float32)
        scaled_sdf = 2.0 * sdf
        return (
            scaled_sdf[None, ...],
            _make_mask_close(scaled_sdf, self.grid_n),
            _make_gt_mask(2.0 * pointcloud, self.grid_n),
            pointcloud,
        )

    def __getitem__(self, index):
        model_id = self.model_ids[index]
        record = self.records.get(model_id)
        if self.preload_mode == "full":
            sdf, close_mask, points, normals, gt_mask = record
            return sdf, close_mask, points, normals, gt_mask, model_id

        with h5py.File(str(self._path(model_id)), "r") as handle:
            if record is None:
                sdf, close_mask, gt_mask, pointcloud = self._load_sdf_record(handle)
            else:
                sdf, close_mask, gt_mask = record
                pointcloud = np.asarray(handle["pointcloud"][:], dtype=np.float32)
            normals = np.asarray(handle["normals"][:], dtype=np.float32)

        points = 2.0 * pointcloud[self.sample_indices]
        sampled_normals = normals[self.sample_indices]
        return sdf, close_mask, points, sampled_normals, gt_mask, model_id


def _rank_shard(model_ids, rank, world_size):
    shard = list(model_ids[rank::world_size])
    target = (len(model_ids) + world_size - 1) // world_size
    if not shard:
        raise ValueError("A distributed rank received no PoNQ training shapes")
    while len(shard) < target:
        shard.append(shard[0])
    return shard


def _seed_everything(seed, cudnn_deterministic=False, cudnn_benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def _check_cuda_memory(device, min_free_gb):
    if min_free_gb <= 0:
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except AttributeError:
        return
    free_gb = free_bytes / float(1024 ** 3)
    total_gb = total_bytes / float(1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            "GPU {} has only {:.2f} GiB free out of {:.2f} GiB; "
            "PoNQ pretraining needs a mostly free GPU for the configured local "
            "batch. Stop other jobs, choose different CUDA_VISIBLE_DEVICES, "
            "lower ponq_training.global_batch_size, or pass --min-free-gb 0 "
            "to skip this guard.".format(device, free_gb, total_gb)
        )


def _worker_seed(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _quadric_loss(
    predicted_points,
    predicted_vstars,
    predicted_normals,
    predicted_quadrics,
    gt_points,
    gt_normals,
):
    pred_to_samples = knn_points(predicted_points, gt_points)
    samples_to_pred = knn_points(gt_points, predicted_points)
    gt_quadrics = torch.matmul(
        gt_normals[:, :, :, None],
        gt_normals[:, :, None, :],
    ).view(gt_normals.shape[0], gt_normals.shape[1], 9)
    predicted_quadrics = predicted_quadrics.view(
        predicted_quadrics.shape[0],
        predicted_quadrics.shape[1],
        9,
    )
    closest_quadric = knn_gather(predicted_quadrics, samples_to_pred.idx).squeeze(-2)
    closest_normal = knn_gather(predicted_normals, samples_to_pred.idx).squeeze(-2)
    closest_to_sample = knn_gather(predicted_vstars, samples_to_pred.idx).squeeze(-2)

    chamfer = (
        pred_to_samples.dists.squeeze(-1).mean(1)
        + samples_to_pred.dists.squeeze(-1).mean(1)
    ).mean()
    quadric = ((closest_quadric - gt_quadrics) ** 2).mean()
    normals = ((closest_normal - gt_normals) ** 2).mean()
    vstars = (((closest_to_sample - gt_points) * gt_normals).sum(-1) ** 2).mean()
    regularizer = ((predicted_points.detach() - predicted_vstars) ** 2).mean()
    return torch.stack((chamfer, vstars, normals, quadric, regularizer))


def _shape_losses(outputs, points, normals, close_mask, gt_mask):
    predicted_points, predicted_vstars, predicted_normals, predicted_quadrics, predicted_bool = outputs
    losses = []
    for batch_index in range(points.shape[0]):
        active = gt_mask[batch_index]
        geometry = _quadric_loss(
            predicted_points[batch_index, active].view(1, -1, 3),
            predicted_vstars[batch_index, active].view(1, -1, 3),
            predicted_normals[batch_index, active].view(1, -1, 3),
            predicted_quadrics[batch_index, active].view(1, -1, 3, 3),
            points[batch_index].view(1, -1, 3),
            normals[batch_index].view(1, -1, 3),
        )
        bool_loss = torch.relu(
            (predicted_bool[batch_index, close_mask[batch_index]] - gt_mask[
                batch_index, close_mask[batch_index]
            ].float()) ** 2
            - 1e-2
        ).mean()
        losses.append(
            torch.stack(
                (
                    geometry[0],
                    geometry[1],
                    geometry[2],
                    geometry[3],
                    bool_loss,
                    geometry[4],
                )
            )
        )
    return torch.stack(losses).mean(0)


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            universal_newlines=True,
        ).strip()
    except Exception:
        return "unknown"


def _save_checkpoint(path, model, optimizer, phase_index, epoch, config, args):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    state = module.state_dict()
    payload = {
        "phase": int(phase_index),
        "epoch": int(epoch),
        "model_state_dict": state,
        "encoder_state_dict": {
            key: value for key, value in state.items() if key.startswith("encoder.")
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "grid_n": 33,
            "k": int(config["ponq_training"]["k"]),
            "feature_dim": 128,
            "encoder_layers": 5,
            "decoder_layers": 3,
        },
        "seed": int(config["seed"]),
        "experiment_config": config,
        "args": vars(args),
        "git_revision": _git_revision(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _load_config(path):
    with Path(path).open("r") as handle:
        config = json.load(handle)
    config["paths"]["hdf5_root"] = str(_resolve_path(config["paths"]["hdf5_root"]))
    config["paths"]["train_split"] = str(_resolve_path(config["paths"]["train_split"]))
    config["paths"]["ponq_output_root"] = str(
        _resolve_path(config["paths"]["ponq_output_root"])
    )
    return config


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Reproduce PoNQ ABC training with DDP.")
    parser.add_argument("--config", default="configs/hybrid_ponq_abc_dccvt_v1.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=12.0,
        help="Abort early if a rank's GPU has less than this much free memory.",
    )
    parser.add_argument(
        "--cudnn-deterministic",
        action="store_true",
        help="Force deterministic cuDNN algorithms. Disabled by default to match PoNQ's legacy trainer.",
    )
    parser.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="Disable cuDNN benchmarking. Benchmarking is enabled by default for PoNQ 3D convolutions.",
    )
    parser.add_argument(
        "--preload-mode",
        choices=("sdf", "none", "full"),
        default="sdf",
        help=(
            "PoNQ data loading mode. 'sdf' caches only SDF masks and GT cell masks; "
            "'none' loads every field on demand; 'full' reproduces the old in-memory "
            "preload and needs substantial host RAM."
        ),
    )
    parser.add_argument(
        "--max-epochs-per-phase",
        type=int,
        default=None,
        help="Smoke-test override; omit for the exact configured schedule.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if not torch.cuda.is_available():
        raise RuntimeError("PoNQ ABC training requires CUDA")
    visible_device_count = torch.cuda.device_count()
    if local_world_size > visible_device_count or local_rank >= visible_device_count:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "torchrun launched {} local processes, but only {} CUDA devices are "
            "visible to this process. Set --nproc_per_node equal to the number "
            "of visible GPUs, or expose more GPUs. LOCAL_RANK={}, "
            "CUDA_VISIBLE_DEVICES={}".format(
                local_world_size,
                visible_device_count,
                local_rank,
                cuda_visible_devices,
            )
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))
    _check_cuda_memory(device, float(args.min_free_gb))
    if world_size > 1:
        dist.init_process_group("nccl")
    _seed_everything(
        int(config["seed"]) + rank,
        cudnn_deterministic=args.cudnn_deterministic,
        cudnn_benchmark=not args.no_cudnn_benchmark,
    )

    global_batch_size = int(config["ponq_training"]["global_batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError("PoNQ global batch size must be divisible by world size")
    local_batch_size = global_batch_size // world_size
    model_ids = _read_ids(config["paths"]["train_split"])
    local_ids = _rank_shard(model_ids, rank, world_size)
    output_root = Path(config["paths"]["ponq_output_root"])
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved = {
            "config": config,
            "args": vars(args),
            "world_size": world_size,
            "local_batch_size": local_batch_size,
            "cudnn_deterministic": bool(args.cudnn_deterministic),
            "cudnn_benchmark": bool(not args.no_cudnn_benchmark),
            "git_revision": _git_revision(),
        }
        (output_root / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True)
        )
        splits = output_root / "rank_splits"
        splits.mkdir(exist_ok=True)
        for shard_rank in range(world_size):
            shard = _rank_shard(model_ids, shard_rank, world_size)
            (splits / "rank_{}.txt".format(shard_rank)).write_text(
                "".join("{}\n".format(model_id) for model_id in shard)
            )
    if world_size > 1:
        dist.barrier()

    model = CNN_3d_multiple_split(
        grid_n=33,
        K=int(config["ponq_training"]["k"]),
        ef_dim=128,
        device=device,
    ).to(device)
    start_phase = 0
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        state = resume_checkpoint.get("model_state_dict", resume_checkpoint)
        model.load_state_dict(state)
        start_phase = int(resume_checkpoint.get("phase", 0))
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
    wrapped = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    loss_weights = torch.tensor(
        config["ponq_training"]["loss_weights"],
        dtype=torch.float32,
        device=device,
    )
    phases = config["ponq_training"]["phases"]
    optimizer = None
    last_phase_index = start_phase
    last_epoch = start_epoch - 1
    for phase_index, phase in enumerate(phases):
        if phase_index < start_phase:
            continue
        phase_start_epoch = start_epoch if phase_index == start_phase else 0
        phase_epochs = int(phase["epochs"])
        if args.max_epochs_per_phase is not None:
            phase_epochs = min(phase_epochs, args.max_epochs_per_phase)
        if phase_start_epoch >= phase_epochs:
            continue

        dataset = RankLocalABCDataset(
            config["paths"]["hdf5_root"],
            local_ids,
            int(phase["sample_count"]),
            33,
            int(config["seed"]) + phase_index,
            preload_mode=args.preload_mode,
        )
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"]) + phase_index + rank)
        loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_seed,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            wrapped.parameters(),
            lr=float(phase["learning_rate"]),
            weight_decay=float(config["ponq_training"]["weight_decay"]),
            betas=(
                float(config["ponq_training"]["beta1"]),
                float(config["ponq_training"]["beta2"]),
            ),
            amsgrad=bool(config["ponq_training"]["amsgrad"]),
        )
        if (
            resume_checkpoint is not None
            and phase_index == start_phase
            and args.resume_optimizer
            and "optimizer_state_dict" in resume_checkpoint
        ):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        for epoch in range(phase_start_epoch, phase_epochs):
            wrapped.train()
            totals = torch.zeros(7, device=device, dtype=torch.float64)
            for sdf, close_mask, points, normals, gt_mask, _ in loader:
                sdf = sdf.to(device, non_blocking=True)
                close_mask = close_mask.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                normals = normals.to(device, non_blocking=True)
                gt_mask = gt_mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = wrapped(sdf)
                losses = _shape_losses(outputs, points, normals, close_mask, gt_mask)
                # The legacy trainer sums shape losses across its batch. DDP
                # averages gradients across ranks, so restore the global sum.
                total = (loss_weights * losses).sum() * points.shape[0] * world_size
                total.backward()
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
                optimizer.step()
                totals[:6] += losses.detach().double()
                totals[6] += 1.0
            if world_size > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            means = (totals[:6] / totals[6].clamp(min=1.0)).cpu().tolist()
            if rank == 0:
                print(
                    "phase={} epoch={}/{} losses={}".format(
                        phase_index + 1,
                        epoch + 1,
                        phase_epochs,
                        means,
                    ),
                    flush=True,
                )
                _save_checkpoint(
                    output_root / "phase_{}_latest.pt".format(phase_index + 1),
                    wrapped,
                    optimizer,
                    phase_index,
                    epoch,
                    config,
                    args,
                )
            if world_size > 1:
                dist.barrier()
            last_phase_index = phase_index
            last_epoch = epoch
        start_epoch = 0
        resume_checkpoint = None

    if rank == 0:
        if optimizer is None:
            raise RuntimeError("No PoNQ training phase was executed")
        _save_checkpoint(
            output_root / "ponq_encoder.pt",
            wrapped,
            optimizer,
            last_phase_index,
            last_epoch,
            config,
            args,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
