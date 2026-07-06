"""Train the PoNQ-style site-only neural DCCVT model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from dccvt.neural.data.datasets import HotspotSDFDataset, resolve_cache_files
from dccvt.neural.losses import dccvt_finetune_loss, stage1_site_loss
from dccvt.neural.models import DCCVTPoNQNet
from dccvt.neural.utils import device_from_value, parse_mesh_ids


def save_checkpoint(
    path: Path,
    *,
    model: DCCVTPoNQNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    stats: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config(),
        "args": vars(args),
        "stats": stats,
    }
    torch.save(payload, path)


def build_model_from_args(args: argparse.Namespace) -> DCCVTPoNQNet:
    return DCCVTPoNQNet(
        grid_n=args.grid_n,
        k=args.k,
        feature_dim=args.feature_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PoNQ-style neural DCCVT site predictor.")
    parser.add_argument("--cache-root", required=True, help="Directory of precomputed HotSpot SDF .npz caches.")
    parser.add_argument("--mesh-ids", default=None, help="Optional comma or space separated cache stems.")
    parser.add_argument("--split-file", default=None, help="Optional text file of cache stems.")
    parser.add_argument("--checkpoint-dir", default="outputs/neural_dccvt/checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true", help="Also restore optimizer state from --resume.")
    parser.add_argument("--stage", choices=("1", "2"), default="1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-subsample", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=6.4e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--grid-n", type=int, default=33)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=5)
    parser.add_argument("--decoder-layers", type=int, default=3)

    parser.add_argument("--w-chamfer", type=float, default=100.0)
    parser.add_argument("--w-occupancy", type=float, default=1.0)
    parser.add_argument("--w-offset", type=float, default=0.1)
    parser.add_argument("--w-domain", type=float, default=1.0)
    parser.add_argument("--w-dccvt-chamfer", type=float, default=1000.0)
    parser.add_argument("--w-dccvt-cvt", type=float, default=100.0)
    parser.add_argument("--w-dccvt", type=float, default=1.0)
    parser.add_argument("--max-dccvt-sites", type=int, default=4096)
    parser.add_argument(
        "--stage2-train-encoder",
        action="store_true",
        help="Allow Stage 2 to update the encoder/activity features. By default Stage 2 updates only the site head.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.stage == "2" and args.device == "auto":
        from dccvt.device import device as dccvt_device
        from dccvt.device import initialize_runtime

        initialize_runtime()
        device = dccvt_device
    else:
        device = device_from_value(args.device)

    cache_files = resolve_cache_files(
        args.cache_root,
        mesh_ids=parse_mesh_ids(args.mesh_ids),
        split_file=args.split_file,
    )
    dataset = HotspotSDFDataset(cache_files, target_subsample=args.target_subsample)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    if resume_checkpoint is not None and resume_checkpoint.get("model_config"):
        model = DCCVTPoNQNet(**resume_checkpoint["model_config"]).to(device)
    else:
        model = build_model_from_args(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if args.resume_optimizer and "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    if args.stage == "2" and not args.stage2_train_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        for param in model.activity_head.parameters():
            param.requires_grad_(False)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        print("Stage 2 default: frozen encoder/activity head; training site head only.")

    stop_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, stop_epoch):
        local_epoch = epoch - start_epoch + 1
        model.train()
        epoch_loss = 0.0
        epoch_stats: dict[str, float] = {}
        for batch in dataloader:
            sdf_grid = batch["sdf_grid"].to(device, non_blocking=True)
            target_points = batch["target_points"].to(device, non_blocking=True)
            near_surface_mask = batch["near_surface_mask"].to(device, non_blocking=True)
            gt_activity_mask = batch["gt_activity_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(sdf_grid)
            occupancy_weight = 0.0 if args.stage == "2" and not args.stage2_train_encoder else args.w_occupancy
            loss, stats = stage1_site_loss(
                outputs,
                target_points,
                gt_activity_mask,
                near_surface_mask,
                chamfer_weight=args.w_chamfer,
                occupancy_weight=occupancy_weight,
                offset_weight=args.w_offset,
                domain_weight=args.w_domain,
            )
            if args.stage == "2":
                active_mask = gt_activity_mask.bool() | near_surface_mask.bool()
                dccvt_loss, dccvt_stats = dccvt_finetune_loss(
                    outputs,
                    sdf_grid,
                    target_points,
                    active_mask,
                    chamfer_weight=args.w_dccvt_chamfer,
                    cvt_weight=args.w_dccvt_cvt,
                    max_sites_per_shape=args.max_dccvt_sites,
                )
                loss = loss + args.w_dccvt * dccvt_loss
                stats.update(dccvt_stats)

            loss.backward()
            optimizer.step()

            batch_loss = float(loss.detach().cpu())
            epoch_loss += batch_loss
            for key, value in stats.items():
                epoch_stats[key] = epoch_stats.get(key, 0.0) + float(value)

        num_batches = max(len(dataloader), 1)
        epoch_loss /= num_batches
        epoch_stats = {key: value / num_batches for key, value in epoch_stats.items()}
        print(f"epoch={epoch} local_epoch={local_epoch}/{args.epochs} loss={epoch_loss:.6g} stats={epoch_stats}")

        latest = checkpoint_dir / "latest.pt"
        save_checkpoint(latest, model=model, optimizer=optimizer, epoch=epoch, args=args, stats=epoch_stats)
        if args.save_every > 0 and (local_epoch % args.save_every == 0 or local_epoch == args.epochs):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                stats=epoch_stats,
            )


if __name__ == "__main__":
    main()
