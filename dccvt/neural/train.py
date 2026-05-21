"""Training utilities for the DCCVT neural prototype."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dccvt.neural.dataset import DCCVTGeneratorDataset, discover_mesh_ids
from dccvt.neural.models import PointNetDCCVT


@dataclass
class TrainConfig:
    label_root: str = "outputs/neural_labels/n32"
    mesh_root: str = "mesh/thingi32"
    output_dir: str = "outputs/neural_runs/pointnet_n32"
    mesh_ids: Optional[List[str]] = None
    num_points: int = 9600
    num_centroids: int = 32
    batch_size: int = 1
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    offset_reg_weight: float = 1e-4
    sign_loss_weight: float = 0.1
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _split_indices(num_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator).tolist()
    if num_items <= 1:
        return indices, []
    val_count = max(1, int(round(num_items * val_fraction)))
    val_count = min(val_count, num_items - 1)
    return indices[val_count:], indices[:val_count]


def _make_loader(dataset, indices: list[int], config: TrainConfig, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    *,
    model: PointNetDCCVT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    offset_reg_weight: float,
    sign_loss_weight: float,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "site_loss": 0.0,
        "sdf_loss": 0.0,
        "sign_loss": 0.0,
        "offset_reg": 0.0,
        "sign_acc": 0.0,
        "pred_neg_frac": 0.0,
    }
    count = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            points = batch["points"].to(device=device, dtype=torch.float32)
            target_sites = batch["target_sites"].to(device=device, dtype=torch.float32)
            target_sdf = batch["target_sdf"].to(device=device, dtype=torch.float32)

            pred = model(points)
            site_loss = criterion(pred["sites"], target_sites)
            sdf_loss = criterion(pred["sites_sdf"], target_sdf)
            sign_loss = _balanced_sign_loss(pred["sites_sdf"], target_sdf)
            offset_reg = pred["offsets"].pow(2).mean()
            loss = site_loss + sdf_loss + sign_loss_weight * sign_loss + offset_reg_weight * offset_reg

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred_inside = pred["sites_sdf"] < 0
                target_inside = target_sdf < 0
                sign_acc = (pred_inside == target_inside).float().mean()
                pred_neg_frac = pred_inside.float().mean()

            batch_size = points.shape[0]
            totals["loss"] += loss.item() * batch_size
            totals["site_loss"] += site_loss.item() * batch_size
            totals["sdf_loss"] += sdf_loss.item() * batch_size
            totals["sign_loss"] += sign_loss.item() * batch_size
            totals["offset_reg"] += offset_reg.item() * batch_size
            totals["sign_acc"] += sign_acc.item() * batch_size
            totals["pred_neg_frac"] += pred_neg_frac.item() * batch_size
            count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def _balanced_sign_loss(pred_sdf: torch.Tensor, target_sdf: torch.Tensor) -> torch.Tensor:
    """Balanced BCE on SDF sign, using `-sdf` as the inside logit."""
    target_inside = (target_sdf < 0).to(dtype=pred_sdf.dtype)
    inside_count = target_inside.sum()
    outside_count = target_inside.numel() - inside_count
    if inside_count.item() > 0:
        pos_weight = outside_count / inside_count.clamp(min=1)
    else:
        pos_weight = pred_sdf.new_tensor(1.0)
    return nn.functional.binary_cross_entropy_with_logits(
        -pred_sdf,
        target_inside,
        pos_weight=pos_weight,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: PointNetDCCVT,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    train_metrics: dict,
    val_metrics: Optional[dict],
    mesh_ids: list[str],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config(),
            "train_config": asdict(config),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "mesh_ids": mesh_ids,
        },
        path,
    )


def train_model(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)

    mesh_ids = config.mesh_ids
    if mesh_ids is None:
        mesh_ids = discover_mesh_ids(config.label_root)
    if not mesh_ids:
        raise ValueError(f"No training labels found under {config.label_root}")

    dataset = DCCVTGeneratorDataset(
        label_root=config.label_root,
        mesh_root=config.mesh_root,
        mesh_ids=mesh_ids,
        num_points=config.num_points,
        num_centroids=config.num_centroids,
        seed=config.seed,
    )
    train_indices, val_indices = _split_indices(len(dataset), config.val_fraction, config.seed)
    train_loader = _make_loader(dataset, train_indices, config, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, config, shuffle=False) if val_indices else None

    model = PointNetDCCVT(num_centroids=config.num_centroids).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            offset_reg_weight=config.offset_reg_weight,
            sign_loss_weight=config.sign_loss_weight,
        )
        val_metrics = None
        score = train_metrics["loss"]
        if val_loader is not None:
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                offset_reg_weight=config.offset_reg_weight,
                sign_loss_weight=config.sign_loss_weight,
            )
            score = val_metrics["loss"]

        print(
            f"epoch {epoch:04d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_sites={train_metrics['site_loss']:.6f} "
            f"train_sdf={train_metrics['sdf_loss']:.6f} "
            f"train_sign={train_metrics['sign_loss']:.6f} "
            f"train_neg={train_metrics['pred_neg_frac']:.4f}"
            + (f" val_loss={val_metrics['loss']:.6f}" if val_metrics else "")
        )

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            mesh_ids=mesh_ids,
        )
        if score <= best_val:
            best_val = score
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                mesh_ids=mesh_ids,
            )

    return best_path


def _parse_mesh_ids(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [part for part in value.replace(",", " ").split() if part]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DCCVT neural generator prototype.")
    parser.add_argument("--label-root", default=TrainConfig.label_root)
    parser.add_argument("--mesh-root", default=TrainConfig.mesh_root)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--mesh-ids", default=None, help="Comma or space separated mesh ids. Defaults to label discovery.")
    parser.add_argument("--num-points", type=int, default=TrainConfig.num_points)
    parser.add_argument("--num-centroids", type=int, default=TrainConfig.num_centroids)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--offset-reg-weight", type=float, default=TrainConfig.offset_reg_weight)
    parser.add_argument("--sign-loss-weight", type=float, default=TrainConfig.sign_loss_weight)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = TrainConfig(
        label_root=args.label_root,
        mesh_root=args.mesh_root,
        output_dir=args.output_dir,
        mesh_ids=_parse_mesh_ids(args.mesh_ids),
        num_points=args.num_points,
        num_centroids=args.num_centroids,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        offset_reg_weight=args.offset_reg_weight,
        sign_loss_weight=args.sign_loss_weight,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    best_path = train_model(config)
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
