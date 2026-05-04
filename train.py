from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from eyegaze.model import EyeGazeNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EyeGaze model from extracted GazeCapture data.")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/eyegaze")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--face-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gaze-loss-weight", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch(
    model: EyeGazeNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    gaze_loss_weight: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    total_gaze_l1 = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            gaze_xy_pred, direction_logits = model(batch["face"], batch["left_eye"], batch["right_eye"], batch["face_grid"], batch["geometry"])
            direction_loss = F.cross_entropy(direction_logits, batch["direction"])
            gaze_loss = F.mse_loss(gaze_xy_pred, batch["gaze_xy"])
            loss = direction_loss + gaze_loss_weight * gaze_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        rows = batch["direction"].numel()
        total_rows += rows
        total_loss += float(loss.detach().item()) * rows
        total_correct += int((direction_logits.argmax(dim=1) == batch["direction"]).sum().item())
        total_gaze_l1 += float(torch.abs(gaze_xy_pred - batch["gaze_xy"]).mean().detach().item()) * rows

    return {
        "loss": total_loss / max(total_rows, 1),
        "accuracy": total_correct / max(total_rows, 1),
        "gaze_l1": total_gaze_l1 / max(total_rows, 1),
    }


def main() -> None:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from eyegaze.dataset import GazeCaptureDataset, load_or_build_manifest, make_weighted_sampler, split_manifest
    from eyegaze.model import EyeGazeNet, save_checkpoint

    torch.manual_seed(args.seed)
    data_root = (ROOT / args.data).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    df = load_or_build_manifest(data_root, manifest_path, rebuild=args.rebuild_manifest)
    train_df = split_manifest(df, "train", args.limit_train, args.seed)
    val_df = split_manifest(df, "val", args.limit_val, args.seed)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Manifest must contain non-empty train and val splits.")

    train_ds = GazeCaptureDataset(train_df, image_size=args.image_size, face_size=args.face_size, train=True)
    val_ds = GazeCaptureDataset(val_df, image_size=args.image_size, face_size=args.face_size, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_df),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeGazeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_acc = -1.0
    history: list[dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Rows: train={len(train_df):,} val={len(val_df):,}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, args.gaze_loss_weight)
        scheduler.step()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, None, scaler, args.gaze_loss_weight)
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(
            f"epoch {epoch:03d} "
            f"train_acc={record['train_accuracy']:.3f} val_acc={record['val_accuracy']:.3f} "
            f"train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f}"
        )
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            save_checkpoint(out_dir / "best.dat", model, image_size=args.image_size, face_size=args.face_size, metrics=record)
        save_checkpoint(out_dir / "last.dat", model, image_size=args.image_size, face_size=args.face_size, metrics=record)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {out_dir / 'best.dat'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
