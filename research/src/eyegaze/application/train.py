"""Training use case for EyeGazeNet v2.

Model selection uses the **subject-mean** validation error (every person counts
once), the same headline metric reported on test. ``last.pt`` carries the full
optimizer/scheduler state so a disconnected Colab session resumes exactly.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from eyegaze.adapters.checkpoint import CheckpointError, save_checkpoint, sha256_text, spec_to_json
from eyegaze.adapters.torch_data import (
    GazeDataset,
    effective_spec,
    geometry_stats,
    load_frame_table,
    select_split,
)
from eyegaze.adapters.torch_model import EyeGazeNet, count_parameters, euclidean_loss
from eyegaze.application.evaluate import accuracy_report, predict, resolve_device
from eyegaze.config import CropSource, TrainConfig, dump_config
from eyegaze.contracts import load_spec
from eyegaze.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainResult:
    run_dir: Path
    best_epoch: int
    best_val_subject_mean: float
    epochs_run: int
    stopped_early: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _loader(ds: GazeDataset, batch_size: int, shuffle: bool, workers: int, device: torch.device, seed: int) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(ds) > batch_size,
        generator=generator,
    )


def train(config: TrainConfig, resume: bool = False) -> TrainResult:
    run_dir = config.run_dir
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run directory {run_dir} is not empty; pass --resume or choose another run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    spec = effective_spec(load_spec(config.data.spec), config.data.spec_overrides, config.data.crop_source)
    spec_json = spec_to_json(spec)
    (run_dir / "spec.json").write_text(spec_json, encoding="utf-8")
    (run_dir / "config.yaml").write_text(dump_config(config), encoding="utf-8")

    table = load_frame_table(config.data.manifest, spec)
    apple = config.data.crop_source is CropSource.APPLE
    train_table = select_split(table, "train", config.data.limit_train, config.seed, require_apple=apple)
    val_table = select_split(table, "val", config.data.limit_val, config.seed, require_apple=apple)
    train_ds = GazeDataset(
        train_table, config.data.frames_root, spec, config.data.crop_source, config.data.augment, config.seed
    )
    val_ds = GazeDataset(val_table, config.data.frames_root, spec, config.data.crop_source, None, config.seed)

    model = EyeGazeNet(config.model)
    mean, std = geometry_stats(train_table, spec)
    model.set_geometry_stats(torch.from_numpy(mean), torch.from_numpy(std))
    model.to(device)

    optim = config.optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // optim.batch_size)
    total_steps = steps_per_epoch * optim.epochs
    warmup_steps = int(optim.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, warmup_steps, total_steps)
    )
    use_amp = optim.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_score, best_epoch, bad_epochs = 1, math.inf, 0, 0
    history: list[dict[str, Any]] = []
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state.get("spec_sha256") != sha256_text(spec_json):
            raise CheckpointError("Cannot resume: preprocess spec changed since the checkpoint was written")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch, bad_epochs = int(state["best_epoch"]), int(state["bad_epochs"])
        history = list(state["history"])
        log.info("resumed", epoch=start_epoch - 1, best_score=best_score)

    log.info(
        "training_start",
        run=config.run_name,
        device=str(device),
        train_frames=len(train_ds),
        val_frames=len(val_ds),
        parameters=count_parameters(model),
    )
    config_dict = config.model_dump(mode="json")
    stopped_early = False
    epoch = start_epoch - 1
    for epoch in range(start_epoch, optim.epochs + 1):
        if bad_epochs >= optim.patience:
            stopped_early = True
            break
        started = time.perf_counter()
        train_ds.set_epoch(epoch)
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in _loader(train_ds, optim.batch_size, True, config.data.num_workers, device, config.seed + epoch):
            target = batch["gaze_cm"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                gaze, _ = model(
                    batch["right_eye"].to(device, non_blocking=True),
                    batch["left_eye"].to(device, non_blocking=True),
                    batch["face"].to(device, non_blocking=True),
                    batch["geometry"].to(device, non_blocking=True),
                )
                loss = euclidean_loss(gaze.float(), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(target)
            seen += len(target)

        val_loader = _loader(val_ds, optim.batch_size, False, config.data.num_workers, device, config.seed)
        val = accuracy_report(val_table, predict(model, val_loader, device))
        score = float(val["subject_mean"])
        record = {
            "epoch": epoch,
            "train_loss_cm": loss_sum / max(seen, 1),
            "val_subject_mean_cm": score,
            "val_frame_mean_cm": val["frame"]["mean"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        log.info("epoch_done", **record)

        if score < best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(best_path, model, spec, epoch, record, config_dict)
        else:
            bad_epochs += 1
        save_checkpoint(
            last_path,
            model,
            spec,
            epoch,
            record,
            config_dict,
            extra={
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "history": history,
            },
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "run": config.run_name,
        "best_epoch": best_epoch,
        "best_val_subject_mean_cm": best_score,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "parameters": count_parameters(model),
        "spec_sha256": sha256_text(spec_json),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("training_done", **summary)
    return TrainResult(run_dir, best_epoch, best_score, len(history), stopped_early)
