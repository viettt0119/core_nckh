"""Evaluation use cases: E1 accuracy report, E2 calibration simulation, E3 ablation table."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from eyegaze.adapters.checkpoint import load_checkpoint
from eyegaze.adapters.torch_data import FrameTable, GazeDataset, load_frame_table, select_split
from eyegaze.adapters.torch_model import EyeGazeNet, count_parameters
from eyegaze.application.calibration_sim import CalibrationMethod, simulate_calibration
from eyegaze.config import CropSource
from eyegaze.domain.metrics import euclidean_errors, gaze_report
from eyegaze.log import get_logger

log = get_logger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


@dataclass(frozen=True, slots=True)
class Predictions:
    index: NDArray[np.int64]  # row indices into the evaluated FrameTable
    gaze_cm: NDArray[np.float64]
    embedding: NDArray[np.float64]


@torch.no_grad()
def predict(model: EyeGazeNet, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device) -> Predictions:
    model.eval()
    idx: list[torch.Tensor] = []
    gaze: list[torch.Tensor] = []
    emb: list[torch.Tensor] = []
    for batch in loader:
        g, e = model(
            batch["right_eye"].to(device, non_blocking=True),
            batch["left_eye"].to(device, non_blocking=True),
            batch["face"].to(device, non_blocking=True),
            batch["geometry"].to(device, non_blocking=True),
        )
        idx.append(batch["index"])
        gaze.append(g.float().cpu())
        emb.append(e.float().cpu())
    if not idx:
        raise ValueError("Empty evaluation loader")
    order = torch.cat(idx).numpy().astype(np.int64)
    return Predictions(order, torch.cat(gaze).double().numpy(), torch.cat(emb).double().numpy())


def accuracy_report(table: FrameTable, preds: Predictions) -> dict[str, Any]:
    target = table.gaze_cm[preds.index].astype(np.float64)
    errors = euclidean_errors(preds.gaze_cm, target)
    report = gaze_report(errors, table.subject_id[preds.index].tolist()).as_dict()
    devices = table.device[preds.index]
    report["per_device"] = {
        str(d): {"n": int((devices == d).sum()), "mean": float(errors[devices == d].mean())} for d in np.unique(devices)
    }
    return report


def calibration_features(table: FrameTable, preds: Predictions) -> NDArray[np.float64]:
    """Features the on-device calibrator uses: ``[embedding, gaze_cm, pose]``."""
    pose = table.keypoints.pose[preds.index]
    return np.concatenate([preds.embedding, preds.gaze_cm, pose], axis=1)


def evaluate_run(
    run_dir: Path,
    manifest: Path,
    frames_root: Path,
    split: str = "test",
    calibration_k: Sequence[int] = (5, 9, 13),
    methods: Sequence[CalibrationMethod] = (CalibrationMethod.RIDGE,),
    batch_size: int = 512,
    num_workers: int = 2,
    limit: int | None = None,
    device_name: str = "auto",
    checkpoint_name: str = "best.pt",
) -> dict[str, Any]:
    device = resolve_device(device_name)
    ckpt = load_checkpoint(run_dir / checkpoint_name, device)
    data_cfg = ckpt.train_config.get("data", {})
    crop_source = CropSource(data_cfg.get("crop_source", CropSource.MEDIAPIPE))
    table = load_frame_table(manifest, ckpt.spec)
    subset = select_split(table, split, limit=limit, require_apple=crop_source is CropSource.APPLE)
    loader = DataLoader(
        GazeDataset(subset, frames_root, ckpt.spec, crop_source, augment=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    preds = predict(ckpt.model, loader, device)
    result: dict[str, Any] = {
        "run": run_dir.name,
        "checkpoint": checkpoint_name,
        "epoch": ckpt.epoch,
        "split": split,
        "crop_source": str(crop_source),
        "spec_overrides": data_cfg.get("spec_overrides", {}),
        "model": ckpt.model_config.model_dump(mode="json"),
        "parameters": count_parameters(ckpt.model),
        "accuracy": accuracy_report(subset, preds),
        "calibration": [],
    }
    features = calibration_features(subset, preds)
    targets = subset.gaze_cm[preds.index].astype(np.float64)
    for method in methods:
        for k in calibration_k:
            try:
                sim = simulate_calibration(
                    features,
                    targets,
                    preds.gaze_cm,
                    subset.dot_num[preds.index],
                    subset.subject_id[preds.index],
                    k=k,
                    method=method,
                )
            except ValueError as exc:
                log.warning("calibration_sim_skipped", k=k, method=str(method), reason=str(exc))
                continue
            result["calibration"].append(sim.as_dict())
    out = run_dir / f"eval_{split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("evaluation_written", path=str(out), subject_mean=result["accuracy"]["subject_mean"])
    return result


# --------------------------------------------------------------------------- E3 ablation table


def _calibrated(result: dict[str, Any], k: int) -> float | None:
    for row in result.get("calibration", []):
        if row["k"] == k and row["method"] == CalibrationMethod.RIDGE:
            return float(row["after_subject_mean"])
    return None


def ablation_table(run_dirs: Sequence[Path], baseline: str, split: str = "test", k: int = 9) -> str:
    """Markdown table of every run's error and its delta vs ``baseline`` (from ``eval_<split>.json``)."""
    results: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        path = run_dir / f"eval_{split}.json"
        if path.is_file():
            results[run_dir.name] = json.loads(path.read_text(encoding="utf-8"))
    if baseline not in results:
        raise ValueError(f"Baseline run {baseline!r} has no eval_{split}.json")
    base = results[baseline]["accuracy"]["subject_mean"]
    lines = [
        f"| Run | Crops | Subject mean (cm) | Δ vs {baseline} | Frame mean (cm) | Calibrated k={k} (cm) | Params |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, res in sorted(results.items(), key=lambda kv: kv[1]["accuracy"]["subject_mean"]):
        acc = res["accuracy"]
        cal = _calibrated(res, k)
        lines.append(
            f"| {name} | {res['crop_source']} | {acc['subject_mean']:.3f} | {acc['subject_mean'] - base:+.3f} | "
            f"{acc['frame']['mean']:.3f} | {'n/a' if cal is None else f'{cal:.3f}'} | {res['parameters']:,} |"
        )
    return "\n".join(lines) + "\n"
