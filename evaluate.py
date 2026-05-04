from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EyeGaze model.")
    parser.add_argument("--checkpoint", default="models/trained/gaze_model.dat")
    parser.add_argument("--data", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest-cache", default="runs/eyegaze/manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from eyegaze.dataset import GazeCaptureDataset, load_or_build_manifest, split_manifest
    from eyegaze.labels import DIRECTION_LABELS
    from eyegaze.model import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    model, ckpt = load_checkpoint(checkpoint_path, device)
    image_size = int(ckpt.get("image_size", 64))
    face_size = int(ckpt.get("face_size", 96))
    manifest_cache = (ROOT / args.manifest_cache).resolve()
    df = load_or_build_manifest(ROOT / args.data, manifest_cache)
    eval_df = split_manifest(df, args.split, args.limit)
    loader = DataLoader(GazeCaptureDataset(eval_df, image_size=image_size, face_size=face_size), batch_size=args.batch_size)

    correct = 0
    total = 0
    gaze_l1_sum = 0.0
    confusion: Counter[str] = Counter()
    with torch.no_grad():
        for batch in loader:
            face = batch["face"].to(device)
            left = batch["left_eye"].to(device)
            right = batch["right_eye"].to(device)
            grid = batch["face_grid"].to(device)
            geom = batch["geometry"].to(device)
            target = batch["direction"].to(device)
            gaze_target = batch["gaze_xy"].to(device)
            outputs = model(face, left, right, grid, geom)
            gaze_xy_pred, direction_logits = outputs
            pred = direction_logits.argmax(dim=1)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
            gaze_l1_sum += float(torch.abs(gaze_xy_pred - gaze_target).mean().item()) * int(target.numel())
            for t, p in zip(target.cpu().tolist(), pred.cpu().tolist()):
                confusion[f"{DIRECTION_LABELS[t]}->{DIRECTION_LABELS[p]}"] += 1

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "rows": total,
        "accuracy": correct / max(total, 1),
        "gaze_l1": gaze_l1_sum / max(total, 1),
        "confusion": dict(sorted(confusion.items())),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    out = Path(args.out) if args.out else checkpoint_path.parent / f"eval_{args.split}.json"
    out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
