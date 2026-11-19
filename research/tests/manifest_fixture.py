"""Synthetic training manifest + frames (train/val/test, several subjects and dots)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eyegaze.adapters.landmark_store import SCHEMA
from eyegaze.domain.observation import HeadPose
from tests.factories import FRAME_H, FRAME_W, make_observation

SPLITS = {"train": ("t1", "t2", "t3"), "val": ("v1", "v2"), "test": ("s1", "s2", "s3")}
FRAMES_PER_SUBJECT = 8
DOTS_PER_SUBJECT = 8  # one frame per dot, so calibration simulation has 8 targets per subject


def write_manifest(root: Path, frames_per_subject: int = FRAMES_PER_SUBJECT) -> tuple[Path, Path]:
    """Return ``(manifest_path, frames_root)``."""
    frames_root = root / "frames_root"
    rng = np.random.default_rng(0)
    rows: dict[str, list[object]] = {name: [] for name in SCHEMA.names}
    for split, subjects in SPLITS.items():
        for subject in subjects:
            frame_dir = frames_root / subject / "frames"
            frame_dir.mkdir(parents=True)
            for i in range(frames_per_subject):
                name = f"{i:05d}.jpg"
                cv2.imwrite(str(frame_dir / name), rng.integers(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8))
                roll = float(rng.uniform(-0.2, 0.2))
                obs = make_observation(image_roll_rad=roll, pose=HeadPose(0.05, -0.1, roll), seed=i)
                values: dict[str, object] = {
                    "subject_id": subject,
                    "frame_index": i,
                    "frame_relpath": f"{subject}/frames/{name}",
                    "split": split,
                    "device": "iPhone 6" if i % 2 else "iPhone 5s",
                    "orientation": 1,
                    "dot_num": i % DOTS_PER_SUBJECT,
                    "gaze_cm_x": float(rng.uniform(-3, 3)),
                    "gaze_cm_y": float(rng.uniform(-10, -1)),
                    "frame_width": FRAME_W,
                    "frame_height": FRAME_H,
                    "status": "ok",
                    "landmarks": obs.landmarks.astype(np.float32).reshape(-1).tolist(),
                    "transform": obs.transform_matrix.astype(np.float32).reshape(-1).tolist(),
                    "blink_left": 0.05,
                    "blink_right": 0.05,
                }
                boxes = {"face": (160, 220, 160, 220), "left_eye": (250, 280, 60, 40), "right_eye": (170, 280, 60, 40)}
                for part, (x, y, w, h) in boxes.items():
                    values.update(
                        {f"apple_{part}_x": x, f"apple_{part}_y": y, f"apple_{part}_w": w, f"apple_{part}_h": h}
                    )
                for key, value in values.items():
                    rows[key].append(value)
    manifest = root / "manifest.parquet"
    pq.write_table(pa.Table.from_pydict(rows, schema=SCHEMA), manifest)
    return manifest, frames_root
