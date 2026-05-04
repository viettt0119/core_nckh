from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from eyegaze.labels import direction_from_xy, xy_for_direction
from eyegaze.paths import DEFAULT_USER_ID, USER_MODEL_DIR

MIN_SAMPLES_PER_TARGET = 3
TARGET_ORDER = ("center", "left", "right", "up", "down")
TARGET_POINTS = {target: xy_for_direction(target) for target in TARGET_ORDER}


@dataclass
class GazeCalibration:
    matrix: np.ndarray

    def apply(self, gaze_xy: tuple[float, float] | np.ndarray) -> np.ndarray:
        xy = np.asarray(gaze_xy, dtype=np.float32)
        aug = np.array([xy[0], xy[1], 1.0], dtype=np.float32)
        corrected = self.matrix @ aug
        return np.clip(corrected, 0.0, 1.0)

    def direction(self, gaze_xy: tuple[float, float] | np.ndarray) -> str:
        corrected = self.apply(gaze_xy)
        return direction_from_xy(float(corrected[0]), float(corrected[1]))

    @classmethod
    def from_json(cls, data: dict) -> "GazeCalibration":
        return cls(matrix=np.asarray(data["matrix"], dtype=np.float32))


def user_dir(user_id: str = DEFAULT_USER_ID) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_id).strip("_")
    return USER_MODEL_DIR / (safe or DEFAULT_USER_ID)


def calibration_path(user_id: str = DEFAULT_USER_ID) -> Path:
    return user_dir(user_id) / "calibration.json"


def fit_calibration(raw_points: list[tuple[float, float]], target_points: list[tuple[float, float]]) -> GazeCalibration:
    if len(raw_points) != len(target_points):
        raise ValueError("Raw and target calibration sample counts must match.")
    if len(raw_points) < 5:
        raise ValueError("Need at least 5 calibration samples.")
    raw = np.asarray(raw_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    design = np.concatenate([raw, np.ones((len(raw), 1), dtype=np.float32)], axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return GazeCalibration(matrix=solution.T.astype(np.float32))


def validate_calibration_samples(
    samples: Iterable[dict],
    targets: Iterable[str] = TARGET_ORDER,
    min_samples_per_target: int = MIN_SAMPLES_PER_TARGET,
) -> None:
    counts = Counter(str(sample.get("target", "")) for sample in samples)
    missing = [target for target in targets if counts[target] < min_samples_per_target]
    if missing:
        details = ", ".join(f"{target}={counts[target]}/{min_samples_per_target}" for target in missing)
        raise ValueError(f"Not enough calibration samples: {details}. Keep your face visible and retry.")


def save_calibration(path: str | Path, calibration: GazeCalibration, samples: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "affine_gaze_xy",
        "matrix": calibration.matrix.tolist(),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: str | Path | None) -> GazeCalibration | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GazeCalibration.from_json(payload)
