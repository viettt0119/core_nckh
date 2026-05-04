from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from eyegaze.labels import DIRECTION_TO_INDEX, direction_from_xy
from eyegaze.preprocess import crop_resize, face_grid_from_box, geometry_vector


def random_rotate(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def random_blur(img: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


MANIFEST_COLUMNS = [
    "subject_id",
    "split",
    "frame_index",
    "frame_path",
    "screen_w",
    "screen_h",
    "gaze_x",
    "gaze_y",
    "direction",
    "face_x",
    "face_y",
    "face_w",
    "face_h",
    "left_eye_x",
    "left_eye_y",
    "left_eye_w",
    "left_eye_h",
    "right_eye_x",
    "right_eye_y",
    "right_eye_w",
    "right_eye_h",
]


def extract_archive(zip_path: str | Path, output_dir: str | Path, skip_existing: bool = True) -> Path:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(output_dir.iterdir()):
        return output_dir
    if not zip_path.exists():
        raise FileNotFoundError(f"GazeCapture archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    return output_dir


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _value(meta: dict[str, list[Any]], key: str, idx: int, default: Any = None) -> Any:
    values = meta.get(key)
    if values is None or idx >= len(values):
        return default
    return values[idx]


def _valid_box(meta: dict[str, list[Any]], idx: int) -> bool:
    return (
        int(_value(meta, "IsValid", idx, 0) or 0) == 1
        and float(_value(meta, "W", idx, 0) or 0) > 0
        and float(_value(meta, "H", idx, 0) or 0) > 0
    )


def _subject_dirs(data_root: Path) -> list[Path]:
    candidates = [data_root, *[p for p in data_root.iterdir() if p.is_dir()]]
    subjects: list[Path] = []
    seen: set[Path] = set()
    required = ["frames.json", "dotInfo.json", "screen.json", "appleFace.json", "appleLeftEye.json", "appleRightEye.json"]
    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            nested = folder / folder.name
            subject = nested if nested.is_dir() else folder
            if subject in seen:
                continue
            if all((subject / name).exists() for name in required):
                subjects.append(subject)
                seen.add(subject)
    return subjects


def _absolute_eye_box(face_box: tuple[float, float, float, float], eye_meta: dict[str, list[Any]], idx: int) -> tuple[float, float, float, float]:
    # In GazeCapture, appleLeftEye/appleRightEye X,Y are relative to the face crop.
    face_x, face_y, _face_w, _face_h = face_box
    return (
        face_x + float(_value(eye_meta, "X", idx, 0) or 0),
        face_y + float(_value(eye_meta, "Y", idx, 0) or 0),
        float(_value(eye_meta, "W", idx, 0) or 0),
        float(_value(eye_meta, "H", idx, 0) or 0),
    )


def build_manifest(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    if not data_root.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    rows: list[dict[str, Any]] = []
    for subject in _subject_dirs(data_root):
        info = _load_json(subject / "info.json") if (subject / "info.json").exists() else {}
        frames = _load_json(subject / "frames.json")
        dot = _load_json(subject / "dotInfo.json")
        screen = _load_json(subject / "screen.json")
        face = _load_json(subject / "appleFace.json")
        left_eye = _load_json(subject / "appleLeftEye.json")
        right_eye = _load_json(subject / "appleRightEye.json")

        for idx, frame_name in enumerate(frames):
            frame_path = subject / "frames" / frame_name
            if not frame_path.exists():
                continue
            if not (_valid_box(face, idx) and _valid_box(left_eye, idx) and _valid_box(right_eye, idx)):
                continue

            screen_w = float(_value(screen, "W", idx, 0) or 0)
            screen_h = float(_value(screen, "H", idx, 0) or 0)
            dot_x = float(_value(dot, "XPts", idx, -1) or -1)
            dot_y = float(_value(dot, "YPts", idx, -1) or -1)
            if screen_w <= 0 or screen_h <= 0 or not (0 <= dot_x <= screen_w and 0 <= dot_y <= screen_h):
                continue

            face_box = (
                float(_value(face, "X", idx, 0) or 0),
                float(_value(face, "Y", idx, 0) or 0),
                float(_value(face, "W", idx, 0) or 0),
                float(_value(face, "H", idx, 0) or 0),
            )
            left_box = _absolute_eye_box(face_box, left_eye, idx)
            right_box = _absolute_eye_box(face_box, right_eye, idx)
            gaze_x = dot_x / screen_w
            gaze_y = dot_y / screen_h
            rows.append(
                {
                    "subject_id": subject.name,
                    "split": str(info.get("Dataset", "unknown")).lower(),
                    "frame_index": idx,
                    "frame_path": str(frame_path),
                    "screen_w": screen_w,
                    "screen_h": screen_h,
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "direction": direction_from_xy(gaze_x, gaze_y),
                    "face_x": face_box[0],
                    "face_y": face_box[1],
                    "face_w": face_box[2],
                    "face_h": face_box[3],
                    "left_eye_x": left_box[0],
                    "left_eye_y": left_box[1],
                    "left_eye_w": left_box[2],
                    "left_eye_h": left_box[3],
                    "right_eye_x": right_box[0],
                    "right_eye_y": right_box[1],
                    "right_eye_w": right_box[2],
                    "right_eye_h": right_box[3],
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def load_or_build_manifest(data_root: str | Path, manifest_path: str | Path, rebuild: bool = False) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not rebuild:
        return pd.read_csv(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_manifest(data_root)
    df.to_csv(manifest_path, index=False)
    return df


def split_manifest(df: pd.DataFrame, split: str, limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    part = df[df["split"] == split].copy()
    if split == "val" and part.empty:
        train = df[df["split"] == "train"].copy()
        if not train.empty:
            part = train.sample(frac=0.1, random_state=seed)
    if limit is not None and limit >= 0 and len(part) > limit:
        part = part.sample(n=limit, random_state=seed)
    return part.reset_index(drop=True)


def _box_from_row(row: pd.Series, prefix: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}_x"]),
        float(row[f"{prefix}_y"]),
        float(row[f"{prefix}_w"]),
        float(row[f"{prefix}_h"]),
    )


class GazeCaptureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 64, face_size: int = 96, train: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.face_size = face_size
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["frame_path"]))
        if image is None:
            image = np.zeros((max(self.face_size, self.image_size), max(self.face_size, self.image_size), 3), dtype=np.uint8)

        face_box = _box_from_row(row, "face")
        left_box = _box_from_row(row, "left_eye")
        right_box = _box_from_row(row, "right_eye")
        face = crop_resize(image, face_box, self.face_size)
        left = crop_resize(image, left_box, self.image_size, flip_horizontal=True)
        right = crop_resize(image, right_box, self.image_size)
        if self.train:
            face = random_rotate(face, max_angle=10.0)
            left = random_rotate(left, max_angle=10.0)
            right = random_rotate(right, max_angle=10.0)

            face = random_blur(face)
            left = random_blur(left)
            right = random_blur(right)

            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-0.04, 0.04)
            face = np.clip(face * gain + bias, 0.0, 1.0)
            left = np.clip(left * gain + bias, 0.0, 1.0)
            right = np.clip(right * gain + bias, 0.0, 1.0)

        grid = face_grid_from_box(image.shape, face_box)
        geom = geometry_vector(image.shape, face_box, left_box, right_box)
        gaze_xy = np.array([float(row["gaze_x"]), float(row["gaze_y"])], dtype=np.float32)
        direction = DIRECTION_TO_INDEX[str(row["direction"])]
        return {
            "face": torch.from_numpy(face).permute(2, 0, 1).float(),
            "left_eye": torch.from_numpy(left).permute(2, 0, 1).float(),
            "right_eye": torch.from_numpy(right).permute(2, 0, 1).float(),
            "face_grid": torch.from_numpy(grid).float(),
            "geometry": torch.from_numpy(geom).float(),
            "gaze_xy": torch.from_numpy(gaze_xy).float(),
            "direction": torch.tensor(direction, dtype=torch.long),
        }


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    if df.empty:
        raise ValueError("Cannot build a weighted sampler from an empty manifest.")
    counts = df["direction"].value_counts().to_dict()
    weights = [1.0 / counts[str(direction)] for direction in df["direction"]]
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
