"""Training data: manifest → compact in-memory table → on-the-fly crops (PyTorch Dataset).

Crops are rendered per sample with the OpenCV twin of the reference sampler
(:mod:`eyegaze.adapters.fast_sampler`, < 0.01 grey levels from what the app
computes, ~4.5x faster). Doing it
on the fly (instead of caching crops) lets every ablation use its own spec
variant without re-materializing tens of GB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from eyegaze.adapters.fast_sampler import render_crops_fast
from eyegaze.adapters.images import OpenCvImageReader
from eyegaze.config import AugmentConfig, CropSource, SpecOverrides
from eyegaze.domain.features import geometry_batch, geometry_vector
from eyegaze.domain.keypoints import KeypointBatch, keypoints_batch
from eyegaze.domain.preprocess import CropTransform, CropTransforms, crop_transforms
from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT, PreprocessSpec

MANIFEST_READ_BATCH = 50_000
APPLE_PARTS = ("face", "left_eye", "right_eye")


class DataError(RuntimeError):
    """Training data is missing, unreadable or inconsistent."""


def effective_spec(base: PreprocessSpec, overrides: SpecOverrides, crop_source: CropSource) -> PreprocessSpec:
    eye, face = base.eye, base.face
    if overrides.eye_size is not None:
        eye = replace(eye, size=overrides.eye_size)
    if overrides.face_size is not None:
        face = replace(face, size=overrides.face_size)
    align = overrides.align_roll
    if crop_source is CropSource.APPLE:
        align = False
    if align is not None:
        eye = replace(eye, align_roll=align)
        face = replace(face, align_roll=align)
    return replace(base, eye=eye, face=face)


# --------------------------------------------------------------------------- frame table


@dataclass(frozen=True, slots=True)
class FrameTable:
    subject_id: NDArray[np.str_]
    split: NDArray[np.str_]
    device: NDArray[np.str_]
    frame_relpath: NDArray[np.str_]
    dot_num: NDArray[np.int64]
    gaze_cm: NDArray[np.float32]  # (N, 2)
    keypoints: KeypointBatch
    apple: NDArray[np.float32]  # (N, 12): face, left_eye, right_eye as x, y, w, h (NaN if missing)

    def __len__(self) -> int:
        return int(self.gaze_cm.shape[0])

    def subset(self, index: NDArray[np.int64]) -> FrameTable:
        return FrameTable(
            self.subject_id[index],
            self.split[index],
            self.device[index],
            self.frame_relpath[index],
            self.dot_num[index],
            self.gaze_cm[index],
            self.keypoints.subset(index),
            self.apple[index],
        )


def load_frame_table(manifest: Path, spec: PreprocessSpec) -> FrameTable:
    """Stream the manifest in batches, keeping only the compact keypoints (not 478 landmarks)."""
    if not manifest.is_file():
        raise DataError(f"Manifest not found: {manifest}")
    apple_cols = [f"apple_{p}_{c}" for p in APPLE_PARTS for c in ("x", "y", "w", "h")]
    columns = [
        "subject_id", "split", "device", "frame_relpath", "dot_num", "gaze_cm_x", "gaze_cm_y",
        "frame_width", "frame_height", "landmarks", "transform", *apple_cols,
    ]  # fmt: skip
    keys = ("sid", "split", "dev", "path", "dot", "gaze", "apple")
    parts: dict[str, list[NDArray[np.generic]]] = {k: [] for k in keys}
    kp_parts: list[KeypointBatch] = []
    parquet = pq.ParquetFile(manifest)
    for batch in parquet.iter_batches(batch_size=MANIFEST_READ_BATCH, columns=columns):
        n = batch.num_rows
        lm = batch.column("landmarks").values.to_numpy(zero_copy_only=False).reshape(n, MEDIAPIPE_LANDMARK_COUNT, 3)
        tf = batch.column("transform").values.to_numpy(zero_copy_only=False).reshape(n, 4, 4)
        kp_parts.append(
            keypoints_batch(
                lm, tf, batch.column("frame_width").to_numpy(), batch.column("frame_height").to_numpy(), spec
            )
        )
        parts["sid"].append(np.asarray(batch.column("subject_id").to_pylist(), dtype=np.str_))
        parts["split"].append(np.asarray(batch.column("split").to_pylist(), dtype=np.str_))
        parts["dev"].append(np.asarray(batch.column("device").to_pylist(), dtype=np.str_))
        parts["path"].append(np.asarray(batch.column("frame_relpath").to_pylist(), dtype=np.str_))
        parts["dot"].append(batch.column("dot_num").to_numpy().astype(np.int64))
        parts["gaze"].append(
            np.stack([batch.column("gaze_cm_x").to_numpy(), batch.column("gaze_cm_y").to_numpy()], axis=1)
        )
        apple = np.stack(
            [batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in apple_cols], axis=1
        )
        parts["apple"].append(apple)
    if not kp_parts:
        raise DataError(f"Manifest {manifest} is empty")
    keypoints = KeypointBatch(
        sizes=np.concatenate([k.sizes for k in kp_parts]),
        points=np.concatenate([k.points for k in kp_parts]),
        bbox=np.concatenate([k.bbox for k in kp_parts]),
        pose=np.concatenate([k.pose for k in kp_parts]),
    )
    return FrameTable(
        subject_id=np.concatenate(parts["sid"]),
        split=np.concatenate(parts["split"]),
        device=np.concatenate(parts["dev"]),
        frame_relpath=np.concatenate(parts["path"]),
        dot_num=np.concatenate(parts["dot"]).astype(np.int64),
        gaze_cm=np.concatenate(parts["gaze"]).astype(np.float32),
        keypoints=keypoints,
        apple=np.concatenate(parts["apple"]),
    )


def select_split(
    table: FrameTable, split: str, limit: int | None = None, seed: int = 0, require_apple: bool = False
) -> FrameTable:
    mask = table.split == split
    if require_apple:
        mask &= np.all(np.isfinite(table.apple), axis=1)
    index = np.flatnonzero(mask)
    if limit is not None and len(index) > limit:
        index = np.sort(np.random.default_rng(seed).choice(index, size=limit, replace=False))
    if len(index) == 0:
        raise DataError(f"No frames for split {split!r}" + (" with valid Apple boxes" if require_apple else ""))
    return table.subset(index.astype(np.int64))


def geometry_stats(table: FrameTable, spec: PreprocessSpec) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    g = geometry_batch(table.keypoints, spec).astype(np.float64)
    return g.mean(axis=0).astype(np.float32), g.std(axis=0).astype(np.float32)


# --------------------------------------------------------------------------- per-sample transforms


def apple_transforms(boxes: NDArray[np.float32]) -> CropTransforms:
    """Axis-aligned square crops from Apple boxes (legacy E0 input definition).

    The eye with the smaller x is the subject's right eye in an un-mirrored frame,
    whatever GazeCapture's file naming means by "left".
    """
    face, eye_a, eye_b = boxes[0:4], boxes[4:8], boxes[8:12]
    if eye_a[0] + eye_a[2] / 2 > eye_b[0] + eye_b[2] / 2:
        eye_a, eye_b = eye_b, eye_a

    def square(b: NDArray[np.float32]) -> CropTransform:
        return CropTransform(float(b[0] + b[2] / 2), float(b[1] + b[3] / 2), float(max(b[2], b[3])), 0.0)

    return CropTransforms(right_eye=square(eye_a), left_eye=square(eye_b), face=square(face))


def jitter_transform(t: CropTransform, draws: NDArray[np.float64], aug: AugmentConfig) -> CropTransform:
    """Perturb a crop with ``draws`` ~ U(-1, 1)^4: shift x/y, scale, rotation (simulates landmark noise)."""
    return CropTransform(
        cx=t.cx + draws[0] * aug.shift * t.side,
        cy=t.cy + draws[1] * aug.shift * t.side,
        side=t.side * (1.0 + draws[2] * aug.scale),
        angle=t.angle + math.radians(draws[3] * aug.rotate_deg),
    )


def photometric(crop: NDArray[np.float32], gain: float, contrast: float) -> NDArray[np.float32]:
    mean = float(crop.mean())
    out: NDArray[np.float32] = np.clip(((crop - mean) * contrast + mean) * gain, 0.0, 1.0).astype(np.float32)
    return out


class GazeDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        table: FrameTable,
        frames_root: Path,
        spec: PreprocessSpec,
        crop_source: CropSource = CropSource.MEDIAPIPE,
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.table = table
        self.frames_root = frames_root
        self.spec = spec
        self.crop_source = crop_source
        self.augment = augment if augment is not None and augment.enabled else None
        self.seed = seed
        self.epoch = 0
        self._reader = OpenCvImageReader()

    def set_epoch(self, epoch: int) -> None:
        """Changes augmentation draws; call before creating the epoch's DataLoader iterator."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.table)

    def _transforms(self, i: int) -> CropTransforms:
        if self.crop_source is CropSource.APPLE:
            return apple_transforms(self.table.apple[i])
        return crop_transforms(self.table.keypoints.item(i), self.spec)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        path = self.frames_root / str(self.table.frame_relpath[i])
        image = self._reader.read_rgb(path)
        if image is None:
            raise DataError(f"Cannot read frame {path}")
        kp = self.table.keypoints.item(i)
        if image.shape[1] != kp.frame_width or image.shape[0] != kp.frame_height:
            raise DataError(f"{path}: size {image.shape[1]}x{image.shape[0]} differs from manifest")
        t = self._transforms(i)
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            draws = rng.uniform(-1.0, 1.0, size=(3, 4))
            parts = (t.right_eye, t.left_eye, t.face)
            jr, jl, jf = (jitter_transform(x, d, self.augment) for x, d in zip(parts, draws, strict=True))
            t = CropTransforms(right_eye=jr, left_eye=jl, face=jf)
        right, left, face = render_crops_fast(image, t, self.spec)
        if self.augment is not None:
            gain = 1.0 + rng.uniform(-1.0, 1.0) * self.augment.brightness
            contrast = 1.0 + rng.uniform(-1.0, 1.0) * self.augment.contrast
            right, left, face = (photometric(c, gain, contrast) for c in (right, left, face))

        def chw(x: NDArray[np.float32]) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

        return {
            "right_eye": chw(right),
            "left_eye": chw(left),
            "face": chw(face),
            "geometry": torch.from_numpy(geometry_vector(kp, self.spec)),
            "gaze_cm": torch.from_numpy(self.table.gaze_cm[i].copy()),
            "index": torch.tensor(i, dtype=torch.long),
        }
