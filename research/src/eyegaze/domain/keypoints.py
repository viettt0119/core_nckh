"""The handful of face measurements the preprocessing actually needs.

A full MediaPipe result is 478x3 floats; crops and geometry features only need
six points, the face bounding box and the head pose. Reducing to
:class:`FaceKeypoints` early keeps a million-frame training set in ~100 MB of
RAM instead of ~6 GB, and gives the Android port a much smaller surface.

:func:`keypoints_batch` is the vectorized twin of
:func:`keypoints_from_observation`; tests assert they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.geometry import Box
from eyegaze.domain.observation import FaceObservation, HeadPose
from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT, PreprocessSpec

FloatArray = NDArray[np.float64]
Point = tuple[float, float]

# Row order of KeypointBatch.points.
POINT_NAMES = ("right_outer", "right_inner", "left_inner", "left_outer", "right_iris", "left_iris")


def point_indices(spec: PreprocessSpec) -> tuple[int, ...]:
    e = spec.eye
    return (e.right.outer, e.right.inner, e.left.inner, e.left.outer, e.right.iris_center, e.left.iris_center)


@dataclass(frozen=True, slots=True)
class FaceKeypoints:
    """Pixel-space keypoints of one face (upright, un-mirrored frame)."""

    frame_width: int
    frame_height: int
    right_outer: Point
    right_inner: Point
    left_inner: Point
    left_outer: Point
    right_iris: Point
    left_iris: Point
    face_bbox: Box
    pose: HeadPose

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Frame size must be positive")

    def points(self) -> tuple[Point, ...]:
        return (self.right_outer, self.right_inner, self.left_inner, self.left_outer, self.right_iris, self.left_iris)


def keypoints_from_observation(obs: FaceObservation, spec: PreprocessSpec) -> FaceKeypoints:
    pts = [obs.px(i) for i in point_indices(spec)]
    lm = obs.landmarks[: spec.face.bbox_landmark_count, :2]
    xs = lm[:, 0] * obs.frame_width
    ys = lm[:, 1] * obs.frame_height
    x0, y0 = float(xs.min()), float(ys.min())
    bbox = Box(x0, y0, max(float(xs.max()) - x0, 1e-6), max(float(ys.max()) - y0, 1e-6))
    return FaceKeypoints(
        obs.frame_width, obs.frame_height, pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], bbox, obs.head_pose()
    )


@dataclass(frozen=True, slots=True)
class KeypointBatch:
    """Column-oriented keypoints for N frames (float64 pixels / radians)."""

    sizes: NDArray[np.int64]  # (N, 2) width, height
    points: FloatArray  # (N, 6, 2) in POINT_NAMES order
    bbox: FloatArray  # (N, 4) x, y, w, h
    pose: FloatArray  # (N, 3) pitch, yaw, roll

    def __post_init__(self) -> None:
        n = self.sizes.shape[0]
        if self.points.shape != (n, 6, 2) or self.bbox.shape != (n, 4) or self.pose.shape != (n, 3):
            raise ValueError("Inconsistent KeypointBatch shapes")

    def __len__(self) -> int:
        return int(self.sizes.shape[0])

    def item(self, i: int) -> FaceKeypoints:
        p = self.points[i]
        b = self.bbox[i]
        pts = [(float(p[k, 0]), float(p[k, 1])) for k in range(6)]
        return FaceKeypoints(
            frame_width=int(self.sizes[i, 0]),
            frame_height=int(self.sizes[i, 1]),
            right_outer=pts[0],
            right_inner=pts[1],
            left_inner=pts[2],
            left_outer=pts[3],
            right_iris=pts[4],
            left_iris=pts[5],
            face_bbox=Box(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            pose=HeadPose(float(self.pose[i, 0]), float(self.pose[i, 1]), float(self.pose[i, 2])),
        )

    def subset(self, index: NDArray[np.int64]) -> KeypointBatch:
        return KeypointBatch(self.sizes[index], self.points[index], self.bbox[index], self.pose[index])


def head_pose_batch(transforms: FloatArray) -> FloatArray:
    """Vectorized :meth:`FaceObservation.head_pose` for ``(N, 4, 4)`` matrices → ``(N, 3)``."""
    m = np.asarray(transforms, dtype=np.float64)
    linear = m[:, :3, :3]
    scales = np.linalg.norm(linear, axis=1, keepdims=True)
    if np.any(scales < 1e-9):
        raise ValueError("Degenerate transform: zero-length basis vector")
    r = linear / scales
    pitch = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    yaw = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
    roll = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    return np.stack([pitch, yaw, roll], axis=1)


def keypoints_batch(
    landmarks: NDArray[np.floating[Any]],
    transforms: NDArray[np.floating[Any]],
    widths: NDArray[np.integer[Any]],
    heights: NDArray[np.integer[Any]],
    spec: PreprocessSpec,
) -> KeypointBatch:
    lm = np.asarray(landmarks, dtype=np.float64)
    n = lm.shape[0]
    if lm.shape != (n, MEDIAPIPE_LANDMARK_COUNT, 3):
        raise ValueError(f"Expected landmarks (N, 478, 3), got {lm.shape}")
    w = np.asarray(widths, dtype=np.int64)
    h = np.asarray(heights, dtype=np.int64)
    if w.shape != (n,) or h.shape != (n,) or np.any(w <= 0) or np.any(h <= 0):
        raise ValueError("widths/heights must be positive arrays of length N")
    scale = np.stack([w, h], axis=1).astype(np.float64)[:, None, :]
    points = lm[:, list(point_indices(spec)), :2] * scale
    face = lm[:, : spec.face.bbox_landmark_count, :2] * scale
    lo = face.min(axis=1)
    hi = face.max(axis=1)
    bbox = np.concatenate([lo, np.maximum(hi - lo, 1e-6)], axis=1)
    return KeypointBatch(np.stack([w, h], axis=1), points, bbox, head_pose_batch(np.asarray(transforms)))
