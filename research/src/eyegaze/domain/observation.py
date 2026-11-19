"""A detected face: landmarks, head pose and frame size."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One face as produced by the landmarker, in the units the spec prescribes."""

    landmarks: FloatArray  # (478, 3), x/y normalized to [0, 1]
    transform_matrix: FloatArray  # (4, 4) row-major
    frame_width: int
    frame_height: int

    def __post_init__(self) -> None:
        if self.landmarks.shape != (MEDIAPIPE_LANDMARK_COUNT, 3):
            raise ValueError(f"Expected landmarks (478, 3), got {self.landmarks.shape}")
        if self.transform_matrix.shape != (4, 4):
            raise ValueError(f"Expected transform matrix (4, 4), got {self.transform_matrix.shape}")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Frame size must be positive")
        if not (np.all(np.isfinite(self.landmarks)) and np.all(np.isfinite(self.transform_matrix))):
            raise ValueError("Landmarks and transform matrix must be finite")

    def px(self, index: int) -> tuple[float, float]:
        lm = self.landmarks[index]
        return float(lm[0]) * self.frame_width, float(lm[1]) * self.frame_height

    def head_pose(self) -> HeadPose:
        return head_pose_from_rotation(rotation_from_transform(self.transform_matrix))


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Euler angles (radians) with ``R = Rz(roll) @ Ry(yaw) @ Rx(pitch)``."""

    pitch: float
    yaw: float
    roll: float


def rotation_from_transform(matrix: FloatArray) -> FloatArray:
    """Extract a pure rotation from a (possibly scaled) 4x4 rigid transform."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) matrix, got {m.shape}")
    linear = m[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales < 1e-9):
        raise ValueError("Degenerate transform: zero-length basis vector")
    rotation: FloatArray = linear / scales
    return rotation


def head_pose_from_rotation(rotation: FloatArray) -> HeadPose:
    r = np.asarray(rotation, dtype=np.float64)
    pitch = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return HeadPose(pitch=pitch, yaw=yaw, roll=roll)


def rotation_from_head_pose(pose: HeadPose) -> FloatArray:
    """Inverse of :func:`head_pose_from_rotation` (used by tests and synthetic data)."""
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    result: FloatArray = rz @ ry @ rx
    return result
