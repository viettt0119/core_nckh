"""Deterministic synthetic inputs shared by tests (no MediaPipe, no dataset needed)."""

from __future__ import annotations

import math

import numpy as np

from eyegaze.domain.observation import FaceObservation, HeadPose, rotation_from_head_pose
from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT

FRAME_W = 480
FRAME_H = 640

# Pixel positions in an upright, un-mirrored frame (subject's right eye on image left).
EYE_POINTS_PX: dict[int, tuple[float, float]] = {
    33: (180.0, 300.0),  # right eye, outer corner
    133: (220.0, 300.0),  # right eye, inner corner
    362: (260.0, 300.0),  # left eye, inner corner
    263: (300.0, 300.0),  # left eye, outer corner
    468: (200.0, 300.0),  # right iris centre
    473: (280.0, 300.0),  # left iris centre
}
FACE_CENTER_PX = (240.0, 330.0)


def make_observation(
    image_roll_rad: float = 0.0,
    pose: HeadPose | None = None,
    seed: int = 0,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> FaceObservation:
    """A face whose landmarks are rotated in-plane by ``image_roll_rad`` around the face centre."""
    rng = np.random.default_rng(seed)
    pts = np.empty((MEDIAPIPE_LANDMARK_COUNT, 2), dtype=np.float64)
    pts[:, 0] = rng.uniform(160.0, 320.0, MEDIAPIPE_LANDMARK_COUNT)
    pts[:, 1] = rng.uniform(220.0, 440.0, MEDIAPIPE_LANDMARK_COUNT)
    for idx, xy in EYE_POINTS_PX.items():
        pts[idx] = xy

    c, s = math.cos(image_roll_rad), math.sin(image_roll_rad)
    centred = pts - np.array(FACE_CENTER_PX)
    rotated = centred @ np.array([[c, s], [-s, c]]) + np.array(FACE_CENTER_PX)

    landmarks = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3), dtype=np.float64)
    landmarks[:, 0] = rotated[:, 0] / frame_w
    landmarks[:, 1] = rotated[:, 1] / frame_h

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_from_head_pose(pose or HeadPose(0.0, 0.0, 0.0))
    matrix[:3, 3] = (0.0, 0.0, -40.0)
    return FaceObservation(landmarks=landmarks, transform_matrix=matrix, frame_width=frame_w, frame_height=frame_h)


def make_image(frame_w: int = FRAME_W, frame_h: int = FRAME_H, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(frame_h, frame_w, 3), dtype=np.uint8)
