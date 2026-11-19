"""Low-dimensional geometric features fed to the network next to the image crops."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.keypoints import FaceKeypoints, KeypointBatch
from eyegaze.domain.spec import PreprocessSpec

Float32Array = NDArray[np.float32]

GEOMETRY_DIM = 17
POSE_SLICE = slice(12, 15)  # pitch, yaw, roll inside the geometry vector


def _check(spec: PreprocessSpec) -> None:
    if spec.geometry.dim != GEOMETRY_DIM:
        raise ValueError(f"This implementation produces {GEOMETRY_DIM} features, spec asks {spec.geometry.dim}")


def geometry_vector(kp: FaceKeypoints, spec: PreprocessSpec) -> Float32Array:
    """Build the vector described by ``spec.geometry.layout``.

    Point coordinates are normalized by frame size so the vector does not depend
    on camera resolution; head pose comes from the landmarker's rigid transform,
    which is far more stable than a pose fitted from 2D points.
    """
    _check(spec)
    w, h = float(kp.frame_width), float(kp.frame_height)
    coords = [v for x, y in kp.points() for v in (x / w, y / h)]
    vec = np.array(
        [*coords, kp.pose.pitch, kp.pose.yaw, kp.pose.roll, kp.face_bbox.w / w, kp.face_bbox.h / h],
        dtype=np.float64,
    )
    out: Float32Array = vec.astype(np.float32)
    return out


def geometry_batch(batch: KeypointBatch, spec: PreprocessSpec) -> Float32Array:
    """Vectorized :func:`geometry_vector` → ``(N, 17)``."""
    _check(spec)
    size = batch.sizes.astype(np.float64)
    coords = (batch.points / size[:, None, :]).reshape(len(batch), 12)
    bbox = batch.bbox[:, 2:4] / size
    out: Float32Array = np.concatenate([coords, batch.pose, bbox], axis=1).astype(np.float32)
    return out
