"""Low-dimensional geometric features fed to the network next to the image crops."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.observation import FaceObservation
from eyegaze.domain.spec import PreprocessSpec

Float32Array = NDArray[np.float32]

GEOMETRY_DIM = 17


def geometry_vector(obs: FaceObservation, spec: PreprocessSpec) -> Float32Array:
    """Build the vector described by ``spec.geometry.layout``.

    Landmark coordinates stay normalized by frame size so the vector does not
    depend on camera resolution; head pose comes from the landmarker's rigid
    transform, which is far more stable than a pose fitted from 2D points.
    """
    if spec.geometry.dim != GEOMETRY_DIM:
        raise ValueError(f"This implementation produces {GEOMETRY_DIM} features, spec asks {spec.geometry.dim}")
    lm = obs.landmarks
    eye = spec.eye
    order = (
        eye.right.outer,
        eye.right.inner,
        eye.left.inner,
        eye.left.outer,
        eye.right.iris_center,
        eye.left.iris_center,
    )
    points = lm[list(order), :2].reshape(-1)

    pose = obs.head_pose()

    face_pts = lm[: spec.face.bbox_landmark_count, :2]
    bbox_w = float(face_pts[:, 0].max() - face_pts[:, 0].min())
    bbox_h = float(face_pts[:, 1].max() - face_pts[:, 1].min())

    vec = np.concatenate([points, [pose.pitch, pose.yaw, pose.roll, bbox_w, bbox_h]])
    out: Float32Array = vec.astype(np.float32)
    return out
