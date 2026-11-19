from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eyegaze.contracts import load_spec
from eyegaze.domain.features import GEOMETRY_DIM, geometry_vector
from eyegaze.domain.keypoints import keypoints_from_observation
from eyegaze.domain.observation import (
    FaceObservation,
    HeadPose,
    head_pose_from_rotation,
    rotation_from_head_pose,
    rotation_from_transform,
)
from tests.factories import EYE_POINTS_PX, FRAME_H, FRAME_W, make_observation

SPEC = load_spec()
angles = st.floats(-1.3, 1.3)


@settings(max_examples=60, deadline=None)
@given(pitch=angles, yaw=angles, roll=angles)
def test_head_pose_roundtrip(pitch: float, yaw: float, roll: float) -> None:
    pose = head_pose_from_rotation(rotation_from_head_pose(HeadPose(pitch, yaw, roll)))
    assert pose.pitch == pytest.approx(pitch, abs=1e-9)
    assert pose.yaw == pytest.approx(yaw, abs=1e-9)
    assert pose.roll == pytest.approx(roll, abs=1e-9)


def test_rotation_from_transform_removes_uniform_scale() -> None:
    r = rotation_from_head_pose(HeadPose(0.1, -0.2, 0.3))
    m = np.eye(4)
    m[:3, :3] = r * 3.5
    np.testing.assert_allclose(rotation_from_transform(m), r, atol=1e-12)


def test_rotation_from_transform_rejects_degenerate_matrix() -> None:
    with pytest.raises(ValueError, match="Degenerate"):
        rotation_from_transform(np.zeros((4, 4)))


def test_observation_validates_shapes_and_finiteness() -> None:
    good = make_observation()
    with pytest.raises(ValueError, match="478"):
        FaceObservation(good.landmarks[:10], good.transform_matrix, FRAME_W, FRAME_H)
    bad = good.landmarks.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        FaceObservation(bad, good.transform_matrix, FRAME_W, FRAME_H)


def test_geometry_vector_layout() -> None:
    pose = HeadPose(0.05, -0.1, 0.02)
    vec = geometry_vector(keypoints_from_observation(make_observation(pose=pose), SPEC), SPEC)
    assert vec.shape == (GEOMETRY_DIM,)
    assert vec.dtype == np.float32
    layout = SPEC.geometry.layout
    assert vec[layout.index("right_outer_x")] == pytest.approx(EYE_POINTS_PX[33][0] / FRAME_W, abs=1e-6)
    assert vec[layout.index("left_iris_y")] == pytest.approx(EYE_POINTS_PX[473][1] / FRAME_H, abs=1e-6)
    assert vec[layout.index("pitch_rad")] == pytest.approx(pose.pitch, abs=1e-6)
    assert vec[layout.index("yaw_rad")] == pytest.approx(pose.yaw, abs=1e-6)
    assert vec[layout.index("roll_rad")] == pytest.approx(pose.roll, abs=1e-6)
    assert 0.0 < vec[layout.index("face_bbox_w")] < 1.0
    assert math.isfinite(float(vec.sum()))
