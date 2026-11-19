from __future__ import annotations

import numpy as np
import pytest

from eyegaze.contracts import load_spec
from eyegaze.domain.features import geometry_batch, geometry_vector
from eyegaze.domain.keypoints import KeypointBatch, keypoints_batch, keypoints_from_observation
from eyegaze.domain.observation import HeadPose
from eyegaze.domain.preprocess import extract_inputs, extract_inputs_from_keypoints
from tests.factories import make_image, make_observation

SPEC = load_spec()


def _batch(observations: list) -> KeypointBatch:  # type: ignore[type-arg]
    return keypoints_batch(
        np.stack([o.landmarks for o in observations]),
        np.stack([o.transform_matrix for o in observations]),
        np.array([o.frame_width for o in observations]),
        np.array([o.frame_height for o in observations]),
        SPEC,
    )


def test_batch_matches_scalar_extraction() -> None:
    observations = [
        make_observation(image_roll_rad=r, pose=HeadPose(0.1 * i, -0.05 * i, r), seed=i)
        for i, r in enumerate((0.0, 0.2, -0.3))
    ]
    batch = _batch(observations)
    assert len(batch) == 3
    for i, obs in enumerate(observations):
        scalar = keypoints_from_observation(obs, SPEC)
        item = batch.item(i)
        np.testing.assert_allclose(item.points(), scalar.points(), atol=1e-9)
        assert item.face_bbox.w == pytest.approx(scalar.face_bbox.w)
        assert item.pose.yaw == pytest.approx(scalar.pose.yaw)
        np.testing.assert_allclose(geometry_batch(batch, SPEC)[i], geometry_vector(scalar, SPEC), atol=1e-6)


def test_subset_keeps_rows_aligned() -> None:
    observations = [make_observation(seed=i, image_roll_rad=0.1 * i) for i in range(4)]
    batch = _batch(observations)
    sub = batch.subset(np.array([3, 1]))
    assert sub.item(0).right_outer == batch.item(3).right_outer
    assert len(sub) == 2


def test_inputs_from_keypoints_equal_inputs_from_observation() -> None:
    obs = make_observation(image_roll_rad=0.15)
    image = make_image()
    a = extract_inputs(image, obs, SPEC)
    b = extract_inputs_from_keypoints(image, keypoints_from_observation(obs, SPEC), SPEC)
    np.testing.assert_array_equal(a.right_eye, b.right_eye)
    np.testing.assert_array_equal(a.geometry, b.geometry)


def test_batch_validation() -> None:
    obs = make_observation()
    with pytest.raises(ValueError, match="478"):
        keypoints_batch(obs.landmarks[None, :10], obs.transform_matrix[None], np.array([1]), np.array([1]), SPEC)
    with pytest.raises(ValueError, match="positive"):
        keypoints_batch(obs.landmarks[None], obs.transform_matrix[None], np.array([0]), np.array([1]), SPEC)
