from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from eyegaze.adapters.fast_sampler import render_crops_fast, warp_crop_fast
from eyegaze.contracts import load_spec
from eyegaze.domain.keypoints import keypoints_from_observation
from eyegaze.domain.preprocess import CropTransform, crop_transforms, render_crops, warp_crop
from tests.factories import make_image, make_observation

SPEC = load_spec()
IMAGE = make_image(seed=5)
TOLERANCE = 0.05 / 255.0  # far below one grey level


@settings(max_examples=40, deadline=None)
@given(
    cx=st.floats(-50.0, 530.0),
    cy=st.floats(-50.0, 690.0),
    side=st.floats(8.0, 300.0),
    angle=st.floats(-math.pi, math.pi),
    flip=st.booleans(),
)
def test_fast_sampler_matches_reference(cx: float, cy: float, side: float, angle: float, flip: bool) -> None:
    t = CropTransform(cx, cy, side, angle)
    ref = warp_crop(IMAGE, t, 32, SPEC.sampling, flip_horizontal=flip)
    fast = warp_crop_fast(IMAGE.astype(np.float32), t, 32, SPEC.sampling, flip_horizontal=flip)
    assert fast.dtype == np.float32
    np.testing.assert_allclose(fast, ref, atol=TOLERANCE)


def test_render_crops_fast_matches_reference() -> None:
    transforms = crop_transforms(keypoints_from_observation(make_observation(image_roll_rad=0.2), SPEC), SPEC)
    reference = render_crops(IMAGE, transforms, SPEC)
    for ref, fast in zip(reference, render_crops_fast(IMAGE, transforms, SPEC), strict=True):
        assert ref.shape == fast.shape
        np.testing.assert_allclose(fast, ref, atol=TOLERANCE)
