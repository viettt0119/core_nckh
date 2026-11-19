from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eyegaze.contracts import load_spec
from eyegaze.domain.preprocess import (
    CropTransform,
    extract_inputs,
    eye_transforms,
    face_transform,
    warp_crop,
)
from eyegaze.domain.spec import SamplingSpec
from tests.factories import EYE_POINTS_PX, make_image, make_observation

SPEC = load_spec()
SAMPLING_1X = SamplingSpec(method="bilinear", supersample=1, border="replicate", pixel_divisor=255.0)


def test_axis_aligned_unit_scale_crop_reproduces_pixels_exactly() -> None:
    image = make_image(64, 48)
    x0, y0, size = 10, 7, 16
    t = CropTransform(cx=x0 + size / 2, cy=y0 + size / 2, side=size, angle=0.0)
    crop = warp_crop(image, t, size, SAMPLING_1X)
    expected = image[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255.0
    np.testing.assert_allclose(crop, expected, atol=1e-6)


def test_supersampling_of_2x_downscale_equals_box_filter() -> None:
    image = make_image(64, 48)
    x0, y0, out = 8, 4, 8
    sampling = SamplingSpec(method="bilinear", supersample=2, border="replicate", pixel_divisor=255.0)
    t = CropTransform(cx=x0 + out, cy=y0 + out, side=2 * out, angle=0.0)
    crop = warp_crop(image, t, out, sampling)
    block = image[y0 : y0 + 2 * out, x0 : x0 + 2 * out].astype(np.float64)
    expected = block.reshape(out, 2, out, 2, 3).mean(axis=(1, 3)) / 255.0
    np.testing.assert_allclose(crop, expected, atol=1e-6)


def test_quarter_turn_matches_rot90() -> None:
    image = make_image(64, 64)
    base = CropTransform(cx=32.0, cy=32.0, side=20.0, angle=0.0)
    turned = CropTransform(cx=32.0, cy=32.0, side=20.0, angle=math.pi / 2)
    a = warp_crop(image, base, 20, SAMPLING_1X)
    b = warp_crop(image, turned, 20, SAMPLING_1X)
    # With e_x = (0, 1), e_y = (-1, 0): B[v, u] = A[u, n-1-v], i.e. numpy rot90 (k=1).
    np.testing.assert_allclose(b, np.rot90(a, k=1, axes=(0, 1)), atol=1e-5)


def test_flip_mirrors_columns() -> None:
    image = make_image()
    t = CropTransform(cx=200.0, cy=300.0, side=40.0, angle=0.3)
    plain = warp_crop(image, t, 32, SPEC.sampling)
    flipped = warp_crop(image, t, 32, SPEC.sampling, flip_horizontal=True)
    np.testing.assert_array_equal(flipped, plain[:, ::-1, :])


def test_border_is_replicated() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, 0] = 200
    t = CropTransform(cx=-50.0, cy=10.0, side=4.0, angle=0.0)
    crop = warp_crop(image, t, 4, SAMPLING_1X)
    np.testing.assert_allclose(crop, 200.0 / 255.0, atol=1e-6)


@settings(max_examples=40, deadline=None)
@given(
    angle=st.floats(-math.pi, math.pi),
    cx=st.floats(30.0, 70.0),
    cy=st.floats(30.0, 70.0),
    side=st.floats(4.0, 40.0),
)
def test_bilinear_is_exact_for_linear_images(angle: float, cx: float, cy: float, side: float) -> None:
    """A ramp ``I(x, y) = x + y`` (pixel centres at k + 0.5) is reproduced exactly away from borders."""
    size = 100
    ramp = np.arange(size)[None, :] + np.arange(size)[:, None]  # max 198, fits uint8 exactly
    image = np.repeat(ramp.astype(np.uint8)[..., None], 3, axis=2)
    t = CropTransform(cx=cx, cy=cy, side=side, angle=angle)
    out = 8
    crop = warp_crop(image, t, out, SAMPLING_1X)[..., 0].astype(np.float64) * 255.0
    grid = (np.arange(out) + 0.5) / out - 0.5
    b, a = np.meshgrid(grid, grid, indexing="ij")
    xs = cx + side * (a * math.cos(angle) - b * math.sin(angle))
    ys = cy + side * (a * math.sin(angle) + b * math.cos(angle))
    expected = (xs - 0.5) + (ys - 0.5)
    np.testing.assert_allclose(crop, expected, atol=1e-3)


def test_warp_rejects_non_uint8_image() -> None:
    with pytest.raises(ValueError, match="uint8"):
        warp_crop(np.zeros((10, 10, 3), dtype=np.float32), CropTransform(5, 5, 4, 0), 4, SAMPLING_1X)


def test_crop_transform_rejects_non_positive_side() -> None:
    with pytest.raises(ValueError, match="side"):
        CropTransform(cx=1.0, cy=1.0, side=0.0, angle=0.0)


@pytest.mark.parametrize("roll", [0.0, 0.2, -0.35])
def test_eye_transforms_follow_corner_axis(roll: float) -> None:
    obs = make_observation(image_roll_rad=roll)
    right, left = eye_transforms(obs, SPEC.eye)
    corner_dist = EYE_POINTS_PX[133][0] - EYE_POINTS_PX[33][0]
    for t in (right, left):
        assert t.angle == pytest.approx(roll, abs=1e-9)
        assert t.side == pytest.approx(corner_dist * SPEC.eye.width_scale, rel=1e-9)
    assert right.cx < left.cx


def test_eye_transform_without_roll_alignment_is_axis_aligned() -> None:
    from dataclasses import replace

    spec = replace(SPEC.eye, align_roll=False)
    right, left = eye_transforms(make_observation(image_roll_rad=0.3), spec)
    assert right.angle == 0.0
    assert left.angle == 0.0


def test_face_transform_contains_face_bbox() -> None:
    obs = make_observation()
    t = face_transform(obs, SPEC.face, SPEC.eye)
    assert t.side >= (440.0 - 220.0) * SPEC.face.scale * 0.99


def test_extract_inputs_shapes_and_range() -> None:
    obs = make_observation(image_roll_rad=0.1)
    inputs = extract_inputs(make_image(), obs, SPEC)
    e, f = SPEC.eye.size, SPEC.face.size
    assert inputs.right_eye.shape == (e, e, 3)
    assert inputs.left_eye.shape == (e, e, 3)
    assert inputs.face.shape == (f, f, 3)
    assert inputs.geometry.shape == (SPEC.geometry.dim,)
    for arr in (inputs.right_eye, inputs.left_eye, inputs.face, inputs.geometry):
        assert arr.dtype == np.float32
    for crop in (inputs.right_eye, inputs.left_eye, inputs.face):
        assert float(crop.min()) >= 0.0
        assert float(crop.max()) <= 1.0


def test_extract_inputs_rejects_frame_size_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        extract_inputs(make_image(100, 100), make_observation(), SPEC)


def test_left_eye_is_mirror_of_right_eye_for_symmetric_face() -> None:
    """A left-right symmetric image must give identical eye crops thanks to the flip."""
    half = make_image(240, 640, seed=3)
    image = np.concatenate([half, half[:, ::-1]], axis=1)
    inputs = extract_inputs(image, make_observation(), SPEC)
    np.testing.assert_allclose(inputs.left_eye, inputs.right_eye, atol=1e-6)
