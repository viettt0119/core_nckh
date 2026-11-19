from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from eyegaze.application.golden import (
    CASES,
    CROP_ORDER,
    delta_decode_x,
    delta_encode_x,
    generate,
    read_gz,
    synthetic_observation,
)
from eyegaze.contracts import CONTRACTS_DIR, load_spec, spec_hash
from eyegaze.domain.observation import FaceObservation
from eyegaze.domain.preprocess import extract_inputs

GOLDEN = CONTRACTS_DIR / "golden"
SPEC = load_spec()


def _assert_json_close(a: object, b: object, path: str = "$") -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_json_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_json_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1e-9, rel=1e-9), path
    else:
        assert a == b, path


@settings(max_examples=30, deadline=None)
@given(arrays(np.uint8, st.tuples(st.integers(1, 6), st.integers(1, 9), st.just(3))))
def test_delta_encoding_roundtrips(image: np.ndarray) -> None:
    np.testing.assert_array_equal(delta_decode_x(delta_encode_x(image)), image)


def test_synthetic_observation_places_spec_points_exactly() -> None:
    case = CASES[0]
    obs = synthetic_observation(case)
    right_outer = obs.px(SPEC.eye.right.outer)
    left_outer = obs.px(SPEC.eye.left.outer)
    assert right_outer[0] < case.center[0] < left_outer[0]
    assert right_outer[1] == pytest.approx(left_outer[1])  # roll 0 → level eyes


def test_committed_fixtures_match_the_reference_implementation(tmp_path: Path) -> None:
    """Regenerating must reproduce contracts/golden; otherwise run `eyegaze golden` and review the diff."""
    generate(tmp_path)
    index = json.loads((GOLDEN / "index.json").read_text(encoding="utf-8"))
    assert index["spec_sha256"] == spec_hash()
    _assert_json_close(json.loads((tmp_path / "index.json").read_text(encoding="utf-8")), index)
    for name in ("calibration.json", "screen_geometry.json", *(f"{c.name}.json" for c in CASES)):
        fresh = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        committed = json.loads((GOLDEN / name).read_text(encoding="utf-8"))
        _assert_json_close(fresh, committed, name)
    for case in CASES:
        assert read_gz(tmp_path / f"{case.name}.rgb.gz") == read_gz(GOLDEN / f"{case.name}.rgb.gz")
        fresh = np.frombuffer(read_gz(tmp_path / f"{case.name}.crops.f32.gz"), dtype="<f4")
        committed = np.frombuffer(read_gz(GOLDEN / f"{case.name}.crops.f32.gz"), dtype="<f4")
        np.testing.assert_allclose(fresh, committed, atol=1e-6)


@pytest.mark.parametrize("case", [c.name for c in CASES])
def test_consumer_path_reproduces_expected_crops(case: str) -> None:
    """Exactly what the Kotlin test does: decode inputs from the files, preprocess, compare."""
    meta = json.loads((GOLDEN / f"{case}.json").read_text(encoding="utf-8"))
    w, h = meta["frame_width"], meta["frame_height"]
    encoded = np.frombuffer(read_gz(GOLDEN / meta["image"]["file"]), dtype=np.uint8).reshape(h, w, 3)
    image = delta_decode_x(encoded)
    obs = FaceObservation(np.array(meta["landmarks"]), np.array(meta["transform_matrix"]), w, h)
    inputs = extract_inputs(image, obs, SPEC)

    crops_meta = meta["expected"]["crops"]
    assert crops_meta["order"] == list(CROP_ORDER)
    flat = np.frombuffer(read_gz(GOLDEN / crops_meta["file"]), dtype="<f4")
    offset = 0
    for name, shape in zip(crops_meta["order"], crops_meta["shapes"], strict=True):
        size = int(np.prod(shape))
        expected = flat[offset : offset + size].reshape(shape)
        offset += size
        np.testing.assert_allclose(getattr(inputs, name), expected, atol=1e-6)
    assert offset == flat.size
    np.testing.assert_allclose(inputs.geometry, meta["expected"]["geometry"], atol=1e-6)
