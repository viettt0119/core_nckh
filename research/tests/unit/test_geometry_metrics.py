from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from eyegaze.domain.geometry import Box, ScreenGeometry
from eyegaze.domain.metrics import euclidean_errors, gaze_report, summarize

PHONE = ScreenGeometry.from_dpi(width_px=1080, height_px=2340, xdpi=400.0, ydpi=400.0)


def test_box_validation() -> None:
    assert Box(1, 2, 4, 6).cx == pytest.approx(3.0)
    with pytest.raises(ValueError, match="positive"):
        Box(0, 0, 0, 1)
    with pytest.raises(ValueError, match="finite"):
        Box(float("nan"), 0, 1, 1)


def test_screen_points_below_camera_have_negative_y() -> None:
    centre = PHONE.screen_norm_to_camera_cm(np.array([0.5, 0.5]))
    assert centre[0] == pytest.approx(0.0, abs=1e-9)  # default camera is top-centre
    assert centre[1] < 0.0


@given(sx=st.floats(-0.5, 1.5), sy=st.floats(-0.5, 1.5))
def test_camera_screen_roundtrip(sx: float, sy: float) -> None:
    s = np.array([sx, sy])
    np.testing.assert_allclose(PHONE.camera_cm_to_screen_norm(PHONE.screen_norm_to_camera_cm(s)), s, atol=1e-9)


def test_screen_delta_to_cm_uses_physical_size() -> None:
    d = PHONE.screen_norm_delta_to_cm(np.array([1.0, 1.0]))
    assert d[0] == pytest.approx(1080 / 400 * 2.54)
    assert d[1] == pytest.approx(2340 / 400 * 2.54)


def test_euclidean_errors_and_summary() -> None:
    e = euclidean_errors(np.array([[0.0, 0.0], [3.0, 4.0]]), np.zeros((2, 2)))
    np.testing.assert_allclose(e, [0.0, 5.0])
    s = summarize(e)
    assert s.mean == pytest.approx(2.5)
    assert s.n == 2


def test_subject_mean_weights_each_person_equally() -> None:
    errors = np.array([1.0] * 9 + [5.0])
    subjects = ["a"] * 9 + ["b"]
    report = gaze_report(errors, subjects)
    assert report.frame.mean == pytest.approx(1.4)
    assert report.subject_mean == pytest.approx(3.0)
    assert set(report.as_dict()["per_subject"]) == {"a", "b"}  # type: ignore[arg-type]


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="matching"):
        euclidean_errors(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        summarize(np.array([]))
    with pytest.raises(ValueError, match="mismatch"):
        gaze_report(np.zeros(2), ["a"])
