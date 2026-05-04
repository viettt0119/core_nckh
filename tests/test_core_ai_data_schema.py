from __future__ import annotations

import numpy as np

from core_ai.data.schema import GazeSample, GazeTarget, validate_sample


def test_validate_sample_accepts_minimal() -> None:
    s = GazeSample(image=np.zeros((96, 96, 3), dtype=np.float32))
    validate_sample(s)


def test_validate_sample_rejects_wrong_dtype() -> None:
    s = GazeSample(image=np.zeros((96, 96, 3), dtype=np.uint8))
    try:
        validate_sample(s)
    except ValueError as e:
        assert "float32" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_validate_sample_checks_gaze_finite() -> None:
    s = GazeSample(image=np.zeros((32, 32, 3), dtype=np.float32), gaze=GazeTarget(pitch=float("nan"), yaw=0.0))
    try:
        validate_sample(s)
    except ValueError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("expected ValueError")

