from __future__ import annotations

import numpy as np
import pytest

from eyegaze.application.calibration_sim import CalibrationMethod, simulate_calibration


def _subjects(n_subjects: int = 4, dots: int = 13, frames_per_dot: int = 5, seed: int = 0) -> dict[str, np.ndarray]:
    """Features linearly related to targets, with a per-subject bias the base model misses."""
    rng = np.random.default_rng(seed)
    feats, targets, base, dot_ids, subjects = [], [], [], [], []
    for s in range(n_subjects):
        bias = rng.normal(scale=2.0, size=2)
        dot_pos = rng.uniform(-5, 5, size=(dots, 2))
        for d in range(dots):
            for _ in range(frames_per_dot):
                t = dot_pos[d]
                f = np.concatenate([t + bias + rng.normal(scale=0.01, size=2), rng.normal(size=3)])
                feats.append(f)
                targets.append(t)
                base.append(t + bias)
                dot_ids.append(d)
                subjects.append(f"s{s}")
    return {
        "features": np.array(feats),
        "targets": np.array(targets),
        "uncalibrated": np.array(base),
        "dot_ids": np.array(dot_ids),
        "subject_ids": np.array(subjects),
    }


def test_calibration_removes_per_subject_bias() -> None:
    result = simulate_calibration(**_subjects(), k=9)
    assert result.subjects_used == 4
    assert result.after_subject_mean < 0.1
    assert result.before_subject_mean > 1.0
    assert result.method == "ridge"


def test_subjects_with_too_few_dots_are_skipped() -> None:
    data = _subjects(dots=6)
    result = simulate_calibration(**data, k=5)
    assert result.subjects_used == 4
    with pytest.raises(ValueError, match="No subject"):
        simulate_calibration(**data, k=6)


def test_svr_method_runs_and_improves() -> None:
    pytest.importorskip("sklearn")
    result = simulate_calibration(**_subjects(n_subjects=2), k=9, method=CalibrationMethod.SVR)
    assert result.method == "svr"
    assert result.after_subject_mean < result.before_subject_mean


def test_input_validation() -> None:
    data = _subjects()
    with pytest.raises(ValueError, match=">= 3"):
        simulate_calibration(**data, k=2)
    data["targets"] = data["targets"][:-1]
    with pytest.raises(ValueError, match="same length"):
        simulate_calibration(**data, k=5)
