from __future__ import annotations

import numpy as np
import pytest

from eyegaze.domain.calibration import fit_ridge


def _calibration_set(
    n_targets: int = 9, per_target: int = 20, dim: int = 6, noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=(dim, 2))
    true_b = np.array([0.5, 0.5])
    centres = rng.normal(size=(n_targets, dim))
    x = np.repeat(centres, per_target, axis=0) + rng.normal(scale=0.05, size=(n_targets * per_target, dim))
    y = x @ true_w + true_b + rng.normal(scale=noise, size=(len(x), 2))
    groups = np.repeat(np.arange(n_targets), per_target)
    return x, y, groups, true_w


def test_recovers_noise_free_linear_map() -> None:
    x, y, groups, _ = _calibration_set()
    model = fit_ridge(x, y, groups, lambdas=(1e-6, 1.0, 100.0))
    assert model.lam == pytest.approx(1e-6)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-3)
    assert model.cv_error < 1e-2


def test_noisy_data_selects_regularization_and_generalizes() -> None:
    x, y, groups, _ = _calibration_set(noise=0.05, seed=1)
    model = fit_ridge(x, y, groups)
    x_test, y_test, _, _ = _calibration_set(noise=0.0, seed=1)
    err = np.linalg.norm(model.predict(x_test) - y_test, axis=1).mean()
    assert err < 0.1
    assert model.cv_error > 0.0


def test_predict_single_vector_and_batch_agree() -> None:
    x, y, groups, _ = _calibration_set()
    model = fit_ridge(x, y, groups)
    np.testing.assert_allclose(model.predict(x[3]), model.predict(x)[3])


def test_constant_feature_column_does_not_break_fit() -> None:
    x, y, groups, _ = _calibration_set()
    x = np.concatenate([x, np.full((len(x), 1), 7.0)], axis=1)
    model = fit_ridge(x, y, groups)
    assert np.all(np.isfinite(model.weights))


def test_requires_at_least_three_targets() -> None:
    x, y, groups, _ = _calibration_set(n_targets=2)
    with pytest.raises(ValueError, match="at least 3"):
        fit_ridge(x, y, groups)


def test_rejects_length_mismatch_and_non_finite() -> None:
    x, y, groups, _ = _calibration_set()
    with pytest.raises(ValueError, match="mismatch"):
        fit_ridge(x[:-1], y, groups)
    y_bad = y.copy()
    y_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_ridge(x, y_bad, groups)


def test_predict_rejects_wrong_dimension() -> None:
    x, y, groups, _ = _calibration_set()
    model = fit_ridge(x, y, groups)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, x.shape[1] + 1)))
