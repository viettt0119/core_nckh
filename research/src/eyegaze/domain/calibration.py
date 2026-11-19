"""Per-user calibration: closed-form ridge regression from model features to targets.

Mirrors the Android ``RidgeCalibrator``. Inputs are typically
``[embedding, gaze_cm, head_pose]`` and targets are normalized screen
coordinates (on device) or camera-space cm (offline simulation on GazeCapture).

The regularization strength is chosen by leave-one-*target*-out cross
validation: frames looking at the same calibration dot are strongly
correlated, so leaving out single frames would grossly under-estimate error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_LAMBDAS: tuple[float, ...] = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
MIN_TARGET_GROUPS = 3
_STD_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class RidgeModel:
    """Standardize → linear map. ``weights`` has shape ``(D + 1, T)``, last row is the bias."""

    feature_mean: FloatArray
    feature_std: FloatArray
    weights: FloatArray
    lam: float
    cv_error: float

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    def predict(self, features: FloatArray) -> FloatArray:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {x.shape[1]}")
        design = _design((x - self.feature_mean) / self.feature_std)
        out: FloatArray = design @ self.weights
        return out[0] if single else out


def _design(z: FloatArray) -> FloatArray:
    return np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)


def _solve(design: FloatArray, targets: FloatArray, lam: float) -> FloatArray:
    """Solve ``(XᵀX + λI') w = Xᵀy`` where ``I'`` leaves the bias unpenalized."""
    d = design.shape[1]
    penalty = np.eye(d, dtype=np.float64) * lam
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        solution: FloatArray = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return solution


def _standardize(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), _STD_FLOOR)
    return mean, std


def fit_ridge(
    features: FloatArray,
    targets: FloatArray,
    groups: Sequence[int] | IntArray,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> RidgeModel:
    """Fit a calibration model, selecting λ by leave-one-group-out CV (mean Euclidean error)."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must be 2-D")
    if not (len(x) == len(y) == len(g)):
        raise ValueError(f"Length mismatch: features={len(x)}, targets={len(y)}, groups={len(g)}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("features and targets must be finite")
    if not lambdas or any(lam <= 0 for lam in lambdas):
        raise ValueError("lambdas must be a non-empty sequence of positive values")
    unique_groups = np.unique(g)
    if len(unique_groups) < MIN_TARGET_GROUPS:
        raise ValueError(
            f"Need samples from at least {MIN_TARGET_GROUPS} calibration targets, got {len(unique_groups)}"
        )

    best_lam = float(lambdas[0])
    best_err = float("inf")
    for lam in lambdas:
        fold_errors: list[float] = []
        for held_out in unique_groups:
            train = g != held_out
            mean, std = _standardize(x[train])
            w = _solve(_design((x[train] - mean) / std), y[train], float(lam))
            pred = _design((x[~train] - mean) / std) @ w
            fold_errors.append(float(np.linalg.norm(pred - y[~train], axis=1).mean()))
        err = float(np.mean(fold_errors))
        if err < best_err:
            best_err, best_lam = err, float(lam)

    mean, std = _standardize(x)
    weights = _solve(_design((x - mean) / std), y, best_lam)
    return RidgeModel(feature_mean=mean, feature_std=std, weights=weights, lam=best_lam, cv_error=best_err)
