"""Offline simulation of per-user calibration on GazeCapture (experiment E2).

For each test subject, ``k`` distinct dot positions play the role of the
on-device calibration targets; a regressor is fitted on their frames and
evaluated on the subject's *other* dots. Pure numpy (plus optional sklearn for
the SVR comparison), so it is unit-testable without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.calibration import fit_ridge

FloatArray = NDArray[np.float64]

MIN_EVAL_DOTS = 1


class CalibrationMethod(StrEnum):
    RIDGE = "ridge"
    SVR = "svr"


@dataclass(frozen=True, slots=True)
class CalibrationSimResult:
    k: int
    method: str
    subjects_used: int
    subjects_skipped: int
    before_subject_mean: float  # uncalibrated error on the same evaluation frames
    after_subject_mean: float
    before_frame_mean: float
    after_frame_mean: float

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _fit_predict(
    method: CalibrationMethod, x_fit: FloatArray, y_fit: FloatArray, groups: NDArray[np.int64], x_eval: FloatArray
) -> FloatArray:
    if method is CalibrationMethod.RIDGE:
        return fit_ridge(x_fit, y_fit, groups).predict(x_eval)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    columns = []
    for dim in range(y_fit.shape[1]):
        svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, epsilon=0.1))
        svr.fit(x_fit, y_fit[:, dim])
        columns.append(svr.predict(x_eval))
    return np.stack(columns, axis=1)


def simulate_calibration(
    features: FloatArray,
    targets: FloatArray,
    uncalibrated: FloatArray,
    dot_ids: NDArray[np.int64],
    subject_ids: Sequence[str] | NDArray[np.str_],
    k: int,
    method: CalibrationMethod = CalibrationMethod.RIDGE,
    seed: int = 0,
) -> CalibrationSimResult:
    """Return mean errors before/after calibrating each subject on ``k`` random dots."""
    if k < 3:
        raise ValueError("k must be >= 3 (ridge CV needs at least 3 calibration targets)")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    base = np.asarray(uncalibrated, dtype=np.float64)
    dots = np.asarray(dot_ids, dtype=np.int64)
    subjects = np.asarray(subject_ids)
    if not (len(x) == len(y) == len(base) == len(dots) == len(subjects)):
        raise ValueError("All inputs must have the same length")

    rng = np.random.default_rng(seed)
    before_means: list[float] = []
    after_means: list[float] = []
    before_all: list[FloatArray] = []
    after_all: list[FloatArray] = []
    skipped = 0
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        unique_dots = np.unique(dots[rows])
        if len(unique_dots) < k + MIN_EVAL_DOTS:
            skipped += 1
            continue
        chosen = rng.choice(unique_dots, size=k, replace=False)
        fit_mask = np.isin(dots[rows], chosen)
        fit_rows, eval_rows = rows[fit_mask], rows[~fit_mask]
        pred = _fit_predict(method, x[fit_rows], y[fit_rows], dots[fit_rows], x[eval_rows])
        err_after = np.linalg.norm(pred - y[eval_rows], axis=1)
        err_before = np.linalg.norm(base[eval_rows] - y[eval_rows], axis=1)
        after_means.append(float(err_after.mean()))
        before_means.append(float(err_before.mean()))
        after_all.append(err_after)
        before_all.append(err_before)

    if not after_means:
        raise ValueError(f"No subject has more than {k} distinct dots; cannot simulate k={k}")
    return CalibrationSimResult(
        k=k,
        method=str(method),
        subjects_used=len(after_means),
        subjects_skipped=skipped,
        before_subject_mean=float(np.mean(before_means)),
        after_subject_mean=float(np.mean(after_means)),
        before_frame_mean=float(np.concatenate(before_all).mean()),
        after_frame_mean=float(np.concatenate(after_all).mean()),
    )
