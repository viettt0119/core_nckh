"""Gaze error metrics.

Two aggregate means are reported because the literature uses both:

* ``frame_mean`` — mean over all frames (iTracker-style, comparable to papers).
* ``subject_mean`` — mean of per-subject means; every person counts once, which
  is what matters for a product that must work for *each* user.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    n: int
    mean: float
    median: float
    p90: float
    std: float

    def as_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "mean": self.mean, "median": self.median, "p90": self.p90, "std": self.std}


@dataclass(frozen=True, slots=True)
class GazeReport:
    frame: ErrorSummary
    subject_mean: float
    subject_median: float
    per_subject: dict[str, ErrorSummary]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame.as_dict(),
            "subject_mean": self.subject_mean,
            "subject_median": self.subject_median,
            "n_subjects": len(self.per_subject),
            "per_subject": {k: v.as_dict() for k, v in sorted(self.per_subject.items())},
        }


def euclidean_errors(pred: FloatArray, target: FloatArray) -> FloatArray:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"Expected matching (N, 2) arrays, got {p.shape} and {t.shape}")
    errors: FloatArray = np.linalg.norm(p - t, axis=1)
    return errors


def summarize(errors: FloatArray) -> ErrorSummary:
    e = np.asarray(errors, dtype=np.float64)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("errors must be a non-empty 1-D array")
    return ErrorSummary(
        n=int(e.size),
        mean=float(e.mean()),
        median=float(np.median(e)),
        p90=float(np.percentile(e, 90)),
        std=float(e.std()),
    )


def gaze_report(errors: FloatArray, subject_ids: Sequence[str]) -> GazeReport:
    e = np.asarray(errors, dtype=np.float64)
    ids = np.asarray(subject_ids)
    if len(ids) != len(e):
        raise ValueError(f"Length mismatch: errors={len(e)}, subject_ids={len(ids)}")
    per_subject = {str(sid): summarize(e[ids == sid]) for sid in np.unique(ids)}
    means = np.array([s.mean for s in per_subject.values()], dtype=np.float64)
    return GazeReport(
        frame=summarize(e),
        subject_mean=float(means.mean()),
        subject_median=float(np.median(means)),
        per_subject=per_subject,
    )
