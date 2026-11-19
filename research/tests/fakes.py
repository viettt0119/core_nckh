"""Deterministic test doubles for the application ports."""

from __future__ import annotations

from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from eyegaze.application.ports import Detection
from tests.factories import make_observation


class FakeDetector:
    """Decides by mean intensity: < 50 → no face, > 180 → eyes closed, else open eyes."""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def detect(self, image_rgb: NDArray[np.uint8]) -> Detection | None:
        self.calls += 1
        mean = float(image_rgb.mean())
        if mean < 50:
            return None
        height, width = image_rgb.shape[:2]
        blink = 0.9 if mean > 180 else 0.05
        return Detection(make_observation(frame_w=width, frame_h=height), blink_left=blink, blink_right=0.05)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FakeDetectorFactory:
    """Top-level and picklable so it also works in spawned worker processes."""

    def __call__(self) -> FakeDetector:
        return FakeDetector()
