"""Ports: the interfaces application use cases depend on.

Adapters implement them; tests substitute deterministic fakes so the dataset
pipeline can be exercised without MediaPipe, GPUs or the real dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.observation import FaceObservation

UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Detection:
    """Landmarker output for one frame."""

    observation: FaceObservation
    blink_left: float  # blendshape eyeBlinkLeft in [0, 1]
    blink_right: float  # blendshape eyeBlinkRight in [0, 1]


class LandmarkDetector(Protocol):
    """Single-image face landmarking. Implementations own native resources."""

    def detect(self, image_rgb: UInt8Array) -> Detection | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> LandmarkDetector: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


# A picklable zero-argument callable, so each worker process builds its own detector.
DetectorFactory = Callable[[], LandmarkDetector]


class ImageReader(Protocol):
    def read_rgb(self, path: Path) -> UInt8Array | None:
        """Return an ``(H, W, 3)`` uint8 RGB image, or ``None`` if unreadable."""
        ...
