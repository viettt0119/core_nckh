"""Coordinate systems shared by training, evaluation and the Android app.

Two spaces matter:

* **Camera space (cm)** — the model's output space, identical to GazeCapture's
  ``XCam``/``YCam``: origin at the front camera, +x to the right of the device,
  +y *up*. Points on a portrait phone screen therefore have negative y.
* **Screen space (normalized)** — ``(0, 0)`` top-left, ``(1, 1)`` bottom-right of
  the display. This is what the UI consumes.

Predicting in camera space keeps the model device-independent; per-device
geometry is applied only at the edge (see :class:`ScreenGeometry`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CM_PER_INCH = 2.54

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel units, ``(x, y)`` is the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError(f"Box values must be finite, got {self}")
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"Box must have positive size, got w={self.w}, h={self.h}")

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """Physical layout of a display relative to its front camera.

    ``camera_x_cm``/``camera_y_cm`` locate the camera in screen coordinates
    (cm, origin at the screen's top-left corner, +y *down*). A camera centred
    above a portrait screen is roughly ``(width_cm / 2, -0.5)``.
    """

    width_px: int
    height_px: int
    px_per_cm_x: float
    px_per_cm_y: float
    camera_x_cm: float
    camera_y_cm: float

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("Screen size must be positive.")
        if self.px_per_cm_x <= 0 or self.px_per_cm_y <= 0:
            raise ValueError("Pixel density must be positive.")

    @classmethod
    def from_dpi(
        cls,
        width_px: int,
        height_px: int,
        xdpi: float,
        ydpi: float,
        camera_x_cm: float | None = None,
        camera_y_cm: float = -0.5,
    ) -> ScreenGeometry:
        """Build from Android ``DisplayMetrics``; defaults assume a top-centre camera."""
        ppcm_x = xdpi / CM_PER_INCH
        ppcm_y = ydpi / CM_PER_INCH
        cam_x = camera_x_cm if camera_x_cm is not None else (width_px / ppcm_x) / 2.0
        return cls(width_px, height_px, ppcm_x, ppcm_y, cam_x, camera_y_cm)

    @property
    def width_cm(self) -> float:
        return self.width_px / self.px_per_cm_x

    @property
    def height_cm(self) -> float:
        return self.height_px / self.px_per_cm_y

    def camera_cm_to_screen_norm(self, gaze_cm: FloatArray) -> FloatArray:
        """Map ``[..., 2]`` camera-space cm to ``[..., 2]`` normalized screen coordinates."""
        g = np.asarray(gaze_cm, dtype=np.float64)
        sx_cm = self.camera_x_cm + g[..., 0]
        sy_cm = self.camera_y_cm - g[..., 1]
        return np.stack([sx_cm / self.width_cm, sy_cm / self.height_cm], axis=-1)

    def screen_norm_to_camera_cm(self, screen_norm: FloatArray) -> FloatArray:
        """Inverse of :meth:`camera_cm_to_screen_norm`."""
        s = np.asarray(screen_norm, dtype=np.float64)
        gx = s[..., 0] * self.width_cm - self.camera_x_cm
        gy = self.camera_y_cm - s[..., 1] * self.height_cm
        return np.stack([gx, gy], axis=-1)

    def screen_norm_delta_to_cm(self, delta_norm: FloatArray) -> FloatArray:
        """Convert normalized-screen displacements to cm (for error reporting)."""
        d = np.asarray(delta_norm, dtype=np.float64)
        return np.stack([d[..., 0] * self.width_cm, d[..., 1] * self.height_cm], axis=-1)
