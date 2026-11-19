"""Reference implementation of ``contracts/preprocess_spec.v1.json``.

This module is the *executable specification* of the input pipeline. The
Android app (``:core:domain`` ``Preprocessor``) re-implements exactly the same
arithmetic and is checked against golden fixtures produced from this code.

Why not ``cv2.resize``/``warpAffine``? OpenCV's interpolation kernels and
Android's Bitmap scaling differ in ways we cannot reproduce bit-for-bit on the
device. A small, explicit sampler (bilinear + fixed supersampling, pixel
centres at ``+0.5``, replicate border) is trivially portable and verifiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.features import geometry_vector
from eyegaze.domain.geometry import Box
from eyegaze.domain.observation import FaceObservation
from eyegaze.domain.spec import EyeSpec, FaceSpec, PreprocessSpec, SamplingSpec

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

# --------------------------------------------------------------------------- sampler


@dataclass(frozen=True, slots=True)
class CropTransform:
    """Square, rotated crop: centre and side in pixels, rotation in radians."""

    cx: float
    cy: float
    side: float
    angle: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.cx) and math.isfinite(self.cy) and math.isfinite(self.angle)):
            raise ValueError("CropTransform values must be finite")
        if not (math.isfinite(self.side) and self.side > 0):
            raise ValueError("CropTransform.side must be positive")


def _bilinear_replicate(image: UInt8Array, xs: FloatArray, ys: FloatArray) -> FloatArray:
    """Sample ``image`` (H, W, C) at continuous pixel coords whose centres sit at ``k + 0.5``."""
    height, width = image.shape[:2]
    x = xs - 0.5
    y = ys - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0i = x0.astype(np.int64)
    y0i = y0.astype(np.int64)
    x1i = np.clip(x0i + 1, 0, width - 1)
    y1i = np.clip(y0i + 1, 0, height - 1)
    x0i = np.clip(x0i, 0, width - 1)
    y0i = np.clip(y0i, 0, height - 1)
    img = image.astype(np.float64)
    top = img[y0i, x0i] * (1.0 - fx) + img[y0i, x1i] * fx
    bottom = img[y1i, x0i] * (1.0 - fx) + img[y1i, x1i] * fx
    result: FloatArray = top * (1.0 - fy) + bottom * fy
    return result


def warp_crop(
    image_rgb: UInt8Array,
    transform: CropTransform,
    out_size: int,
    sampling: SamplingSpec,
    flip_horizontal: bool = False,
) -> Float32Array:
    """Extract a rotated square crop as ``float32 (out_size, out_size, 3)`` in ``[0, 1]``.

    Output pixel ``(u, v)`` is the mean of ``supersample²`` bilinear samples at
    ``centre + side * (a * e_x + b * e_y)`` with ``a, b`` in ``[-0.5, 0.5)`` and
    ``e_x = (cos θ, sin θ)``, ``e_y = (-sin θ, cos θ)``.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 (H, W, 3) image, got {image_rgb.dtype} {image_rgb.shape}")
    if out_size <= 0:
        raise ValueError("out_size must be positive")

    s = sampling.supersample
    sub = (np.arange(s, dtype=np.float64) + 0.5) / s
    grid = (np.arange(out_size, dtype=np.float64)[:, None] + sub[None, :]).reshape(-1) / out_size - 0.5
    b, a = np.meshgrid(grid, grid, indexing="ij")  # b: rows (v), a: columns (u)

    cos_t = math.cos(transform.angle)
    sin_t = math.sin(transform.angle)
    xs = transform.cx + transform.side * (a * cos_t - b * sin_t)
    ys = transform.cy + transform.side * (a * sin_t + b * cos_t)

    samples = _bilinear_replicate(image_rgb, xs, ys)
    pooled = samples.reshape(out_size, s, out_size, s, 3).mean(axis=(1, 3))
    if flip_horizontal:
        pooled = pooled[:, ::-1, :]
    out: Float32Array = (pooled / sampling.pixel_divisor).astype(np.float32)
    return out


# --------------------------------------------------------------------------- landmarks → crops


def _eye_transform(
    obs: FaceObservation, left_px: tuple[float, float], right_px: tuple[float, float], spec: EyeSpec
) -> CropTransform:
    """Crop around the segment from the image-left corner to the image-right corner."""
    dx = right_px[0] - left_px[0]
    dy = right_px[1] - left_px[1]
    width = math.hypot(dx, dy)
    if width <= 1e-6:
        raise ValueError("Degenerate eye: corners coincide")
    angle = math.atan2(dy, dx) if spec.align_roll else 0.0
    return CropTransform(
        cx=(left_px[0] + right_px[0]) / 2.0,
        cy=(left_px[1] + right_px[1]) / 2.0,
        side=width * spec.width_scale,
        angle=angle,
    )


def eye_transforms(obs: FaceObservation, spec: EyeSpec) -> tuple[CropTransform, CropTransform]:
    """Return ``(right_eye, left_eye)`` crop transforms (subject's right/left).

    In an un-mirrored frame the subject's right eye is on the image left, so its
    outer corner is the image-left point; for the left eye the inner corner is.
    """
    right = _eye_transform(obs, obs.px(spec.right.outer), obs.px(spec.right.inner), spec)
    left = _eye_transform(obs, obs.px(spec.left.inner), obs.px(spec.left.outer), spec)
    return right, left


def face_bbox(obs: FaceObservation, spec: FaceSpec) -> Box:
    pts = obs.landmarks[: spec.bbox_landmark_count, :2]
    xs = pts[:, 0] * obs.frame_width
    ys = pts[:, 1] * obs.frame_height
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return Box(x0, y0, max(x1 - x0, 1e-6), max(y1 - y0, 1e-6))


def face_transform(obs: FaceObservation, face: FaceSpec, eye: EyeSpec) -> CropTransform:
    box = face_bbox(obs, face)
    angle = 0.0
    if face.align_roll:
        rx, ry = obs.px(eye.right.outer)
        lx, ly = obs.px(eye.left.outer)
        angle = math.atan2(ly - ry, lx - rx)
    return CropTransform(cx=box.cx, cy=box.cy, side=max(box.w, box.h) * face.scale, angle=angle)


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """Network inputs for one frame (NHWC without the batch axis)."""

    right_eye: Float32Array  # (E, E, 3)
    left_eye: Float32Array  # (E, E, 3), horizontally flipped when spec.eye.flip_left
    face: Float32Array  # (F, F, 3)
    geometry: Float32Array  # (17,)


def extract_inputs(image_rgb: UInt8Array, obs: FaceObservation, spec: PreprocessSpec) -> ModelInputs:
    """Full per-frame preprocessing: three crops plus the geometry vector."""
    if image_rgb.shape[0] != obs.frame_height or image_rgb.shape[1] != obs.frame_width:
        raise ValueError(
            f"Image {image_rgb.shape[1]}x{image_rgb.shape[0]} does not match observation "
            f"{obs.frame_width}x{obs.frame_height}"
        )
    right_t, left_t = eye_transforms(obs, spec.eye)
    face_t = face_transform(obs, spec.face, spec.eye)
    return ModelInputs(
        right_eye=warp_crop(image_rgb, right_t, spec.eye.size, spec.sampling),
        left_eye=warp_crop(image_rgb, left_t, spec.eye.size, spec.sampling, flip_horizontal=spec.eye.flip_left),
        face=warp_crop(image_rgb, face_t, spec.face.size, spec.sampling),
        geometry=geometry_vector(obs, spec),
    )
