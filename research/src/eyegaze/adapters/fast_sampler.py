"""OpenCV-accelerated twin of :func:`eyegaze.domain.preprocess.warp_crop` for training throughput.

Same sampling geometry (supersampled grid, pixel centres at +0.5, replicate
border, box pooling); ``cv2.remap`` quantizes sub-pixel offsets to 1/32 px,
so results differ from the reference by well under one grey level. A unit test
bounds that difference. The numpy reference remains the contract for golden
fixtures and Android parity; this backend is only used to feed the trainer.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from eyegaze.domain.preprocess import CropTransform, CropTransforms
from eyegaze.domain.spec import PreprocessSpec, SamplingSpec

Float32Array = NDArray[np.float32]

_grid_cache: dict[tuple[int, int], tuple[Float32Array, Float32Array]] = {}


def _unit_grid(out_size: int, supersample: int) -> tuple[Float32Array, Float32Array]:
    key = (out_size, supersample)
    if key not in _grid_cache:
        sub = (np.arange(supersample, dtype=np.float64) + 0.5) / supersample
        g = (np.arange(out_size, dtype=np.float64)[:, None] + sub[None, :]).reshape(-1) / out_size - 0.5
        b, a = np.meshgrid(g, g, indexing="ij")
        _grid_cache[key] = (a.astype(np.float32), b.astype(np.float32))
    return _grid_cache[key]


def warp_crop_fast(
    image_f32: Float32Array,
    transform: CropTransform,
    out_size: int,
    sampling: SamplingSpec,
    flip_horizontal: bool = False,
) -> Float32Array:
    """``image_f32`` is the RGB frame already converted to float32 in [0, 255]."""
    a, b = _unit_grid(out_size, sampling.supersample)
    c, s = math.cos(transform.angle), math.sin(transform.angle)
    side = transform.side
    map_x = (transform.cx - 0.5) + side * (a * c - b * s)
    map_y = (transform.cy - 0.5) + side * (a * s + b * c)
    samples = cv2.remap(
        image_f32, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )  # fmt: skip
    k = sampling.supersample
    pooled = samples.reshape(out_size, k, out_size, k, 3).mean(axis=(1, 3))
    if flip_horizontal:
        pooled = pooled[:, ::-1, :]
    out: Float32Array = np.ascontiguousarray(pooled / np.float32(sampling.pixel_divisor), dtype=np.float32)
    return out


def render_crops_fast(
    image_rgb: NDArray[np.uint8], transforms: CropTransforms, spec: PreprocessSpec
) -> tuple[Float32Array, Float32Array, Float32Array]:
    img = image_rgb.astype(np.float32)
    return (
        warp_crop_fast(img, transforms.right_eye, spec.eye.size, spec.sampling),
        warp_crop_fast(img, transforms.left_eye, spec.eye.size, spec.sampling, flip_horizontal=spec.eye.flip_left),
        warp_crop_fast(img, transforms.face, spec.face.size, spec.sampling),
    )
