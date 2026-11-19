"""Typed mirror of ``contracts/preprocess_spec.v1.json``.

Plain frozen dataclasses keep the domain free of I/O and framework code; the
JSON file is parsed and validated at the boundary (:mod:`eyegaze.contracts`).
"""

from __future__ import annotations

from dataclasses import dataclass

MEDIAPIPE_LANDMARK_COUNT = 478


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    method: str
    supersample: int
    border: str
    pixel_divisor: float

    def __post_init__(self) -> None:
        if self.method != "bilinear":
            raise ValueError(f"Unsupported sampling method {self.method!r}")
        if self.border != "replicate":
            raise ValueError(f"Unsupported border mode {self.border!r}")
        if not 1 <= self.supersample <= 4:
            raise ValueError("supersample must be in [1, 4]")
        if self.pixel_divisor <= 0:
            raise ValueError("pixel_divisor must be positive")


@dataclass(frozen=True, slots=True)
class EyeLandmarks:
    outer: int
    inner: int
    iris_center: int


@dataclass(frozen=True, slots=True)
class EyeSpec:
    size: int
    width_scale: float
    align_roll: bool
    right: EyeLandmarks
    left: EyeLandmarks
    flip_left: bool

    def __post_init__(self) -> None:
        if self.size < 16:
            raise ValueError("eye.size must be >= 16")
        if self.width_scale <= 1.0:
            raise ValueError("eye.width_scale must be > 1 so the crop contains both corners")


@dataclass(frozen=True, slots=True)
class FaceSpec:
    size: int
    scale: float
    align_roll: bool
    bbox_landmark_count: int

    def __post_init__(self) -> None:
        if self.size < 16:
            raise ValueError("face.size must be >= 16")
        if self.scale <= 0:
            raise ValueError("face.scale must be positive")
        if not 0 < self.bbox_landmark_count <= MEDIAPIPE_LANDMARK_COUNT:
            raise ValueError("face.bbox_landmark_count out of range")


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    dim: int
    layout: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.layout) != self.dim:
            raise ValueError(f"geometry.layout has {len(self.layout)} entries, expected {self.dim}")


@dataclass(frozen=True, slots=True)
class FrameSpec:
    color_order: str
    orientation: str
    landmarks: str
    transform_matrix: str

    def __post_init__(self) -> None:
        if self.color_order != "RGB":
            raise ValueError("Only RGB frames are supported")


@dataclass(frozen=True, slots=True)
class PreprocessSpec:
    version: int
    frame: FrameSpec
    sampling: SamplingSpec
    eye: EyeSpec
    face: FaceSpec
    geometry: GeometrySpec
