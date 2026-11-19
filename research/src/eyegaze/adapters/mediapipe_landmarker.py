"""MediaPipe Face Landmarker adapter (Tasks API, IMAGE mode).

The Android app runs the very same ``face_landmarker.task`` in LIVE_STREAM
mode; IMAGE mode is used offline because GazeCapture frames are sparse
samples, not a continuous stream (tracking state would leak across gaps).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from numpy.typing import NDArray

from eyegaze.application.ports import Detection
from eyegaze.domain.observation import FaceObservation
from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT

BLINK_LEFT = "eyeBlinkLeft"
BLINK_RIGHT = "eyeBlinkRight"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LandmarkerConfig:
    model_path: Path
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Face landmarker model not found: {self.model_path}. Run `make models` to download it."
            )
        for value in (self.min_detection_confidence, self.min_presence_confidence):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Confidence thresholds must be in [0, 1]")


class MediaPipeLandmarker:
    def __init__(self, config: LandmarkerConfig) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_presence_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._closed = False

    def detect(self, image_rgb: NDArray[np.uint8]) -> Detection | None:
        if self._closed:
            raise RuntimeError("Landmarker is closed")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 (H, W, 3) RGB image, got {image_rgb.dtype} {image_rgb.shape}")
        height, width = image_rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        points = result.face_landmarks[0]
        if len(points) != MEDIAPIPE_LANDMARK_COUNT:
            return None
        landmarks = np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64)
        if not result.facial_transformation_matrixes:
            return None
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(4, 4)
        scores = (
            {c.category_name: float(c.score) for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
        )
        try:
            observation = FaceObservation(landmarks, matrix, width, height)
        except ValueError:
            return None
        return Detection(
            observation=observation,
            blink_left=scores.get(BLINK_LEFT, 0.0),
            blink_right=scores.get(BLINK_RIGHT, 0.0),
        )

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> MediaPipeLandmarker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class MediaPipeLandmarkerFactory:
    """Picklable factory so every worker process creates its own native landmarker."""

    config: LandmarkerConfig

    def __call__(self) -> MediaPipeLandmarker:
        return MediaPipeLandmarker(self.config)
