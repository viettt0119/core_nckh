from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

# ============================================================================
# OpenCV Flags Reference for C++ Parity
# ============================================================================
# When reimplementing this pipeline in C++, use these exact OpenCV constants
# to guarantee bit-identical preprocessing:
#
# crop_resize():
#   - cv2.COLOR_BGR2RGB  → cv::COLOR_BGR2RGB  (= 4)
#   - cv2.INTER_AREA     → cv::INTER_AREA     (= 3)
#   - cv2.flip(img, 1)   → cv::flip(img, dst, 1)  // horizontal flip
#   - Normalize: pixel.astype(float32) / 255.0
#
# face_grid_from_box():
#   - np.floor → std::floor
#   - np.ceil  → std::ceil
#   - Grid size: 25x25 binary float32 matrix
#
# random_rotate() [TRAINING ONLY — do NOT use in C++ inference]:
#   - cv2.getRotationMatrix2D → cv::getRotationMatrix2D
#   - cv2.warpAffine flags=cv2.INTER_LINEAR → cv::INTER_LINEAR (= 1)
#   - borderMode=cv2.BORDER_REPLICATE → cv::BORDER_REPLICATE (= 1)
#
# random_blur() [TRAINING ONLY]:
#   - cv2.GaussianBlur(img, (k,k), 0) → cv::GaussianBlur
#
# Input tensor layout (PyTorch): [B, C, H, W] — channels-first, RGB
# Input tensor layout (TFLite):  [B, H, W, C] — channels-last, RGB
# ============================================================================

RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_BOX = [33, 133, 159, 145, 153, 144, 160, 158]
LEFT_EYE_BOX = [362, 263, 386, 374, 373, 380, 385, 387]


@dataclass(frozen=True)
class FrameFeatures:
    face_box: tuple[float, float, float, float]
    left_eye_box: tuple[float, float, float, float]
    right_eye_box: tuple[float, float, float, float]
    face: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    face_grid: np.ndarray
    geometry: np.ndarray


def clamp_box(box: tuple[float, float, float, float], image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    height, width = image_shape[:2]
    x0 = max(0, min(width, int(round(x))))
    y0 = max(0, min(height, int(round(y))))
    x1 = max(0, min(width, int(round(x + w))))
    y1 = max(0, min(height, int(round(y + h))))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_resize(
    image_bgr: np.ndarray,
    box: tuple[float, float, float, float],
    image_size: int,
    flip_horizontal: bool = False,
) -> np.ndarray:
    bounds = clamp_box(box, image_bgr.shape)
    if bounds is None:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    x0, y0, x1, y1 = bounds
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)
    if flip_horizontal:
        crop = cv2.flip(crop, 1)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0


def geometry_vector(
    image_shape: tuple[int, ...],
    face_box: tuple[float, float, float, float],
    left_eye_box: tuple[float, float, float, float],
    right_eye_box: tuple[float, float, float, float],
) -> np.ndarray:
    height, width = image_shape[:2]
    scale = np.array([width, height, width, height] * 3, dtype=np.float32)
    raw = np.array([*face_box, *left_eye_box, *right_eye_box], dtype=np.float32)
    return np.clip(raw / np.maximum(scale, 1.0), 0.0, 1.5).astype(np.float32)


def face_grid_from_box(
    image_shape: tuple[int, ...],
    face_box: tuple[float, float, float, float],
    grid_size: int = 25,
) -> np.ndarray:
    height, width = image_shape[:2]
    x, y, w, h = face_box
    gx0 = max(0, min(grid_size, int(np.floor(x / max(width, 1) * grid_size))))
    gy0 = max(0, min(grid_size, int(np.floor(y / max(height, 1) * grid_size))))
    gx1 = max(0, min(grid_size, int(np.ceil((x + w) / max(width, 1) * grid_size))))
    gy1 = max(0, min(grid_size, int(np.ceil((y + h) / max(height, 1) * grid_size))))
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if gx1 > gx0 and gy1 > gy0:
        grid[gy0:gy1, gx0:gx1] = 1.0
    return grid


def landmark_xy(landmarks: list, idx: int, width: int, height: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * width, lm.y * height


def box_from_indices(
    landmarks: list,
    indices: list[int],
    width: int,
    height: int,
    pad: float = 0.45,
) -> tuple[float, float, float, float]:
    pts = np.array([landmark_xy(landmarks, idx, width, height) for idx in indices], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw = x1 - x0
    bh = y1 - y0
    size_pad = max(bw, bh) * pad
    return float(x0 - size_pad), float(y0 - size_pad), float(bw + 2 * size_pad), float(bh + 2 * size_pad)


def face_box_from_landmarks(landmarks: list, width: int, height: int) -> tuple[float, float, float, float]:
    pts = np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def eye_aspect_ratio(points: list[tuple[float, float]]) -> float:
    pts = np.asarray(points, dtype=np.float32)
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal <= 1e-6:
        return 0.0
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def extract_frame_features(frame_bgr: np.ndarray, landmarks: list, image_size: int, face_size: int) -> FrameFeatures:
    height, width = frame_bgr.shape[:2]
    face_box = face_box_from_landmarks(landmarks, width, height)
    left_box = box_from_indices(landmarks, LEFT_EYE_BOX, width, height)
    right_box = box_from_indices(landmarks, RIGHT_EYE_BOX, width, height)
    face = crop_resize(frame_bgr, face_box, face_size)
    left_eye = crop_resize(frame_bgr, left_box, image_size, flip_horizontal=True)
    right_eye = crop_resize(frame_bgr, right_box, image_size)
    grid = face_grid_from_box(frame_bgr.shape, face_box)
    geom = geometry_vector(frame_bgr.shape, face_box, left_box, right_box)
    return FrameFeatures(face_box, left_box, right_box, face, left_eye, right_eye, grid, geom)


def batch_from_features(features: FrameFeatures, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "face": torch.from_numpy(features.face).permute(2, 0, 1).unsqueeze(0).float().to(device),
        "left_eye": torch.from_numpy(features.left_eye).permute(2, 0, 1).unsqueeze(0).float().to(device),
        "right_eye": torch.from_numpy(features.right_eye).permute(2, 0, 1).unsqueeze(0).float().to(device),
        "face_grid": torch.from_numpy(features.face_grid).unsqueeze(0).float().to(device),
        "geometry": torch.from_numpy(features.geometry).unsqueeze(0).float().to(device),
    }
