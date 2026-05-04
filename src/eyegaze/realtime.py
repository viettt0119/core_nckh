from __future__ import annotations

import os
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import mediapipe as mp
import numpy as np
import torch

from eyegaze.calibration import load_calibration
from eyegaze.labels import DIRECTION_LABELS, direction_from_xy
from eyegaze.model import load_checkpoint
from eyegaze.preprocess import (
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    batch_from_features,
    extract_frame_features,
    eye_aspect_ratio,
    landmark_xy,
)


class BlinkDetector:
    def __init__(self, min_closed_ms: int = 250, double_window_ms: int = 1500) -> None:
        self.min_closed_ms = min_closed_ms
        self.double_window_ms = double_window_ms
        self.baseline: float | None = None
        self.closed_since: int | None = None
        self.last_blink: int | None = None

    def reset(self) -> None:
        self.baseline = None
        self.closed_since = None
        self.last_blink = None

    def update(self, ear: float, timestamp_ms: int) -> tuple[str, bool]:
        if self.baseline is None:
            self.baseline = ear
        threshold = max(0.18, self.baseline * 0.65)
        closed = ear < threshold
        if not closed:
            self.baseline = 0.97 * self.baseline + 0.03 * ear
        if closed and self.closed_since is None:
            self.closed_since = timestamp_ms
            return "closed", False
        if closed:
            return "closed", False
        if self.closed_since is None:
            return "open", False
        duration = timestamp_ms - self.closed_since
        self.closed_since = None
        if duration < self.min_closed_ms:
            return "open", False
        if self.last_blink is not None and timestamp_ms - self.last_blink <= self.double_window_ms:
            self.last_blink = None
            return "double_blink", True
        self.last_blink = timestamp_ms
        return "slow_blink", False


class DirectionSmoother:
    def __init__(self, window: int = 7) -> None:
        self.items: deque[str] = deque(maxlen=window)

    def reset(self) -> None:
        self.items.clear()

    def update(self, direction: str) -> str:
        self.items.append(direction)
        return Counter(self.items).most_common(1)[0][0]


class DwellSelector:
    def __init__(self, dwell_ms: int = 1000) -> None:
        self.dwell_ms = dwell_ms
        self.target: str | None = None
        self.started_ms: int | None = None
        self.fired = False

    def reset(self) -> None:
        self.target = None
        self.started_ms = None
        self.fired = False

    def update(self, target: str, timestamp_ms: int) -> bool:
        if target != self.target:
            self.target = target
            self.started_ms = timestamp_ms
            self.fired = False
            return False
        if self.started_ms is None:
            self.started_ms = timestamp_ms
            return False
        if not self.fired and timestamp_ms - self.started_ms >= self.dwell_ms:
            self.fired = True
            return True
        return False


@dataclass
class RealtimeState:
    direction: str = "center"
    confidence: float = 0.0
    raw_gaze: tuple[float, float] = (0.5, 0.5)
    corrected_gaze: tuple[float, float] = (0.5, 0.5)
    blink: str = "open"
    confirmed: bool = False
    selected: bool = False
    model_ready: bool = True


def heuristic_gaze_from_face(face_box: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    x, y, w, h = face_box
    face_center_x = (x + w * 0.5) / max(width, 1)
    face_center_y = (y + h * 0.42) / max(height, 1)
    # Mirror head position into a rough screen gaze estimate. This keeps the
    # realtime demo usable before a trained model is installed.
    gaze_x = np.clip(1.0 - face_center_x, 0.0, 1.0)
    gaze_y = np.clip(face_center_y, 0.0, 1.0)
    return np.array([gaze_x, gaze_y], dtype=np.float32)


def _draw_overlay(frame: np.ndarray, state: RealtimeState) -> np.ndarray:
    height, width = frame.shape[:2]
    zones = {
        "up": (width // 2 - 90, 20, 180, 70),
        "left": (20, height // 2 - 35, 180, 70),
        "center": (width // 2 - 90, height // 2 - 35, 180, 70),
        "right": (width - 200, height // 2 - 35, 180, 70),
        "down": (width // 2 - 90, height - 90, 180, 70),
    }
    names = {"up": "UP", "left": "LEFT", "center": "CENTER", "right": "RIGHT", "down": "DOWN"}
    for direction, (x, y, w, h) in zones.items():
        active = direction == state.direction
        color = (35, 185, 95) if state.selected and active else ((0, 170, 255) if active else (55, 55, 55))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2 if not active else -1)
        text_color = (25, 25, 25) if active else (235, 235, 235)
        cv2.putText(frame, names[direction], (x + 18, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)

    cv2.putText(
        frame,
        f"dir={state.direction} conf={state.confidence:.2f} blink={state.blink}",
        (12, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"gaze={state.corrected_gaze[0]:.2f},{state.corrected_gaze[1]:.2f}",
        (12, height - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    if not state.model_ready:
        cv2.putText(frame, "FALLBACK: install trained gaze_model.dat", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    elif state.confirmed:
        cv2.putText(frame, "DOUBLE BLINK CONFIRMED", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
    return frame


def run_realtime(
    checkpoint: str | Path,
    camera: int = 0,
    calibration: str | Path | None = None,
    allow_fallback: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    image_size = 64
    face_size = 96
    model_ready = True
    try:
        model, ckpt = load_checkpoint(checkpoint, device)
        image_size = int(ckpt.get("image_size", image_size))
        face_size = int(ckpt.get("face_size", face_size))
    except FileNotFoundError:
        if not allow_fallback:
            raise
        model_ready = False

    calibrator = load_calibration(calibration)
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    smoother = DirectionSmoother()
    blink = BlinkDetector()
    dwell = DwellSelector()
    state = RealtimeState(model_ready=model_ready)

    try:
        with face_mesh, torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                state.confirmed = False
                state.selected = False

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    features = extract_frame_features(frame, landmarks, image_size, face_size)
                    if model is None:
                        raw_gaze = heuristic_gaze_from_face(features.face_box, frame.shape)
                        confidence = 0.0
                        predicted_direction = direction_from_xy(float(raw_gaze[0]), float(raw_gaze[1]))
                    else:
                        batch = batch_from_features(features, device)
                        outputs = model(
                            batch["face"],
                            batch["left_eye"],
                            batch["right_eye"],
                            batch["face_grid"],
                            batch["geometry"],
                        )
                        gaze_xy_pred, direction_logits = outputs
                        probs = torch.softmax(direction_logits, dim=1)[0]
                        pred_idx = int(torch.argmax(probs).item())
                        raw_gaze = gaze_xy_pred[0].detach().cpu().numpy().astype(np.float32)
                        confidence = float(probs[pred_idx].item())
                        predicted_direction = DIRECTION_LABELS[pred_idx]

                    corrected = calibrator.apply(raw_gaze) if calibrator is not None else raw_gaze
                    if calibrator is not None:
                        predicted_direction = direction_from_xy(float(corrected[0]), float(corrected[1]))
                    state.raw_gaze = (float(raw_gaze[0]), float(raw_gaze[1]))
                    state.corrected_gaze = (float(corrected[0]), float(corrected[1]))
                    state.direction = smoother.update(predicted_direction)
                    state.confidence = confidence
                    state.model_ready = model_ready

                    left_pts = [landmark_xy(landmarks, idx, width, height) for idx in LEFT_EYE_EAR]
                    right_pts = [landmark_xy(landmarks, idx, width, height) for idx in RIGHT_EYE_EAR]
                    ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0
                    state.blink, state.confirmed = blink.update(ear, timestamp_ms)
                    state.selected = dwell.update(state.direction, timestamp_ms)
                else:
                    smoother.reset()
                    blink.reset()
                    dwell.reset()
                    state = RealtimeState(model_ready=model_ready)
                    cv2.putText(frame, "No face detected", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("EyeGaze realtime - press q to quit", _draw_overlay(frame, state))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
