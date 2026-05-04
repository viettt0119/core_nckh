from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-user EyeGaze calibration samples.")
    parser.add_argument("--checkpoint", default="models/trained/gaze_model.dat")
    parser.add_argument("--user", default="default")
    parser.add_argument("--out", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds-per-target", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--allow-fallback", action="store_true", help="Allow calibration using fallback predictor if no model is installed.")
    return parser.parse_args()


def draw_target(frame: np.ndarray, target: str, remaining: float) -> np.ndarray:
    import cv2
    from eyegaze.calibration import TARGET_POINTS

    h, w = frame.shape[:2]
    tx, ty = TARGET_POINTS[target]
    px, py = int(tx * w), int(ty * h)
    cv2.circle(frame, (px, py), 26, (0, 180, 255), -1)
    cv2.circle(frame, (px, py), 38, (255, 255, 255), 3)
    cv2.putText(frame, f"Look at: {target.upper()}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"{remaining:.1f}s", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def main() -> None:
    args = parse_args()
    import cv2
    import mediapipe as mp
    import torch

    from eyegaze.calibration import (
        TARGET_ORDER,
        TARGET_POINTS,
        calibration_path,
        fit_calibration,
        save_calibration,
        validate_calibration_samples,
    )
    from eyegaze.model import load_checkpoint
    from eyegaze.preprocess import batch_from_features, extract_frame_features
    from eyegaze.realtime import heuristic_gaze_from_face

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    image_size = 64
    face_size = 96
    try:
        model, ckpt = load_checkpoint(args.checkpoint, device)
        image_size = int(ckpt.get("image_size", image_size))
        face_size = int(ckpt.get("face_size", face_size))
    except FileNotFoundError:
        if not args.allow_fallback:
            raise

    out = Path(args.out) if args.out else calibration_path(args.user)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    raw_points: list[tuple[float, float]] = []
    target_points: list[tuple[float, float]] = []
    saved_samples: list[dict] = []

    try:
        with face_mesh, torch.no_grad():
            for target in TARGET_ORDER:
                start = time.time()
                while time.time() - start < args.seconds_per_target:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    elapsed = time.time() - start
                    remaining = max(0.0, args.seconds_per_target - elapsed)
                    collect = elapsed >= args.warmup_seconds
                    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.multi_face_landmarks and collect:
                        landmarks = result.multi_face_landmarks[0].landmark
                        features = extract_frame_features(frame, landmarks, image_size, face_size)
                        if model is None:
                            raw = heuristic_gaze_from_face(features.face_box, frame.shape)
                        else:
                            batch = batch_from_features(features, device)
                            outputs = model(
                                batch["face"],
                                batch["left_eye"],
                                batch["right_eye"],
                                batch["face_grid"],
                                batch["geometry"],
                            )
                            raw = outputs["gaze_xy"][0].detach().cpu().numpy()
                        raw_tuple = (float(raw[0]), float(raw[1]))
                        target_tuple = TARGET_POINTS[target]
                        raw_points.append(raw_tuple)
                        target_points.append(target_tuple)
                        saved_samples.append({"target": target, "raw": list(raw_tuple), "expected": list(target_tuple)})
                    cv2.imshow("EyeGaze calibration - press q to stop", draw_target(frame, target, remaining))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
    finally:
        cap.release()
        cv2.destroyAllWindows()

    validate_calibration_samples(saved_samples)
    calibration = fit_calibration(raw_points, target_points)
    save_calibration(out, calibration, saved_samples)
    print(f"Saved calibration: {out.resolve()}")
    print(f"Samples: {len(saved_samples)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
