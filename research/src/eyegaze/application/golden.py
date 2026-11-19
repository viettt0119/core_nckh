"""Golden fixtures: the cross-language contract tests for the Android port.

Everything here is produced by the Python *reference* implementations and is
committed under ``contracts/golden/``. Kotlin unit tests load the same files
and must reproduce them (crops to ≤1e-3, scalars to ≤1e-6). Re-running the
generator must reproduce the committed files; a test enforces that, so the
fixtures can never silently drift from the reference code.

Files
-----
``index.json``                  case list, spec hash, generator version
``<case>.json``                 inputs (landmarks, transform) + expected keypoints/transforms/geometry
``<case>.rgb.gz``               input frame, uint8 RGB, HWC, delta-x encoded, gzip (mtime 0)
``<case>.crops.f32.gz``         expected crops right_eye|left_eye|face, float32 LE, HWC, gzip
``calibration.json``            ridge calibration inputs and expected model/predictions
``screen_geometry.json``        camera-cm ↔ normalized-screen conversions
``golden_model.tflite`` + ``golden_model_expected.json``   (optional) LiteRT outputs for each case
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eyegaze.contracts import DEFAULT_SPEC_PATH, load_spec, spec_hash
from eyegaze.domain.calibration import DEFAULT_LAMBDAS, fit_ridge
from eyegaze.domain.geometry import ScreenGeometry
from eyegaze.domain.keypoints import FaceKeypoints, keypoints_from_observation
from eyegaze.domain.observation import FaceObservation, HeadPose, rotation_from_head_pose
from eyegaze.domain.preprocess import CropTransform, crop_transforms, extract_inputs
from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT, PreprocessSpec

GENERATOR_VERSION = 1
CROP_ORDER = ("right_eye", "left_eye", "face")


@dataclass(frozen=True, slots=True)
class FaceCase:
    name: str
    frame_width: int
    frame_height: int
    center: tuple[float, float]
    eye_distance: float  # px between eye centres
    roll: float  # in-image rotation, radians
    pose: HeadPose
    iris_shift: tuple[float, float]  # iris offset from eye centre, in eye-widths
    seed: int


CASES: tuple[FaceCase, ...] = (
    FaceCase("frontal_portrait", 480, 640, (240.0, 320.0), 80.0, 0.0, HeadPose(0.0, 0.0, 0.0), (0.0, 0.0), 1),
    FaceCase("roll_positive", 480, 640, (250.0, 300.0), 84.0, 0.35, HeadPose(0.05, 0.1, 0.35), (0.1, 0.0), 2),
    FaceCase("roll_negative_pose", 480, 640, (230.0, 340.0), 76.0, -0.25, HeadPose(0.2, -0.3, -0.25), (-0.15, 0.05), 3),
    FaceCase("near_border", 480, 640, (55.0, 90.0), 70.0, 0.1, HeadPose(-0.1, 0.4, 0.1), (0.05, -0.05), 4),
    FaceCase("small_face_landscape", 320, 240, (200.0, 110.0), 30.0, -0.05, HeadPose(0.0, 0.0, -0.05), (0.0, 0.1), 5),
    FaceCase("large_face", 480, 640, (240.0, 330.0), 160.0, 0.05, HeadPose(0.1, 0.05, 0.05), (-0.1, -0.1), 6),
)


def synthetic_observation(case: FaceCase) -> FaceObservation:
    """Plausible landmark layout: the six spec points are exact, the rest fill the face region."""
    rng = np.random.default_rng(case.seed)
    d = case.eye_distance
    eye_w = 0.5 * d
    eye_y = -0.3 * d
    local = np.empty((MEDIAPIPE_LANDMARK_COUNT, 2), dtype=np.float64)
    local[:, 0] = rng.uniform(-1.1 * d, 1.1 * d, MEDIAPIPE_LANDMARK_COUNT)
    local[:, 1] = rng.uniform(-1.2 * d, 1.6 * d, MEDIAPIPE_LANDMARK_COUNT)
    shift = np.array(case.iris_shift) * eye_w
    fixed = {
        33: (-0.5 * d - 0.5 * eye_w, eye_y),  # right eye outer (image left)
        133: (-0.5 * d + 0.5 * eye_w, eye_y),
        362: (0.5 * d - 0.5 * eye_w, eye_y),
        263: (0.5 * d + 0.5 * eye_w, eye_y),
        468: (-0.5 * d + shift[0], eye_y + shift[1]),
        473: (0.5 * d + shift[0], eye_y + shift[1]),
    }
    for idx, xy in fixed.items():
        local[idx] = xy
    c, s = math.cos(case.roll), math.sin(case.roll)
    px = local @ np.array([[c, s], [-s, c]]) + np.array(case.center)
    landmarks = np.zeros((MEDIAPIPE_LANDMARK_COUNT, 3), dtype=np.float64)
    landmarks[:, 0] = px[:, 0] / case.frame_width
    landmarks[:, 1] = px[:, 1] / case.frame_height
    landmarks[:, 2] = rng.normal(scale=0.02, size=MEDIAPIPE_LANDMARK_COUNT)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_from_head_pose(case.pose) * 1.03  # MediaPipe matrices may carry a scale
    matrix[:3, 3] = (1.5, -2.0, -45.0)
    return FaceObservation(landmarks, matrix, case.frame_width, case.frame_height)


def synthetic_image(width: int, height: int, seed: int) -> NDArray[np.uint8]:
    """Smooth, textured RGB pattern: exercises interpolation and keeps the fixtures small."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    channels = []
    for k in range(3):
        fx, fy, phase = rng.uniform(0.02, 0.09), rng.uniform(0.02, 0.09), rng.uniform(0, 2 * math.pi)
        channels.append(128 + 70 * np.sin(fx * x + phase) * np.cos(fy * y + 0.5 * k) + 0.08 * (x - y))
    img = np.stack(channels, axis=2)
    out: NDArray[np.uint8] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return out


def _transform_dict(t: CropTransform) -> dict[str, float]:
    return {"cx": t.cx, "cy": t.cy, "side": t.side, "angle": t.angle}


def _keypoints_dict(kp: FaceKeypoints) -> dict[str, Any]:
    return {
        "right_outer": list(kp.right_outer),
        "right_inner": list(kp.right_inner),
        "left_inner": list(kp.left_inner),
        "left_outer": list(kp.left_outer),
        "right_iris": list(kp.right_iris),
        "left_iris": list(kp.left_iris),
        "face_bbox": {"x": kp.face_bbox.x, "y": kp.face_bbox.y, "w": kp.face_bbox.w, "h": kp.face_bbox.h},
        "pose": {"pitch": kp.pose.pitch, "yaw": kp.pose.yaw, "roll": kp.pose.roll},
    }


def delta_encode_x(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Per row and channel: first pixel verbatim, then ``(p[x] - p[x-1]) mod 256`` (PNG "Sub" filter)."""
    d = image.astype(np.int16)
    d[:, 1:] -= image[:, :-1].astype(np.int16)
    out: NDArray[np.uint8] = (d % 256).astype(np.uint8)
    return out


def delta_decode_x(encoded: NDArray[np.uint8]) -> NDArray[np.uint8]:
    out: NDArray[np.uint8] = (np.cumsum(encoded.astype(np.int64), axis=1) % 256).astype(np.uint8)
    return out


def _write_gz(path: Path, data: bytes) -> None:
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(data)


def read_gz(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class CaseArtifacts:
    image: NDArray[np.uint8]
    crops: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    geometry: NDArray[np.float32]


def write_case(out_dir: Path, case: FaceCase, spec: PreprocessSpec) -> CaseArtifacts:
    obs = synthetic_observation(case)
    image = synthetic_image(case.frame_width, case.frame_height, case.seed)
    kp = keypoints_from_observation(obs, spec)
    transforms = crop_transforms(kp, spec)
    inputs = extract_inputs(image, obs, spec)
    crops = (inputs.right_eye, inputs.left_eye, inputs.face)
    _write_gz(out_dir / f"{case.name}.rgb.gz", delta_encode_x(image).tobytes())
    _write_gz(out_dir / f"{case.name}.crops.f32.gz", b"".join(c.astype("<f4").tobytes() for c in crops))
    _dump(
        out_dir / f"{case.name}.json",
        {
            "name": case.name,
            "frame_width": case.frame_width,
            "frame_height": case.frame_height,
            "image": {
                "file": f"{case.name}.rgb.gz",
                "dtype": "uint8",
                "layout": "HWC",
                "color": "RGB",
                "encoding": "delta-x",  # decode: running sum mod 256 along x, per row and channel
            },
            "landmarks": obs.landmarks.tolist(),
            "transform_matrix": obs.transform_matrix.tolist(),
            "expected": {
                "keypoints": _keypoints_dict(kp),
                "transforms": {
                    "right_eye": _transform_dict(transforms.right_eye),
                    "left_eye": _transform_dict(transforms.left_eye),
                    "face": _transform_dict(transforms.face),
                },
                "geometry": inputs.geometry.astype(np.float64).tolist(),
                "crops": {
                    "file": f"{case.name}.crops.f32.gz",
                    "dtype": "float32-le",
                    "layout": "HWC",
                    "order": list(CROP_ORDER),
                    "shapes": [list(c.shape) for c in crops],
                },
            },
        },
    )
    return CaseArtifacts(image, crops, inputs.geometry)


def write_calibration(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n_targets, per_target, dim = 9, 12, 8
    w_true = rng.normal(size=(dim, 2))
    centres = rng.normal(size=(n_targets, dim))
    x = np.repeat(centres, per_target, axis=0) + rng.normal(scale=0.1, size=(n_targets * per_target, dim))
    y = x @ w_true + np.array([0.5, 0.4]) + rng.normal(scale=0.05, size=(len(x), 2))
    groups = np.repeat(np.arange(n_targets), per_target)
    model = fit_ridge(x, y, groups)
    probes = rng.normal(size=(5, dim))
    _dump(
        out_dir / "calibration.json",
        {
            "lambdas": list(DEFAULT_LAMBDAS),
            "features": x.tolist(),
            "targets": y.tolist(),
            "groups": groups.tolist(),
            "expected": {
                "lambda": model.lam,
                "cv_error": model.cv_error,
                "feature_mean": model.feature_mean.tolist(),
                "feature_std": model.feature_std.tolist(),
                "weights": model.weights.tolist(),
                "probes": probes.tolist(),
                "predictions": model.predict(probes).tolist(),
            },
        },
    )


def write_screen_geometry(out_dir: Path) -> None:
    screens = [
        {"width_px": 1080, "height_px": 2340, "xdpi": 400.0, "ydpi": 400.0},
        {"width_px": 720, "height_px": 1600, "xdpi": 268.0, "ydpi": 270.5},
    ]
    gaze_cm = [[0.0, -0.5], [-3.1, -7.4], [2.2, -12.9], [0.4, 1.0]]
    cases = []
    for s in screens:
        geo = ScreenGeometry.from_dpi(s["width_px"], s["height_px"], s["xdpi"], s["ydpi"])  # type: ignore[arg-type]
        cases.append(
            {
                "screen": s,
                "expected_camera_x_cm": geo.camera_x_cm,
                "expected_camera_y_cm": geo.camera_y_cm,
                "gaze_cm": gaze_cm,
                "expected_screen_norm": geo.camera_cm_to_screen_norm(np.array(gaze_cm)).tolist(),
            }
        )
    _dump(out_dir / "screen_geometry.json", {"default_camera_y_cm": -0.5, "cases": cases})


def generate(out_dir: Path, spec_path: Path = DEFAULT_SPEC_PATH) -> dict[str, CaseArtifacts]:
    """Write all preprocessing/calibration/geometry fixtures; returns case artifacts for the model step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(spec_path)
    artifacts = {case.name: write_case(out_dir, case, spec) for case in CASES}
    write_calibration(out_dir)
    write_screen_geometry(out_dir)
    _dump(
        out_dir / "index.json",
        {
            "generator_version": GENERATOR_VERSION,
            "spec_file": "../" + spec_path.name,
            "spec_sha256": spec_hash(spec_path),
            "tolerances": {"crops": 1e-3, "scalars": 1e-6, "model_outputs": 1e-4},
            "cases": [c.name for c in CASES],
        },
    )
    return artifacts


def generate_golden_model(
    out_dir: Path, artifacts: dict[str, CaseArtifacts], spec_path: Path = DEFAULT_SPEC_PATH
) -> None:
    """Export a tiny deterministic model and record its LiteRT outputs on every case (needs [export])."""
    import torch

    from eyegaze.adapters.litert import LiteRtRunner, export_litert
    from eyegaze.adapters.torch_model import EyeGazeNet
    from eyegaze.config import ModelConfig

    spec = load_spec(spec_path)
    torch.manual_seed(1234)
    model = EyeGazeNet(ModelConfig(width=0.25, embedding_dim=16, dropout=0.0)).eval()
    geoms = np.stack([a.geometry for a in artifacts.values()]).astype(np.float64)
    model.set_geometry_stats(
        torch.from_numpy(geoms.mean(axis=0).astype(np.float32)), torch.from_numpy(geoms.std(axis=0).astype(np.float32))
    )
    model_path = out_dir / "golden_model.tflite"
    export_litert(model, spec, model_path)
    runner = LiteRtRunner(model_path)
    expected = {}
    for name, art in artifacts.items():
        gaze, emb = runner.run(*art.crops, art.geometry)
        expected[name] = {"gaze_cm": gaze.astype(np.float64).tolist(), "embedding": emb.astype(np.float64).tolist()}
    payload = {"model_file": model_path.name, "embedding_dim": 16, "cases": expected}
    _dump(out_dir / "golden_model_expected.json", payload)
