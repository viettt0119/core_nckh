from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyegaze.calibration import fit_calibration, validate_calibration_samples
from eyegaze.dataset import build_manifest
from eyegaze.model import EyeGazeNet, load_checkpoint, save_checkpoint
from eyegaze.preprocess import face_grid_from_box
from export_tflite import make_dummy_inputs


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _create_subject(root: Path) -> None:
    subject = root / "00001" / "00001"
    frames = subject / "frames"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "00000.jpg"), np.full((100, 120, 3), 160, np.uint8))
    _write_json(subject / "info.json", {"Dataset": "train"})
    _write_json(subject / "frames.json", ["00000.jpg"])
    _write_json(subject / "dotInfo.json", {"XPts": [60], "YPts": [50]})
    _write_json(subject / "screen.json", {"W": [120], "H": [100]})
    _write_json(subject / "appleFace.json", {"X": [10], "Y": [20], "W": [80], "H": [70], "IsValid": [1]})
    _write_json(subject / "appleLeftEye.json", {"X": [15], "Y": [10], "W": [20], "H": [12], "IsValid": [1]})
    _write_json(subject / "appleRightEye.json", {"X": [50], "Y": [10], "W": [20], "H": [12], "IsValid": [1]})


def test_manifest_converts_gazecapture_eye_boxes_to_absolute(tmp_path: Path) -> None:
    _create_subject(tmp_path)
    df = build_manifest(tmp_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["direction"] == "center"
    assert row["left_eye_x"] == 25
    assert row["left_eye_y"] == 30
    assert row["right_eye_x"] == 60
    assert row["right_eye_y"] == 30


def test_face_grid_shape() -> None:
    grid = face_grid_from_box((100, 100, 3), (20, 20, 40, 40))

    assert grid.shape == (25, 25)
    assert grid.sum() > 0


def test_model_forward_returns_tuple() -> None:
    """model.forward() must return (gaze_xy, direction_logits) tuple."""
    model = EyeGazeNet()
    gaze_xy, direction_logits = model(
        torch.rand(2, 3, 96, 96),
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 25, 25),
        torch.rand(2, 12),
    )
    assert isinstance(gaze_xy, torch.Tensor)
    assert isinstance(direction_logits, torch.Tensor)
    assert gaze_xy.shape == (2, 2)
    assert direction_logits.shape == (2, 5)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = EyeGazeNet()
    path = tmp_path / "best.dat"
    save_checkpoint(path, model, image_size=64, face_size=96, metrics={"accuracy": 1.0})
    loaded, ckpt = load_checkpoint(path, torch.device("cpu"))
    assert isinstance(loaded, EyeGazeNet)
    assert ckpt["format"] == "eyegaze-1.1"

    # Verify loaded model also returns tuple
    gaze_xy, _ = loaded(
        torch.rand(1, 3, 96, 96),
        torch.rand(1, 3, 64, 64),
        torch.rand(1, 3, 64, 64),
        torch.rand(1, 25, 25),
        torch.rand(1, 12),
    )
    assert gaze_xy.shape == (1, 2)


def test_model_onnx_exportable() -> None:
    """Model can be traced with torch.onnx.export without a wrapper."""
    pytest.importorskip("onnxscript", reason="onnxscript required for ONNX export")
    import io
    model = EyeGazeNet().eval()
    dummy = make_dummy_inputs(image_size=64, face_size=96)
    buffer = io.BytesIO()
    torch.onnx.export(model, dummy, buffer, opset_version=17,
                      input_names=["face", "left_eye", "right_eye", "face_grid", "geometry"],
                      output_names=["gaze_xy", "direction_logits"])
    assert buffer.tell() > 0


def test_calibration_validation() -> None:
    raw = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
    calibration = fit_calibration(raw, raw)
    np.testing.assert_allclose(calibration.apply((0.25, 0.75)), [0.25, 0.75], atol=1e-5)

    with pytest.raises(ValueError, match="left=0/3"):
        validate_calibration_samples([{"target": "center"}] * 3, ["center", "left"])
