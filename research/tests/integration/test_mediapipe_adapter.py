from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from eyegaze.cli import app
from eyegaze.cli_data import DEFAULT_LANDMARKER
from tests.gazecapture_fixture import write_fixture

pytestmark = [
    pytest.mark.requires_mediapipe,
    pytest.mark.skipif(not DEFAULT_LANDMARKER.is_file(), reason="run `make models` first"),
]


def test_blank_image_has_no_face() -> None:
    from eyegaze.adapters.mediapipe_landmarker import LandmarkerConfig, MediaPipeLandmarker

    with MediaPipeLandmarker(LandmarkerConfig(DEFAULT_LANDMARKER)) as landmarker:
        assert landmarker.detect(np.full((640, 480, 3), 127, np.uint8)) is None
        with pytest.raises(ValueError, match="uint8"):
            landmarker.detect(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        landmarker.detect(np.zeros((10, 10, 3), np.uint8))


def test_missing_model_fails_fast(tmp_path: Path) -> None:
    from eyegaze.adapters.mediapipe_landmarker import LandmarkerConfig

    with pytest.raises(FileNotFoundError, match="make models"):
        LandmarkerConfig(tmp_path / "missing.task")


def test_cli_end_to_end_with_real_landmarker(tmp_path: Path) -> None:
    """Synthetic frames contain no faces: the run must finish and account for them as no_face."""
    data = write_fixture(tmp_path / "gc")
    runner = CliRunner()
    result = runner.invoke(app, ["data", "landmarks", "--data", str(data), "--out", str(tmp_path / "lm")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["frames_detected"] == 0

    result = runner.invoke(
        app, ["data", "manifest", "--landmarks", str(tmp_path / "lm"), "--out", str(tmp_path / "m.parquet")]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["kept"] == 0
    assert report["dropped"]["no_face"] == 5
