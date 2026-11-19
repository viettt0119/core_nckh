"""Writes a tiny synthetic GazeCapture tree that exercises every filter path.

Frame pixel intensity tells :class:`tests.fakes.FakeDetector` what to return:
dark → no face, bright → eyes closed, mid-grey → a valid face.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

NO_FACE = 10
OK = 120
EYES_CLOSED = 230

# Expected accounting for the fixture below (used by several tests).
EXPECTED = {
    "frames_total": 11,
    "kept": 3,
    "dropped": {
        "non_phone_device": 3,
        "non_portrait": 1,
        "frame_stride": 0,
        "invalid_label": 1,
        "unreadable_image": 1,
        "no_face": 1,
        "eyes_closed": 1,
    },
}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _subject(
    root: Path,
    subject_id: str,
    *,
    nested: bool,
    split: str,
    device: str,
    intensities: list[int],
    orientations: list[int],
    x_cam: list[float],
    missing_frames: tuple[int, ...] = (),
    with_apple: bool = True,
) -> Path:
    base = root / subject_id / subject_id if nested else root / subject_id
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True)
    names = [f"{i:05d}.jpg" for i in range(len(intensities))]
    for i, (name, value) in enumerate(zip(names, intensities, strict=True)):
        if i in missing_frames:
            continue
        cv2.imwrite(str(frames_dir / name), np.full((64, 48, 3), value, np.uint8))
    n = len(names)
    _write_json(base / "info.json", {"Dataset": split, "DeviceName": device, "TotalFrames": n})
    _write_json(base / "frames.json", names)
    _write_json(
        base / "dotInfo.json",
        {"DotNum": list(range(n)), "XCam": x_cam, "YCam": [-5.0] * n, "XPts": [100] * n, "YPts": [200] * n},
    )
    _write_json(base / "screen.json", {"Orientation": orientations, "W": [320] * n, "H": [568] * n})
    if with_apple:

        def boxes(x: int, y: int, size: int, valid: int) -> dict[str, list[int]]:
            return {"X": [x] * n, "Y": [y] * n, "W": [size] * n, "H": [size] * n, "IsValid": [valid] * n}

        _write_json(base / "appleFace.json", boxes(10, 20, 30, 1))
        _write_json(base / "appleLeftEye.json", boxes(15, 5, 8, 1))
        _write_json(base / "appleRightEye.json", boxes(3, 5, 8, 0))
    return base


def write_fixture(root: Path) -> Path:
    _subject(
        root,
        "00001",
        nested=False,
        split="train",
        device="iPhone 6",
        intensities=[OK, OK, NO_FACE, OK, EYES_CLOSED, OK],
        orientations=[1, 1, 1, 3, 1, 1],
        x_cam=[1.0, 2.0, 3.0, 4.0, 5.0, math.nan],
    )
    _subject(
        root,
        "00002",
        nested=False,
        split="test",
        device="iPad Air 2",
        intensities=[OK, OK, OK],
        orientations=[1, 1, 1],
        x_cam=[0.0, 0.0, 0.0],
    )
    _subject(
        root,
        "00003",
        nested=True,
        split="val",
        device="iPhone 5s",
        intensities=[OK, OK],
        orientations=[1, 1],
        x_cam=[0.5, 0.5],
        missing_frames=(1,),
        with_apple=False,
    )
    return root
