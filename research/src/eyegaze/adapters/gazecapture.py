"""Reader for the raw GazeCapture dataset (Krafka et al., CVPR 2016).

Each subject directory holds per-frame parallel arrays in JSON files:
``info.json`` (split, device), ``frames.json`` (file names), ``dotInfo.json``
(target in points and in camera-space cm), ``screen.json`` (orientation) and
Apple's face/eye detections (kept only for the legacy-crop ablation, E0).

The JSON is validated at this boundary; anything inconsistent fails loudly
with the subject id instead of producing silently misaligned labels.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PORTRAIT = 1
REQUIRED_FILES = ("info.json", "frames.json", "dotInfo.json", "screen.json")
APPLE_FILES = ("appleFace.json", "appleLeftEye.json", "appleRightEye.json")


class GazeCaptureError(ValueError):
    """Malformed or inconsistent GazeCapture data."""


# --------------------------------------------------------------------------- JSON schemas


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class SubjectInfo(_Lenient):
    dataset: str = Field(alias="Dataset")
    device_name: str = Field(alias="DeviceName")


class DotInfo(_Lenient):
    x_cam: list[float] = Field(alias="XCam")
    y_cam: list[float] = Field(alias="YCam")
    dot_num: list[int] = Field(alias="DotNum")

    @model_validator(mode="after")
    def _same_length(self) -> DotInfo:
        if not (len(self.x_cam) == len(self.y_cam) == len(self.dot_num)):
            raise ValueError("dotInfo arrays differ in length")
        return self


class ScreenInfo(_Lenient):
    orientation: list[int] = Field(alias="Orientation")


class AppleBoxes(_Lenient):
    x: list[float] = Field(alias="X")
    y: list[float] = Field(alias="Y")
    w: list[float] = Field(alias="W")
    h: list[float] = Field(alias="H")
    is_valid: list[int] = Field(alias="IsValid")

    @model_validator(mode="after")
    def _same_length(self) -> AppleBoxes:
        if not (len(self.x) == len(self.y) == len(self.w) == len(self.h) == len(self.is_valid)):
            raise ValueError("Apple box arrays differ in length")
        return self


# --------------------------------------------------------------------------- domain-facing records


@dataclass(frozen=True, slots=True)
class AppleBox:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class FrameRecord:
    subject_id: str
    frame_index: int
    frame_relpath: str  # relative to the dataset root, POSIX separators
    split: str
    device: str
    orientation: int
    dot_num: int
    gaze_cm_x: float
    gaze_cm_y: float
    apple_face: AppleBox | None  # absolute pixels
    apple_left_eye: AppleBox | None  # absolute pixels (GazeCapture stores them face-relative)
    apple_right_eye: AppleBox | None


@dataclass(frozen=True, slots=True)
class SubjectFilter:
    """Which frames to keep before running the (expensive) landmarker."""

    phones_only: bool = True
    portrait_only: bool = True
    frame_stride: int = 1

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")


@dataclass(frozen=True, slots=True)
class SubjectFrames:
    subject_id: str
    split: str
    device: str
    total_frames: int
    records: tuple[FrameRecord, ...]
    skipped_device: int
    skipped_orientation: int
    skipped_stride: int
    skipped_label: int


# --------------------------------------------------------------------------- archive + discovery


def extract_archive(zip_path: Path, output_dir: Path, include: Collection[str] | None = None) -> Path:
    """Extract ``zip_path`` into ``output_dir``; resumable and zip-slip safe.

    ``include`` (archive-relative POSIX paths, e.g. a manifest's ``frame_relpath``
    column) limits extraction to those files — on Colab only the frames that
    survived filtering need to be copied to the fast local disk. Files already
    present with the right size are skipped, so an interrupted run resumes.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"GazeCapture archive not found: {zip_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    wanted = set(include) if include is not None else None
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for member in members:
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise GazeCaptureError(f"Archive entry escapes the output directory: {member.filename!r}")
        for member in members:
            if member.is_dir():
                continue
            if wanted is not None and member.filename not in wanted:
                continue
            target = root / member.filename
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            zf.extract(member, root)
    return root


def _is_subject_dir(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES)


def discover_subjects(data_root: Path) -> list[Path]:
    """Find subject directories under ``data_root``.

    Supports the official flat layout (``root/00002/frames.json``), the doubly
    nested layout produced by extracting per-subject tarballs
    (``root/00002/00002/frames.json``) and one extra wrapper directory
    (``root/gazecapture/00002/...``), as commonly found in Drive copies.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    found: dict[str, Path] = {}

    def visit(base: Path, depth: int) -> None:
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            if _is_subject_dir(child):
                found.setdefault(child.name, child)
            elif depth > 0:
                visit(child, depth - 1)

    if _is_subject_dir(data_root):
        return [data_root]
    visit(data_root, depth=2)
    return [found[k] for k in sorted(found)]


# --------------------------------------------------------------------------- per-subject parsing


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise GazeCaptureError(f"Cannot read {path}: {exc}") from exc


def _load_apple(subject_dir: Path, name: str, n_frames: int) -> AppleBoxes | None:
    path = subject_dir / name
    if not path.is_file():
        return None
    boxes = AppleBoxes.model_validate(_load_json(path))
    if len(boxes.x) != n_frames:
        raise GazeCaptureError(f"{path}: {len(boxes.x)} entries for {n_frames} frames")
    return boxes


def _apple_box(boxes: AppleBoxes | None, i: int, offset: tuple[float, float] = (0.0, 0.0)) -> AppleBox | None:
    if boxes is None or boxes.is_valid[i] != 1 or boxes.w[i] <= 0 or boxes.h[i] <= 0:
        return None
    return AppleBox(boxes.x[i] + offset[0], boxes.y[i] + offset[1], boxes.w[i], boxes.h[i])


def read_subject(subject_dir: Path, data_root: Path, keep: SubjectFilter) -> SubjectFrames:
    subject_id = subject_dir.name
    try:
        info = SubjectInfo.model_validate(_load_json(subject_dir / "info.json"))
        frames = [str(f) for f in _load_json(subject_dir / "frames.json")]
        dots = DotInfo.model_validate(_load_json(subject_dir / "dotInfo.json"))
        screen = ScreenInfo.model_validate(_load_json(subject_dir / "screen.json"))
        n = len(frames)
        if len(dots.x_cam) != n or len(screen.orientation) != n:
            raise GazeCaptureError(
                f"frames={n}, dotInfo={len(dots.x_cam)}, screen={len(screen.orientation)} are not aligned"
            )
        face = _load_apple(subject_dir, "appleFace.json", n)
        left = _load_apple(subject_dir, "appleLeftEye.json", n)
        right = _load_apple(subject_dir, "appleRightEye.json", n)
    except (ValidationError, GazeCaptureError) as exc:
        raise GazeCaptureError(f"Subject {subject_id}: {exc}") from exc

    split = info.dataset.strip().lower()
    device = info.device_name.strip()
    is_phone = device.lower().startswith("iphone")
    if keep.phones_only and not is_phone:
        return SubjectFrames(subject_id, split, device, n, (), n, 0, 0, 0)

    records: list[FrameRecord] = []
    skipped_orientation = skipped_stride = skipped_label = 0
    rel_dir = subject_dir.relative_to(data_root).as_posix()
    for i, name in enumerate(frames):
        if keep.portrait_only and screen.orientation[i] != PORTRAIT:
            skipped_orientation += 1
            continue
        if i % keep.frame_stride != 0:
            skipped_stride += 1
            continue
        gx, gy = dots.x_cam[i], dots.y_cam[i]
        if not (math.isfinite(gx) and math.isfinite(gy)):
            skipped_label += 1
            continue
        face_box = _apple_box(face, i)
        offset = (face_box.x, face_box.y) if face_box else (0.0, 0.0)
        records.append(
            FrameRecord(
                subject_id=subject_id,
                frame_index=i,
                frame_relpath=f"{rel_dir}/frames/{name}" if rel_dir != "." else f"frames/{name}",
                split=split,
                device=device,
                orientation=screen.orientation[i],
                dot_num=dots.dot_num[i],
                gaze_cm_x=gx,
                gaze_cm_y=gy,
                apple_face=face_box,
                apple_left_eye=_apple_box(left, i, offset) if face_box else None,
                apple_right_eye=_apple_box(right, i, offset) if face_box else None,
            )
        )
    return SubjectFrames(
        subject_id=subject_id,
        split=split,
        device=device,
        total_frames=n,
        records=tuple(records),
        skipped_device=0,
        skipped_orientation=skipped_orientation,
        skipped_stride=skipped_stride,
        skipped_label=skipped_label,
    )


def iter_subjects(data_root: Path, keep: SubjectFilter) -> Iterator[SubjectFrames]:
    for subject_dir in discover_subjects(data_root):
        yield read_subject(subject_dir, data_root, keep)
