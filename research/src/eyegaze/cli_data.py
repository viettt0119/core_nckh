"""``eyegaze data ...``: dataset preparation commands (needs the ``[data]`` extra).

Heavy imports happen inside commands so the base CLI works without them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from eyegaze.contracts import REPO_ROOT

data_app = typer.Typer(no_args_is_help=True, help="Prepare GazeCapture: extract, landmark, build manifest.")

DEFAULT_LANDMARKER = REPO_ROOT / "models" / "face_landmarker.task"


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


@data_app.command("extract")
def extract(
    zip_path: Annotated[Path, typer.Option("--zip", help="GazeCapture archive")],
    out: Annotated[Path, typer.Option(help="Extraction directory")],
) -> None:
    """Safely extract a GazeCapture zip archive."""
    from eyegaze.adapters.gazecapture import GazeCaptureError, extract_archive

    try:
        root = extract_archive(zip_path, out)
    except (FileNotFoundError, GazeCaptureError, OSError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(str(root))


@data_app.command("landmarks")
def landmarks(
    data: Annotated[Path, typer.Option(help="GazeCapture root (extracted)")],
    out: Annotated[Path, typer.Option(help="Per-subject Parquet output directory")],
    model: Annotated[Path, typer.Option(help="face_landmarker.task")] = DEFAULT_LANDMARKER,
    workers: Annotated[int, typer.Option(min=1, help="Parallel processes")] = 1,
    frame_stride: Annotated[int, typer.Option(min=1, help="Keep every N-th frame")] = 1,
    all_devices: Annotated[bool, typer.Option(help="Include tablets/iPods")] = False,
    all_orientations: Annotated[bool, typer.Option(help="Include landscape frames")] = False,
    overwrite: Annotated[bool, typer.Option(help="Recompute finished subjects")] = False,
) -> None:
    """Run MediaPipe Face Landmarker over every kept frame (resumable)."""
    from eyegaze.adapters.gazecapture import GazeCaptureError, SubjectFilter
    from eyegaze.adapters.images import OpenCvImageReader
    from eyegaze.adapters.landmark_store import Fingerprint, LandmarkStoreError
    from eyegaze.adapters.mediapipe_landmarker import LandmarkerConfig, MediaPipeLandmarkerFactory, file_sha256
    from eyegaze.application.build_dataset import landmark_dataset

    try:
        config = LandmarkerConfig(model_path=model)
        keep = SubjectFilter(phones_only=not all_devices, portrait_only=not all_orientations, frame_stride=frame_stride)
        fingerprint = Fingerprint(
            landmarker_sha256=file_sha256(model),
            phones_only=keep.phones_only,
            portrait_only=keep.portrait_only,
            frame_stride=keep.frame_stride,
        )
        summary = landmark_dataset(
            data_root=data,
            store_dir=out,
            detector_factory=MediaPipeLandmarkerFactory(config),
            reader=OpenCvImageReader(),
            keep=keep,
            fingerprint=fingerprint,
            workers=workers,
            overwrite=overwrite,
        )
    except (FileNotFoundError, GazeCaptureError, LandmarkStoreError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(json.dumps(asdict(summary), indent=2))


@data_app.command("manifest")
def manifest(
    landmarks_dir: Annotated[Path, typer.Option("--landmarks", help="Output of `eyegaze data landmarks`")],
    out: Annotated[Path, typer.Option(help="Manifest Parquet path")],
    max_blink: Annotated[float, typer.Option(help="Drop frames whose eyeBlink score reaches this")] = 0.5,
) -> None:
    """Filter landmarked frames into the training manifest and write a drop report."""
    from eyegaze.adapters.landmark_store import LandmarkStoreError
    from eyegaze.application.build_dataset import build_manifest

    try:
        report = build_manifest(landmarks_dir, out, max_blink=max_blink)
    except (LandmarkStoreError, ValueError, RuntimeError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(json.dumps(report.as_dict(), indent=2))
