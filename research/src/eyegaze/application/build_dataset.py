"""Use cases: landmark the GazeCapture frames, then assemble the training manifest.

Stage 1 (``landmark_dataset``) is the expensive pass (~1M frames): parallel
over subjects, resumable, fingerprinted. Stage 2 (``build_manifest``) is cheap
and re-runnable: it applies quality filters and writes the single Parquet file
training reads, plus a JSON report accounting for every dropped frame.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from eyegaze.adapters import landmark_store as store
from eyegaze.adapters.gazecapture import AppleBox, FrameRecord, SubjectFilter, discover_subjects, read_subject
from eyegaze.application.ports import Detection, DetectorFactory, ImageReader, LandmarkDetector
from eyegaze.log import get_logger

log = get_logger(__name__)

STATUS_OK = "ok"
STATUS_NO_FACE = "no_face"
STATUS_UNREADABLE = "unreadable"
DEFAULT_MAX_BLINK = 0.5


# --------------------------------------------------------------------------- stage 1: landmarking


@dataclass(frozen=True, slots=True)
class SubjectResult:
    subject_id: str
    skipped: bool  # already complete from a previous run
    frames_considered: int
    detected: int
    seconds: float


@dataclass(frozen=True, slots=True)
class LandmarkRunSummary:
    subjects_total: int
    subjects_processed: int
    subjects_resumed: int
    frames_considered: int
    frames_detected: int
    seconds: float


def _box_values(box: AppleBox | None) -> tuple[float | None, ...]:
    return (box.x, box.y, box.w, box.h) if box else (None, None, None, None)


def _row(
    record: FrameRecord, width: int | None, height: int | None, status: str, det: Detection | None
) -> dict[str, object]:
    row: dict[str, object] = {
        "subject_id": record.subject_id,
        "frame_index": record.frame_index,
        "frame_relpath": record.frame_relpath,
        "split": record.split,
        "device": record.device,
        "orientation": record.orientation,
        "dot_num": record.dot_num,
        "gaze_cm_x": record.gaze_cm_x,
        "gaze_cm_y": record.gaze_cm_y,
        "frame_width": width,
        "frame_height": height,
        "status": status,
        "landmarks": None,
        "transform": None,
        "blink_left": None,
        "blink_right": None,
    }
    for part, box in (
        ("face", record.apple_face),
        ("left_eye", record.apple_left_eye),
        ("right_eye", record.apple_right_eye),
    ):
        for name, value in zip(("x", "y", "w", "h"), _box_values(box), strict=True):
            row[f"apple_{part}_{name}"] = value
    if det is not None:
        row["landmarks"] = det.observation.landmarks.astype("float32").reshape(-1).tolist()
        row["transform"] = det.observation.transform_matrix.astype("float32").reshape(-1).tolist()
        row["blink_left"] = det.blink_left
        row["blink_right"] = det.blink_right
    return row


def process_subject(
    subject_dir: Path,
    data_root: Path,
    store_dir: Path,
    keep: SubjectFilter,
    fingerprint: store.Fingerprint,
    detector: LandmarkDetector,
    reader: ImageReader,
    overwrite: bool = False,
) -> SubjectResult:
    subject_id = subject_dir.name
    if not overwrite and store.is_complete(store_dir, subject_id, fingerprint):
        return SubjectResult(subject_id, skipped=True, frames_considered=0, detected=0, seconds=0.0)

    started = time.perf_counter()
    frames = read_subject(subject_dir, data_root, keep)
    rows: list[dict[str, object]] = []
    detected = 0
    for record in frames.records:
        image = reader.read_rgb(data_root / record.frame_relpath)
        if image is None:
            rows.append(_row(record, None, None, STATUS_UNREADABLE, None))
            continue
        height, width = image.shape[:2]
        det = detector.detect(image)
        if det is None:
            rows.append(_row(record, width, height, STATUS_NO_FACE, None))
            continue
        detected += 1
        rows.append(_row(record, width, height, STATUS_OK, det))

    stats: dict[str, int | str] = {
        "subject_id": subject_id,
        "split": frames.split,
        "device": frames.device,
        "total_frames": frames.total_frames,
        "skipped_device": frames.skipped_device,
        "skipped_orientation": frames.skipped_orientation,
        "skipped_stride": frames.skipped_stride,
        "skipped_label": frames.skipped_label,
    }
    store.write_subject(store_dir, subject_id, rows, fingerprint, stats)
    return SubjectResult(subject_id, False, len(frames.records), detected, time.perf_counter() - started)


# Worker-process state: one native landmarker per process, created once.
_worker_detector: LandmarkDetector | None = None


def _init_worker(factory: DetectorFactory) -> None:
    global _worker_detector
    _worker_detector = factory()


def _worker_process(
    args: tuple[Path, Path, Path, SubjectFilter, store.Fingerprint, ImageReader, bool],
) -> SubjectResult:
    if _worker_detector is None:
        raise RuntimeError("Worker detector not initialised")
    subject_dir, data_root, store_dir, keep, fp, reader, overwrite = args
    return process_subject(subject_dir, data_root, store_dir, keep, fp, _worker_detector, reader, overwrite)


def landmark_dataset(
    data_root: Path,
    store_dir: Path,
    detector_factory: DetectorFactory,
    reader: ImageReader,
    keep: SubjectFilter,
    fingerprint: store.Fingerprint,
    workers: int = 1,
    overwrite: bool = False,
    on_progress: Callable[[SubjectResult, int, int], None] | None = None,
) -> LandmarkRunSummary:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    subjects = discover_subjects(data_root)
    if not subjects:
        raise FileNotFoundError(f"No GazeCapture subjects found under {data_root}")
    started = time.perf_counter()
    results: list[SubjectResult] = []

    def record(result: SubjectResult) -> None:
        results.append(result)
        if on_progress:
            on_progress(result, len(results), len(subjects))
        log.info("subject_done", **asdict(result), progress=f"{len(results)}/{len(subjects)}")

    if workers == 1:
        with detector_factory() as detector:
            for subject_dir in subjects:
                record(
                    process_subject(subject_dir, data_root, store_dir, keep, fingerprint, detector, reader, overwrite)
                )
    else:
        # spawn: MediaPipe's native graph is not fork-safe.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_init_worker, initargs=(detector_factory,)
        ) as pool:
            futures = [
                pool.submit(_worker_process, (s, data_root, store_dir, keep, fingerprint, reader, overwrite))
                for s in subjects
            ]
            for future in as_completed(futures):
                record(future.result())

    return LandmarkRunSummary(
        subjects_total=len(subjects),
        subjects_processed=sum(not r.skipped for r in results),
        subjects_resumed=sum(r.skipped for r in results),
        frames_considered=sum(r.frames_considered for r in results),
        frames_detected=sum(r.detected for r in results),
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- stage 2: manifest


@dataclass(frozen=True, slots=True)
class ManifestReport:
    fingerprint: dict[str, object]
    max_blink: float
    subjects: int
    frames_total: int
    dropped: dict[str, int]
    kept: int
    kept_by_split: dict[str, int]
    subjects_by_split: dict[str, int]
    devices: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_manifest(store_dir: Path, manifest_path: Path, max_blink: float = DEFAULT_MAX_BLINK) -> ManifestReport:
    """Filter landmarked frames into ``manifest_path`` and write ``<manifest>.report.json``."""
    if not 0.0 < max_blink <= 1.0:
        raise ValueError("max_blink must be in (0, 1]")
    files = store.subject_files(store_dir)
    table = store.read_store(store_dir)
    fingerprint = store.Fingerprint.decode((table.schema.metadata or {})[store.FINGERPRINT_KEY])

    stage1: Counter[str] = Counter()
    for path in files:
        stats = store.read_stats(path)
        for key in ("total_frames", "skipped_device", "skipped_orientation", "skipped_stride", "skipped_label"):
            value = stats[key]
            stage1[key] += int(value)

    status = table.column("status")
    ok = pc.equal(status, STATUS_OK)
    blink_max = pc.max_element_wise(table.column("blink_left"), table.column("blink_right"))
    eyes_open = pc.fill_null(pc.less(blink_max, max_blink), False)
    keep_mask = pc.and_(ok, eyes_open)
    kept = table.filter(keep_mask)

    dropped = {
        "non_phone_device": stage1["skipped_device"],
        "non_portrait": stage1["skipped_orientation"],
        "frame_stride": stage1["skipped_stride"],
        "invalid_label": stage1["skipped_label"],
        "unreadable_image": int(pc.sum(pc.equal(status, STATUS_UNREADABLE)).as_py() or 0),
        "no_face": int(pc.sum(pc.equal(status, STATUS_NO_FACE)).as_py() or 0),
        "eyes_closed": int(pc.sum(pc.and_(ok, pc.invert(eyes_open))).as_py() or 0),
    }

    def counts(column: pa.ChunkedArray) -> dict[str, int]:
        return {str(k): int(v) for k, v in sorted(Counter(column.to_pylist()).items())}

    split_subjects: Counter[str] = Counter()
    for split, _subject in {(r["split"], r["subject_id"]) for r in kept.select(["split", "subject_id"]).to_pylist()}:
        split_subjects[str(split)] += 1

    report = ManifestReport(
        fingerprint=asdict(fingerprint),
        max_blink=max_blink,
        subjects=len(files),
        frames_total=stage1["total_frames"],
        dropped=dropped,
        kept=kept.num_rows,
        kept_by_split=counts(kept.column("split")),
        subjects_by_split=dict(sorted(split_subjects.items())),
        devices=counts(kept.column("device")),
    )
    accounted = report.kept + sum(dropped.values())
    if accounted != report.frames_total:
        raise RuntimeError(f"Frame accounting mismatch: kept+dropped={accounted}, total={report.frames_total}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".parquet.tmp")
    pq.write_table(kept, tmp, compression="zstd")
    tmp.replace(manifest_path)
    report_path = manifest_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    log.info("manifest_written", path=str(manifest_path), kept=report.kept, dropped=dropped)
    return report
