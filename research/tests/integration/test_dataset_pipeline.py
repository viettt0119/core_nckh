from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from eyegaze.adapters import landmark_store as store
from eyegaze.adapters.gazecapture import (
    GazeCaptureError,
    SubjectFilter,
    discover_subjects,
    extract_archive,
    read_subject,
)
from eyegaze.adapters.images import OpenCvImageReader
from eyegaze.application.build_dataset import LandmarkRunSummary, build_manifest, landmark_dataset
from tests.fakes import FakeDetectorFactory
from tests.gazecapture_fixture import EXPECTED, write_fixture

FINGERPRINT = store.Fingerprint(landmarker_sha256="test", phones_only=True, portrait_only=True, frame_stride=1)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_fixture(tmp_path / "gazecapture")


def _run(data: Path, out: Path, workers: int = 1, fp: store.Fingerprint = FINGERPRINT) -> LandmarkRunSummary:
    return landmark_dataset(
        data_root=data,
        store_dir=out,
        detector_factory=FakeDetectorFactory(),
        reader=OpenCvImageReader(),
        keep=SubjectFilter(),
        fingerprint=fp,
        workers=workers,
    )


# --------------------------------------------------------------------------- reader


def test_discovers_flat_and_nested_subjects(dataset: Path) -> None:
    subjects = discover_subjects(dataset)
    assert [s.name for s in subjects] == ["00001", "00002", "00003"]
    assert subjects[2] == dataset / "00003" / "00003"


def test_discovers_subjects_below_a_wrapper_directory(tmp_path: Path) -> None:
    write_fixture(tmp_path / "wrapper" / "inner")
    assert len(discover_subjects(tmp_path / "wrapper")) == 3


def test_read_subject_filters_and_converts_apple_eye_boxes(dataset: Path) -> None:
    frames = read_subject(dataset / "00001", dataset, SubjectFilter())
    assert [r.frame_index for r in frames.records] == [0, 1, 2, 4]
    assert frames.skipped_orientation == 1
    assert frames.skipped_label == 1
    first = frames.records[0]
    assert first.frame_relpath == "00001/frames/00000.jpg"
    assert first.gaze_cm_y == -5.0
    assert first.apple_left_eye is not None
    assert (first.apple_left_eye.x, first.apple_left_eye.y) == (25.0, 25.0)  # face-relative → absolute
    assert first.apple_right_eye is None  # IsValid == 0


def test_read_subject_skips_tablets_and_applies_stride(dataset: Path) -> None:
    tablet = read_subject(dataset / "00002", dataset, SubjectFilter())
    assert tablet.records == ()
    assert tablet.skipped_device == 3
    strided = read_subject(dataset / "00001", dataset, SubjectFilter(frame_stride=2))
    assert [r.frame_index for r in strided.records] == [0, 2, 4]


def test_misaligned_json_names_the_subject(dataset: Path) -> None:
    (dataset / "00001" / "screen.json").write_text(json.dumps({"Orientation": [1]}), encoding="utf-8")
    with pytest.raises(GazeCaptureError, match="00001"):
        read_subject(dataset / "00001", dataset, SubjectFilter())


def test_extract_archive_roundtrip_and_zip_slip(tmp_path: Path, dataset: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in dataset.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(dataset.parent))
    out = extract_archive(archive, tmp_path / "extracted")
    assert len(discover_subjects(out)) == 3

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(GazeCaptureError, match="escapes"):
        extract_archive(evil, tmp_path / "evil_out")
    assert not (tmp_path / "escape.txt").exists()


# --------------------------------------------------------------------------- landmarking + manifest


def test_landmarking_writes_one_file_per_subject_and_resumes(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "landmarks"
    first = _run(dataset, out)
    assert first.subjects_processed == 3
    assert first.frames_considered == 6  # 4 + 0 + 2
    assert sorted(p.name for p in out.glob("*.parquet")) == ["00001.parquet", "00002.parquet", "00003.parquet"]
    assert not list(out.glob("*.tmp"))

    second = _run(dataset, out)
    assert second.subjects_resumed == 3
    assert second.subjects_processed == 0


def test_resume_with_different_settings_is_refused(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "landmarks"
    _run(dataset, out)
    other = store.Fingerprint(landmarker_sha256="other", phones_only=True, portrait_only=True, frame_stride=1)
    with pytest.raises(store.LandmarkStoreError, match="different settings"):
        _run(dataset, out, fp=other)


def test_manifest_accounts_for_every_frame(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "landmarks"
    _run(dataset, out)
    manifest_path = tmp_path / "manifest.parquet"
    report = build_manifest(out, manifest_path)

    assert report.frames_total == EXPECTED["frames_total"]
    assert report.kept == EXPECTED["kept"]
    assert report.dropped == EXPECTED["dropped"]
    assert report.kept_by_split == {"train": 2, "val": 1}
    assert report.subjects_by_split == {"train": 1, "val": 1}

    table = pq.read_table(manifest_path)
    assert table.num_rows == 3
    assert set(table.column("status").to_pylist()) == {"ok"}
    assert store.landmarks_array(table).shape == (3, 478, 3)
    assert store.transforms_array(table).shape == (3, 4, 4)
    saved = json.loads(manifest_path.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert saved["kept"] == 3


def test_landmarks_survive_parquet_roundtrip(dataset: Path, tmp_path: Path) -> None:
    from tests.factories import make_observation

    out = tmp_path / "landmarks"
    _run(dataset, out)
    table = store.read_store(out)
    ok = table.filter(np.array([s == "ok" for s in table.column("status").to_pylist()]))
    expected = make_observation(frame_w=48, frame_h=64).landmarks.astype(np.float32)
    np.testing.assert_allclose(store.landmarks_array(ok)[0], expected, atol=1e-6)


@pytest.mark.slow
def test_parallel_workers_match_serial_run(dataset: Path, tmp_path: Path) -> None:
    _run(dataset, tmp_path / "serial", workers=1)
    _run(dataset, tmp_path / "parallel", workers=2)
    serial = build_manifest(tmp_path / "serial", tmp_path / "s.parquet")
    parallel = build_manifest(tmp_path / "parallel", tmp_path / "p.parquet")
    assert serial.as_dict() == parallel.as_dict()
