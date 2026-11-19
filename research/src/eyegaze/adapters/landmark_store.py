"""Parquet storage of per-frame landmarking results.

One file per subject makes the landmarking pass resumable (a finished subject
is a file that exists) and parallel (subjects are independent). Files are
written to a temporary name and atomically renamed, so a crash never leaves a
half-written file that would be mistaken for a finished subject.

Every file carries a *fingerprint* (landmarker model hash + filter settings) in
its schema metadata; resuming with different settings is refused instead of
silently mixing incompatible rows.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from eyegaze.domain.spec import MEDIAPIPE_LANDMARK_COUNT

SCHEMA_VERSION = 1
FINGERPRINT_KEY = b"eyegaze.fingerprint"
STATS_KEY = b"eyegaze.stats"
LANDMARK_VALUES = MEDIAPIPE_LANDMARK_COUNT * 3

_BOX = ("x", "y", "w", "h")
_APPLE_COLUMNS = [f"apple_{part}_{c}" for part in ("face", "left_eye", "right_eye") for c in _BOX]

SCHEMA = pa.schema(
    [
        ("subject_id", pa.string()),
        ("frame_index", pa.int32()),
        ("frame_relpath", pa.string()),
        ("split", pa.string()),
        ("device", pa.string()),
        ("orientation", pa.int8()),
        ("dot_num", pa.int16()),
        ("gaze_cm_x", pa.float32()),
        ("gaze_cm_y", pa.float32()),
        *[(name, pa.float32()) for name in _APPLE_COLUMNS],
        ("frame_width", pa.int32()),
        ("frame_height", pa.int32()),
        ("status", pa.string()),  # ok | no_face | unreadable
        ("landmarks", pa.list_(pa.float32(), LANDMARK_VALUES)),
        ("transform", pa.list_(pa.float32(), 16)),
        ("blink_left", pa.float32()),
        ("blink_right", pa.float32()),
    ]
)
COLUMNS = tuple(SCHEMA.names)


class LandmarkStoreError(RuntimeError):
    """Store content is incompatible with the requested operation."""


@dataclass(frozen=True, slots=True)
class Fingerprint:
    landmarker_sha256: str
    phones_only: bool
    portrait_only: bool
    frame_stride: int
    schema_version: int = SCHEMA_VERSION

    def encode(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True).encode("utf-8")

    @classmethod
    def decode(cls, raw: bytes) -> Fingerprint:
        return cls(**json.loads(raw.decode("utf-8")))


def subject_path(store_dir: Path, subject_id: str) -> Path:
    return store_dir / f"{subject_id}.parquet"


def read_fingerprint(path: Path) -> Fingerprint:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(FINGERPRINT_KEY)
    if raw is None:
        raise LandmarkStoreError(f"{path} has no eyegaze fingerprint")
    return Fingerprint.decode(raw)


def is_complete(store_dir: Path, subject_id: str, expected: Fingerprint) -> bool:
    """True if the subject was already processed with identical settings; raises on mismatch."""
    path = subject_path(store_dir, subject_id)
    if not path.is_file():
        return False
    found = read_fingerprint(path)
    if found != expected:
        raise LandmarkStoreError(
            f"{path} was produced with different settings ({found}); expected {expected}. "
            "Use a new output directory or --overwrite."
        )
    return True


def write_subject(
    store_dir: Path,
    subject_id: str,
    rows: Sequence[Mapping[str, object]],
    fp: Fingerprint,
    stats: Mapping[str, int | str],
) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[object]] = {name: [] for name in COLUMNS}
    for row in rows:
        missing = set(COLUMNS) - row.keys()
        if missing:
            raise LandmarkStoreError(f"Row for subject {subject_id} lacks columns {sorted(missing)}")
        for name in COLUMNS:
            columns[name].append(row[name])
    schema = SCHEMA.with_metadata(
        {FINGERPRINT_KEY: fp.encode(), STATS_KEY: json.dumps(dict(stats), sort_keys=True).encode("utf-8")}
    )
    table = pa.Table.from_pydict(columns, schema=schema)
    final = subject_path(store_dir, subject_id)
    tmp = final.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, final)
    return final


def read_stats(path: Path) -> dict[str, int | str]:
    """Per-subject counters written alongside the rows (frames skipped before landmarking etc.)."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(STATS_KEY)
    if raw is None:
        raise LandmarkStoreError(f"{path} has no eyegaze stats")
    stats: dict[str, int | str] = json.loads(raw.decode("utf-8"))
    return stats


def subject_files(store_dir: Path) -> list[Path]:
    return sorted(store_dir.glob("*.parquet"))


def read_store(store_dir: Path, columns: Sequence[str] | None = None) -> pa.Table:
    files = subject_files(store_dir)
    if not files:
        raise LandmarkStoreError(f"No subject files in {store_dir}")
    fingerprints = {read_fingerprint(f) for f in files}
    if len(fingerprints) != 1:
        raise LandmarkStoreError(f"{store_dir} mixes {len(fingerprints)} different fingerprints")
    tables = [pq.read_table(f, columns=list(columns) if columns else None) for f in files]
    return pa.concat_tables(tables).replace_schema_metadata({FINGERPRINT_KEY: fingerprints.pop().encode()})


def landmarks_array(table: pa.Table) -> NDArray[np.float32]:
    """``(N, 478, 3)`` float32 view of the ``landmarks`` column (rows must be non-null)."""
    col = table.column("landmarks").combine_chunks()
    if col.null_count:
        raise LandmarkStoreError("landmarks column contains nulls; filter status == 'ok' first")
    flat: NDArray[np.float32] = col.values.to_numpy(zero_copy_only=False)
    return flat.reshape(len(table), MEDIAPIPE_LANDMARK_COUNT, 3)


def transforms_array(table: pa.Table) -> NDArray[np.float32]:
    col = table.column("transform").combine_chunks()
    if col.null_count:
        raise LandmarkStoreError("transform column contains nulls; filter status == 'ok' first")
    flat: NDArray[np.float32] = col.values.to_numpy(zero_copy_only=False)
    return flat.reshape(len(table), 4, 4)
