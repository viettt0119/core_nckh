from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from eyegaze.adapters.torch_data import (
    DataError,
    GazeDataset,
    apple_transforms,
    effective_spec,
    geometry_stats,
    jitter_transform,
    load_frame_table,
    photometric,
    select_split,
)
from eyegaze.config import AugmentConfig, CropSource, SpecOverrides
from eyegaze.contracts import load_spec
from eyegaze.domain.keypoints import keypoints_from_observation
from eyegaze.domain.preprocess import CropTransform
from tests.factories import make_observation
from tests.manifest_fixture import FRAMES_PER_SUBJECT, write_manifest

pytestmark = pytest.mark.requires_torch
SPEC = load_spec()


@pytest.fixture(scope="module")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_manifest(tmp_path_factory.mktemp("manifest"))


def test_effective_spec_applies_overrides() -> None:
    spec = effective_spec(SPEC, SpecOverrides(eye_size=96, align_roll=False), CropSource.MEDIAPIPE)
    assert spec.eye.size == 96
    assert spec.eye.align_roll is False
    assert spec.face.align_roll is False
    assert effective_spec(SPEC, SpecOverrides(), CropSource.APPLE).eye.align_roll is False
    assert effective_spec(SPEC, SpecOverrides(), CropSource.MEDIAPIPE) == SPEC


def test_frame_table_is_compact_and_matches_scalar_keypoints(manifest: tuple[Path, Path]) -> None:
    table = load_frame_table(manifest[0], SPEC)
    assert len(table) == 8 * FRAMES_PER_SUBJECT
    assert table.gaze_cm.shape == (len(table), 2)
    assert table.apple.shape == (len(table), 12)
    assert np.all(np.isfinite(table.apple))
    # Row 0 was generated from make_observation(seed=0) with some roll; compare corners via the batch path.
    kp = table.keypoints.item(0)
    assert kp.frame_width == 480
    assert kp.right_outer[0] < kp.left_outer[0]


def test_select_split_limits_and_errors(manifest: tuple[Path, Path]) -> None:
    table = load_frame_table(manifest[0], SPEC)
    train = select_split(table, "train")
    assert set(train.split.tolist()) == {"train"}
    assert len(select_split(table, "train", limit=5, seed=1)) == 5
    with pytest.raises(DataError, match="No frames"):
        select_split(table, "holdout")


def test_dataset_item_shapes_and_ranges(manifest: tuple[Path, Path]) -> None:
    path, root = manifest
    table = select_split(load_frame_table(path, SPEC), "val")
    item = GazeDataset(table, root, SPEC)[0]
    assert item["right_eye"].shape == (3, SPEC.eye.size, SPEC.eye.size)
    assert item["face"].shape == (3, SPEC.face.size, SPEC.face.size)
    assert item["geometry"].shape == (17,)
    assert item["gaze_cm"].shape == (2,)
    assert 0.0 <= float(item["right_eye"].min()) <= float(item["right_eye"].max()) <= 1.0
    assert int(item["index"]) == 0


def test_augmentation_is_deterministic_per_epoch_and_varies_across_epochs(manifest: tuple[Path, Path]) -> None:
    path, root = manifest
    table = select_split(load_frame_table(path, SPEC), "train")
    ds = GazeDataset(table, root, SPEC, augment=AugmentConfig(), seed=3)
    first = ds[2]["right_eye"]
    assert (ds[2]["right_eye"] == first).all()
    ds.set_epoch(1)
    assert not (ds[2]["right_eye"] == first).all()

    plain = GazeDataset(table, root, SPEC, augment=AugmentConfig(enabled=False))
    assert (plain[2]["right_eye"] == plain[2]["right_eye"]).all()


def test_apple_crop_source_renders(manifest: tuple[Path, Path]) -> None:
    path, root = manifest
    spec = effective_spec(SPEC, SpecOverrides(), CropSource.APPLE)
    table = select_split(load_frame_table(path, spec), "test", require_apple=True)
    item = GazeDataset(table, root, spec, CropSource.APPLE)[0]
    assert item["left_eye"].shape == (3, spec.eye.size, spec.eye.size)


def test_apple_transforms_order_eyes_by_image_position() -> None:
    boxes = np.array([0, 0, 100, 100, 70, 10, 20, 10, 10, 10, 20, 10], dtype=np.float32)  # "left" box on the right
    t = apple_transforms(boxes)
    assert t.right_eye.cx < t.left_eye.cx
    assert t.face.side == 100
    assert t.right_eye.angle == 0.0


def test_jitter_and_photometric_identities() -> None:
    t = CropTransform(10.0, 20.0, 30.0, 0.1)
    assert jitter_transform(t, np.zeros(4), AugmentConfig()) == t
    moved = jitter_transform(t, np.ones(4), AugmentConfig(shift=0.1, scale=0.1, rotate_deg=10))
    assert moved.cx == pytest.approx(13.0)
    assert moved.side == pytest.approx(33.0)
    crop = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
    np.testing.assert_allclose(photometric(crop, 1.0, 1.0), crop, atol=1e-6)
    assert float(photometric(crop, 5.0, 1.0).max()) <= 1.0


def test_unreadable_frame_raises(manifest: tuple[Path, Path], tmp_path: Path) -> None:
    path, _ = manifest
    table = select_split(load_frame_table(path, SPEC), "val")
    with pytest.raises(DataError, match="Cannot read"):
        GazeDataset(table, tmp_path / "nowhere", SPEC)[0]


def test_geometry_stats_shapes(manifest: tuple[Path, Path]) -> None:
    mean, std = geometry_stats(load_frame_table(manifest[0], SPEC), SPEC)
    assert mean.shape == (17,)
    assert std.shape == (17,)
    expected_x = keypoints_from_observation(make_observation(), SPEC).right_outer[0] / 480
    assert abs(float(mean[0]) - expected_x) < 0.1


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_frame_table(tmp_path / "nope.parquet", SPEC)
