from __future__ import annotations

import torch

from core_ai.data.synthetic import SyntheticGazeDataset


def test_synthetic_dataset_shapes() -> None:
    ds = SyntheticGazeDataset(n=4, seed=1)
    item = ds[0]
    assert item["image"].shape[0] == 3
    assert item["image"].shape[1] == item["image"].shape[2]
    assert item["landmarks_xy"].ndim == 2 and item["landmarks_xy"].shape[1] == 2
    assert item["gaze_pitch_yaw"].shape == (2,)

    batch = [ds[i] for i in range(4)]
    images = torch.stack([b["image"] for b in batch], dim=0)
    assert images.shape[0] == 4

