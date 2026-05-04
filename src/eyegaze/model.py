from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from eyegaze.labels import DIRECTION_LABELS

CHECKPOINT_VERSION = "eyegaze-1.1"
REQUIRED_KEYS = {"format", "model_state", "image_size", "face_size", "direction_labels"}


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageTower(nn.Module):
    def __init__(self, out_features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EyeGazeNet(nn.Module):
    """Compact iTracker-style network for laptop realtime inference.

    GazeCapture uses face, eye crops and face-grid context. This smaller model
    keeps those signals but is light enough for CPU webcam demos.
    """

    def __init__(self) -> None:
        super().__init__()
        self.eye_tower = ImageTower(64)
        self.face_tower = ImageTower(64)
        self.grid_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25 * 25, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.geometry_net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 64 + 32, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
        )
        self.gaze_xy = nn.Sequential(nn.Linear(96, 2), nn.Sigmoid())
        self.direction = nn.Linear(96, len(DIRECTION_LABELS))

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        face_grid: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gaze_xy, direction_logits) as a tuple for ONNX/TFLite export."""
        feat = torch.cat(
            [
                self.face_tower(face),
                self.eye_tower(left_eye),
                self.eye_tower(right_eye),
                self.grid_net(face_grid),
                self.geometry_net(geometry),
            ],
            dim=1,
        )
        feat = self.head(feat)
        return self.gaze_xy(feat), self.direction(feat)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    image_size: int,
    face_size: int,
    metrics: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "image_size": int(image_size),
            "face_size": int(face_size),
            "direction_labels": DIRECTION_LABELS,
            "metrics": metrics,
        },
        path,
    )


def validate_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[EyeGazeNet, dict[str, Any]]:
    """Validate and load a checkpoint. Returns (model, checkpoint_dict).

    The model is instantiated exactly once and returned to avoid redundant
    memory allocation.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid gaze model: checkpoint payload must be a dictionary.")
    missing = sorted(REQUIRED_KEYS - set(checkpoint.keys()))
    if missing:
        raise RuntimeError(f"Invalid gaze model: missing keys {missing}.")
    if checkpoint["format"] != CHECKPOINT_VERSION:
        raise RuntimeError(f"Invalid gaze model: unsupported format {checkpoint['format']!r}.")
    if tuple(checkpoint["direction_labels"]) != DIRECTION_LABELS:
        raise RuntimeError(f"Invalid gaze model: direction_labels must be {list(DIRECTION_LABELS)}.")
    resolved_device = torch.device(device) if isinstance(device, str) else device
    model = EyeGazeNet().to(resolved_device)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid gaze model: model_state does not match EyeGazeNet. {exc}") from exc
    return model, checkpoint


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[EyeGazeNet, dict[str, Any]]:
    """Load a trained checkpoint. Model is created only once (no double-init)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gaze model not found: {path}. Put the trained model in models/trained/gaze_model.dat "
            "or run install_model.py --src path\\to\\best.pt."
        )
    model, checkpoint = validate_checkpoint(path, device)
    model.eval()
    return model, checkpoint
