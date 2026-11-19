"""EyeGazeNet v2 (PyTorch).

Inputs (channels-first here; the LiteRT export flips I/O to NHWC):
``right_eye``/``left_eye`` ``(B, 3, E, E)``, ``face`` ``(B, 3, F, F)`` in
``[0, 1]`` and ``geometry`` ``(B, 17)``. Outputs: ``gaze_cm`` ``(B, 2)`` and
``embedding`` ``(B, D)``.

The I/O signature is fixed even when an ablation disables the face tower or
the geometry branch, so every variant runs through the same export/app path.
Input normalization (pixels and geometry standardization) lives *inside* the
network as buffers, so the app feeds raw spec outputs and cannot get it wrong.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from eyegaze.config import GeometryFeatures, ModelConfig
from eyegaze.domain.features import GEOMETRY_DIM, POSE_SLICE

PIXEL_MEAN = 0.5
PIXEL_STD = 0.25


def _channels(base: int, width: float) -> int:
    return max(8, math.ceil(base * width / 8.0) * 8)


class InvertedResidual(nn.Module):
    """MobileNetV2 block: 1x1 expand → 3x3 depthwise → 1x1 project (+ skip when shapes match)."""

    def __init__(self, c_in: int, c_out: int, stride: int, expand: int = 4) -> None:
        super().__init__()
        hidden = c_in * expand
        self.use_skip = stride == 1 and c_in == c_out
        self.block = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.block(x)
        return x + out if self.use_skip else out


class ImageTower(nn.Module):
    """Input-size agnostic CNN (global average pooling) producing a feature vector."""

    def __init__(self, width: float, out_features: int) -> None:
        super().__init__()
        c = [_channels(v, width) for v in (24, 32, 48, 64, 96, 128)]
        self.features = nn.Sequential(
            nn.Conv2d(3, c[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU6(inplace=True),
            InvertedResidual(c[0], c[1], stride=2),
            InvertedResidual(c[1], c[1], stride=1),
            InvertedResidual(c[1], c[2], stride=2),
            InvertedResidual(c[2], c[2], stride=1),
            InvertedResidual(c[2], c[3], stride=2),
            InvertedResidual(c[3], c[4], stride=1),
            InvertedResidual(c[4], c[5], stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(c[5], out_features), nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.proj(self.features(x))
        return out


class EyeGazeNet(nn.Module):
    geometry_mask: Tensor
    geometry_mean: Tensor
    geometry_std: Tensor
    face_tower: ImageTower | None
    geometry_net: nn.Sequential | None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        eye_feat, face_feat, geom_feat = 128, 64, 64
        self.eye_tower = ImageTower(config.width, eye_feat)  # shared by both eyes (left is mirrored)
        self.face_tower = ImageTower(config.width * 0.75, face_feat) if config.use_face else None
        self.geometry_net = (
            nn.Sequential(nn.Linear(GEOMETRY_DIM, 64), nn.ReLU(inplace=True), nn.Linear(64, geom_feat), nn.ReLU())
            if config.geometry is not GeometryFeatures.NONE
            else None
        )
        fused = 2 * eye_feat + (face_feat if config.use_face else 0) + (geom_feat if self.geometry_net else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.embedding_dim),
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(config.embedding_dim, 2))

        mask = torch.ones(GEOMETRY_DIM)
        if config.geometry is GeometryFeatures.NO_POSE:
            mask[POSE_SLICE] = 0.0
        self.register_buffer("geometry_mask", mask)
        self.register_buffer("geometry_mean", torch.zeros(GEOMETRY_DIM))
        self.register_buffer("geometry_std", torch.ones(GEOMETRY_DIM))

    def set_geometry_stats(self, mean: Tensor, std: Tensor) -> None:
        """Freeze training-set statistics used to standardize the geometry vector."""
        if mean.shape != (GEOMETRY_DIM,) or std.shape != (GEOMETRY_DIM,):
            raise ValueError(f"Geometry stats must have shape ({GEOMETRY_DIM},)")
        self.geometry_mean.copy_(mean)
        self.geometry_std.copy_(torch.clamp(std, min=1e-6))

    def forward(self, right_eye: Tensor, left_eye: Tensor, face: Tensor, geometry: Tensor) -> tuple[Tensor, Tensor]:
        def norm(x: Tensor) -> Tensor:
            return (x - PIXEL_MEAN) / PIXEL_STD

        parts = [self.eye_tower(norm(right_eye)), self.eye_tower(norm(left_eye))]
        if self.face_tower is not None:
            parts.append(self.face_tower(norm(face)))
        if self.geometry_net is not None:
            g = (geometry - self.geometry_mean) / self.geometry_std * self.geometry_mask
            parts.append(self.geometry_net(g))
        embedding = self.fusion(torch.cat(parts, dim=1))
        gaze_cm = self.head(embedding)
        return gaze_cm, embedding


def euclidean_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean Euclidean distance; ``eps`` keeps the gradient finite at zero error."""
    return torch.sqrt(((pred - target) ** 2).sum(dim=1) + eps).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
