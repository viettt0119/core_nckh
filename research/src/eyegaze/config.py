"""Experiment configuration (YAML → validated Pydantic models).

A config may name a ``base`` file; its values are deep-merged underneath, so an
ablation is a few lines on top of ``configs/base.yaml``. ``--set a.b=value``
overrides from the CLI are applied last. The fully resolved config is written
into every run directory, which makes each result reproducible.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eyegaze.contracts import DEFAULT_SPEC_PATH


class ConfigError(ValueError):
    """Invalid or unreadable experiment configuration."""


class CropSource(StrEnum):
    MEDIAPIPE = "mediapipe"  # v2: landmark-anchored, roll-aligned crops
    APPLE = "apple"  # E0: legacy GazeCapture Apple boxes, axis-aligned


class GeometryFeatures(StrEnum):
    FULL = "full"
    NO_POSE = "no_pose"
    NONE = "none"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpecOverrides(_Strict):
    """Variations of the preprocess spec for ablations (E3)."""

    eye_size: int | None = Field(default=None, ge=16, le=256)
    face_size: int | None = Field(default=None, ge=16, le=256)
    align_roll: bool | None = None


class ModelConfig(_Strict):
    width: float = Field(default=1.0, gt=0.0, le=4.0)
    embedding_dim: int = Field(default=64, ge=8, le=512)
    use_face: bool = True
    geometry: GeometryFeatures = GeometryFeatures.FULL
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class AugmentConfig(_Strict):
    enabled: bool = True
    shift: float = Field(default=0.04, ge=0.0, le=0.3)  # fraction of crop side
    scale: float = Field(default=0.05, ge=0.0, le=0.3)  # relative
    rotate_deg: float = Field(default=3.0, ge=0.0, le=30.0)
    brightness: float = Field(default=0.15, ge=0.0, le=1.0)  # multiplicative gain range
    contrast: float = Field(default=0.15, ge=0.0, le=1.0)


class DataConfig(_Strict):
    manifest: Path
    frames_root: Path
    spec: Path = DEFAULT_SPEC_PATH
    spec_overrides: SpecOverrides = SpecOverrides()
    crop_source: CropSource = CropSource.MEDIAPIPE
    num_workers: int = Field(default=2, ge=0, le=64)
    limit_train: int | None = Field(default=None, ge=1)
    limit_val: int | None = Field(default=None, ge=1)
    augment: AugmentConfig = AugmentConfig()


class OptimConfig(_Strict):
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=256, ge=1)
    lr: float = Field(default=2e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    warmup_epochs: float = Field(default=1.0, ge=0.0)
    grad_clip: float = Field(default=5.0, gt=0.0)
    patience: int = Field(default=6, ge=1)  # early stopping on val subject-mean error
    amp: bool = True


class TrainConfig(_Strict):
    run_name: str = Field(pattern=r"^[A-Za-z0-9._-]{1,80}$")
    output_dir: Path = Path("runs")
    seed: int = 42
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(:\d+)?|mps)$")
    data: DataConfig
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()

    @model_validator(mode="after")
    def _apple_crops_have_no_roll(self) -> TrainConfig:
        if self.data.crop_source is CropSource.APPLE and self.data.spec_overrides.align_roll:
            raise ValueError("Apple boxes are axis-aligned; align_roll cannot be true with crop_source=apple")
        return self

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


# --------------------------------------------------------------------------- loading


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"Config inheritance cycle: {' -> '.join(str(p) for p in (*seen, resolved))}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} must be a mapping")
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    return deep_merge(_read_yaml(resolved.parent / str(base_name), (*seen, resolved)), raw)


def parse_override(item: str) -> dict[str, Any]:
    """``"optim.lr=0.001"`` → ``{"optim": {"lr": 0.001}}`` (value parsed as YAML scalar)."""
    if "=" not in item:
        raise ConfigError(f"Override must look like key.path=value, got {item!r}")
    key, raw_value = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"Empty override key in {item!r}")
    value: Any = yaml.safe_load(raw_value)
    for part in reversed(parts):
        value = {part: value}
    result: dict[str, Any] = value
    return result


def load_config(path: Path, overrides: list[str] | tuple[str, ...] = ()) -> TrainConfig:
    data = _read_yaml(path)
    for item in overrides:
        data = deep_merge(data, parse_override(item))
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}: {exc}") from exc


def dump_config(config: TrainConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
