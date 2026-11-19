"""Checkpoint persistence for EyeGazeNet v2.

A checkpoint is self-describing: model config, the *effective* preprocess spec
(JSON text + SHA-256) and metrics travel with the weights, so evaluation and
export never have to guess how a model was trained. Loading uses
``weights_only=True`` (no arbitrary pickle execution).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import TypeAdapter, ValidationError

from eyegaze.adapters.torch_model import EyeGazeNet
from eyegaze.config import ModelConfig
from eyegaze.domain.spec import PreprocessSpec

CHECKPOINT_FORMAT = "eyegaze-v2"
_SPEC_ADAPTER: TypeAdapter[PreprocessSpec] = TypeAdapter(PreprocessSpec)


class CheckpointError(RuntimeError):
    """Missing, foreign or corrupted checkpoint."""


def spec_to_json(spec: PreprocessSpec) -> str:
    return json.dumps(_SPEC_ADAPTER.dump_python(spec, mode="json"), indent=2, sort_keys=True) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    model: EyeGazeNet
    model_config: ModelConfig
    spec: PreprocessSpec
    spec_json: str
    epoch: int
    metrics: dict[str, Any]
    train_config: dict[str, Any]
    raw: dict[str, Any]


def save_checkpoint(
    path: Path,
    model: EyeGazeNet,
    spec: PreprocessSpec,
    epoch: int,
    metrics: dict[str, Any],
    train_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    spec_json = spec_to_json(spec)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.config.model_dump(mode="json"),
        "state_dict": model.state_dict(),
        "spec_json": spec_json,
        "spec_sha256": sha256_text(spec_json),
        "epoch": epoch,
        "metrics": metrics,
        "train_config": train_config,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # never leave a truncated checkpoint behind (Colab disconnects)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for corrupt files
        raise CheckpointError(f"Cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path} is not an {CHECKPOINT_FORMAT} checkpoint")
    try:
        model_config = ModelConfig.model_validate(raw["model_config"])
        spec_json = str(raw["spec_json"])
        if sha256_text(spec_json) != raw["spec_sha256"]:
            raise CheckpointError(f"{path}: spec hash mismatch (corrupted checkpoint)")
        spec = _SPEC_ADAPTER.validate_python(json.loads(spec_json))
        model = EyeGazeNet(model_config)
        model.load_state_dict(raw["state_dict"], strict=True)
    except (KeyError, ValidationError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"{path}: invalid checkpoint content: {exc}") from exc
    model.to(device).eval()
    return LoadedCheckpoint(
        model=model,
        model_config=model_config,
        spec=spec,
        spec_json=spec_json,
        epoch=int(raw["epoch"]),
        metrics=dict(raw["metrics"]),
        train_config=dict(raw["train_config"]),
        raw=raw,
    )
