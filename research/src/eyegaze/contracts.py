"""Loading and fingerprinting of the cross-language contracts in ``contracts/``.

The SHA-256 of the *raw spec file bytes* is the compatibility key between a
trained model and an app build: it is written into ``model_card.json`` at export
and re-computed by the app from its bundled copy of the same file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from eyegaze.domain.spec import PreprocessSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_DIR = REPO_ROOT / "contracts"
DEFAULT_SPEC_PATH = CONTRACTS_DIR / "preprocess_spec.v1.json"

_SPEC_ADAPTER: TypeAdapter[PreprocessSpec] = TypeAdapter(PreprocessSpec)


class ContractError(ValueError):
    """A contract file is missing, malformed, or violates its schema."""


def spec_hash(path: Path = DEFAULT_SPEC_PATH) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f"Cannot read spec {path}: {exc}") from exc


def load_spec(path: Path = DEFAULT_SPEC_PATH) -> PreprocessSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Cannot read spec {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"Spec {path} is not valid JSON: {exc}") from exc
    try:
        return _SPEC_ADAPTER.validate_python(raw, strict=False)
    except (ValidationError, ValueError) as exc:
        raise ContractError(f"Spec {path} is invalid: {exc}") from exc


# --------------------------------------------------------------------------- model I/O contract

DEFAULT_MODEL_IO_PATH = CONTRACTS_DIR / "model_io.v1.json"


class TensorContract(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    shape: list[int | str]
    dtype: str


class ModelIoContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    signature: str
    inputs: list[TensorContract]
    outputs: list[TensorContract]


def load_model_io(path: Path = DEFAULT_MODEL_IO_PATH) -> ModelIoContract:
    try:
        return ModelIoContract.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"Cannot read model I/O contract {path}: {exc}") from exc
    except ValidationError as exc:
        raise ContractError(f"Model I/O contract {path} is invalid: {exc}") from exc


def resolve_shape(shape: list[int | str], spec: PreprocessSpec, embedding_dim: int) -> tuple[int, ...]:
    """Replace symbolic dims (``eye.size``, ``face.size``, ``geometry.dim``, ``embedding_dim``) by values."""
    symbols = {
        "eye.size": spec.eye.size,
        "face.size": spec.face.size,
        "geometry.dim": spec.geometry.dim,
        "embedding_dim": embedding_dim,
    }
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim in symbols:
            resolved.append(symbols[dim])
        else:
            raise ContractError(f"Unknown symbolic dimension {dim!r}")
    return tuple(resolved)


def check_model_io(
    contract: ModelIoContract,
    inputs: dict[str, tuple[tuple[int, ...], str]],
    outputs: dict[str, tuple[tuple[int, ...], str]],
    spec: PreprocessSpec,
    embedding_dim: int,
) -> list[str]:
    """Return human-readable violations (empty list = the model honours the contract)."""
    problems: list[str] = []
    for kind, expected, actual in (("input", contract.inputs, inputs), ("output", contract.outputs, outputs)):
        names = {t.name for t in expected}
        for extra in sorted(set(actual) - names):
            problems.append(f"unexpected {kind} {extra!r}")
        for tensor in expected:
            if tensor.name not in actual:
                problems.append(f"missing {kind} {tensor.name!r}")
                continue
            shape, dtype = actual[tensor.name]
            want = resolve_shape(tensor.shape, spec, embedding_dim)
            if shape != want:
                problems.append(f"{kind} {tensor.name!r}: shape {shape} != {want}")
            if dtype != tensor.dtype:
                problems.append(f"{kind} {tensor.name!r}: dtype {dtype} != {tensor.dtype}")
    return problems


# --------------------------------------------------------------------------- model card


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str  # "random" | "test_frames"
    samples: int
    max_abs_gaze_cm: float
    mean_abs_gaze_cm: float
    max_abs_embedding: float


class ModelCard(BaseModel):
    """Everything the app (and a reviewer) needs to trust a shipped model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: str = "eyegaze-model-card-v1"
    model_file: str
    model_sha256: str
    model_bytes: int
    precision: str
    spec_file: str
    spec_sha256: str
    model_io_version: int
    embedding_dim: int
    parameters: int
    source_run: str
    source_epoch: int
    created_utc: str
    package_version: str
    git_commit: str | None
    train_config: dict[str, object]
    validation: dict[str, object]
    test_metrics: dict[str, object] | None
    parity: list[ParityReport]
