"""Loading and fingerprinting of the cross-language contracts in ``contracts/``.

The SHA-256 of the *raw spec file bytes* is the compatibility key between a
trained model and an app build: it is written into ``model_card.json`` at export
and re-computed by the app from its bundled copy of the same file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

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
