from __future__ import annotations

import pytest

from eyegaze.contracts import (
    ContractError,
    ModelCard,
    ParityReport,
    check_model_io,
    load_model_io,
    load_spec,
    resolve_shape,
)

SPEC = load_spec()
CONTRACT = load_model_io()


def _good_io() -> tuple[dict[str, tuple[tuple[int, ...], str]], dict[str, tuple[tuple[int, ...], str]]]:
    e, f = SPEC.eye.size, SPEC.face.size
    inputs = {
        "right_eye": ((1, e, e, 3), "float32"),
        "left_eye": ((1, e, e, 3), "float32"),
        "face": ((1, f, f, 3), "float32"),
        "geometry": ((1, 17), "float32"),
    }
    outputs = {"gaze_cm": ((1, 2), "float32"), "embedding": ((1, 64), "float32")}
    return inputs, outputs


def test_contract_file_loads() -> None:
    assert CONTRACT.signature == "serving_default"
    assert [t.name for t in CONTRACT.inputs] == ["right_eye", "left_eye", "face", "geometry"]
    assert [t.name for t in CONTRACT.outputs] == ["gaze_cm", "embedding"]


def test_resolve_shape_substitutes_symbols() -> None:
    assert resolve_shape([1, "eye.size", "eye.size", 3], SPEC, 64) == (1, SPEC.eye.size, SPEC.eye.size, 3)
    assert resolve_shape([1, "embedding_dim"], SPEC, 32) == (1, 32)
    with pytest.raises(ContractError, match="Unknown"):
        resolve_shape([1, "mystery"], SPEC, 8)


def test_conforming_model_has_no_problems() -> None:
    inputs, outputs = _good_io()
    assert check_model_io(CONTRACT, inputs, outputs, SPEC, 64) == []


def test_violations_are_reported() -> None:
    inputs, outputs = _good_io()
    inputs.pop("face")
    inputs["extra"] = ((1,), "float32")
    outputs["gaze_cm"] = ((1, 3), "float32")
    outputs["embedding"] = ((1, 64), "float16")
    problems = check_model_io(CONTRACT, inputs, outputs, SPEC, 64)
    assert "missing input 'face'" in problems
    assert "unexpected input 'extra'" in problems
    assert any("gaze_cm" in p and "shape" in p for p in problems)
    assert any("embedding" in p and "dtype" in p for p in problems)


def test_model_card_roundtrip_and_strictness() -> None:
    card = ModelCard(
        model_file="m.tflite",
        model_sha256="a" * 64,
        model_bytes=10,
        precision="fp32",
        spec_file="s.json",
        spec_sha256="b" * 64,
        model_io_version=1,
        embedding_dim=64,
        parameters=1,
        source_run="base",
        source_epoch=3,
        created_utc="2026-01-01T00:00:00+00:00",
        package_version="2",
        git_commit=None,
        train_config={},
        validation={},
        test_metrics=None,
        parity=[ParityReport(source="random", samples=1, max_abs_gaze_cm=0, mean_abs_gaze_cm=0, max_abs_embedding=0)],
    )
    assert ModelCard.model_validate_json(card.model_dump_json()) == card
    with pytest.raises(ValueError, match="unknown"):
        ModelCard.model_validate({**card.model_dump(), "unknown": 1})
