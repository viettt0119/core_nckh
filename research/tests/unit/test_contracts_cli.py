from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eyegaze import __version__
from eyegaze.cli import app
from eyegaze.contracts import DEFAULT_SPEC_PATH, ContractError, load_spec, spec_hash


def test_bundled_spec_is_valid() -> None:
    spec = load_spec()
    assert spec.version == 1
    assert spec.geometry.dim == len(spec.geometry.layout) == 17
    assert spec.eye.right.outer == 33


def test_spec_hash_is_stable_sha256() -> None:
    digest = spec_hash()
    assert len(digest) == 64
    assert digest == spec_hash(DEFAULT_SPEC_PATH)


def test_invalid_spec_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    raw["eye"]["width_scale"] = 0.5
    bad = tmp_path / "spec.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ContractError, match="width_scale"):
        load_spec(bad)


def test_missing_and_malformed_spec(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="Cannot read"):
        load_spec(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ContractError, match="not valid JSON"):
        load_spec(broken)


def test_cli_version_and_spec() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout

    result = runner.invoke(app, ["spec"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["sha256"] == spec_hash()


def test_cli_spec_reports_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["spec", "--path", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
