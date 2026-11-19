from __future__ import annotations

from pathlib import Path

import pytest

from eyegaze.config import (
    ConfigError,
    CropSource,
    GeometryFeatures,
    TrainConfig,
    deep_merge,
    dump_config,
    load_config,
    parse_override,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_every_shipped_config_is_valid() -> None:
    names = sorted(p.name for p in CONFIGS.glob("*.yaml"))
    assert "base.yaml" in names
    for path in CONFIGS.glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.run_name == path.stem


def test_inheritance_merges_nested_values() -> None:
    cfg = load_config(CONFIGS / "e0_legacy_crops.yaml")
    assert cfg.data.crop_source is CropSource.APPLE
    assert cfg.model.geometry is GeometryFeatures.NONE
    assert cfg.optim.epochs == load_config(CONFIGS / "base.yaml").optim.epochs  # inherited


def test_overrides_are_typed_and_applied_last() -> None:
    cfg = load_config(CONFIGS / "base.yaml", ["optim.lr=0.0005", "model.use_face=false", "run_name=x1"])
    assert cfg.optim.lr == pytest.approx(5e-4)
    assert cfg.model.use_face is False
    assert cfg.run_dir == Path("runs/x1")


def test_parse_override_and_deep_merge() -> None:
    assert parse_override("a.b.c=3") == {"a": {"b": {"c": 3}}}
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
    with pytest.raises(ConfigError, match=r"key\.path=value"):
        parse_override("novalue")


def test_invalid_values_and_unknown_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="lr"):
        load_config(CONFIGS / "base.yaml", ["optim.lr=-1"])
    with pytest.raises(ConfigError, match="typo"):
        load_config(CONFIGS / "base.yaml", ["optim.typo=1"])


def test_apple_crops_cannot_be_roll_aligned() -> None:
    with pytest.raises(ConfigError, match="align_roll"):
        load_config(CONFIGS / "e0_legacy_crops.yaml", ["data.spec_overrides.align_roll=true"])


def test_inheritance_cycle_is_detected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("base: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_unreadable_and_non_mapping_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_dump_roundtrip() -> None:
    import yaml

    cfg = load_config(CONFIGS / "e3_eye96.yaml")
    again = TrainConfig.model_validate(yaml.safe_load(dump_config(cfg)))
    assert again == cfg
