from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from typer.testing import CliRunner  # noqa: E402

from eyegaze.adapters.checkpoint import CheckpointError, load_checkpoint, save_checkpoint  # noqa: E402
from eyegaze.adapters.torch_model import EyeGazeNet  # noqa: E402
from eyegaze.application.evaluate import ablation_table, evaluate_run  # noqa: E402
from eyegaze.application.train import train, warmup_cosine  # noqa: E402
from eyegaze.cli import app  # noqa: E402
from eyegaze.config import ModelConfig, load_config  # noqa: E402
from eyegaze.contracts import load_spec  # noqa: E402
from tests.manifest_fixture import write_manifest  # noqa: E402

pytestmark = pytest.mark.requires_torch
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_manifest(tmp_path_factory.mktemp("data"))


def _overrides(manifest: tuple[Path, Path], out: Path, name: str, epochs: int = 2) -> list[str]:
    return [
        f"run_name={name}",
        f"output_dir={out}",
        f"data.manifest={manifest[0]}",
        f"data.frames_root={manifest[1]}",
        "data.num_workers=0",
        "device=cpu",
        f"optim.epochs={epochs}",
        "optim.batch_size=8",
        "optim.patience=10",
        "model.width=0.5",
    ]


def test_warmup_cosine_schedule() -> None:
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(10, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="positive"):
        warmup_cosine(0, 0, 0)


def test_train_resume_evaluate_and_report(manifest: tuple[Path, Path], tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "base.yaml", _overrides(manifest, tmp_path, "base", epochs=2))
    result = train(cfg)
    run = result.run_dir
    for name in ("best.pt", "last.pt", "history.json", "metrics.json", "config.yaml", "spec.json"):
        assert (run / name).is_file(), name
    assert result.epochs_run == 2
    assert result.best_val_subject_mean > 0

    with pytest.raises(FileExistsError, match="--resume"):
        train(cfg)

    longer = load_config(CONFIGS / "base.yaml", _overrides(manifest, tmp_path, "base", epochs=3))
    resumed = train(longer, resume=True)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert resumed.epochs_run == 3

    report = evaluate_run(run, manifest[0], manifest[1], split="test", calibration_k=(5,), num_workers=0,
                          device_name="cpu")  # fmt: skip
    assert report["accuracy"]["n_subjects"] == 3
    assert report["calibration"][0]["k"] == 5
    assert (run / "eval_test.json").is_file()

    # A second run (E0 legacy crops) for the ablation table.
    e0 = load_config(CONFIGS / "e0_legacy_crops.yaml", _overrides(manifest, tmp_path, "e0_legacy_crops", epochs=1))
    train(e0)
    evaluate_run(e0.run_dir, manifest[0], manifest[1], calibration_k=(5,), num_workers=0, device_name="cpu")
    table = ablation_table([run, e0.run_dir], baseline="base", k=5)
    assert "| base | mediapipe |" in table
    assert "| e0_legacy_crops | apple |" in table
    assert "+0.000" in table


def test_checkpoint_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    model = EyeGazeNet(ModelConfig(width=0.5)).eval()
    spec = load_spec()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, spec, epoch=3, metrics={"x": 1.0}, train_config={"data": {}})
    loaded = load_checkpoint(path)
    inputs = (torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64), torch.rand(1, 17))
    torch.testing.assert_close(loaded.model(*inputs)[0], model(*inputs)[0])
    assert loaded.spec == spec
    assert loaded.epoch == 3

    raw = torch.load(path, weights_only=True)
    raw["spec_json"] = raw["spec_json"].replace('"size": 64', '"size": 65', 1)
    torch.save(raw, path)
    with pytest.raises(CheckpointError, match="hash"):
        load_checkpoint(path)

    torch.save({"format": "other"}, path)
    with pytest.raises(CheckpointError, match="not an"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_cli_train_and_evaluate(manifest: tuple[Path, Path], tmp_path: Path) -> None:
    runner = CliRunner()
    args = ["train", "--config", str(CONFIGS / "e3_no_face.yaml")]
    for item in _overrides(manifest, tmp_path, "cli_run", epochs=1):
        args += ["--set", item]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    run = tmp_path / "cli_run"
    result = runner.invoke(
        app,
        ["evaluate", "--run", str(run), "--manifest", str(manifest[0]), "--frames-root", str(manifest[1]),
         "--k", "5", "--workers", "0", "--device", "cpu"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "subject_mean_cm" in json.loads(result.stdout)

    bad = runner.invoke(app, ["train", "--config", str(CONFIGS / "base.yaml"), "--set", "optim.lr=-1"])
    assert bad.exit_code == 1
