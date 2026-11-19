from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eyegaze.adapters.checkpoint import save_checkpoint  # noqa: E402
from eyegaze.adapters.litert import NhwcGazeModel, sample_inputs  # noqa: E402
from eyegaze.adapters.torch_model import EyeGazeNet  # noqa: E402
from eyegaze.config import ModelConfig  # noqa: E402
from eyegaze.contracts import CONTRACTS_DIR, load_spec  # noqa: E402

SPEC = load_spec()
GOLDEN = CONTRACTS_DIR / "golden"


def test_nhwc_wrapper_matches_channels_first_model() -> None:
    model = EyeGazeNet(ModelConfig(width=0.5)).eval()
    kw = sample_inputs(SPEC)
    out = NhwcGazeModel(model)(**kw)
    gaze, emb = model(*(kw[k].permute(0, 3, 1, 2) for k in ("right_eye", "left_eye", "face")), kw["geometry"])
    torch.testing.assert_close(out["gaze_cm"], gaze)
    torch.testing.assert_close(out["embedding"], emb)


# --------------------------------------------------------------------------- requires the [export] extra

litert = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("litert_torch", "ai_edge_litert")),
    reason="needs the [export] extra",
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("runs") / "tiny"
    torch.manual_seed(0)
    model = EyeGazeNet(ModelConfig(width=0.25, embedding_dim=16)).eval()
    save_checkpoint(run / "best.pt", model, SPEC, epoch=1, metrics={"val_subject_mean_cm": 3.0},
                    train_config={"data": {"crop_source": "mediapipe"}})  # fmt: skip
    return run


@litert
def test_export_bundle_parity_and_verification(run_dir: Path, tmp_path: Path) -> None:
    from eyegaze.application.export import CARD_FILE, MODEL_FILE, SPEC_FILE, export_run, verify_bundle
    from tests.manifest_fixture import write_manifest

    manifest, frames_root = write_manifest(tmp_path / "data")
    out = tmp_path / "bundle"
    result = export_run(run_dir, out, parity_samples=8, manifest=manifest, frames_root=frames_root)
    for name in (MODEL_FILE, SPEC_FILE, CARD_FILE):
        assert (out / name).is_file()
    card = result.card
    assert [p.source for p in card.parity] == ["random", "test_frames"]
    assert all(p.max_abs_gaze_cm < 1e-3 for p in card.parity)
    assert card.embedding_dim == 16
    assert card.source_epoch == 1
    assert verify_bundle(out) == card

    spec_text = (out / SPEC_FILE).read_text(encoding="utf-8")
    (out / SPEC_FILE).write_text(spec_text.replace('"size": 64', '"size": 96', 1), encoding="utf-8")
    from eyegaze.adapters.litert import LiteRtExportError

    with pytest.raises(LiteRtExportError, match="SHA-256"):
        verify_bundle(out)


@litert
def test_contract_violation_aborts_export(run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from eyegaze.adapters.litert import LiteRtExportError
    from eyegaze.application import export as export_module
    from eyegaze.contracts import load_model_io

    contract = load_model_io()
    broken = contract.model_copy(update={"outputs": [*contract.outputs[:1]]})  # pretends "embedding" is not allowed
    monkeypatch.setattr(export_module, "load_model_io", lambda: broken)
    with pytest.raises(LiteRtExportError, match="unexpected output 'embedding'"):
        export_module.export_run(run_dir, tmp_path / "bundle", parity_samples=2)
    assert not (tmp_path / "bundle" / export_module.MODEL_FILE).exists()


@litert
def test_cli_export_and_verify(run_dir: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from eyegaze.cli import app

    runner = CliRunner()
    out = tmp_path / "bundle"
    result = runner.invoke(app, ["export", "--run", str(run_dir), "--out", str(out), "--parity-samples", "2"])
    assert result.exit_code == 0, result.output
    # The converter prints progress lines to stdout before our JSON summary.
    summary = json.loads(result.stdout[result.stdout.index('{\n  "out_dir"') :])
    assert summary["model_bytes"] > 0
    result = runner.invoke(app, ["verify-bundle", "--dir", str(out)])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["export", "--run", str(run_dir), "--out", str(out), "--manifest", "x"]).exit_code == 1


@litert
def test_committed_golden_model_outputs_are_reproducible() -> None:
    """The Android golden test relies on these numbers; they must match the committed .tflite."""
    from eyegaze.adapters.litert import LiteRtRunner
    from eyegaze.application.golden import read_gz

    expected = json.loads((GOLDEN / "golden_model_expected.json").read_text(encoding="utf-8"))
    runner = LiteRtRunner(GOLDEN / expected["model_file"])
    for name, outputs in expected["cases"].items():
        meta = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
        flat = np.frombuffer(read_gz(GOLDEN / meta["expected"]["crops"]["file"]), dtype="<f4")
        crops, offset = [], 0
        for shape in meta["expected"]["crops"]["shapes"]:
            size = int(np.prod(shape))
            crops.append(flat[offset : offset + size].reshape(shape))
            offset += size
        gaze, emb = runner.run(*crops, np.array(meta["expected"]["geometry"], dtype=np.float32))
        np.testing.assert_allclose(gaze, outputs["gaze_cm"], atol=1e-5)
        np.testing.assert_allclose(emb, outputs["embedding"], atol=1e-5)
