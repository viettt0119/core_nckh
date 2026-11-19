"""Export use case: checkpoint → LiteRT model bundle the app can trust.

Bundle layout (``out_dir``)::

    gaze_model.tflite      FP32, NHWC, named signature (see contracts/model_io.v1.json)
    preprocess_spec.json   the exact spec the model was trained with
    model_card.json        hashes, I/O, parity, metrics, provenance

Export *fails* (no bundle written) if the converted model violates the I/O
contract or disagrees with PyTorch beyond the parity tolerance — a broken
model must never reach the device. FP16 is not shipped separately: the GPU
delegate can compute an FP32 model in FP16 at runtime, and INT8 is deferred
until it can be validated on real data (ADR-0006).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from eyegaze import __version__
from eyegaze.adapters.checkpoint import LoadedCheckpoint, load_checkpoint
from eyegaze.adapters.litert import LiteRtExportError, LiteRtRunner, export_litert
from eyegaze.adapters.torch_model import count_parameters
from eyegaze.contracts import (
    REPO_ROOT,
    ModelCard,
    ParityReport,
    check_model_io,
    load_model_io,
)
from eyegaze.domain.features import GEOMETRY_DIM
from eyegaze.log import get_logger

log = get_logger(__name__)

MODEL_FILE = "gaze_model.tflite"
SPEC_FILE = "preprocess_spec.json"
CARD_FILE = "model_card.json"
MAX_PARITY_GAZE_CM = 1e-3  # FP32 conversion must be numerically equivalent (≪ 0.1 cm product tolerance)

Sample = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def random_samples(ckpt: LoadedCheckpoint, n: int, seed: int = 0) -> Iterator[Sample]:
    """Uniform crops and geometry drawn around the training statistics stored in the model."""
    rng = np.random.default_rng(seed)
    e, f = ckpt.spec.eye.size, ckpt.spec.face.size
    mean = ckpt.model.geometry_mean.cpu().numpy()
    std = ckpt.model.geometry_std.cpu().numpy()
    for _ in range(n):
        yield (
            rng.random((e, e, 3), dtype=np.float32),
            rng.random((e, e, 3), dtype=np.float32),
            rng.random((f, f, 3), dtype=np.float32),
            (mean + std * rng.standard_normal(GEOMETRY_DIM)).astype(np.float32),
        )


def frame_samples(ckpt: LoadedCheckpoint, manifest: Path, frames_root: Path, n: int, seed: int = 0) -> Iterator[Sample]:
    """Real preprocessed test frames (E4: parity on the actual input distribution)."""
    from eyegaze.adapters.torch_data import GazeDataset, load_frame_table, select_split
    from eyegaze.config import CropSource

    crop_source = CropSource(ckpt.train_config.get("data", {}).get("crop_source", CropSource.MEDIAPIPE))
    table = select_split(
        load_frame_table(manifest, ckpt.spec), "test", limit=n, seed=seed, require_apple=crop_source is CropSource.APPLE
    )
    dataset = GazeDataset(table, frames_root, ckpt.spec, crop_source)
    for i in range(len(dataset)):
        item = dataset[i]

        def hwc(x: torch.Tensor) -> NDArray[np.float32]:
            out: NDArray[np.float32] = x.permute(1, 2, 0).numpy()
            return out

        yield hwc(item["right_eye"]), hwc(item["left_eye"]), hwc(item["face"]), item["geometry"].numpy()


@torch.no_grad()
def measure_parity(
    ckpt: LoadedCheckpoint, runner: LiteRtRunner, samples: Iterator[Sample], source: str
) -> ParityReport:
    model = ckpt.model.cpu().eval()
    gaze_diffs: list[float] = []
    emb_max = 0.0
    for right, left, face, geom in samples:
        lite_gaze, lite_emb = runner.run(right, left, face, geom)

        def chw(x: NDArray[np.float32]) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

        ref_gaze, ref_emb = model(chw(right), chw(left), chw(face), torch.from_numpy(geom)[None])
        gaze_diffs.append(float(np.abs(lite_gaze - ref_gaze[0].numpy()).max()))
        emb_max = max(emb_max, float(np.abs(lite_emb - ref_emb[0].numpy()).max()))
    if not gaze_diffs:
        raise ValueError(f"No samples for parity source {source!r}")
    return ParityReport(
        source=source,
        samples=len(gaze_diffs),
        max_abs_gaze_cm=max(gaze_diffs),
        mean_abs_gaze_cm=float(np.mean(gaze_diffs)),
        max_abs_embedding=emb_max,
    )


@dataclass(frozen=True, slots=True)
class ExportResult:
    out_dir: Path
    card: ModelCard


def export_run(
    run_dir: Path,
    out_dir: Path,
    parity_samples: int = 64,
    manifest: Path | None = None,
    frames_root: Path | None = None,
    checkpoint_name: str = "best.pt",
) -> ExportResult:
    ckpt = load_checkpoint(run_dir / checkpoint_name, "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    spec_path = out_dir / SPEC_FILE
    for stale in (model_path, spec_path, out_dir / CARD_FILE):
        stale.unlink(missing_ok=True)  # never leave a bundle whose parts come from different exports

    export_litert(ckpt.model, ckpt.spec, model_path)
    runner = LiteRtRunner(model_path)
    inputs = {t.name: (t.shape, t.dtype) for t in runner.inputs()}
    outputs = {t.name: (t.shape, t.dtype) for t in runner.outputs()}
    contract = load_model_io()
    problems = check_model_io(contract, inputs, outputs, ckpt.spec, ckpt.model_config.embedding_dim)
    if problems:
        model_path.unlink(missing_ok=True)
        raise LiteRtExportError("Exported model violates contracts/model_io.v1.json: " + "; ".join(problems))

    parity = [measure_parity(ckpt, runner, random_samples(ckpt, parity_samples), "random")]
    if manifest is not None and frames_root is not None:
        parity.append(
            measure_parity(ckpt, runner, frame_samples(ckpt, manifest, frames_root, parity_samples), "test_frames")
        )
    worst = max(p.max_abs_gaze_cm for p in parity)
    if worst > MAX_PARITY_GAZE_CM:
        model_path.unlink(missing_ok=True)
        raise LiteRtExportError(f"LiteRT/PyTorch gaze mismatch {worst:.2e} cm exceeds {MAX_PARITY_GAZE_CM:.0e} cm")

    spec_path.write_text(ckpt.spec_json, encoding="utf-8")
    eval_path = run_dir / "eval_test.json"
    test_metrics = None
    if eval_path.is_file():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        acc = evaluation["accuracy"]
        test_metrics = {
            "subject_mean_cm": acc["subject_mean"],
            "frame_mean_cm": acc["frame"]["mean"],
            "frame_median_cm": acc["frame"]["median"],
            "n_subjects": acc["n_subjects"],
            "calibration": evaluation.get("calibration", []),
        }
    card = ModelCard(
        model_file=MODEL_FILE,
        model_sha256=file_sha256(model_path),
        model_bytes=model_path.stat().st_size,
        precision="fp32",
        spec_file=SPEC_FILE,
        spec_sha256=file_sha256(spec_path),
        model_io_version=contract.version,
        embedding_dim=ckpt.model_config.embedding_dim,
        parameters=count_parameters(ckpt.model),
        source_run=run_dir.name,
        source_epoch=ckpt.epoch,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        package_version=__version__,
        git_commit=git_commit(),
        train_config=ckpt.train_config,
        validation=ckpt.metrics,
        test_metrics=test_metrics,
        parity=parity,
    )
    (out_dir / CARD_FILE).write_text(card.model_dump_json(indent=2), encoding="utf-8")
    log.info("export_done", out_dir=str(out_dir), model_bytes=card.model_bytes, parity_max_cm=worst)
    return ExportResult(out_dir, card)


def verify_bundle(out_dir: Path) -> ModelCard:
    """Re-check a bundle's hashes (what the app does at start-up), raising on any mismatch."""
    card = ModelCard.model_validate_json((out_dir / CARD_FILE).read_text(encoding="utf-8"))
    for name, expected in ((card.model_file, card.model_sha256), (card.spec_file, card.spec_sha256)):
        actual = file_sha256(out_dir / name)
        if actual != expected:
            raise LiteRtExportError(f"{name}: SHA-256 {actual} does not match model card {expected}")
    return card
