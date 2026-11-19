# EyeGaze — gaze selection on Android

Look at a cell on the phone screen to select it, close your eyes briefly to confirm.
A research project covering the full loop: dataset preparation, model training, rigorous evaluation,
on-device deployment (LiteRT) and a user study — all on-device, no network permission.

> **Status:** v2 rewrite in progress. Milestones M0 (repo, tooling, CI, ADRs) and M1 (domain layer)
> are done; see [Roadmap](#roadmap). The legacy v1 code is preserved at commit `db82694`.

## Repository layout

```text
contracts/   Language-neutral specs binding Python and Kotlin (preprocessing, model I/O, golden fixtures)
research/    Python: GazeCapture → dataset → training → evaluation → LiteRT export
android/     Kotlin app: camera → landmarks → gaze → calibration → cell selection + blink confirm
docs/        Architecture, ADRs, evaluation protocol, model card
```

## Quick start (research)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
make setup
```

```bash
make check
```

```bash
uv --directory research run eyegaze spec
```

## Key design points

* **Train/serve parity by construction** — the same MediaPipe Face Landmarker and the same crop
  specification ([`contracts/preprocess_spec.v1.json`](contracts/preprocess_spec.v1.json)) are used for
  training data and on the phone; Kotlin is tested against golden tensors produced by the Python reference.
* **Errors in centimetres, per subject** — comparable with published work, and honest about the worst users.
* **Per-user calibration on device** — ridge regression on network embeddings, λ chosen by
  leave-one-target-out cross-validation.
* **Deliberate-blink confirmation** — long blink (400–1500 ms) via blendshapes, cell locked at blink onset.

Details: [docs/architecture.md](docs/architecture.md) · [ADRs](docs/adr/) ·
[evaluation protocol](docs/evaluation-protocol.md).

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Repo restructure, tooling, CI, ADRs | done |
| M1 | Domain: preprocessing spec + reference sampler, features, geometry, calibration, metrics | done |
| M2 | Dataset pipeline: GazeCapture reader, sharded landmarking, crop shards | next |
| M3 | Model v2, training, evaluation (E0–E3), Colab wrapper | |
| M4 | LiteRT export, parity tests, golden fixtures | |
| M5 | Android `:core:domain` | |
| M6 | Android camera + ML adapters, Board screen | |
| M7 | Calibration flow, profiles | |
| M8 | Evaluation mode + session export, laptop demo | |
| M9 | Hardening, model card, results | |

## Data and license

GazeCapture is licensed for **non-commercial research only**; it is not redistributed here.
