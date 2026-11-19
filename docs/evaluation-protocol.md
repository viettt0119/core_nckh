# Evaluation protocol

All numbers in reports come from the experiments below. Every run directory stores its config snapshot,
`metrics.json`, and the git commit, so each figure is reproducible.

## Data

* GazeCapture, **phones only, portrait only** (`orientation == 1`), official subject-disjoint train/val/test splits.
* Frames where MediaPipe finds no face or both eyes are closed (`eyeBlink* > 0.5`) are excluded and counted.
* Labels: `dotInfo.XCam/YCam` in cm.

## Metrics

* Euclidean error in cm between predicted and true gaze.
* **Frame mean** (comparable with iTracker-style papers) and **subject mean** (mean of per-subject means;
  every person counts once). Median and P90 are reported alongside.

## Experiments

| ID | Question | Procedure | Reported |
|---|---|---|---|
| E0 | What did the legacy design achieve? | Legacy input definition (Apple boxes, no roll alignment, no pose) on the v2 trainer | test error (cm) |
| E1 | How accurate is the v2 model? | Train on train, select on val subject-mean, evaluate once on test | frame/subject mean, median, P90, per-subject histogram |
| E2 | How much does calibration help? | Per test subject: k ∈ {0, 5, 9, 13} calibration targets drawn from the subject's dot positions, fit on those frames, evaluate on the remaining targets; ridge vs SVR | error vs k |
| E3 | Which design choices matter? | One change at a time vs E1: Apple vs MediaPipe crops; roll alignment on/off; face tower on/off; pose on/off; eye size 64/96/128 | Δ cm table |
| E4 | Does export preserve accuracy, and is it fast? | PyTorch vs LiteRT FP32/FP16/INT8 on 2 000 test frames; on-device latency per stage | max/mean Δ cm; ms/frame |
| E5 | Does the product work for people? | On-device study: N participants, grids 2×2, 2×3, 3×3, 3×4; random target cell; select by gaze + long blink | selection accuracy, time-to-select, false confirms/min, point error (cm) after calibration |

## E5 session procedure

1. Participant at a comfortable distance (~30–40 cm), indoor lighting, phone hand-held or on a stand (record which).
2. Calibration (3×3 × 3 rounds) followed by the 5-point validation; record the validation error.
3. For each grid size: 20 trials with random targets, 2 s rest between trials.
4. Export the session (CSV/JSON) from the app's Evaluation screen; aggregate with `eyegaze eval-sessions`.
