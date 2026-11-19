# Architecture

EyeGaze turns a phone's front camera into a gaze pointer: **look at a cell to select it, blink to confirm**.
Everything runs on-device. Design decisions are recorded in [`adr/`](adr/).

## System overview

```mermaid
flowchart LR
    subgraph Research["research/ (Python)"]
        GC[(GazeCapture<br/>frames + JSON)] --> LM[Landmarking<br/>MediaPipe Face Landmarker<br/>sharded, resumable]
        LM --> DS[(Dataset<br/>Parquet metadata<br/>+ uint8 crop shards)]
        DS --> TR[Train · PyTorch]
        TR --> EV[Evaluate<br/>cm error per subject<br/>calibration simulation, ablations]
        TR --> EX[Export · ai-edge-torch<br/>.tflite + model_card.json]
        EX --> PAR[Parity test<br/>PyTorch ↔ LiteRT]
    end
    subgraph Contracts["contracts/"]
        SPEC[preprocess_spec.v1.json]
        GOLD[golden fixtures]
    end
    subgraph Android["android/ (Kotlin)"]
        APP[EyeGaze app<br/>Calibration · Board · Evaluation]
    end
    SPEC -.-> LM
    SPEC -.-> APP
    GOLD -.-> APP
    EX -- gaze_model.tflite --> APP
    APP -- exported session logs --> EV
```

## Research package layers (`research/src/eyegaze`)

| Layer | Contents | Rules |
|---|---|---|
| `domain/` | `spec`, `observation` (landmarks, head pose), `preprocess` (reference crop sampler), `features`, `geometry` (cm ↔ screen), `calibration` (ridge + LOO-target CV), `metrics` | numpy + stdlib only, no I/O, `mypy --strict` |
| `application/` | use cases: build dataset, train, evaluate, export; `ports.py` protocols | depends on domain + ports; dataset use cases also use the GazeCapture/landmark-store adapters directly, since those *are* the data format |
| `adapters/` | GazeCapture reader, MediaPipe landmarker, shard storage, PyTorch model, LiteRT export | the only place touching files/frameworks |
| `contracts.py`, `cli.py`, `log.py` | boundary parsing (Pydantic), Typer CLI, structlog | thin |

## Dataset pipeline

```mermaid
flowchart LR
    Z[archive.zip] -->|eyegaze data extract<br/>zip-slip safe| R[(GazeCapture root<br/>flat or nested)]
    R -->|discover + validate JSON<br/>phone · portrait · stride · finite label| S[subject frames]
    S -->|MediaPipe IMAGE mode<br/>N spawn workers| L[(landmarks/&lt;subject&gt;.parquet<br/>atomic write · fingerprint · stats)]
    L -->|eyegaze data manifest<br/>status == ok · eyes open| M[(manifest.parquet)]
    L --> REP[manifest.report.json<br/>every dropped frame accounted for]
```

Resumability: a finished subject is a Parquet file that exists; its schema metadata stores the
fingerprint (landmarker SHA-256 + filters). Resuming with different settings is refused.

## Training and evaluation

```mermaid
flowchart LR
    M[(manifest.parquet)] -->|stream in 50k-row batches| T[FrameTable<br/>6 keypoints + bbox + pose per frame<br/>~100 B/frame]
    T --> D[GazeDataset<br/>crop transforms → jitter → cv2.remap twin of the reference sampler<br/>→ photometric aug]
    D --> N[EyeGazeNet v2<br/>shared eye tower · face tower · geometry MLP<br/>→ gaze_cm, embedding]
    N -->|Euclidean cm loss, AdamW, warmup+cosine, AMP| CK[(best.pt / last.pt<br/>config · effective spec + SHA-256)]
    CK --> E1[E1 accuracy<br/>frame + subject mean, per device]
    CK --> E2[E2 calibration simulation<br/>k dots per subject, ridge vs SVR]
    E1 & E2 --> E3[ablation-table]
```

* **Model selection** on validation *subject-mean* error; early stopping with patience.
* **Resumable**: `last.pt` holds optimizer/scheduler/scaler state and history; resuming with a different
  preprocess spec is refused.
* **Ablations without re-materializing data**: crops are rendered on the fly from keypoints, so each config may
  use its own spec variant (crop size, roll alignment) or the legacy Apple boxes (E0).
* **Sampler backends**: the numpy reference (`domain/preprocess.py`) defines the contract and golden fixtures;
  training uses an OpenCV `remap` twin (`adapters/fast_sampler.py`), ~4.5x faster and within 0.05 grey levels
  (property-tested).

## Per-frame pipeline (Android)

```mermaid
sequenceDiagram
    participant Cam as CameraX (KEEP_ONLY_LATEST)
    participant LM as FaceLandmarker (LIVE_STREAM)
    participant Pre as Preprocessor
    participant M as LiteRtGazeModel
    participant Cal as RidgeCalibrator
    participant F as OneEuroFilter
    participant Sel as CellSelector + BlinkFsm
    participant UI as Board UI
    Cam->>LM: frame (640x480, timestamp)
    LM-->>Pre: 478 landmarks, blendshapes, transform matrix
    alt no face
        Pre-->>Sel: FaceLost → reset
    else face
        Pre->>M: eyes, face (NHWC float32), geometry[17]
        M-->>Cal: gaze_cm[2], embedding[D]
        Cal-->>F: screen_xy
        F-->>Sel: smoothed_xy
        LM-->>Sel: eyeBlinkLeft/Right
        Sel-->>UI: hoverCell, blinkProgress, confirmedCell?
    end
```

Latency budget: landmarker ≤ 25 ms, preprocessing ≤ 3 ms, inference ≤ 10 ms, total ≤ 50 ms (≥ 20 FPS).

## Blink confirmation state machine

```mermaid
stateDiagram-v2
    [*] --> Tracking
    Tracking --> Closing: both blink scores > T_close<br/>lock cell from pre-closure frames
    Closing --> Tracking: reopened < 300 ms (natural blink)
    Closing --> Armed: closed ≥ 400 ms (show progress)
    Armed --> Confirmed: reopened, total ≤ 1500 ms
    Armed --> Tracking: closed > 1500 ms (cancel)
    Confirmed --> Cooldown: emit Select(lockedCell)
    Cooldown --> Tracking: 500 ms
```

## Calibration flow

```mermaid
sequenceDiagram
    actor U as User
    participant C as CalibrationScreen
    participant P as Gaze pipeline
    participant R as RidgeCalibrator
    participant DB as Room
    loop 3 rounds × 9 targets (3x3, shuffled)
        C->>U: shrinking target, 1.5 s
        P-->>C: (embedding, gaze_cm, pose) from the last 1.0 s
    end
    C->>R: fit, λ by leave-one-target-out CV
    R-->>C: weights + CV error (cm)
    C->>U: 5-point validation, show measured error
    C->>DB: save CalibrationModel if error ≤ threshold
```

## Coordinate systems

* **Camera space (cm)** — model output; GazeCapture `XCam/YCam`; origin at the camera, +x right, +y up.
* **Normalized screen** — `(0,0)` top-left, `(1,1)` bottom-right; used by UI and calibration targets.
* **Frame pixels** — upright, *un-mirrored* camera frame; landmarks normalized by frame width/height.

## Data contracts

* [`contracts/preprocess_spec.v1.json`](../contracts/preprocess_spec.v1.json) — crop geometry, sampler, geometry-vector layout.
  Its SHA-256 is the model ↔ app compatibility key.
* `contracts/model_io.v1.json` — tensor names/shapes/layouts (added with the exporter).
* `contracts/golden/` — frames + landmarks + expected tensors for cross-language parity tests.
