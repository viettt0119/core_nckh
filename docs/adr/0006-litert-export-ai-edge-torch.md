# ADR-0006: Export with litert-torch to LiteRT

- **Status:** Accepted (revised during M4)
- **Date:** 2026-11-19

## Context

The legacy exporter went PyTorch→ONNX→onnx2tf→TFLite, producing NHWC/NCHW confusion and an INT8 model whose
uint8 input the C++ runtime never fed. Google's `ai-edge-torch` has since been renamed **`litert-torch`**
(the old package is a deprecated shim).

## Decision

* Convert directly with `litert-torch` through a thin `NhwcGazeModel` wrapper: inputs are NHWC float32 in
  `[0, 1]` and the signature `serving_default` uses **named** tensors (`right_eye`, `left_eye`, `face`,
  `geometry` → `gaze_cm`, `embedding`). The app binds by name, never by index.
* Ship **FP32 only**. FP16 weights would need TensorFlow's converter flags (not available without TF), and the
  Android GPU delegate can already execute an FP32 model in FP16 (`precisionLossAllowed`). INT8 is deferred until
  it can be validated against real test frames (accept only if ≤ 0.1 cm worse).
* `eyegaze export` writes a bundle: `gaze_model.tflite`, the exact `preprocess_spec.json` the model was
  trained with, and `model_card.json` (SHA-256 of both files, I/O, parity, metrics, provenance).
* Export **fails** (nothing is left behind) if the model violates `contracts/model_io.v1.json` or its outputs
  differ from PyTorch by more than 1e-3 cm on random inputs and, when given, real test frames.

## Consequences

One conversion step, verified automatically on every export. The app loads the bundled spec at runtime and
refuses to start if any hash in the model card does not match, so model and preprocessing can never drift apart.
