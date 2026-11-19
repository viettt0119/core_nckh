# ADR-0006: Export with ai-edge-torch to LiteRT

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The legacy exporter went PyTorch→ONNX→onnx2tf→TFLite, producing NHWC/NCHW confusion and an INT8 model whose uint8 input the C++ runtime never fed.

## Decision

Export directly with `ai-edge-torch` with channel-last I/O. Ship FP32 and FP16; INT8 only if its error vs FP32 is ≤0.1 cm on real representative data. Each export writes `model_card.json` (version, spec hash, model SHA-256, I/O signature, metrics).

## Consequences

One conversion step, verified by an automatic PyTorch↔LiteRT parity test. The app fails fast on spec/model mismatch.
