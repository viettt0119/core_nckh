# ADR-0004: Preprocessing as a portable, tested specification

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

`cv2.resize(INTER_AREA)` and Android Bitmap scaling are not reproducible bit-for-bit on the other platform.

## Decision

Crops are rotated squares anchored on landmarks (eye: corner axis × `width_scale`, roll-aligned; left eye mirrored; face: landmark bbox × `scale`). Sampling is an explicit bilinear sampler with fixed supersampling, pixel centres at +0.5 and replicate borders, specified in `contracts/preprocess_spec.v1.json`. `research/src/eyegaze/domain/preprocess.py` is the reference implementation; Android re-implements it and must match golden fixtures to ≤1e-3.

## Consequences

Parity is verifiable in CI. For training throughput an OpenCV `remap` twin of the sampler is used; a property test bounds its deviation from the reference to < 0.05 grey levels, so the train/serve gap stays far below JPEG noise. Roll alignment removes head-tilt variance from eye crops. The spec file's SHA-256 links a model to compatible app builds.
