# ADR-0005: Multi-input CNN with gaze and embedding outputs

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

We need high accuracy within a mobile latency budget (≤10 ms inference) and features for per-user calibration.

## Decision

Shared-weight eye tower for both eyes, a face tower, and an MLP over the 17-D geometry vector (corners, iris, pose, face size), fused into a head with two outputs: `gaze_cm` (2) and `embedding` (D≈64). Loss: Euclidean distance in cm. Model selection: subject-mean validation error. Eye/face crop sizes and tower depth are decided by ablation (experiment E3).

## Consequences

Embedding output enables calibration without retraining. Architecture choices are backed by measured ablations rather than intuition.
