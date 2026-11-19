# ADR-0007: On-device ridge-regression calibration

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Legacy calibration fitted an affine map on 5 direction targets. EyeMU fits SVR on CNN embeddings plus eye-corner and face geometry from a multi-round 3×3 grid.

## Decision

Closed-form ridge regression from standardized `[embedding, gaze_cm, pose]` to normalized screen coordinates, bias unpenalized, λ chosen by leave-one-*target*-out CV. 3×3 grid × 3 rounds plus a 5-point validation. SVR is evaluated offline (E2) and adopted only if clearly better.

## Consequences

Tens of milliseconds in plain Kotlin, no native ML library. Leave-one-target-out CV gives an honest error estimate shown to the user.
