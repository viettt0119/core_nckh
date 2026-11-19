# ADR-0003: Predict gaze in camera-space centimetres

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Legacy labels were `dot / screen_size`, mixing phones, tablets and orientations in one normalized space, and checkpoints were selected on 5-way direction accuracy. Neither is comparable with the literature.

## Decision

The model predicts GazeCapture `XCam/YCam` (cm, origin at the camera, +x right, +y up). The 5-direction head is removed. Screen mapping happens at the edge: per-user calibration learns camera→screen directly; uncalibrated mode uses `DisplayMetrics` dpi and a top-centre camera assumption (`ScreenGeometry`).

## Consequences

Errors are reported in cm, per subject, comparable with iTracker/EyeMU. The model is device-independent; device geometry is absorbed by calibration.
