# ADR-0002: Face landmarks: MediaPipe Face Landmarker on both sides

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The legacy model trained on Apple's face/eye boxes (GazeCapture JSON) but ran on boxes built from MediaPipe landmarks with ad-hoc padding: a train/serve skew that silently costs accuracy. Google ML Kit Face Detection was considered for Android.

## Decision

Use MediaPipe Face Landmarker (Tasks API, same `face_landmarker.task` model) offline on every GazeCapture frame and on-device in the app. ML Kit is rejected: it has no Python SDK (so training crops could not be reproduced), no iris landmarks and no blendshapes.

## Consequences

Training and serving crops come from the same detector and the same code path. (On Android the landmarker runs in VIDEO mode — synchronous, still tracking — instead of LIVE_STREAM; MediaPipe's Android API returns the transform column-major and the adapter converts it to the row-major matrix Python produces.) We get iris centres, a rigid head transform (pose features) and `eyeBlinkLeft/Right` blendshapes (blink confirmation) for free. Cost: a one-off, sharded, resumable landmarking pass over ~1M frames.
