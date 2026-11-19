# Golden fixtures

Cross-language contract tests. Produced by the Python reference implementation
(`eyegaze golden --with-model`); a Python test fails if regenerating would change them.
The Android port must reproduce every expected value within `index.json → tolerances`.

## Files

| File | Content |
|---|---|
| `index.json` | case list, SHA-256 of `../preprocess_spec.v1.json`, tolerances |
| `<case>.json` | inputs and expected values for one face (see below) |
| `<case>.rgb.gz` | input frame: gzip of `H×W×3` uint8 RGB, **delta-x encoded** |
| `<case>.crops.f32.gz` | expected crops: gzip of float32 little-endian, HWC, concatenated in `order` |
| `calibration.json` | ridge-calibration inputs (`features`, `targets`, `groups`, `lambdas`) and expected `lambda`, `cv_error`, `feature_mean`, `feature_std`, `weights` (`(D+1)×2`, last row = bias), `predictions` for `probes` |
| `screen_geometry.json` | `ScreenGeometry.from_dpi` cases: expected camera position and camera-cm → normalized-screen mapping |
| `golden_model.tflite` | tiny deterministic EyeGazeNet (embedding 16) exported like production models |
| `golden_model_expected.json` | LiteRT outputs (`gaze_cm`, `embedding`) of the golden model on each case's expected crops + geometry |

## Decoding a frame

```text
encoded = gunzip(<case>.rgb.gz)            // H*W*3 bytes, row-major, RGB interleaved
for each row y, channel c:
    acc = 0
    for x in 0 until W:
        acc = (acc + encoded[y][x][c]) mod 256
        pixel[y][x][c] = acc
```

## `<case>.json`

```text
frame_width, frame_height
landmarks            478 × [x, y, z]   (x, y normalized by frame width/height)
transform_matrix     4 × 4, row-major (may include a uniform scale)
expected.keypoints   pixel-space points, face_bbox {x, y, w, h}, pose {pitch, yaw, roll} (radians)
expected.transforms  right_eye / left_eye / face: {cx, cy, side, angle}
expected.geometry    17 floats in preprocess_spec geometry.layout order
expected.crops       {file, dtype, layout, order, shapes}
```

## Tolerances

* crops: max abs error ≤ 1e-3 (values in [0, 1])
* keypoints, transforms, geometry, calibration, screen geometry: ≤ 1e-6
* golden model outputs: ≤ 1e-4
