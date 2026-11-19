# models/

Local, git-ignored location for exported artifacts:

```text
models/face_landmarker.task      MediaPipe Face Landmarker (downloaded by `make models`)
models/<run>/gaze_model.tflite   exported LiteRT model
models/<run>/model_card.json     version, preprocess-spec SHA-256, model SHA-256, I/O signature, metrics
```

Training checkpoints live in `runs/`. Per-user calibration lives on the device (Room database), never here.
