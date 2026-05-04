# EyeGaze Realtime Demo

EyeGaze is a Python 3.11 project for webcam-based gaze direction demos.

The project has two main parts:

- GazeCapture training: builds a model from `appleFace`, `appleLeftEye`, `appleRightEye`, `dotInfo`, `screen`, and frame images.
- User calibration: keeps the trained base model separate from per-user calibration files.

## Structure

```text
src/eyegaze/              Runtime package: dataset, preprocessing, model, calibration, realtime
Colab1.1/                 Upload package for Google Colab training
models/trained/           Installed runtime model: gaze_model.dat
models/users/             Per-user calibration JSON files
models/landmarks/         Reserved for future landmark .dat files
storage/                  Local runtime logs/sessions
data/                     Local GazeCapture data
runs/                     Training outputs
tests/                    Unit tests

train.py                  Train locally from data/
evaluate.py               Evaluate a checkpoint
install_model.py          Install best.dat as models/trained/gaze_model.dat
calibrate_user.py         Collect calibration samples for a user
demo_realtime.py          Run webcam realtime demo
check_env.py              Check demo/train environment
export_tflite.py          Export the PyTorch checkpoint to TFLite
```

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Realtime Demo

The demo can run without a trained model by using the MediaPipe fallback predictor.

```powershell
python check_env.py --mode demo
python demo_realtime.py
```

With a trained checkpoint:

```powershell
python install_model.py --src path\to\best.dat
python check_env.py --mode demo
python calibrate_user.py --user default
python demo_realtime.py --user default
```

## Local Training

Put extracted GazeCapture subjects under `data/`, then run:

```powershell
python check_env.py --mode train
python train.py --data data --out runs/eyegaze --epochs 20 --batch-size 64
python install_model.py --src runs/eyegaze/best.dat
```

For a quick smoke test:

```powershell
python train.py --out runs/smoke --epochs 1 --batch-size 8 --limit-train 16 --limit-val 8
```

## Evaluation

```powershell
python evaluate.py --checkpoint models/trained/gaze_model.dat --split test
```

The evaluator writes JSON metrics next to the checkpoint unless `--out` is provided.

## Colab Training

Upload these items to Google Drive:

```text
archive.zip
Colab1.1/
```

Open:

```text
Colab1.1/GazeCapture_Train_Colab1_1.ipynb
```

The notebook trains and exports:

```text
best.dat
last.dat
history.json
eval_test.json
manifest.csv
```

Download `best.dat` to the laptop and install it with `install_model.py`.

## Calibration

Calibration is stored separately from the base model:

```text
models/users/default/calibration.json
```

Create or refresh calibration:

```powershell
python calibrate_user.py --user default
```

Use a separate `--user` value for each person.

## TFLite Export

Install the optional converter first:

```powershell
python -m pip install ai-edge-torch
```

Then export:

```powershell
python export_tflite.py --checkpoint models/trained/gaze_model.dat --out models/trained/gaze_model.tflite
```

The exported output tensors are ordered as:

```text
gaze_xy, direction_logits
```

## Tests

```powershell
python -m pytest -q
```

## Checkpoint Format

`models/trained/gaze_model.dat` is a PyTorch checkpoint with a deploy-style extension. Required keys:

```text
format
model_state
image_size
face_size
direction_labels
metrics
```
