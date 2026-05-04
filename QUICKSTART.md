# Quickstart

## Laptop Demo

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_env.py --mode demo
python demo_realtime.py
```

If you already have a trained model:

```powershell
python install_model.py --src path\to\best.dat
python calibrate_user.py --user default
python demo_realtime.py --user default
```

## Local Train Smoke Test

```powershell
python check_env.py --mode train
python train.py --out runs/smoke --epochs 1 --batch-size 8 --limit-train 16 --limit-val 8
python evaluate.py --checkpoint runs/smoke/best.dat --split val --limit 8 --out runs/smoke/eval_val.json --manifest-cache runs/smoke/manifest.csv
```

## Colab Training

Upload to Drive:

```text
archive.zip
Colab1.1/
```

Open:

```text
Colab1.1/GazeCapture_Train_Colab1_1.ipynb
```

After training, download:

```text
best.dat
```

Install it on the laptop:

```powershell
python install_model.py --src "$env:USERPROFILE\Downloads\best.dat"
```

## TFLite Export

```powershell
python -m pip install ai-edge-torch
python export_tflite.py --checkpoint models/trained/gaze_model.dat --out models/trained/gaze_model.tflite
```
