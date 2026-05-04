# Colab1.1 Training Package

Upload this folder's notebook to Google Colab together with your GazeCapture archive:

```text
archive.zip
Colab1.1/GazeCapture_Train_Colab1_1.ipynb
```

The notebook is self-contained. It extracts `archive.zip`, builds the GazeCapture manifest, trains the EyeGaze model, and writes:

```text
best.dat
last.dat
history.json
eval_test.json
manifest.csv
```

After training, download or copy `best.dat` to the laptop and install it:

```powershell
python install_model.py --src path\to\best.dat
```
