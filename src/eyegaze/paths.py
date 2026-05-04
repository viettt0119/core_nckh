from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
TRAINED_MODEL_DIR = MODEL_DIR / "trained"
USER_MODEL_DIR = MODEL_DIR / "users"
LANDMARK_MODEL_DIR = MODEL_DIR / "landmarks"
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"

DEFAULT_MODEL_PATH = TRAINED_MODEL_DIR / "gaze_model.dat"
DEFAULT_USER_ID = "default"
