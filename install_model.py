from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DEST = ROOT / "models" / "trained" / "gaze_model.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install a trained EyeGaze checkpoint into models/trained.")
    parser.add_argument("--src", required=True, help="Trained checkpoint from Colab/local training, usually best.dat.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Runtime model path. Default: models/trained/gaze_model.dat")
    return parser.parse_args()


def main() -> None:
    from eyegaze.model import validate_checkpoint

    args = parse_args()
    src = Path(args.src).resolve()
    dest = Path(args.dest).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")

    checkpoint = validate_checkpoint(src, "cpu")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src != dest:
        shutil.copyfile(src, dest)

    info = {
        "model_path": str(dest),
        "source_path": str(src),
        "format": checkpoint.get("format"),
        "image_size": checkpoint.get("image_size"),
        "face_size": checkpoint.get("face_size"),
        "direction_labels": list(checkpoint.get("direction_labels", [])),
        "metrics": checkpoint.get("metrics", {}),
    }
    (dest.parent / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Installed model: {dest}")
    print(f"Model info: {dest.parent / 'model_info.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
