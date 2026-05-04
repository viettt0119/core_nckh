from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run webcam realtime EyeGaze demo.")
    parser.add_argument("--checkpoint", default="models/trained/gaze_model.dat")
    parser.add_argument("--calibration", default=None, help="Calibration JSON. Default: models/users/<user>/calibration.json")
    parser.add_argument("--user", default="default")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-fallback", action="store_true", help="Fail if no trained model is installed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eyegaze.calibration import calibration_path
    from eyegaze.realtime import run_realtime

    calibration = args.calibration or calibration_path(args.user)
    run_realtime(args.checkpoint, args.camera, calibration, allow_fallback=not args.no_fallback)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
