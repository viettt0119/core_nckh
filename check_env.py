from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check EyeGaze demo/training environment.")
    parser.add_argument("--mode", choices=("demo", "train"), default="demo")
    parser.add_argument("--checkpoint", default="models/trained/gaze_model.dat")
    parser.add_argument("--data", default="data")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _import_required(name: str, hint: str) -> ModuleType | None:
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        print(f"{name}: NOT INSTALLED")
        print(hint)
        if exc.name != name:
            print(f"Import failed while loading dependency: {exc.name}")
        return None


def _camera_ok(cv2: ModuleType, camera: int) -> tuple[bool, str]:
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
    finally:
        cap.release()
    return bool(ok), str(frame.shape) if ok and frame is not None else ""


def main() -> None:
    args = parse_args()
    ok = True
    cv2 = _import_required("cv2", "Install deps: python -m pip install -r requirements.txt")
    mp = _import_required("mediapipe", "Install deps: python -m pip install -r requirements.txt")
    torch = _import_required("torch", "Install deps: python -m pip install -r requirements.txt")

    print(f"Python: {sys.version.split()[0]}")
    if cv2 is not None:
        print(f"OpenCV: {cv2.__version__}")
    else:
        ok = False
    if mp is not None:
        print(f"MediaPipe: {mp.__version__}")
    else:
        ok = False
    if torch is not None:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if args.require_cuda and not torch.cuda.is_available():
            print("CUDA is required but not available.")
            ok = False
    else:
        ok = False

    if args.mode == "demo" and cv2 is not None:
        camera_ok, detail = _camera_ok(cv2, args.camera)
        print(f"Camera {args.camera}: {'OK' if camera_ok else 'FAIL'} {detail}")
        ok = ok and camera_ok
        model_path = (ROOT / args.checkpoint).resolve()
        print(f"Runtime model: {model_path}")
        if model_path.exists() and torch is not None:
            from eyegaze.model import validate_checkpoint

            try:
                _model, ckpt = validate_checkpoint(model_path, "cpu")
                print(f"Model: OK format={ckpt['format']} image_size={ckpt['image_size']} face_size={ckpt['face_size']}")
            except RuntimeError as exc:
                print(f"Model: FAIL ({exc})")
                ok = False
        else:
            print("Model: MISSING. demo_realtime.py can still run in fallback mode; install_model.py enables trained gaze.")

    if args.mode == "train":
        from eyegaze.dataset import build_manifest

        data_root = (ROOT / args.data).resolve()
        print(f"Data root: {data_root}")
        manifest = build_manifest(data_root)
        print(f"Manifest rows: {len(manifest):,}")
        if manifest.empty:
            ok = False
        else:
            print(manifest.groupby(["split", "direction"]).size().to_string())

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
