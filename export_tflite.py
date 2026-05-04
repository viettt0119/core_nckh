from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def make_dummy_inputs(image_size: int, face_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, 3, face_size, face_size),
        torch.randn(1, 3, image_size, image_size),
        torch.randn(1, 3, image_size, image_size),
        torch.randn(1, 25, 25),
        torch.randn(1, 12),
    )


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch model to TFLite via ONNX for Flutter/Mobile.")
    parser.add_argument("--checkpoint", default="models/trained/gaze_model.dat", help="Path to the trained PyTorch .dat model.")
    parser.add_argument("--out", default="models/trained/gaze_model.tflite", help="Path to the output .tflite file.")
    parser.add_argument("--onnx-out", default=None, help="Path to save intermediate ONNX file. Default: next to --out.")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 post-training quantization.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args()


def export_onnx(model: torch.nn.Module, dummy_inputs: tuple, onnx_path: Path, opset: int) -> None:
    """Export PyTorch model to ONNX format."""
    print(f"Exporting ONNX (opset {opset}) to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        opset_version=opset,
        input_names=["face", "left_eye", "right_eye", "face_grid", "geometry"],
        output_names=["gaze_xy", "direction_logits"],
        dynamic_axes=None,  # Fixed batch=1 for mobile
    )
    print(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")


def onnx_to_tflite(onnx_path: Path, tflite_path: Path, quantize_int8: bool = False) -> None:
    """Convert ONNX → TFLite using onnx2tf. Optionally quantize to INT8."""
    try:
        import onnx2tf
    except ImportError:
        print("Error: onnx2tf is not installed.")
        print("Please install it using: pip install onnx2tf tensorflow")
        sys.exit(1)

    print(f"Converting ONNX → TFLite: {onnx_path} → {tflite_path}...")
    # onnx2tf converts to SavedModel first, then to TFLite
    saved_model_dir = tflite_path.parent / "saved_model_tmp"

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
    )

    if quantize_int8:
        _quantize_saved_model(saved_model_dir, tflite_path)
    else:
        _convert_saved_model(saved_model_dir, tflite_path)

    # Clean up temporary SavedModel directory
    import shutil
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)


def _convert_saved_model(saved_model_dir: Path, tflite_path: Path) -> None:
    """Convert SavedModel → TFLite (float32)."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"TFLite (float32) saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")


def _quantize_saved_model(saved_model_dir: Path, tflite_path: Path) -> None:
    """Convert SavedModel → TFLite with INT8 post-training quantization."""
    import numpy as np
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for calibration (random samples are sufficient
    # for PTQ since we just need the value range)
    def representative_dataset():
        for _ in range(100):
            yield [
                np.random.randn(1, 3, 96, 96).astype(np.float32),   # face
                np.random.randn(1, 3, 64, 64).astype(np.float32),   # left_eye
                np.random.randn(1, 3, 64, 64).astype(np.float32),   # right_eye
                np.random.randn(1, 25, 25).astype(np.float32),      # face_grid
                np.random.randn(1, 12).astype(np.float32),           # geometry
            ]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32  # Keep output as float for gaze coordinates

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    print(f"TFLite (INT8 quantized) saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")


def main() -> None:
    args = parse_args()

    from eyegaze.model import load_checkpoint

    device = torch.device("cpu")
    checkpoint_path = project_path(args.checkpoint)
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    try:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    model.eval()
    image_size = int(checkpoint["image_size"])
    face_size = int(checkpoint["face_size"])

    print(f"Creating dummy inputs (Image size: {image_size}, Face size: {face_size})...")
    dummy_inputs = make_dummy_inputs(image_size, face_size)

    # Step 1: Export to ONNX
    out_path = project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx_out) if args.onnx_out else out_path.with_suffix(".onnx")
    export_onnx(model, dummy_inputs, onnx_path, args.opset)

    # Step 2: Convert ONNX → TFLite (with optional INT8 quantization)
    onnx_to_tflite(onnx_path, out_path, quantize_int8=args.quantize)

    print(f"\nSuccess! TFLite model saved to: {out_path.resolve()}")
    print("Output tensors are ordered as: gaze_xy, direction_logits")
    if args.quantize:
        print("Model has been quantized to INT8 for optimal mobile performance.")
    print("You can now integrate this file into your Flutter app using the tflite_flutter package.")


if __name__ == "__main__":
    main()
