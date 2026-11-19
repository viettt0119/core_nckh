"""LiteRT (TFLite) export and inference.

The exported graph takes **NHWC** float32 crops in ``[0, 1]`` (what the app's
Preprocessor produces, no transposes on device) and has a named signature
``serving_default``: inputs ``right_eye, left_eye, face, geometry``; outputs
``gaze_cm, embedding``. Names, not positional order, are the contract —
converters are free to reorder tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from eyegaze.adapters.torch_model import EyeGazeNet
from eyegaze.domain.features import GEOMETRY_DIM
from eyegaze.domain.spec import PreprocessSpec

SIGNATURE = "serving_default"
INPUT_NAMES = ("right_eye", "left_eye", "face", "geometry")
OUTPUT_NAMES = ("gaze_cm", "embedding")

Float32Array = NDArray[np.float32]


class LiteRtExportError(RuntimeError):
    """Conversion failed or the converted model violates the I/O contract."""


class NhwcGazeModel(nn.Module):
    """Channels-last façade over :class:`EyeGazeNet` with dict outputs (named LiteRT outputs)."""

    def __init__(self, model: EyeGazeNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, right_eye: Tensor, left_eye: Tensor, face: Tensor, geometry: Tensor) -> dict[str, Tensor]:
        gaze, embedding = self.model(
            right_eye.permute(0, 3, 1, 2), left_eye.permute(0, 3, 1, 2), face.permute(0, 3, 1, 2), geometry
        )
        return {"gaze_cm": gaze, "embedding": embedding}


def sample_inputs(spec: PreprocessSpec, seed: int = 0) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    e, f = spec.eye.size, spec.face.size
    return {
        "right_eye": torch.rand(1, e, e, 3, generator=g),
        "left_eye": torch.rand(1, e, e, 3, generator=g),
        "face": torch.rand(1, f, f, 3, generator=g),
        "geometry": torch.rand(1, GEOMETRY_DIM, generator=g),
    }


def export_litert(model: EyeGazeNet, spec: PreprocessSpec, path: Path) -> None:
    """Convert ``model`` (eval mode, CPU) to a ``.tflite`` file at ``path``."""
    import litert_torch

    wrapper = NhwcGazeModel(model.cpu().eval()).eval()
    try:
        edge = litert_torch.convert(wrapper, sample_kwargs=sample_inputs(spec))
    except Exception as exc:  # the converter raises many unrelated exception types
        raise LiteRtExportError(f"LiteRT conversion failed: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tflite.tmp")
    edge.export(str(tmp))
    tmp.replace(path)


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str


class LiteRtRunner:
    """Thin wrapper over the LiteRT interpreter's signature runner (batch size 1)."""

    def __init__(self, path: Path, num_threads: int = 1) -> None:
        from ai_edge_litert.interpreter import Interpreter

        if not path.is_file():
            raise FileNotFoundError(f"LiteRT model not found: {path}")
        self._interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
        signatures = self._interpreter.get_signature_list()
        if SIGNATURE not in signatures:
            raise LiteRtExportError(f"{path} has no {SIGNATURE!r} signature (found {list(signatures)})")
        self._runner = self._interpreter.get_signature_runner(SIGNATURE)

    def _infos(self, details: dict[str, dict[str, object]]) -> list[TensorInfo]:
        infos = []
        for name, d in sorted(details.items()):
            shape = tuple(int(v) for v in np.asarray(d["shape"]).tolist())
            infos.append(TensorInfo(name=name, shape=shape, dtype=np.dtype(d["dtype"]).name))  # type: ignore[call-overload]
        return infos

    def inputs(self) -> list[TensorInfo]:
        return self._infos(self._runner.get_input_details())

    def outputs(self) -> list[TensorInfo]:
        return self._infos(self._runner.get_output_details())

    def run(
        self, right_eye: Float32Array, left_eye: Float32Array, face: Float32Array, geometry: Float32Array
    ) -> tuple[Float32Array, Float32Array]:
        """Inputs without batch axis (HWC crops, 17-vector); returns ``(gaze_cm[2], embedding[D])``."""
        out = self._runner(
            right_eye=np.ascontiguousarray(right_eye[None], dtype=np.float32),
            left_eye=np.ascontiguousarray(left_eye[None], dtype=np.float32),
            face=np.ascontiguousarray(face[None], dtype=np.float32),
            geometry=np.ascontiguousarray(geometry[None], dtype=np.float32),
        )
        return np.asarray(out["gaze_cm"][0], dtype=np.float32), np.asarray(out["embedding"][0], dtype=np.float32)
