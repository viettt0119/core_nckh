"""Image decoding (OpenCV) behind the :class:`~eyegaze.application.ports.ImageReader` port."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class OpenCvImageReader:
    def read_rgb(self, path: Path) -> NDArray[np.uint8] | None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb: NDArray[np.uint8] = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return rgb
