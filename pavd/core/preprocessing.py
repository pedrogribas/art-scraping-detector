from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def image_array(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.uint8)
