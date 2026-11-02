"""Load images with PIL and convert them to normalised numpy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def to_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height), convert to RGB and scale to [0, 1] float32."""
    rgb = img.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def load_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Open one image file and return an array of shape (height, width, 3)."""
    with Image.open(path) as img:
        return to_array(img, size)


def load_batch(paths: Iterable[str | Path], size: tuple[int, int]) -> np.ndarray:
    """Stack several images into one array of shape (n, height, width, 3)."""
    return np.stack([load_image(p, size) for p in paths])
