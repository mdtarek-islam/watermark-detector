import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_images(tmp_path):
    """Write small random RGB images and return their paths."""

    def _make(n, folder="clean", size=(48, 40)):
        out = tmp_path / folder
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        paths = []
        for i in range(n):
            arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
            p = out / f"img_{i:03d}.jpg"
            Image.fromarray(arr).save(p)
            paths.append(p)
        return paths

    return _make
