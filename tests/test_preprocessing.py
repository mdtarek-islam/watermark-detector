import numpy as np
import pytest
from PIL import Image

from watermark_detector.preprocessing import load_batch, load_image, to_array


def test_to_array_shape_dtype_and_range():
    img = Image.new("RGB", (50, 30), (255, 0, 128))
    arr = to_array(img, (64, 64))
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert arr[0, 0, 0] == pytest.approx(1.0)
    assert arr[0, 0, 1] == pytest.approx(0.0)


def test_greyscale_converted_to_rgb():
    img = Image.new("L", (20, 20), 100)
    assert to_array(img, (16, 16)).shape == (16, 16, 3)


def test_image_size_is_width_height(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(p)
    assert load_image(p, (32, 24)).shape == (24, 32, 3)


def test_load_batch_stacks(make_images):
    paths = make_images(3)
    batch = load_batch(paths, (16, 16))
    assert batch.shape == (3, 16, 16, 3)
