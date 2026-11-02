import numpy as np
from PIL import Image

from watermark_detector.watermark_synth import (
    add_logo_watermark,
    add_text_watermark,
    apply_random_watermark,
    synthesise_dataset,
)


def _plain(size=(120, 90), colour=(40, 120, 200)):
    return Image.new("RGB", size, colour)


def test_text_watermark_changes_pixels_and_keeps_size():
    img = _plain()
    out = add_text_watermark(img, np.random.default_rng(1), text="SAMPLE")
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_logo_watermark_changes_pixels_and_keeps_size():
    img = _plain()
    out = add_logo_watermark(img, np.random.default_rng(2))
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_original_image_is_not_mutated():
    img = _plain()
    before = np.asarray(img).copy()
    apply_random_watermark(img, np.random.default_rng(3))
    assert np.array_equal(np.asarray(img), before)


def test_random_watermark_is_deterministic_for_seed():
    img = _plain()
    a = apply_random_watermark(img, np.random.default_rng(7))
    b = apply_random_watermark(img, np.random.default_rng(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_synthesise_dataset_writes_labelled_images(make_images, tmp_path):
    clean = make_images(4)
    records = synthesise_dataset(clean, tmp_path / "labelled", seed=5)
    assert len(records) == 8
    labels = [label for _, label in records]
    assert labels.count(0) == 4 and labels.count(1) == 4
    for path, label in records:
        assert path.exists()
        assert path.parent.name == ("watermarked" if label else "clean")
        Image.open(path).verify()
