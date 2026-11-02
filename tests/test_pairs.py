import numpy as np
import pytest

from watermark_detector.pairs import PairGenerator, make_pairs


def _labels(n_clean=10, n_wm=10):
    return np.array([0] * n_clean + [1] * n_wm)


def test_pairs_are_balanced():
    pairs = make_pairs(_labels(), n_pairs=100, seed=1)
    assert len(pairs) == 100
    y = np.array([p[2] for p in pairs])
    assert (y == 1).sum() == 50 and (y == 0).sum() == 50


def test_pair_labels_match_class_agreement():
    labels = _labels()
    for i, j, y in make_pairs(labels, n_pairs=60, seed=2):
        assert i != j
        assert y == int(labels[i] == labels[j])


def test_pairs_are_deterministic():
    labels = _labels()
    assert make_pairs(labels, 40, seed=3) == make_pairs(labels, 40, seed=3)
    assert make_pairs(labels, 40, seed=3) != make_pairs(labels, 40, seed=4)


def test_single_class_rejected():
    with pytest.raises(ValueError):
        make_pairs(np.zeros(5, dtype=int), 10, seed=0)


def test_generator_yields_batches_of_32(make_images):
    paths = make_images(8)
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    gen = PairGenerator(paths, labels, n_pairs=70, batch_size=32, image_size=(16, 16), seed=0)
    assert len(gen) == 3
    batches = list(gen.batches())
    (a, b), y = batches[0]
    assert a.shape == (32, 16, 16, 3) and b.shape == (32, 16, 16, 3)
    assert y.shape == (32,) and y.dtype == np.float32
    assert [len(bt[1]) for bt in batches] == [32, 32, 6]


def test_generator_repeat_is_infinite(make_images):
    paths = make_images(4)
    gen = PairGenerator(paths, [0, 0, 1, 1], n_pairs=8, batch_size=4, image_size=(8, 8), seed=0)
    it = gen.batches(repeat=True)
    for _ in range(5):
        next(it)
