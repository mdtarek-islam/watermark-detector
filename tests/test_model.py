import numpy as np

from watermark_detector.model import build_siamese


def test_output_shape_and_range():
    model = build_siamese((32, 32, 3))
    a = np.random.rand(4, 32, 32, 3).astype("float32")
    b = np.random.rand(4, 32, 32, 3).astype("float32")
    out = model.predict([a, b], verbose=0)
    assert out.shape == (4, 1)
    assert np.all((out >= 0) & (out <= 1))


def test_l1_head_is_symmetric():
    model = build_siamese((32, 32, 3))
    a = np.random.rand(2, 32, 32, 3).astype("float32")
    b = np.random.rand(2, 32, 32, 3).astype("float32")
    ab = model.predict([a, b], verbose=0)
    ba = model.predict([b, a], verbose=0)
    np.testing.assert_allclose(ab, ba, rtol=1e-5, atol=1e-6)
