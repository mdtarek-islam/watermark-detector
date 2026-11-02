"""Overfit check: 64 pairs, 30 epochs. Also times one step with cached arrays."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from watermark_detector.model import build_siamese  # noqa: E402
from watermark_detector.pairs import PairGenerator  # noqa: E402
from watermark_detector.train import split_records  # noqa: E402
from watermark_detector.watermark_synth import synthesise_dataset  # noqa: E402

raw = sorted((ROOT / "data" / "raw").glob("*.jpg"))
recs = synthesise_dataset(raw, ROOT / "data" / "labelled", seed=42)
paths, labels = zip(*split_records(recs, 42)["train"])
gen = PairGenerator(paths, labels, 64, 32, (128, 128), 42)
t = time.time()
data = list(gen.batches())
print("load s", round(time.time() - t, 1), flush=True)
a = np.concatenate([d[0][0] for d in data]); b = np.concatenate([d[0][1] for d in data]); y = np.concatenate([d[1] for d in data])
print("label mean", y.mean(), flush=True)
m = build_siamese((128, 128, 3))
t = time.time()
h = m.fit([a, b], y, batch_size=32, epochs=30, verbose=0)
print("fit s", round(time.time() - t, 1), "acc", [round(x, 2) for x in h.history["accuracy"][::5]], "loss", round(h.history["loss"][-1], 3))
