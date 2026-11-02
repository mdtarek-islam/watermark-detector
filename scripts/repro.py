"""Reproduction run: train on collected photos, then write local metrics and charts.

Output goes to docs/local/ (gitignored) and the model to the workspace models folder.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from watermark_detector.config import Config  # noqa: E402
from watermark_detector.evaluate import predict_all, roc_auc  # noqa: E402
from watermark_detector.model import load_model  # noqa: E402
from watermark_detector.pairs import PairGenerator  # noqa: E402
from watermark_detector.train import run, split_records  # noqa: E402
from watermark_detector.watermark_synth import synthesise_dataset  # noqa: E402

MODEL_DIR = Path(os.environ.get("WMD_MODEL_DIR", "D:/Machine Learning/models/watermark-detector"))
OUT = ROOT / "docs" / "local"


def main() -> None:
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    cfg = Config.from_env(model_dir=MODEL_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = run(cfg, skip_collect=True)
    wall = time.time() - t0

    # Recompute test predictions with the same seeded split to draw the ROC curve.
    raw = sorted((cfg.data_dir / "raw").glob("*.jpg"))[: cfg.dataset_size]
    records = synthesise_dataset(raw, cfg.data_dir / "labelled", seed=cfg.seed)
    test = split_records(records, cfg.seed)["test"]
    paths, labels = zip(*test)
    n_eval = max(cfg.batch_size, cfg.n_pairs * 15 // 70)
    gen = PairGenerator(paths, labels, n_eval, cfg.batch_size, cfg.image_size, cfg.seed + 2)
    y, p = predict_all(load_model(MODEL_DIR / "watermark_siamese.keras"), gen)
    order = np.argsort(-p)
    ys = y[order]
    tpr = np.concatenate([[0], np.cumsum(ys) / ys.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - ys) / (1 - ys).sum()])
    auc = roc_auc(y, p)

    h = m["history"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["loss"], label="train loss")
    ax.plot(h["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy")
    ax.legend()
    fig.savefig(OUT / "loss-curves.svg", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    fig.savefig(OUT / "roc-curve.svg", bbox_inches="tight")
    plt.close(fig)

    t = m["test"]
    prec, rec = t["precision"], t["recall"]
    out = {
        "test_accuracy": t["accuracy"],
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "test_roc_auc": auc,
        "test_pairs": t["n"],
        "threshold": t["threshold"],
        "final_train_loss": h["loss"][-1],
        "final_val_loss": h["val_loss"][-1],
        "final_train_accuracy": h["accuracy"][-1],
        "final_val_accuracy": h["val_accuracy"][-1],
        "epochs": m["epochs"],
        "batch_size": m["batch_size"],
        "image_count": m["photos"],
        "split_sizes_images": m["split_sizes"],
        "n_pairs_train": cfg.n_pairs,
        "image_size": m["image_size"],
        "seed": cfg.seed,
        "fit_seconds": m["train_seconds"],
        "wall_seconds": round(wall, 1),
        "command": "python scripts/collect.py; python scripts/repro.py",
        "data_source": "Unsplash search API, query 'landscape', small variant, synthetic text/logo watermarks",
        "hardware": f"CPU only: {platform.processor()} ({os.cpu_count()} logical cores), {platform.platform()}",
        "model_path": m["model_path"],
    }
    (OUT / "repro-metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    keys = ("test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "wall_seconds")
    print(json.dumps({k: out[k] for k in keys}))


if __name__ == "__main__":
    main()
