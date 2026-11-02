"""Training CLI: collect, synthesise, pair, fit, then evaluate on held-out pairs.

Every metric in metrics.json comes from predictions on the test split of
this run. Nothing is copied from earlier reports.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .evaluate import evaluate_model
from .pairs import PairGenerator
from .unsplash_collector import UnsplashCollector
from .watermark_synth import synthesise_dataset

Record = tuple[Path, int]


def split_records(
    records: list[Record], seed: int, fractions: tuple[float, float] = (0.7, 0.15)
) -> dict[str, list[Record]]:
    """Split by source image, so a clean photo and its watermarked copy share one split."""
    sources = sorted({p.stem for p, _ in records})
    order = np.random.default_rng(seed).permutation(len(sources))
    n_train = int(len(sources) * fractions[0])
    n_val = int(len(sources) * fractions[1])
    bucket = {}
    for rank, idx in enumerate(order):
        bucket[sources[idx]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    splits: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    for path, label in records:
        splits[bucket[path.stem]].append((path, label))
    return splits


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(prog="wmd-train", description="Train the Siamese watermark detector.")
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--model-dir", type=Path, default=d.model_dir)
    p.add_argument("--dataset-size", type=int, default=d.dataset_size, help="clean photos to collect")
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--image-size", type=int, nargs=2, default=list(d.image_size), metavar=("W", "H"))
    p.add_argument("--n-pairs", type=int, default=d.n_pairs, help="training pairs per epoch")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--query", default=d.query)
    p.add_argument("--skip-collect", action="store_true", help="use photos already in data/raw")
    return p


def run(cfg: Config, skip_collect: bool = False) -> dict:
    """Run the full pipeline and return the metrics written to disk."""
    raw_dir = cfg.data_dir / "raw"
    if skip_collect:
        raw = sorted(raw_dir.glob("*.jpg"))[: cfg.dataset_size]
    else:
        raw = UnsplashCollector(cfg.require_access_key(), raw_dir).collect(cfg.dataset_size, query=cfg.query)
    if len(raw) < 10:
        raise RuntimeError(f"Only {len(raw)} photos in {raw_dir}. Collect more before training.")

    records = synthesise_dataset(raw, cfg.data_dir / "labelled", seed=cfg.seed)
    splits = split_records(records, cfg.seed)

    def gen(name: str, n_pairs: int, offset: int) -> PairGenerator:
        paths, labels = zip(*splits[name])
        return PairGenerator(paths, labels, n_pairs, cfg.batch_size, cfg.image_size, cfg.seed + offset)

    n_eval = max(cfg.batch_size, cfg.n_pairs * 15 // 70)
    train_gen, val_gen, test_gen = gen("train", cfg.n_pairs, 0), gen("val", n_eval, 1), gen("test", n_eval, 2)

    from tensorflow import keras

    from .model import build_siamese

    keras.utils.set_random_seed(cfg.seed)
    w, h = cfg.image_size
    model = build_siamese((h, w, 3))
    started = time.time()
    history = model.fit(
        train_gen.batches(repeat=True),
        steps_per_epoch=len(train_gen),
        validation_data=val_gen.batches(repeat=True),
        validation_steps=len(val_gen),
        epochs=cfg.epochs,
        verbose=2,
    )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.model_dir / "watermark_siamese.keras"
    model.save(model_path)

    metrics = {
        "test": evaluate_model(model, test_gen),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "photos": len(raw),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "image_size": list(cfg.image_size),
        "seed": cfg.seed,
        "train_seconds": round(time.time() - started, 1),
        "model_path": str(model_path),
    }
    (cfg.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config.from_env(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        n_pairs=args.n_pairs,
        seed=args.seed,
        query=args.query,
    )
    metrics = run(cfg, skip_collect=args.skip_collect)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
