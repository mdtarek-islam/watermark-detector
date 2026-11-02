"""Inference CLI: compare one image with a reference image of known class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config
from .preprocessing import load_image


def verdict(score: float, reference_label: str, threshold: float = 0.5) -> str:
    """Turn a same-class score into 'watermarked' or 'clean'."""
    same = score >= threshold
    other = "clean" if reference_label == "watermarked" else "watermarked"
    return reference_label if same else other


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wmd-predict", description="Detect a watermark against a reference image.")
    p.add_argument("image", type=Path)
    p.add_argument("--reference", type=Path, required=True, help="image whose class you know")
    p.add_argument("--reference-label", choices=["watermarked", "clean"], default="watermarked")
    p.add_argument("--model", type=Path, default=Config().model_dir / "watermark_siamese.keras")
    p.add_argument("--image-size", type=int, nargs=2, default=list(Config().image_size), metavar=("W", "H"))
    p.add_argument("--threshold", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .model import load_model

    model = load_model(args.model)
    size = tuple(args.image_size)
    a = load_image(args.image, size)[np.newaxis]
    b = load_image(args.reference, size)[np.newaxis]
    score = float(model.predict([a, b], verbose=0).ravel()[0])
    print(json.dumps({
        "image": str(args.image),
        "same_class_score": round(score, 4),
        "prediction": verdict(score, args.reference_label, args.threshold),
    }, indent=2))


if __name__ == "__main__":
    main()
