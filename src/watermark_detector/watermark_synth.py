"""Apply synthetic text and logo watermarks to clean images to make labelled positives."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORDS = ("SAMPLE", "PREVIEW", "COPYRIGHT", "STOCK", "PROOF", "DRAFT")


def _overlay(base: Image.Image, layer: Image.Image) -> Image.Image:
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


def _colour(rng: np.random.Generator, opacity: int) -> tuple[int, int, int, int]:
    shade = int(rng.choice([255, 0]))
    return (shade, shade, shade, opacity)


def add_text_watermark(
    img: Image.Image, rng: np.random.Generator, text: str | None = None
) -> Image.Image:
    """Draw semi-transparent text, tiled or single, at a random place and angle."""
    w, h = img.size
    text = text or str(rng.choice(WORDS))
    size = max(10, int(min(w, h) * rng.uniform(0.08, 0.2)))
    font = ImageFont.load_default(size=size)
    opacity = int(rng.integers(90, 200))

    layer = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if rng.random() < 0.5:
        step_x, step_y = size * len(text), size * 3
        for y in range(0, h * 2, step_y):
            for x in range(0, w * 2, step_x):
                draw.text((x, y), text, font=font, fill=_colour(rng, opacity))
    else:
        x = int(rng.integers(w // 2, w + w // 2))
        y = int(rng.integers(h // 2, h + h // 2))
        draw.text((x, y), text, font=font, fill=_colour(rng, opacity), anchor="mm")
    angle = float(rng.uniform(-45, 45))
    layer = layer.rotate(angle, resample=Image.Resampling.BILINEAR)
    layer = layer.crop((w // 2, h // 2, w // 2 + w, h // 2 + h))
    return _overlay(img, layer)


def add_logo_watermark(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Draw a simple semi-transparent geometric logo in a random corner or the centre."""
    w, h = img.size
    side = max(8, int(min(w, h) * rng.uniform(0.15, 0.35)))
    opacity = int(rng.integers(90, 200))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    spots = [(0, 0), (w - side, 0), (0, h - side), (w - side, h - side), ((w - side) // 2, (h - side) // 2)]
    x0, y0 = spots[int(rng.integers(len(spots)))]
    box = (x0, y0, x0 + side - 1, y0 + side - 1)
    fill = _colour(rng, opacity)
    width = max(2, side // 10)
    if rng.random() < 0.5:
        draw.ellipse(box, outline=fill, width=width)
        draw.line((box[0], box[1], box[2], box[3]), fill=fill, width=width)
    else:
        draw.rectangle(box, outline=fill, width=width)
        draw.polygon([(x0 + side // 2, y0 + width), (x0 + width, box[3] - width), (box[2] - width, box[3] - width)], fill=fill)
    return _overlay(img, layer)


def apply_random_watermark(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Choose a text or logo watermark at random. Return a new image."""
    if rng.random() < 0.5:
        return add_text_watermark(img, rng)
    return add_logo_watermark(img, rng)


def synthesise_dataset(
    clean_paths: Iterable[str | Path], out_dir: str | Path, seed: int = 42
) -> list[tuple[Path, int]]:
    """Write a clean copy (label 0) and a watermarked copy (label 1) of each image."""
    out_dir = Path(out_dir)
    clean_dir, wm_dir = out_dir / "clean", out_dir / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records: list[tuple[Path, int]] = []
    for src in clean_paths:
        src = Path(src)
        with Image.open(src) as opened:
            img = opened.convert("RGB")
        name = src.stem + ".png"
        img.save(clean_dir / name)
        apply_random_watermark(img, rng).save(wm_dir / name)
        records.append((clean_dir / name, 0))
        records.append((wm_dir / name, 1))
    return records
