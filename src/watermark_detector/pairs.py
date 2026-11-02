"""Build balanced Siamese pairs and stream them in batches.

A pair has label 1 when both images share a class (both clean or both
watermarked) and label 0 when the classes differ.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .preprocessing import load_image

Pair = tuple[int, int, int]


def make_pairs(labels: Sequence[int], n_pairs: int, seed: int) -> list[Pair]:
    """Return n_pairs index pairs, half same-class and half cross-class, in shuffled order."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("make_pairs needs exactly two classes")
    groups = {int(c): np.flatnonzero(labels == c) for c in classes}
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError("each class needs at least two images")

    rng = np.random.default_rng(seed)
    n_pos = n_pairs // 2
    pairs: list[Pair] = []
    for k in range(n_pairs):
        if k < n_pos:
            group = groups[int(classes[k % 2])]
            i, j = rng.choice(group, size=2, replace=False)
            pairs.append((int(i), int(j), 1))
        else:
            i = rng.choice(groups[int(classes[0])])
            j = rng.choice(groups[int(classes[1])])
            if rng.random() < 0.5:
                i, j = j, i
            pairs.append((int(i), int(j), 0))
    order = rng.permutation(len(pairs))
    return [pairs[o] for o in order]


class PairGenerator:
    """Decode images on first use and cache them, so later epochs skip disk reads."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        labels: Sequence[int],
        n_pairs: int,
        batch_size: int = 32,
        image_size: tuple[int, int] = (128, 128),
        seed: int = 42,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.batch_size = batch_size
        self.image_size = image_size
        self.pairs = make_pairs(labels, n_pairs, seed)
        self._cache: dict[int, np.ndarray] = {}

    def _get(self, idx: int) -> np.ndarray:
        """Decode each image once. A split holds a few hundred small arrays."""
        if idx not in self._cache:
            self._cache[idx] = load_image(self.paths[idx], self.image_size)
        return self._cache[idx]

    def __len__(self) -> int:
        return math.ceil(len(self.pairs) / self.batch_size)

    def batches(self, repeat: bool = False) -> Iterator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]]:
        """Yield ((img_a, img_b), label) batches. Loop for ever when repeat is True."""
        while True:
            for start in range(0, len(self.pairs), self.batch_size):
                chunk = self.pairs[start : start + self.batch_size]
                a = np.stack([self._get(i) for i, _, _ in chunk])
                b = np.stack([self._get(j) for _, j, _ in chunk])
                y = np.array([lab for _, _, lab in chunk], dtype=np.float32)
                yield (a, b), y
            if not repeat:
                return
