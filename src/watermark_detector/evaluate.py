"""Compute accuracy, precision and recall from real model predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def binary_metrics(y_true: Sequence[float], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    """Score probabilities against true labels at the given threshold."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = int(len(y_true))
    return {
        "accuracy": float((y_pred == y_true).mean()) if n else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "threshold": threshold,
        "n": n,
    }


def roc_auc(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Return the area under the ROC curve by rank statistic. Ties get average ranks."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob, kind="mergesort")
    sorted_p = y_prob[order]
    ranks = np.empty(len(y_prob))
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def predict_all(model, generator):
    """Return (y_true, y_prob) arrays over every batch of a PairGenerator."""
    y_true, y_prob = [], []
    for (a, b), y in generator.batches():
        y_prob.append(model.predict([a, b], verbose=0).ravel())
        y_true.append(y)
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate_model(model, generator) -> dict:
    """Run the model over every batch of a PairGenerator and score the predictions."""
    y_true, y_prob = predict_all(model, generator)
    out = binary_metrics(y_true, y_prob)
    out["roc_auc"] = roc_auc(y_true, y_prob)
    return out
