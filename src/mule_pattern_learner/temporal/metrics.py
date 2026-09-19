"""Imbalance-aware, threshold-locked evaluation and grouped uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def select_threshold(y: NDArray[np.int64], score: NDArray[np.float64]) -> float:
    if len(np.unique(y)) != 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y: NDArray[np.int64], score: NDArray[np.float64], threshold: float) -> dict[str, Any]:
    if not len(y):
        return {"n": 0, "positives": 0, "average_precision": None, "roc_auc": None}
    prediction = score >= threshold
    tp = int(((y == 1) & prediction).sum())
    precision = tp / max(int(prediction.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    order = np.argsort(-score, kind="stable")
    result: dict[str, Any] = {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
    for fraction in (0.01, 0.05):
        k = max(1, int(np.ceil(fraction * len(y))))
        hits = int(y[order[:k]].sum())
        result[f"precision_at_{int(fraction * 100)}pct"] = hits / k
        result[f"recall_at_{int(fraction * 100)}pct"] = hits / max(int(y.sum()), 1)
    return result


def grouped_ap_interval(
    y: NDArray[np.int64],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
    seed: int = 42,
    draws: int = 200,
) -> list[float] | None:
    unique = np.unique(groups)
    if y.sum() == 0 or len(unique) < 2:
        return None
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sample = np.concatenate([indices[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[sample])) == 2:
            values.append(average_precision_score(y[sample], scores[sample]))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None
