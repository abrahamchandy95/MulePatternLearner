"""Threshold-locked proxy metrics and the weighted metrics of an audit sample.

Everything here is pure numpy and scikit-learn: arrays in, numbers out. The proxy
metrics (proxy_metrics) score observed labels as they are; the weighted metrics estimate population
values from a sample in which each account stands for ``weight`` accounts. Both share
the thresholded precision, recall and F1 (threshold_metrics) and the tie-aware review
budgets (capture_at_budgets).
"""

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


def threshold_metrics(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any], threshold: float
) -> dict[str, float]:
    """Weighted precision, recall and F1 of the accounts scored at or above threshold."""
    predicted = score >= threshold
    positives, tp = float(weight[y == 1].sum()), float(weight[(y == 1) & predicted].sum())
    precision, recall = tp / max(float(weight[predicted].sum()), 1), tp / max(positives, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def proxy_metrics(
    y: NDArray[np.int64], score: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    """AP, ROC AUC and the thresholded and review-budget metrics of observed labels.

    The review budgets are the audit's (capture_at_budgets with unit weights), so tied
    scores share a budget that ends among them.
    """
    if not len(y):
        return {"n": 0, "positives": 0, "average_precision": None, "roc_auc": None}
    unit = np.ones(len(y))
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, score)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "threshold": threshold,
        **threshold_metrics(y, score, unit, threshold),
        **capture_at_budgets(y, score, unit),
    }


# The review budgets: the top 1, 5 and 10% of the (estimated) population.
REVIEW_BUDGETS = (0.01, 0.05, 0.10)


def capture_curve(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Population accounts reviewed and weighted positives found, ranked by score.

    Accounts rank by score, highest first, and each sampled account stands for
    ``weight`` population accounts. Accounts with tied scores form one block, as one
    threshold of sklearn's average precision, so the curve has a point at the end of
    each block, after a first point at zero.
    """
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    # The last rank of each block of tied scores.
    ends = np.flatnonzero(np.append(ranked[1:] != ranked[:-1], True))
    # Population accounts and weighted positives above each block end, starting at zero.
    reviewed = np.concatenate(([0.0], np.cumsum(weight[order])[ends]))
    found = np.concatenate(([0.0], np.cumsum(np.where(y == 1, weight, 0.0)[order])[ends]))
    return reviewed, found


def capture_at_budgets(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> dict[str, float]:
    """Weighted precision and recall in the top 1, 5 and 10% of the estimated population.

    Accounts rank by score, highest first. Each sampled account stands for ``weight``
    (1 / inclusion probability) population accounts, so the top fraction f is the
    ranked prefix whose weights add up to f times the estimated population W. Accounts
    with tied scores form one block, as one threshold of sklearn's average precision:
    a budget that ends inside the block takes the same share of each of its accounts,
    the expected result of ordering the tied population accounts at random. The block
    that straddles the boundary counts only for the part inside it: a sampled negative
    stands for many accounts. Precision is the weighted positives inside over f * W,
    recall the same over all weighted positives. With unit weights, no ties and a whole
    f * n these are the precision and recall of the top f * n accounts; proxy_metrics
    reports them for observed labels.
    """
    reviewed, found = capture_curve(y, score, weight)
    result: dict[str, float] = {}
    for fraction in REVIEW_BUDGETS:
        size = fraction * float(reviewed[-1])
        hits = float(np.interp(size, reviewed, found))
        name = f"{round(fraction * 100)}pct"
        result[f"precision_at_{name}"] = hits / size
        result[f"recall_at_{name}"] = hits / max(float(found[-1]), 1)
    return result


def weighted_metrics(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any], threshold: float
) -> dict[str, Any]:
    """Weighted AP, ROC AUC, threshold and top-fraction metrics of a weighted sample.

    Each sampled account stands for ``weight`` population accounts, so these are
    estimates, not census measurements. The top-fraction metrics
    (``capture_at_budgets``) share a budget that ends among tied scores evenly
    across them, so row order does not matter.
    """
    positives = float(weight[y == 1].sum())
    return {
        "sample_accounts": len(y),
        "sample_positives": int(y.sum()),
        "estimated_population": float(weight.sum()),
        "weighted_prevalence": positives / float(weight.sum()),
        "average_precision": float(average_precision_score(y, score, sample_weight=weight))
        if y.any()
        else None,
        "roc_auc": float(roc_auc_score(y, score, sample_weight=weight))
        if len(np.unique(y)) == 2
        else None,
        "threshold": threshold,
        **threshold_metrics(y, score, weight, threshold),
        **capture_at_budgets(y, score, weight),
    }
