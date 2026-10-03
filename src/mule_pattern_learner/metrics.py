"""Threshold-locked proxy metrics and the weighted metrics of an audit sample.

Everything here is pure numpy and scikit-learn: arrays in, numbers out. The proxy
metrics (proxy_metrics) score observed labels as they are; the weighted metrics estimate population
values from a sample in which each account stands for ``weight`` accounts. Both share
the thresholded precision, recall and F1 (threshold_metrics) and the tie-aware review
budgets (capture_at_budgets). weighted_quantiles summarises such a sample's values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
import sklearn.metrics
from sklearn.metrics import average_precision_score, roc_auc_score


def select_threshold(y: NDArray[np.int64], score: NDArray[np.float64]) -> float:
    if len(np.unique(y)) != 2:
        return 0.5
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, score)
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
        "average_precision": average_precision(y, score),
        "roc_auc": roc_auc(y, score),
        "threshold": threshold,
        **threshold_metrics(y, score, unit, threshold),
        **capture_at_budgets(y, score, unit),
    }


# The review budgets: the top 1, 5 and 10% of the (estimated) population.
REVIEW_BUDGETS = (0.01, 0.05, 0.10)
# The intervals of the audit metrics: their level, and the bootstrap replicates and the
# seed of their draws.
INTERVAL = 0.90
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 0


def ranked_blocks(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The blocks of tied scores, highest first: the score of each, and what lies above it.

    Accounts rank by score, highest first, and each sampled account stands for
    ``weight`` population accounts. Accounts with tied scores form one block, as one
    threshold of sklearn's average precision. For each block this returns its score, the
    population accounts that score at least as high and the weighted positives among
    them. The curves below are these arrays in other units.
    """
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    # The last rank of each block of tied scores.
    ends = np.flatnonzero(np.append(ranked[1:] != ranked[:-1], True))
    reviewed = np.cumsum(weight[order])[ends].astype(np.float64)
    found = np.cumsum(np.where(y == 1, weight, 0.0)[order])[ends].astype(np.float64)
    return ranked[ends].astype(np.float64), reviewed, found


def capture_curve(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Population accounts reviewed and weighted positives found, ranked by score.

    The curve has a point at the end of each block of tied scores (ranked_blocks),
    after a first point at zero.
    """
    _, reviewed, found = ranked_blocks(y, score, weight)
    return np.concatenate(([0.0], reviewed)), np.concatenate(([0.0], found))


def precision_recall_curve(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Weighted recall and precision of each score threshold, highest threshold first.

    There is one point per block of tied scores (ranked_blocks), counting the accounts
    scored at or above its score, so recall only grows along the arrays. The average
    precision is the sum of each point's precision times its gain in recall. Without
    positives recall is zero throughout.
    """
    thresholds, reviewed, found = ranked_blocks(y, score, weight)
    return _share(found), found / reviewed, thresholds


def roc_curve(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Weighted false and true positive rates of each score threshold, from (0, 0).

    The first point has the threshold infinity and flags nothing; then there is one
    point per block of tied scores (ranked_blocks), so tied scores draw one diagonal
    segment, and the area under the curve is the weighted ROC AUC.
    """
    thresholds, reviewed, found = ranked_blocks(y, score, weight)
    return (
        np.concatenate(([0.0], _share(reviewed - found))),
        np.concatenate(([0.0], _share(found))),
        np.concatenate(([np.inf], thresholds)),
    )


def _share(cumulative: NDArray[np.float64]) -> NDArray[np.float64]:
    """A cumulative sum as a share of its total; all zero when the total is zero."""
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    return cumulative / total if total else np.zeros_like(cumulative)


def budget_name(fraction: float) -> str:
    """How the metrics of a review budget name it: 0.05 is "5pct" (precision_at_5pct)."""
    return f"{round(fraction * 100)}pct"


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
        name = budget_name(fraction)
        result[f"precision_at_{name}"] = hits / size
        result[f"recall_at_{name}"] = hits / max(float(found[-1]), 1)
    return result


def average_precision(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any] | None = None
) -> float | None:
    """Average precision (sklearn's, with sample weights if given); None without positives."""
    return float(average_precision_score(y, score, sample_weight=weight)) if y.any() else None


def roc_auc(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any] | None = None
) -> float | None:
    """ROC AUC (sklearn's, with sample weights if given); None without both classes."""
    if len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y, score, sample_weight=weight))


def ranking_metrics(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]
) -> dict[str, float | None]:
    """Weighted AP, ROC AUC and the review budgets: the metrics that get intervals.

    AP is None without positives and ROC AUC None without both classes.
    """
    return {
        "average_precision": average_precision(y, score, weight),
        "roc_auc": roc_auc(y, score, weight),
        **capture_at_budgets(y, score, weight),
    }


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
    ranking = ranking_metrics(y, score, weight)
    return {
        "sample_accounts": len(y),
        "sample_positives": int(y.sum()),
        "estimated_population": float(weight.sum()),
        "weighted_prevalence": positives / float(weight.sum()),
        "average_precision": ranking.pop("average_precision"),
        "roc_auc": ranking.pop("roc_auc"),
        "threshold": threshold,
        **threshold_metrics(y, score, weight, threshold),
        **ranking,
    }


def ring_clusters(y: NDArray[Any], rings: NDArray[Any] | None) -> list[NDArray[np.intp]]:
    """The rows of each positive cluster: a ring's positives together, the others alone.

    ``rings`` holds each row's ring id, where -1 (or any negative id) means none; without
    it every positive is a cluster of its own. Rings come in the order of their ids,
    then the positives without a ring in row order.
    """
    positives = np.flatnonzero(y == 1)
    if rings is None:
        return [positives[i : i + 1] for i in range(len(positives))]
    ring = np.asarray(rings)[positives]
    grouped = [positives[ring == r] for r in np.unique(ring[ring >= 0])]
    return grouped + [positives[i : i + 1] for i in np.flatnonzero(ring < 0)]


def resamples(
    y: NDArray[Any],
    rings: NDArray[Any] | None = None,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Iterator[NDArray[np.intp]]:
    """The rows of each bootstrap replicate: positives by ring, negatives within their class.

    A replicate draws as many positive clusters (ring_clusters) as there are, with
    replacement, and takes every row of each; without ``rings`` that is the stratified
    bootstrap of the positives. It then draws as many negatives as there are, one by
    one, with replacement. Rows keep their weights, so each replicate is a sample of the
    same design, and the same seed draws the same replicates for any scores: applying
    them to several runs over the same accounts pairs the runs (paired_replicates).
    """
    rng = np.random.default_rng(seed)
    clusters = ring_clusters(y, rings)
    negatives = np.flatnonzero(y == 0)
    empty = np.zeros(0, dtype=np.intp)
    for _ in range(replicates):
        drawn = rng.integers(0, len(clusters), len(clusters)) if clusters else empty
        positives = [clusters[i] for i in drawn]
        chosen = (
            negatives[rng.integers(0, len(negatives), len(negatives))] if len(negatives) else empty
        )
        yield np.concatenate([*positives, chosen]).astype(np.intp)


def percentile_interval(values: NDArray[Any], level: float = INTERVAL) -> list[float] | None:
    """The central ``level`` interval of replicate values; None when none is defined.

    Replicates whose value is undefined (NaN) are left out.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    low, high = np.quantile(finite, [(1 - level) / 2, (1 + level) / 2])
    return [float(low), float(high)]


def bootstrap_intervals(
    y: NDArray[Any],
    score: NDArray[Any],
    weight: NDArray[Any],
    rings: NDArray[Any] | None = None,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    level: float = INTERVAL,
) -> dict[str, list[float] | None]:
    """The ``level`` interval of each ranking metric over bootstrap replicates (resamples).

    With ``rings`` the positives are resampled by ring (the ring-clustered bootstrap),
    without it one by one (stratified). A metric that is undefined for the sample has no
    interval (None).
    """
    values: dict[str, list[float]] = {}
    for rows in resamples(y, rings, replicates=replicates, seed=seed):
        metrics = ranking_metrics(y[rows], score[rows], weight[rows])
        for name, value in metrics.items():
            values.setdefault(name, []).append(np.nan if value is None else value)
    return {name: percentile_interval(np.array(found), level) for name, found in values.items()}


def paired_replicates(
    y: NDArray[Any],
    weight: NDArray[Any],
    scores: Sequence[NDArray[Any]],
    statistic: Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], float | None],
    rings: NDArray[Any] | None = None,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> NDArray[np.float64]:
    """The statistic of every run on each shared bootstrap replicate: replicates by runs.

    Every run scores the same accounts (``y``, ``weight`` and ``rings`` row by row), and
    each replicate draws one resample of those accounts and rings (resamples) and applies
    it to every run, so differences between the columns of a row are paired: they vary
    with the audit sample only through what the runs rank differently. ``statistic``
    takes (y, score, weight); None becomes NaN.
    """
    result = np.full((replicates, len(scores)), np.nan)
    for row, rows in enumerate(resamples(y, rings, replicates=replicates, seed=seed)):
        for column, score in enumerate(scores):
            value = statistic(y[rows], score[rows], weight[rows])
            result[row, column] = np.nan if value is None else value
    return result


def weighted_quantiles(
    values: NDArray[Any], weight: NDArray[Any], quantiles: Sequence[float]
) -> NDArray[np.float64]:
    """Quantiles of a weighted sample: each value stands for ``weight`` population values.

    The values are sorted, and each takes the middle of its weight's share of the
    cumulative weight; a quantile between two of them is interpolated linearly.
    """
    x, w = np.asarray(values, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    middle = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(quantiles, dtype=np.float64), middle, x)
