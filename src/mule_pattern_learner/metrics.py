"""Threshold-locked proxy metrics, the weighted metrics of an audit sample, and the Monday
rankings of the test world.

Everything here is pure numpy and scikit-learn: arrays in, numbers out. The proxy
metrics (proxy_metrics) score observed labels as they are; the weighted metrics estimate population
values from a sample in which each account stands for ``weight`` accounts. Both share
the thresholded precision, recall and F1 (threshold_metrics) and the tie-aware review
budgets (capture_at_budgets). weighted_quantiles summarises such a sample's values, and
hidden_name names a metric of the hidden mules alone.

The test world's metrics (docs/design.md, "Rankings" and "Metrics") rank each Monday apart:
MondayRanking orders each Monday's account-Mondays by logit, highest first, in blocks of tied
logits, and gives each the share of it that each budget, the top 1% or 5% of its Monday's
ranked weight, finds: a budget that ends inside a block finds the same share of each of its
accounts, the expected result of ordering them at random. From those shares: recall and
precision per Monday and pooled over Mondays (budget_found), the chance that an account is
flagged on any of its Mondays (ever_flagged), and the expected days to its first flag
(first_flag).
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


def sample_metrics(y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any]) -> dict[str, Any]:
    """A weighted sample's size, estimated population and prevalence, and ranking metrics.

    Each sampled account stands for ``weight`` population accounts, so these are
    estimates, not census measurements. The ranking metrics are ranking_metrics'.
    """
    positives = float(weight[y == 1].sum())
    return {
        "sample_accounts": len(y),
        "sample_positives": int(y.sum()),
        "estimated_population": float(weight.sum()),
        "weighted_prevalence": positives / float(weight.sum()),
        **ranking_metrics(y, score, weight),
    }


def weighted_metrics(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any], threshold: float
) -> dict[str, Any]:
    """Weighted AP, ROC AUC, threshold and top-fraction metrics of a weighted sample.

    sample_metrics, and the precision, recall and F1 of the accounts scored at or above
    ``threshold``. The top-fraction metrics (``capture_at_budgets``) share a budget that
    ends among tied scores evenly across them, so row order does not matter.
    """
    found = sample_metrics(y, score, weight)
    # The review budgets' metrics, precision_at_1pct and the like, come last.
    budgets = {name: value for name, value in found.items() if "_at_" in name}
    return {
        **{name: value for name, value in found.items() if name not in budgets},
        "threshold": threshold,
        **threshold_metrics(y, score, weight, threshold),
        **budgets,
    }


def hidden_name(metric: str) -> str:
    """How a metric of the hidden mules alone is named beside the one of every mule.

    The hidden mules are those not yet revealed at the split's cutoff, ranked against
    the non-mules with the revealed mules removed: "average_precision" of the hidden
    mules is "hidden_average_precision".
    """
    return f"hidden_{metric}"


def hidden_first(
    y: NDArray[Any], score: NDArray[Any], weight: NDArray[Any], revealed: NDArray[Any]
) -> dict[str, float | None]:
    """ranking_metrics of the hidden mules (hidden_name), then of every mule.

    The hidden mules' rank without the accounts ``revealed`` marks (the mules revealed by
    the cutoff), as an investigator would remove the cases already known; every mule's
    rank all the accounts.
    """
    kept = ~np.asarray(revealed, dtype=bool)
    hidden = ranking_metrics(y[kept], score[kept], weight[kept])
    return {
        **{hidden_name(name): value for name, value in hidden.items()},
        **ranking_metrics(y, score, weight),
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


# How close to 0 or 1 log_odds_mean lets a score be: 2 ** -50, about 8.9e-16 or 34.7 in
# log-odds, near where float64 can still tell a score from 1. A score of exactly 0 or 1
# has no finite log-odds; a power of two makes 1 - clip exact, so both ends clip alike.
LOG_ODDS_CLIP = 2.0**-50


def log_odds_mean(scores: NDArray[Any], clip: float = LOG_ODDS_CLIP) -> NDArray[np.float64]:
    """The scores of several models of the same accounts, averaged on the log-odds scale.

    ``scores`` holds one column per model (a seed of a variant) and one row per account,
    as probabilities. Each is clipped to [clip, 1 - clip], turned into its log-odds,
    averaged over the columns and turned back into a probability: the geometric mean of
    the odds. The trade-off: on the log-odds scale a model that is confident about an
    account, scoring it near 0 or 1, moves the mean further than one that is unsure, so a
    confident seed weighs more than a hesitant one; averaging the models' ranks instead
    would give every seed the same say about every account, whatever its confidence. The
    clip bounds how far one model can pull.
    """
    clipped = np.clip(np.asarray(scores, dtype=np.float64), clip, 1 - clip)
    mean = (np.log(clipped) - np.log1p(-clipped)).mean(axis=1)
    return 1 / (1 + np.exp(-mean))


# The stream of the seeds' draws (seed_draws), apart from the accounts' resamples drawn
# with the same seed, so that the two are independent.
SEED_STREAM = 1


def seed_draws(
    count: int, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED
) -> NDArray[np.intp]:
    """The seeds each bootstrap replicate draws: replicates by ``count`` positions.

    Each replicate draws ``count`` of the seeds' positions with replacement, from a
    generator of its own (``seed`` and SEED_STREAM), independent of the accounts'
    resamples.
    """
    rng = np.random.default_rng([seed, SEED_STREAM])
    return rng.integers(0, count, size=(replicates, count)).astype(np.intp)


def two_source_replicates(
    differences: NDArray[Any], draws: NDArray[Any] | None = None
) -> NDArray[np.float64]:
    """The mean paired difference on each replicate, the seeds and the accounts resampled.

    ``differences`` has one row per bootstrap replicate of the audit sample, each one
    resample of the accounts and rings applied to every run (paired_replicates), and one
    column per seed: the difference between two runs of that seed on that resample. Each
    replicate also draws as many seeds as there are, with replacement (``draws``, by
    default seed_draws), and averages their differences, so the replicates vary with
    both sources of uncertainty: which seeds were trained, and which accounts audited.
    """
    values = np.asarray(differences, dtype=np.float64)
    if draws is None:
        draws = seed_draws(values.shape[1], replicates=values.shape[0])
    return np.take_along_axis(values, np.asarray(draws, dtype=np.intp), axis=1).mean(axis=1)


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


# The budgets of the test rankings: the top 1% and 5% of each Monday's ranked weight.
TEST_BUDGETS = (0.01, 0.05)


def _blocks_of(
    monday: NDArray[np.int64], score: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.intp]]:
    """The order of rows by Monday, then score highest first; where each Monday and each block
    of tied scores starts in that order."""
    order = np.lexsort((-score, monday)).astype(np.intp)
    mondays, scores = monday[order], score[order]
    new_monday = np.r_[True, mondays[1:] != mondays[:-1]] if len(order) else np.zeros(0, bool)
    new_block = new_monday | (np.r_[True, scores[1:] != scores[:-1]] if len(order) else new_monday)
    return order, new_monday, new_block, np.cumsum(new_block) - 1


class MondayRanking:
    """The account-Mondays of one ranking: each Monday's ordered by score, highest first, in
    blocks of tied scores. Scores are compared within a Monday only."""

    def __init__(self, monday: NDArray[np.int64], score: NDArray[np.floating]) -> None:
        values = np.asarray(score, dtype=np.float64)
        mondays = np.asarray(monday, dtype=np.int64)
        if values.shape != mondays.shape or values.ndim != 1:
            raise ValueError(f"{values.shape} scores for {mondays.shape} Mondays")
        if not np.isfinite(values).all():
            raise ValueError("A ranked score is not finite")
        order, new_monday, new_block, block = _blocks_of(mondays, values)
        self.size = len(values)
        self.block = np.empty(self.size, dtype=np.intp)
        self.block[order] = block
        self.blocks = int(new_block.sum())
        monday_index = np.cumsum(new_monday) - 1
        self.monday_of_row = np.empty(self.size, dtype=np.intp)
        self.monday_of_row[order] = monday_index
        # Each block's Monday and score, and each Monday's value and first block.
        self.block_monday = monday_index[new_block]
        self.block_score = values[order][new_block]
        self.mondays = mondays[order][new_monday]
        self.first_block = block[new_monday]

    def _weights(
        self, weight: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Each block's weight, the weight ranked above it in its Monday, and each Monday's
        weight."""
        if weight.shape != (self.size,):
            raise ValueError(f"{weight.shape} weights for {self.size} ranked rows")
        block = np.bincount(self.block, weights=weight, minlength=self.blocks).astype(np.float64)
        before = np.cumsum(block) - block
        above = before - before[self.first_block][self.block_monday]
        monday = np.bincount(self.block_monday, weights=block, minlength=len(self.mondays))
        return block, above, monday.astype(np.float64)

    @staticmethod
    def _share(
        size: NDArray[np.float64], above: NDArray[np.float64], tied: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The share of a block of weight tied, with above ranked over it, that a budget of
        size finds; a block of no weight is found whole when the budget reaches past what is
        above it."""
        found = np.divide(size - above, tied, out=np.zeros_like(size), where=tied > 0)
        return np.where(tied > 0, np.clip(found, 0.0, 1.0), (above < size).astype(np.float64))

    def shares(
        self, weight: NDArray[np.float64], budgets: Sequence[float] = TEST_BUDGETS
    ) -> NDArray[np.float64]:
        """The share of each row each budget finds: a row per row, a column per budget."""
        block, above, monday = self._weights(weight)
        found = np.stack(
            [self._share(b * monday[self.block_monday], above, block) for b in budgets], axis=1
        )
        return found[self.block]

    def rank_fraction(self, weight: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each row's place in its Monday as a share of the Monday's weight: the weight ranked
        above its block and half its block's."""
        block, above, monday = self._weights(weight)
        total = monday[self.block_monday]
        place = np.divide(above + block / 2, total, out=np.zeros_like(total), where=total > 0)
        return place[self.block]

    def against(
        self,
        weight: NDArray[np.float64],
        monday: NDArray[np.int64],
        score: NDArray[np.floating],
        budgets: Sequence[float] = TEST_BUDGETS,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """For rows outside the ranking, each on a Monday of it, the share each budget of its
        Monday finds and its rank fraction, against the ranked rows alone, as if each were
        added with no weight."""
        block, _, totals = self._weights(weight)
        values = np.asarray(score, dtype=np.float64)
        shares = np.zeros((len(values), len(budgets)))
        fractions = np.zeros(len(values))
        index = {int(value): position for position, value in enumerate(self.mondays.tolist())}
        ends = np.r_[self.first_block[1:], self.blocks]
        for row, (day, value) in enumerate(zip(monday.tolist(), values.tolist(), strict=True)):
            if int(day) not in index:
                raise ValueError(f"No ranked row is on the Monday {day}")
            position = index[int(day)]
            first, last = int(self.first_block[position]), int(ends[position])
            scores = self.block_score[first:last]
            # The blocks of the Monday run highest first: those scoring above the row, then
            # the block it ties with, if any.
            higher = int(np.searchsorted(-scores, -value, side="left"))
            ties = higher < len(scores) and scores[higher] == value
            over = float(block[first : first + higher].sum())
            tied = float(block[first + higher]) if ties else 0.0
            total = float(totals[position])
            for column, budget in enumerate(budgets):
                shares[row, column] = self._share(
                    np.array([budget * total]), np.array([over]), np.array([tied])
                )[0]
            fractions[row] = (over + tied / 2) / total if total > 0 else 0.0
        return shares, fractions

    def budget_found(
        self,
        shares: NDArray[np.float64],
        positive: NDArray[np.bool_],
        weight: NDArray[np.float64],
        budgets: Sequence[float] = TEST_BUDGETS,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Per Monday (in increasing order): the positive weight each budget found (a column
        per budget), the positive weight, and each budget's size in weight."""
        _, _, totals = self._weights(weight)
        rows = self.monday_of_row
        found = np.stack(
            [
                np.bincount(
                    rows,
                    weights=np.where(positive, weight * shares[:, column], 0.0),
                    minlength=len(self.mondays),
                )
                for column in range(len(budgets))
            ],
            axis=1,
        )
        positives = np.bincount(
            rows, weights=np.where(positive, weight, 0.0), minlength=len(self.mondays)
        )
        sizes = np.outer(totals, np.asarray(budgets, dtype=np.float64))
        return found.astype(np.float64), positives.astype(np.float64), sizes


def ever_flagged(
    shares: NDArray[np.float64], group: NDArray[np.intp], groups: int
) -> NDArray[np.float64]:
    """The chance each group (an account) is flagged on at least one of its rows, each row
    flagged with its share apart from the others: 1 minus the product of 1 minus each
    share."""
    with np.errstate(divide="ignore"):
        missed = np.log1p(-np.clip(shares, 0.0, 1.0))
    total = np.zeros(groups)
    np.add.at(total, group, missed)
    return 1.0 - np.exp(total)


def first_flag(days: NDArray[np.float64], shares: NDArray[np.float64]) -> tuple[float, float]:
    """Of one account's rows, in increasing days, each flagged with its share apart from the
    others: the chance it is flagged on any, and the expected days to its first flag given
    that it is (NaN when it never is)."""
    order = np.argsort(days, kind="stable")
    ordered, chances = days[order], np.clip(shares[order], 0.0, 1.0)
    unflagged = np.r_[1.0, np.cumprod(1.0 - chances)[:-1]]
    first = chances * unflagged
    ever = float(first.sum())
    return ever, float(first @ ordered) / ever if ever > 0 else float("nan")
