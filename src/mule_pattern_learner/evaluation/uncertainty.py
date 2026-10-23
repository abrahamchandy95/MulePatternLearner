"""The ring bootstrap (docs/design.md, "Uncertainty"), and the ranking statistics it resamples.

A replicate resamples whole clusters with replacement, the positive and the negative clusters
apart, as many of each as there are: a ring with all its account-Mondays, each ringless true
mule with all its account-Mondays as a cluster of its own, and each non-mule account with all
its Mondays, since an account's 52 Mondays are correlated. A cluster drawn k times weighs k
times its rows' weights in the replicate, which is the same as taking its rows k times.
Replicates are drawn from a generator of the seed in a fixed order, so the same clusters, seed
and replicates give the same draws for any scores: applied to several runs over the same
account-Mondays, they pair the runs (paired_interval).

Ring ids are the answer key's, so this is evaluation's alone; selection's paired interval
resamples identified mule accounts instead (training.selection).

Ranked holds account-Mondays ordered by score in blocks of tied scores, so that the weighted
ROC AUC and the weighted average precision of any weights, a replicate's among them, take one
pass over the rows: tied scores count half, as one threshold of the curves (metrics.roc_curve
and metrics.precision_recall_curve).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Replicates drawn at a time: their weights take replicates by clusters numbers.
CHUNK = 32


def cluster_keys(
    accounts: Sequence[str], is_mule: NDArray[np.int64], rings: NDArray[np.int64]
) -> list[str]:
    """Each account-Monday's cluster: its ring for a true mule of a ring, the account for a
    ringless true mule or a non-mule."""
    found: list[str] = []
    for account, mule, ring in zip(accounts, is_mule.tolist(), rings.tolist(), strict=True):
        if mule == 1 and ring >= 0:
            found.append(f"ring:{ring}")
        elif mule == 1:
            found.append(f"mule:{account}")
        else:
            found.append(f"account:{account}")
    return found


@dataclass(frozen=True)
class Clusters:
    """Each row's cluster, by number, and each cluster's class: positive (rings and ringless
    true mules) or negative (non-mule accounts)."""

    row: NDArray[np.intp]
    positive: NDArray[np.bool_]

    @classmethod
    def of(cls, keys: Sequence[str], positive: NDArray[np.bool_]) -> Clusters:
        """The clusters of rows named by keys, positive when their rows are; a cluster whose
        rows are of both classes is refused."""
        if len(keys) != len(positive):
            raise ValueError(f"{len(keys)} cluster keys for {len(positive)} rows")
        names, row = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
        count = len(names)
        positives = np.bincount(row, weights=positive.astype(np.float64), minlength=count)
        sizes = np.bincount(row, minlength=count)
        if ((positives > 0) & (positives < sizes)).any():
            raise ValueError("A cluster holds positive and negative account-Mondays")
        return cls(row.astype(np.intp), positives > 0)

    def __len__(self) -> int:
        return len(self.positive)


def replicate_weights(
    clusters: Clusters, replicates: int, seed: int
) -> Iterator[NDArray[np.float64]]:
    """For each replicate in turn, chunks of at most CHUNK rows, each the number of times each
    cluster is drawn: as many positive clusters as there are, with replacement, among the
    positive ones, and likewise the negative ones."""
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(clusters.positive), np.flatnonzero(~clusters.positive)]
    for start in range(0, replicates, CHUNK):
        size = min(CHUNK, replicates - start)
        drawn = np.zeros((size, len(clusters)))
        for members in strata:
            if len(members):
                uniform = np.full(len(members), 1 / len(members))
                drawn[:, members] = rng.multinomial(len(members), uniform, size=size)
        yield drawn


def percentile(values: NDArray[np.float64], level: float) -> tuple[float, float] | None:
    """The central level interval of replicate values, NaN ones left out; None with none."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    low, high = np.quantile(finite, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def replicate_values(
    statistic: Callable[[NDArray[np.float64]], float],
    clusters: Clusters,
    weight: NDArray[np.float64],
    *,
    replicates: int,
    seed: int,
) -> NDArray[np.float64]:
    """The statistic of each replicate: statistic takes each row's weight in the replicate, its
    weight times its cluster's draws, and gives NaN where it is undefined."""
    found: list[float] = []
    for chunk in replicate_weights(clusters, replicates, seed):
        for draws in chunk:
            found.append(statistic(weight * draws[clusters.row]))
    return np.asarray(found, dtype=np.float64)


def interval(
    statistic: Callable[[NDArray[np.float64]], float],
    clusters: Clusters,
    weight: NDArray[np.float64],
    *,
    replicates: int,
    level: float,
    seed: int,
) -> tuple[float, float] | None:
    """The central level interval of the statistic over the replicates; None when it is
    undefined on every one."""
    values = replicate_values(statistic, clusters, weight, replicates=replicates, seed=seed)
    return percentile(values, level)


def intervals(
    statistics: Sequence[Callable[[NDArray[np.float64]], float]],
    clusters: Clusters,
    weight: NDArray[np.float64],
    *,
    replicates: int,
    level: float,
    seed: int,
) -> list[tuple[float, float] | None]:
    """The interval of each statistic, all on the same replicates, which are drawn once."""
    found = np.full((len(statistics), replicates), np.nan)
    done = 0
    for chunk in replicate_weights(clusters, replicates, seed):
        for draws in chunk:
            weights = weight * draws[clusters.row]
            for row, statistic in enumerate(statistics):
                found[row, done] = statistic(weights)
            done += 1
    return [percentile(values, level) for values in found]


def replicate_matrix(
    statistics: Sequence[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
    clusters: Clusters,
    weight: NDArray[np.float64],
    *,
    replicates: int,
    seed: int,
) -> NDArray[np.float64]:
    """Each statistic's values on each replicate, all on the same draws: replicates by
    statistics by the statistic's values. Applied to several runs over the same rows, the
    statistics' values on one replicate are paired."""
    rows: list[NDArray[np.float64]] = []
    for chunk in replicate_weights(clusters, replicates, seed):
        for draws in chunk:
            weights = weight * draws[clusters.row]
            rows.append(np.stack([statistic(weights) for statistic in statistics]))
    return np.stack(rows)


def percentiles(
    values: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The central level interval of replicate values along their first axis, NaN ones left
    out: its low and high ends, NaN where no replicate is defined."""
    flat = values.reshape(len(values), -1)
    low = np.full(flat.shape[1], np.nan)
    high = np.full(flat.shape[1], np.nan)
    for column in range(flat.shape[1]):
        found = percentile(flat[:, column], level)
        if found is not None:
            low[column], high[column] = found
    return low.reshape(values.shape[1:]), high.reshape(values.shape[1:])


def spread(values: Sequence[float]) -> dict[str, float]:
    """The mean, standard deviation, least and greatest of values over seeds or rotations,
    NaN ones left out."""
    found = np.asarray(values, dtype=np.float64)
    found = found[np.isfinite(found)]
    if not len(found):
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(found.mean()),
        "std": float(found.std(ddof=1)) if len(found) > 1 else 0.0,
        "min": float(found.min()),
        "max": float(found.max()),
    }


def paired_interval(
    first: Callable[[NDArray[np.float64]], float],
    second: Callable[[NDArray[np.float64]], float],
    clusters: Clusters,
    weight: NDArray[np.float64],
    *,
    replicates: int,
    level: float,
    seed: int,
) -> tuple[float, float] | None:
    """The central level interval of first minus second, both computed on each replicate's
    same draws: a paired interval of two runs over the same account-Mondays."""

    def difference(weights: NDArray[np.float64]) -> float:
        return first(weights) - second(weights)

    return interval(difference, clusters, weight, replicates=replicates, level=level, seed=seed)


class Ranked:
    """Rows ordered by score in blocks of tied scores, lowest first, with each row's class."""

    def __init__(self, score: NDArray[np.floating], positive: NDArray[np.bool_]) -> None:
        values = np.asarray(score, dtype=np.float64)
        if values.shape != positive.shape or values.ndim != 1:
            raise ValueError(f"{values.shape} scores for {positive.shape} classes")
        if not np.isfinite(values).all():
            raise ValueError("A ranked score is not finite")
        order = np.argsort(values, kind="stable")
        ranked = values[order]
        starts = np.r_[True, ranked[1:] != ranked[:-1]] if len(ranked) else np.zeros(0, bool)
        self.block = np.empty(len(values), dtype=np.intp)
        self.block[order] = np.cumsum(starts) - 1
        self.blocks = int(starts.sum())
        self.positive = positive.astype(bool)

    def _sums(self, weight: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Each block's positive and negative weight."""
        positives = np.bincount(
            self.block, weights=np.where(self.positive, weight, 0.0), minlength=self.blocks
        )
        negatives = np.bincount(
            self.block, weights=np.where(self.positive, 0.0, weight), minlength=self.blocks
        )
        return positives.astype(np.float64), negatives.astype(np.float64)

    def roc_auc(self, weight: NDArray[np.float64]) -> float:
        """The weighted ROC AUC: of every positive and negative pair, by weight, the share in
        which the positive scores higher, a tie counting half; NaN without both classes."""
        positives, negatives = self._sums(weight)
        total = float(positives.sum()) * float(negatives.sum())
        if total <= 0:
            return float("nan")
        below = np.cumsum(negatives) - negatives
        return float(positives @ (below + negatives / 2)) / total

    def average_precision(self, weight: NDArray[np.float64]) -> float:
        """The weighted average precision: from the highest block down, the precision at each
        block times the share of the positives it adds; NaN without a positive."""
        positives, negatives = self._sums(weight)
        found = float(positives.sum())
        if found <= 0:
            return float("nan")
        down_positives = np.cumsum(positives[::-1])
        down_all = np.cumsum((positives + negatives)[::-1])
        precision = np.divide(
            down_positives, down_all, out=np.zeros_like(down_all), where=down_all > 0
        )
        return float(precision @ positives[::-1]) / found
