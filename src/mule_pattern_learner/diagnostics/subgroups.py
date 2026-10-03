"""Which mules a run's ranking finds: revealed or hidden, how few make its AP, which rings.

From a run's audit samples (audit/<split>.parquet), each account weighted by 1 / its
inclusion probability, for validation and test:
- revealed and hidden: each kind of mule against every non-mule, their weighted ROC
  AUC, the mules and how many of them rank in the top 1, 5 and 10% of the split's
  population, and their median population rank (the estimated number of non-mules
  scoring at least as high). A proxy trained on revealed mules that finds only them
  measures the reveal, not mule detection.
- AP concentration: the average precision is a sum over the mules, each adding the
  precision at its score times its share of the mules, so its running sum over the mules
  ranked highest first says how few mules make most of the AP (cumulative_average_precision
  at each rank; the last one is the split's AP).
- ring coverage: the share of the split's rings (ring_id 0 and up) with a member in the
  top 1, 5 and 10%, beside the share of the mules there.

An account is in the top fraction f when the population accounts scoring at least as
high as it, itself and every tie included, are at most f of the split's population.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import DIAGNOSTIC_TABLES
from ..metrics import REVIEW_BUDGETS, average_precision, budget_name, roc_auc

COLUMNS = DIAGNOSTIC_TABLES["subgroups"]


def share_above(score: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Each account's share of the population scoring at least as high as it, ties included."""
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    cumulative = np.cumsum(weight[order])
    # The weight up to the end of each account's block of tied scores.
    ends = np.searchsorted(-ranked, -ranked, side="right") - 1
    share = np.empty(len(score))
    share[order] = cumulative[ends] / weight.sum()
    return share


def population_rank(score: np.ndarray, weight: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The estimated non-mules of the population scoring at least as high as each mule."""
    negative = y == 0
    order = np.argsort(-score[negative], kind="stable")
    ranked = score[negative][order]
    cumulative = np.concatenate([[0.0], np.cumsum(weight[negative][order])])
    # The non-mules scoring at least as high as each mule lead the ranked ones.
    above = np.searchsorted(-ranked, -score[y == 1], side="right")
    return cumulative[above]


def ap_contributions(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Each mule's part of the weighted AP, the mules ranked highest first.

    A mule adds the precision of its block of tied scores (the weighted mules over the
    weighted accounts scoring at least as high) times its weight's share of the mules;
    the parts add up to sklearn's weighted average precision.
    """
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    ends = np.searchsorted(-ranked, -ranked, side="right") - 1
    reviewed = np.cumsum(weight[order])[ends]
    found = np.cumsum(np.where(y == 1, weight, 0.0)[order])[ends]
    mule = y[order] == 1
    parts = (found / reviewed * weight[order] / weight[y == 1].sum())[mule]
    return parts


def split_rows(split: str, frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    """The subgroup rows of one audited split."""
    y = frame.is_mule.to_numpy(np.int64)
    score = frame.score.to_numpy(np.float64)
    weight = 1 / frame.inclusion_probability.to_numpy(np.float64)
    revealed = frame.revealed.to_numpy(bool)
    rings = frame.ring_id.to_numpy(np.int64)
    share = share_above(score, weight)
    records: list[tuple[Any, ...]] = []

    def add(subset: str, metric: str, value: float | None, rank: float = np.nan) -> None:
        if value is not None:
            records.append((split, subset, rank, metric, float(value)))

    for subset, chosen in (("revealed", revealed), ("hidden", ~revealed)):
        mules = (y == 1) & chosen
        keep = (y == 0) | mules
        add(subset, "mules", int(mules.sum()))
        add(subset, "roc_auc", roc_auc(y[keep], score[keep], weight[keep]))
        for fraction in REVIEW_BUDGETS:
            add(subset, f"in_top_{budget_name(fraction)}", int((mules & (share <= fraction)).sum()))
        if mules.any():
            ranks = population_rank(score[keep], weight[keep], y[keep])
            add(subset, "median_population_rank", float(np.median(ranks)))
    parts = ap_contributions(y, score, weight)
    for rank, value in enumerate(np.cumsum(parts), start=1):
        add("mules", "cumulative_average_precision", value, float(rank))
    add("mules", "average_precision", average_precision(y, score, weight))
    ringed = (y == 1) & (rings >= 0)
    names = np.unique(rings[ringed])
    add("rings", "rings", len(names))
    for fraction in REVIEW_BUDGETS:
        name = budget_name(fraction)
        top = share <= fraction
        if len(names):
            covered = np.unique(rings[ringed & top])
            add("rings", f"coverage_at_{name}", len(covered) / len(names))
        if (y == 1).any():
            add("mules", f"coverage_at_{name}", float(((y == 1) & top).sum() / (y == 1).sum()))
    return records


def subgroups(audits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The subgroups table of a run's scored audit samples, by split."""
    records = [row for split, frame in audits.items() for row in split_rows(split, frame)]
    return pd.DataFrame(records, columns=list(COLUMNS))
