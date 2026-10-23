"""The selection score, from pu_label alone (docs/design.md, "Training and selection" and
"Selection world").

The selection world's keys (data.examples' selection table) hold, on every Monday, each
account scored then (every scorable account, or under the fallback the identified mules and a
weighted sample of the rest), whether it is a positive (an identified mule active that
Monday), whether it is ranked (an identified mule's inactive Monday is not), and its weight
(1, or the sample's weight under the fallback). Hidden mules are ranked like anyone else; the
answer key is never read.

On each Monday the ranked accounts are ordered by their logits in float64, highest first, and
a positive is found by a budget when it lies within the Monday's top 1% or 5% of the ranked
weight. Accounts with tied logits form one block, and a budget that ends inside a block finds
the same share of each of its accounts, the expected result of ordering them at random. Recall
is pooled over Mondays: the positives found on every Monday divided by the positives on every
Monday, so a Monday without a positive never fails it. The selection score is the mean of the
two pooled recalls; a later epoch or prior must be strictly better to be kept.

paired_interval gives the interval of the difference between two runs' scores, the identified
mule accounts resampled with all their positive Mondays (ring ids are evaluation's alone), for
the prior grid's decision (training.priors).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# The budgets of the selection score: the top 1% and 5% of each Monday's ranking.
BUDGETS = (0.01, 0.05)
# The columns of the selection keys.
SELECTION_COLUMNS = ("t_ms", "account_id", "positive", "ranked", "weight")
# The resamples of a paired interval, its level, and the seed of its draws.
REPLICATES = 1_000
LEVEL = 0.95
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Found:
    """What a ranking of the selection world found: for each positive account-Monday, in the
    order of the keys, its account, its weight, and the share of it each budget found (a row
    per positive, a column per budget of BUDGETS)."""

    accounts: tuple[str, ...]
    weights: NDArray[np.float64]
    shares: NDArray[np.float64]

    def recall(self) -> NDArray[np.float64]:
        """The pooled recall of each budget: the weight found over every positive's weight."""
        found = (self.shares * self.weights[:, None]).sum(axis=0)
        return found / self.weights.sum()

    @property
    def score(self) -> float:
        """The selection score: the mean of the pooled recalls."""
        return float(self.recall().mean())


def ranked_keys(keys: pd.DataFrame) -> pd.DataFrame:
    """The selection keys a ranking scores: the ranked ones, by Monday and account."""
    missing = [column for column in SELECTION_COLUMNS if column not in keys.columns]
    if missing:
        raise ValueError(f"The selection keys lack the columns {missing}")
    if (keys["positive"] & ~keys["ranked"]).any():
        raise ValueError("A positive of the selection keys is left out of the ranking")
    held = keys.loc[keys["ranked"], list(SELECTION_COLUMNS)].astype(
        {
            "t_ms": "int64",
            "account_id": "str",
            "positive": "bool",
            "ranked": "bool",
            "weight": "float64",
        }
    )
    if not (held["weight"] > 0).all():
        raise ValueError("A selection weight is a number above 0")
    return held.sort_values(["t_ms", "account_id"]).reset_index(drop=True)


def found(keys: pd.DataFrame, logits: NDArray[np.floating]) -> Found:
    """What the logits of the ranked selection keys found, row by row: keys as ranked_keys
    gives them."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (len(keys),):
        raise ValueError(f"{scores.shape[0]} logits for {len(keys)} selection keys")
    if not np.isfinite(scores).all():
        raise ValueError("A selection logit is not finite")
    positive = keys["positive"].to_numpy(dtype=bool)
    if not positive.any():
        raise ValueError("The selection world has no positive to recall")
    weights = keys["weight"].to_numpy(dtype=np.float64)
    shares = np.zeros((len(keys), len(BUDGETS)))
    t_ms = keys["t_ms"].to_numpy(dtype=np.int64)
    for day in np.unique(t_ms):
        rows = np.flatnonzero(t_ms == day)
        shares[rows] = _monday_shares(scores[rows], weights[rows])
    return Found(
        tuple(str(account) for account in keys["account_id"].to_numpy()[positive]),
        weights[positive],
        shares[positive],
    )


def _monday_shares(
    scores: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The share of each account of one Monday that each budget's top finds."""
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    # Each account's block of tied logits: the weight ranked above the block, and its weight.
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    block = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ranked)]))
    cumulative = np.r_[0.0, np.cumsum(weights[order])]
    before = cumulative[starts][block]
    within = (cumulative[np.r_[starts[1:], len(ranked)]] - cumulative[starts])[block]
    total = cumulative[-1]
    shares = np.empty((len(scores), len(BUDGETS)))
    for column, budget in enumerate(BUDGETS):
        shares[order, column] = np.clip((budget * total - before) / within, 0.0, 1.0)
    return shares


def selection_score(keys: pd.DataFrame, logits: NDArray[np.floating]) -> float:
    """The selection score of the logits of the ranked selection keys."""
    return found(keys, logits).score


def _by_account(runs: Sequence[Found], accounts: Sequence[str]) -> list[NDArray[np.float64]]:
    """Each run's found weight per budget, summed per mule account (a row per account)."""
    index = {account: row for row, account in enumerate(accounts)}
    rows = np.array([index[account] for account in runs[0].accounts])
    found: list[NDArray[np.float64]] = []
    for run in runs:
        sums = np.zeros((len(accounts), len(BUDGETS)))
        np.add.at(sums, rows, run.shares * run.weights[:, None])
        found.append(sums)
    return found


def paired_interval(
    first: Sequence[Found],
    second: Sequence[Found],
    *,
    replicates: int = REPLICATES,
    level: float = LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """The central level interval of the difference between the mean score of the first runs
    and of the second (several seeds each, all on one selection world), over replicates that
    resample the identified mule accounts with replacement, each with all its positive
    Mondays, the same draws for every run."""
    runs = [*first, *second]
    if not first or not second:
        raise ValueError("A paired interval compares at least one run with another")
    if any(run.accounts != runs[0].accounts for run in runs):
        raise ValueError("The runs of a paired interval ranked different positives")
    accounts = sorted(set(runs[0].accounts))
    sums = _by_account(runs, accounts)
    positives = np.zeros(len(accounts))
    index = {account: row for row, account in enumerate(accounts)}
    np.add.at(positives, [index[account] for account in runs[0].accounts], runs[0].weights)
    rng = np.random.default_rng(seed)
    drawn = rng.multinomial(len(accounts), np.full(len(accounts), 1 / len(accounts)), replicates)
    total = drawn @ positives
    scores = np.stack([((drawn @ found) / total[:, None]).mean(axis=1) for found in sums])
    difference = scores[: len(first)].mean(axis=0) - scores[len(first) :].mean(axis=0)
    low, high = np.quantile(difference, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)
