"""mule audit's label checks and its identification horizon check (docs/design.md, "Checks on
every world" and "Identification horizon"), read from the answer key, read_labels and the
scored populations.

The label checks, each failing on any account that breaks it:

- label.identified_not_mule: an account read_labels names (pu_label 1) whose is_mule is not
  1, or which the answer key lacks;
- label.mule_dates_and_ring: an is_mule 1 account without 0 < mule_onset_ms <= mule_last_ms,
  or without a ring value (a ring id, or -1 for none);
- label.scorable_without_truth: a scorable account of some week the answer key lacks;
- label.truth_join: a join of what evaluation scores with the truth that shares a column other
  than the account id, which a merge would rename. It is checked on the columns evaluation
  really joins (JOINED_COLUMNS: the test keys, their weight and the logit of each, which
  rankings.evaluation_frame builds and refuses to build otherwise) against the stored truth,
  so a column added to either side fails the audit.

Reported beside them, never failing: the true mules counted by account_type and by whether
they are owned, and every true mule outside the scored population in a scored week of its
activity, which the agreed population (every mule an owned deposit account) does not score.

The identification horizon: of the true mules whose onset falls in a calendar month (UTC) of
the scored year, from the month of the first scoring time to that of the last, the share PU
labels identified, month by month. Each month's identified count must lie inside its binomial
band at audit.horizon_level around the pooled share; a month without a mule has no band. The
check fails when a month falls outside, or when no true mule has its onset in the scored year.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import math

import numpy as np
import pandas as pd

from .answer_key import ACTIVE, AnswerKey, status

IDENTIFIED_NOT_MULE = "label.identified_not_mule"
MULE_DATES_AND_RING = "label.mule_dates_and_ring"
SCORABLE_WITHOUT_TRUTH = "label.scorable_without_truth"
TRUTH_JOIN = "label.truth_join"
LABEL_CHECKS = (IDENTIFIED_NOT_MULE, MULE_DATES_AND_RING, SCORABLE_WITHOUT_TRUTH, TRUTH_JOIN)
IDENTIFICATION_HORIZON = "identification_horizon"
# The column every truth join is made on.
JOIN_COLUMN = "account_id"
# The columns of what evaluation joins with the truth: the test keys (a t and an account)
# with their weight, and the logit that a model scored each with.
JOINED_COLUMNS = ("t_ms", "account_id", "weight", "logit")


@dataclass(frozen=True)
class LabelChecks:
    """What the label checks found: the accounts breaking each label check (empty when it
    passed), the report tables, and the identification horizon's table."""

    breaking: Mapping[str, tuple[str, ...]]
    counts: pd.DataFrame
    unscored: pd.DataFrame
    horizon: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def results(self) -> dict[str, bool]:
        """Pass or fail of each label check and of the identification horizon."""
        found = {name: not self.breaking[name] for name in LABEL_CHECKS}
        found[IDENTIFICATION_HORIZON] = horizon_passed(self.horizon)
        return found


def truth_join_columns(joined: Iterable[object], truth: pd.DataFrame) -> tuple[str, ...]:
    """The columns other than the account id that a frame holding the columns joined (their
    names) shares with the truth, sorted: those a merge of the two would rename."""
    shared = (set(map(str, joined)) & set(map(str, truth.columns))) - {JOIN_COLUMN}
    return tuple(sorted(shared))


def binomial_band(n: int, p: float, level: float) -> tuple[int, int]:
    """The central level band of a binomial of n trials of chance p: the smallest counts whose
    cumulative chance reaches (1 - level) / 2 and (1 + level) / 2."""
    if n == 0 or p <= 0:
        return 0, 0
    if p >= 1:
        return n, n
    k = np.arange(n + 1)
    log_choose = np.array(
        [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1)]
    )
    cumulative = np.cumsum(np.exp(log_choose + k * math.log(p) + (n - k) * math.log1p(-p)))
    low = int(np.searchsorted(cumulative, (1 - level) / 2))
    high = int(np.searchsorted(cumulative, (1 + level) / 2))
    return min(low, n), min(high, n)


def _month(t_ms: int) -> str:
    return datetime.fromtimestamp(t_ms / 1000, UTC).strftime("%Y-%m")


def scored_months(times: Sequence[int]) -> list[str]:
    """The calendar months of the scored year, from the first scoring time's to the last's."""
    first = datetime.fromtimestamp(times[0] / 1000, UTC)
    last = datetime.fromtimestamp(times[-1] / 1000, UTC)
    months: list[str] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def horizon(
    key: AnswerKey, identified: frozenset[str], times: Sequence[int], level: float
) -> pd.DataFrame:
    """The identification horizon's table: per month of the scored year, the true mules whose
    onset falls in it, the identified among them, the band around the pooled share and whether
    the month lies inside it."""
    months = scored_months(times)
    mules = Counter[str]()
    found = Counter[str]()
    for truth in key.mules:
        month = _month(truth.mule_onset_ms)
        if month in months:
            mules[month] += 1
            found[month] += truth.account_id in identified
    total = sum(mules.values())
    pooled = sum(found.values()) / total if total else float("nan")
    rows: list[tuple[str, int, int, int, int, bool]] = []
    for month in months:
        n, k = mules[month], found[month]
        low, high = binomial_band(n, pooled, level) if total else (0, 0)
        rows.append((month, n, k, low, high, low <= k <= high))
    return pd.DataFrame(
        rows, columns=["month", "mules", "identified", "low", "high", "inside"]
    ).astype({"inside": "bool"})


def horizon_passed(table: pd.DataFrame) -> bool:
    """Whether every month lies inside its band and some true mule's onset is in the scored
    year."""
    return bool(not table.empty and table["mules"].sum() > 0 and table["inside"].all())


def check_labels(
    key: AnswerKey,
    identified: frozenset[str],
    populations: Mapping[int, Sequence[str]],
    scoring_times: Sequence[int],
    level: float,
    joined: Sequence[str] = JOINED_COLUMNS,
) -> LabelChecks:
    """The label checks, their reports and the identification horizon: populations gives the
    scorable accounts of each scored week, scoring_times the world's scoring times in
    increasing order, and joined the columns of what evaluation joins with the truth."""
    truths = key.by_id
    breaking: dict[str, tuple[str, ...]] = {
        IDENTIFIED_NOT_MULE: tuple(
            sorted(
                account
                for account in identified
                if account not in truths or truths[account].is_mule != 1
            )
        ),
        MULE_DATES_AND_RING: tuple(
            truth.account_id
            for truth in key.mules
            if not 0 < truth.mule_onset_ms <= truth.mule_last_ms or truth.mule_ring_id < -1
        ),
        SCORABLE_WITHOUT_TRUTH: tuple(
            sorted(
                {
                    account
                    for accounts in populations.values()
                    for account in accounts
                    if account not in truths
                }
            )
        ),
        TRUTH_JOIN: truth_join_columns(joined, key.frame()),
    }
    mules = key.frame()
    mules = mules[mules["is_mule"] == 1]
    counts = (
        mules.groupby(["account_type", "owned"]).size().rename("mules").reset_index()
        if not mules.empty
        else pd.DataFrame({"account_type": [], "owned": [], "mules": []})
    )
    scorable = {t_ms: set(accounts) for t_ms, accounts in populations.items()}
    missing = [
        (t_ms, truth.account_id)
        for truth in key.mules
        for t_ms in sorted(populations)
        if status(truth, t_ms, scoring_times) == ACTIVE and truth.account_id not in scorable[t_ms]
    ]
    unscored = pd.DataFrame(missing, columns=["t_ms", "account_id"]).astype(
        {"t_ms": "int64", "account_id": "str"}
    )
    return LabelChecks(breaking, counts, unscored, horizon(key, identified, scoring_times, level))
