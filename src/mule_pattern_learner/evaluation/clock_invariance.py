"""The clock invariance test on the test world (docs/design.md, the Clock guard): mule audit stores
the audit sample at the start of each week and fetches its rows, and mule evaluate scores it at
both times with the trained GNN.

Each week is scored at a time of its own inside it (data.calendar, scoring_times), and the model
reads durations measured to t, so it cannot read an event's clock time from a duration. The gain
test of mule audit (evaluation.clock_gain) measures whether the clock habits of the data could
separate mules from non-mules beyond plain activity. This test judges the trained model
directly: it scores the same account-weeks at the start of their week, a Monday at 00:00 UTC,
where every duration gives its event's clock time back as it did for every example while t was
always a Monday, and finds how far the model's ranking moves.

On a world that plays a test role, mule audit stores the audit sample (evaluation.sample) with
each row's week start and whether its account is scorable then (with_week_starts), and fetches
the rows of those at the week start, which the cache keeps beside the rows at the scoring time.
A row not scorable at the week start (an account opened or linked between the two) is left out of
the test.

mule evaluate scores the sample at both times with the GNN that mule score uses (the best
selection score), and ranks each week at each time as the test rankings do (evaluation.ranking),
but only the rows in the same class at both times: the sampled non-mules scorable at the start of
their week, each standing for its weight, and the true mules active at their scoring time
(answer_key.status with the world's scoring times, which is the sample's own label) and at the
start of their week (the same rule with the week starts as the scoring times, where the window
is 7 days: the mule's onset is before the start and its last payment is at most 7 days before
it). Every other row is left out of both rankings: a non-mule or a mule not scorable at the week
start, and a mule not active at one of the two times. The mule whose onset falls between the week
start and the scoring time is the typical one: the newest and hardest, and ranked at the scoring
time alone it would move the difference for a reason that is not the clock. So the two rankings
hold the same account-weeks with the same labels, and only t differs. A row left out is not
scored at all.

The test reads two numbers of each ranking: recall at 1% pooled over the weeks, and ROC AUC. Each
has a difference, its value at the scoring times minus its value at the week starts, and a
ring-resampled interval of it from resampling the clusters of the compared rows once and
computing both values on each resample (evaluation.uncertainty), so that the two are paired. The
test passes only when the compared rows hold at least audit.clock_min_mule_clusters (30) mule
clusters (the positive clusters of evaluation.uncertainty.Clusters: each ring, and each ringless
mule alone) and the interval of the difference in recall lies inside plus and minus
audit.clock_invariance_margin (0.05); the difference in ROC AUC is reported beside it. A few
mules give a wide interval, or a point (3 mules ranked at the top at both times differ by exactly
0 on every resample), so below the minimum the test fails whatever its interval says, and low
power never passes. The report table gives the account-weeks compared, the true mules' among
them, the clusters and mule clusters of those, and the mule and non-mule rows left out.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..config import AuditConfig
from ..metrics import TEST_BUDGETS, budget_name
from .answer_key import ACTIVE, NON_MULE, AnswerKey, statuses
from .ranking import Ranking
from .rankings import HEADLINE, evaluation_frame
from .uncertainty import Clusters, percentiles, replicate_matrix

# The name of the test in the report tables and the verdict.
CLOCK_INVARIANCE = "clock_invariance"
# The columns of the stored sample: the audit sample's own, then the start of the row's week and
# whether its account is scorable then.
COLUMNS = (
    "t_ms",
    "account_id",
    "positive",
    "weight",
    "cluster",
    "week_ms",
    "at_week_start",
)
# The two numbers the test reads of a ranking, in the order of Ranking.headline's values.
RECALL = f"recall_at_{budget_name(TEST_BUDGETS[0])}"
ROC_AUC = "roc_auc"
READ = (HEADLINE.index(RECALL), HEADLINE.index(ROC_AUC))


def with_week_starts(
    sample: pd.DataFrame,
    starts: Mapping[int, int],
    scorable: Mapping[int, Collection[str]],
) -> pd.DataFrame:
    """The audit sample (evaluation.sample) as the store keeps it: its t_ms, account_id,
    positive, weight and cluster, then week_ms, the start of the week of each row's scoring time
    (starts maps a scoring time to it), and at_week_start, whether the account is scorable at
    that start (scorable gives the accounts of each week start)."""
    week = [starts[int(t_ms)] for t_ms in sample["t_ms"].tolist()]
    listed = {start: set(accounts) for start, accounts in scorable.items()}
    found = sample.loc[:, ["t_ms", "account_id", "positive", "weight", "cluster"]].copy()
    found["week_ms"] = pd.Series(week, dtype="int64", index=found.index)
    found["at_week_start"] = [
        account in listed.get(start, ())
        for account, start in zip(found["account_id"].tolist(), week, strict=True)
    ]
    return found.loc[:, list(COLUMNS)].reset_index(drop=True)


def week_start_keys(stored: pd.DataFrame) -> pd.DataFrame:
    """The sampled account-weeks scored at the start of their week: (t_ms, account_id), t_ms
    being the week start, by week and account."""
    chosen = stored[stored["at_week_start"]]
    found = pd.DataFrame(
        {
            "t_ms": chosen["week_ms"].astype("int64"),
            "account_id": chosen["account_id"].astype("str"),
        }
    )
    return found.sort_values(["t_ms", "account_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class Compared:
    """One number at the two times: at the scoring times, at the week starts, their difference
    (the first minus the second) and its paired ring-resampled interval, None when it is
    undefined on every resample."""

    scoring_time: float
    week_start: float
    difference: float
    interval: tuple[float, float] | None


@dataclass(frozen=True)
class Invariance:
    """What the invariance test found: recall at 1% and ROC AUC at the two times, whether it
    passed, the margin it was judged by, the account-weeks compared (rows, the true mules' among
    them in mule_rows), the clusters of those and the mule clusters among them, and the true-mule
    and non-mule rows of the sample left out of both rankings."""

    recall: Compared
    roc_auc: Compared
    margin: float
    passed: bool
    rows: int
    mule_rows: int
    clusters: int
    mule_clusters: int
    mule_rows_left_out: int
    nonmule_rows_left_out: int

    def frame(self) -> pd.DataFrame:
        """The test as a one-row report table."""
        row: dict[str, list[float | bool | int]] = {}
        for name, found in ((RECALL, self.recall), (ROC_AUC, self.roc_auc)):
            low, high = found.interval if found.interval is not None else (np.nan, np.nan)
            row |= {
                f"{name}_scoring_time": [found.scoring_time],
                f"{name}_week_start": [found.week_start],
                f"{name}_difference": [found.difference],
                f"{name}_low": [low],
                f"{name}_high": [high],
            }
        row |= {
            "margin": [self.margin],
            "passed": [self.passed],
            "rows": [self.rows],
            "mule_rows": [self.mule_rows],
            "clusters": [self.clusters],
            "mule_clusters": [self.mule_clusters],
            "mule_rows_left_out": [self.mule_rows_left_out],
            "nonmule_rows_left_out": [self.nonmule_rows_left_out],
        }
        return pd.DataFrame(row)


@dataclass(frozen=True)
class _Ranked:
    """The compared rows ranked at one time: their ranking, and each ranked row's row of the
    compared rows."""

    ranking: Ranking
    rows: NDArray[np.intp]

    @classmethod
    def of(cls, frame: pd.DataFrame, rows: NDArray[np.intp]) -> _Ranked:
        """The ranking of a test frame's non-mules and active true mules, each row of the frame
        given with its row of the compared rows."""
        positive = (frame["status"] == ACTIVE).to_numpy()
        kept = positive | (frame["status"] == NON_MULE).to_numpy()
        return cls(Ranking.of(frame, positive), rows[kept])

    def read(self, weight: NDArray[np.float64]) -> NDArray[np.float64]:
        """Recall at 1% and ROC AUC under the compared rows' weights, or a replicate's."""
        return self.ranking.headline(weight[self.rows])[list(READ)]


def _frame(
    scores: pd.DataFrame,
    held: pd.DataFrame,
    times: Sequence[int],
    key: AnswerKey,
    scoring_times: Sequence[int],
) -> tuple[pd.DataFrame, NDArray[np.intp]]:
    """The test frame (evaluation.rankings.evaluation_frame) of the rows held, each at the time
    given for it, and each of its rows' row of held (its index)."""
    keys = pd.DataFrame(
        {
            "t_ms": pd.Series(list(times), dtype="int64"),
            "account_id": held["account_id"].astype("str").reset_index(drop=True),
            "weight": held["weight"].astype("float64").reset_index(drop=True),
        }
    )
    frame = evaluation_frame(scores, keys, key, frozenset[str](), scoring_times)
    position = {
        (int(t_ms), str(account)): int(row)
        for row, t_ms, account in zip(
            held.index.tolist(), keys["t_ms"].tolist(), keys["account_id"].tolist(), strict=True
        )
    }
    rows = np.array(
        [
            position[(int(t_ms), str(account))]
            for t_ms, account in zip(
                frame["t_ms"].tolist(), frame["account_id"].tolist(), strict=True
            )
        ],
        dtype=np.intp,
    )
    return frame, rows


def _same_class(
    sample: pd.DataFrame,
    key: AnswerKey,
    scoring_times: Sequence[int],
    week_starts: Sequence[int],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Of each row of the sample, whether the test compares it, and whether its account is a true
    mule. A row is compared when its account is scorable at the week start and is either a
    non-mule or a true mule active at the row's scoring time and at the start of its week (the
    states of answer_key.status under the scoring times and under the week starts)."""
    truth = key.frame().set_index("account_id").reindex(sample["account_id"].astype("str"))
    if truth["is_mule"].isna().any():
        raise ValueError("A sampled account has no truth in the answer key")
    mule = truth["is_mule"].to_numpy(dtype=np.int64)
    onset = truth["mule_onset_ms"].to_numpy(dtype=np.int64)
    last = truth["mule_last_ms"].to_numpy(dtype=np.int64)
    at_scoring = statuses(sample["t_ms"].to_numpy(dtype=np.int64), mule, onset, last, scoring_times)
    at_start = statuses(sample["week_ms"].to_numpy(dtype=np.int64), mule, onset, last, week_starts)
    same = (at_scoring == NON_MULE) | ((at_scoring == ACTIVE) & (at_start == ACTIVE))
    return same & sample["at_week_start"].to_numpy(dtype=bool), mule == 1


def _failed(margin: float, mules_left_out: int, nonmules_left_out: int) -> Invariance:
    """A test with nothing to compare: undefined everywhere, and not passed."""
    nothing = Compared(float("nan"), float("nan"), float("nan"), None)
    return Invariance(
        nothing, nothing, margin, False, 0, 0, 0, 0, mules_left_out, nonmules_left_out
    )


def invariance(
    stored: pd.DataFrame,
    key: AnswerKey,
    scoring_times: Sequence[int],
    week_starts: Sequence[int],
    score: Callable[[pd.DataFrame], pd.DataFrame],
    audit: AuditConfig,
    seed: int,
) -> Invariance:
    """The invariance test on the stored sample (COLUMNS). scoring_times are the test world's
    and week_starts the starts of its scored weeks, in increasing order, both of which the
    active state reads (answer_key.status). Only the rows in the same class at both times are
    scored and ranked (_same_class); the others are counted as left out. score gives the logit of
    each key it is given ((t_ms, account_id), at most once each) as a frame of t_ms, account_id
    and logit; mule evaluate gives it the GNN's scoring of the test world. It passes only with
    at least audit.clock_min_mule_clusters mule clusters among the rows compared and an interval
    of the difference in recall inside the margin."""
    if tuple(stored.columns) != COLUMNS:
        raise ValueError(f"The stored sample has the columns {list(stored.columns)}, not {COLUMNS}")
    sample = stored.reset_index(drop=True)
    kept, mule = _same_class(sample, key, scoring_times, week_starts)
    left_out = (int((mule & ~kept).sum()), int((~mule & ~kept).sum()))
    if not kept.any():
        return _failed(audit.clock_invariance_margin, *left_out)
    rows = sample[kept].reset_index(drop=True)
    positive = mule[kept]
    weight = rows["weight"].to_numpy(dtype=np.float64)
    clusters = Clusters.of(rows["cluster"].tolist(), positive)

    scoring = _frame(
        score(rows.loc[:, ["t_ms", "account_id"]]),
        rows,
        rows["t_ms"].tolist(),
        key,
        scoring_times,
    )
    starts = _frame(
        score(week_start_keys(rows)),
        rows,
        rows["week_ms"].tolist(),
        key,
        week_starts,
    )
    at_scoring = _Ranked.of(*scoring)
    at_starts = _Ranked.of(*starts)

    values = replicate_matrix(
        [at_scoring.read, at_starts.read],
        clusters,
        weight,
        replicates=audit.replicates,
        seed=seed,
    )
    differences = values[:, 0, :] - values[:, 1, :]
    low, high = percentiles(differences, audit.interval)
    point = (at_scoring.read(weight), at_starts.read(weight))

    def compared(column: int) -> Compared:
        found = (
            (float(low[column]), float(high[column]))
            if np.isfinite(low[column]) and np.isfinite(high[column])
            else None
        )
        return Compared(
            float(point[0][column]),
            float(point[1][column]),
            float(point[0][column] - point[1][column]),
            found,
        )

    # The columns are READ's: recall at 1%, then ROC AUC.
    recall, roc_auc = compared(0), compared(1)
    margin = audit.clock_invariance_margin
    inside = recall.interval is not None and (
        -margin <= recall.interval[0] and recall.interval[1] <= margin
    )
    mule_clusters = int(clusters.positive.sum())
    return Invariance(
        recall,
        roc_auc,
        margin,
        inside and mule_clusters >= audit.clock_min_mule_clusters,
        len(rows),
        int(positive.sum()),
        len(clusters),
        mule_clusters,
        *left_out,
    )
