"""The answer key (docs/design.md, "read_answer_key"): every internal account's truth, read once
per world by mule audit through the evaluation executor, which runs read_answer_key alone, a
page at a time by strictly increasing id, and kept in the evaluation store (evaluation.store) so
that no world is loaded again for evaluation.

Beside the reader, what evaluation asks of a true mule at a scoring time t (docs/design.md,
"Positives, dropped weeks and the activity period" and "Metrics"):

- active when mule_onset_ms < t <= mule_last_ms + W(t), W(t) being the longer of 7 days and the
  time since the previous scoring time, as an identified mule is positive (data.labels, which the
  world's scoring times are passed to, never recomputed here);
- at its early-warning time when t <= mule_onset_ms < next(t), next(t) being the first scoring
  time after t, or t + 7 days when there is none: the last scoring time at or before its first
  ring payment, so a mule has exactly one when the first scoring time is no later than its
  onset and its onset is before the last scoring time plus 7 days;
- before activity at every other time up to its onset, and after activity at every later time at
  which it is not active.

An account with is_mule 0 is a non-mule at every time. True mules outside their activity are
left out of every ranking; their scores at the early-warning time and after activity are
reported apart.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..data.labels import active_at, active_rows
from ..tigergraph.executor import EvaluationExecutor, QueryExecutor
from ..tigergraph.queries import READ_ANSWER_KEY
from ..tigergraph.queries import read_answer_key as answer_key_query
from ..tigergraph.queries.read_answer_key import AccountTruth
from ..tigergraph.queries.rows import DAY_MS, PAGE_LIMIT
from .statuses import ACTIVE, AFTER, BEFORE, EARLY_WARNING, NON_MULE

# How far after t the next scoring time is taken to be when none follows it: a week.
WEEK_MS = 7 * DAY_MS
# The answer key's columns, in the order of its rows' fields.
COLUMNS = tuple(field.name for field in fields(AccountTruth))


def active(truth: AccountTruth, t_ms: int, scoring_times: Sequence[int]) -> bool:
    """Whether a true mule is active at the scoring time t (data.labels.active_at) in the world
    whose scoring times, in increasing order, are given."""
    return truth.is_mule == 1 and active_at(
        truth.mule_onset_ms, truth.mule_last_ms, t_ms, scoring_times
    )


def early_warning_at(onset_ms: int, t_ms: int, scoring_times: Sequence[int]) -> bool:
    """Whether t is the early-warning time of a mule whose first ring payment is at onset_ms:
    t <= onset_ms < next(t), next(t) the first of scoring_times (in increasing order) after t, or
    t + 7 days when none is. So an onset exactly at a scoring time makes that time its
    early-warning time, and not the one before."""
    after = bisect_right(scoring_times, t_ms)
    following = scoring_times[after] if after < len(scoring_times) else t_ms + WEEK_MS
    return t_ms <= onset_ms < following


def early_warning_rows(
    onset_ms: NDArray[np.int64], t_ms: NDArray[np.int64], scoring_times: Sequence[int]
) -> NDArray[np.bool_]:
    """early_warning_at for every row of aligned arrays, as a vectorised form of it."""
    following = t_ms + WEEK_MS
    times = np.asarray(scoring_times, dtype=np.int64)
    if len(times):
        after = np.searchsorted(times, t_ms, side="right")
        following = np.where(
            after < len(times), times[np.minimum(after, len(times) - 1)], following
        )
    return (t_ms <= onset_ms) & (onset_ms < following)


def status(truth: AccountTruth, t_ms: int, scoring_times: Sequence[int]) -> str:
    """What the account is at the scoring time t: a non-mule, or a true mule active, at its
    early-warning time, before it or after its activity."""
    if truth.is_mule != 1:
        return NON_MULE
    if active(truth, t_ms, scoring_times):
        return ACTIVE
    if early_warning_at(truth.mule_onset_ms, t_ms, scoring_times):
        return EARLY_WARNING
    return BEFORE if t_ms <= truth.mule_onset_ms else AFTER


def statuses(
    t_ms: NDArray[np.int64],
    is_mule: NDArray[np.int64],
    onset_ms: NDArray[np.int64],
    last_ms: NDArray[np.int64],
    scoring_times: Sequence[int],
) -> NDArray[np.str_]:
    """status for every row of aligned arrays, as a vectorised form of it."""
    mule = is_mule == 1
    active = active_rows(onset_ms, last_ms, t_ms, scoring_times)
    found = np.full(len(t_ms), NON_MULE, dtype=object)
    found[mule & (t_ms <= onset_ms)] = BEFORE
    found[mule & early_warning_rows(onset_ms, t_ms, scoring_times)] = EARLY_WARNING
    found[mule & active] = ACTIVE
    found[mule & (onset_ms < t_ms) & ~active] = AFTER
    return found.astype(str)


@dataclass(frozen=True)
class AnswerKey:
    """Every internal account's truth, by increasing id."""

    truths: tuple[AccountTruth, ...]

    @cached_property
    def by_id(self) -> dict[str, AccountTruth]:
        return {truth.account_id: truth for truth in self.truths}

    @property
    def mules(self) -> tuple[AccountTruth, ...]:
        """The true mules, by id."""
        return tuple(truth for truth in self.truths if truth.is_mule == 1)

    def frame(self) -> pd.DataFrame:
        """The truths as a frame, a column per field."""
        rows = [asdict(truth) for truth in self.truths]
        found = pd.DataFrame(rows, columns=list(COLUMNS))
        return found.astype(
            {
                "account_id": "str",
                "is_mule": "int64",
                "mule_ring_id": "int64",
                "mule_onset_ms": "int64",
                "mule_last_ms": "int64",
                "opened_ms": "int64",
                "account_type": "str",
                "owned": "bool",
            }
        )


def read_answer_key(executor: QueryExecutor, page: int = PAGE_LIMIT) -> AnswerKey:
    """Every internal account's truth, from every page of read_answer_key until one comes back
    short, through the evaluation executor, which runs nothing else."""
    evaluation = EvaluationExecutor(executor)
    found: list[AccountTruth] = []
    after_id = ""
    while True:
        rows = evaluation.run(READ_ANSWER_KEY, answer_key_query.params(after_id, page))
        truths = answer_key_query.parse(rows, after_id, page)
        found.extend(truths)
        if len(truths) < page:
            return AnswerKey(tuple(found))
        after_id = truths[-1].account_id


def parse_rows(rows: Iterable[dict[str, object]]) -> AnswerKey:
    """The answer key of stored rows, reparsed by read_answer_key's parser as one page of every
    account, so that stored rows pass the live checks."""
    listed = list(rows)
    response: list[dict[str, object]] = [
        {"status": "ok"},
        {"accounts": [{"attributes": row} for row in listed]},
    ]
    return AnswerKey(answer_key_query.parse(response, "", max(1, len(listed))))
