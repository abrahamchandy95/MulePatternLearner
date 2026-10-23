"""The answer key (docs/design.md, "read_answer_key"): every internal account's truth, read once
per world by mule audit through the evaluation executor, which runs read_answer_key alone, a
page at a time by strictly increasing id, and kept in the evaluation store (evaluation.store) so
that no world is loaded again for evaluation.

Beside the reader, what evaluation asks of a true mule on a Monday t (docs/design.md,
"Positives, dropped Mondays and the activity period" and "Metrics"):

- active when mule_onset_ms < t <= mule_last_ms + 7 days, as an identified mule is positive;
- on its early-warning Monday, the one Monday with mule_onset_ms - 7 days < t <= mule_onset_ms;
- before activity on every earlier Monday, and after activity on every Monday after
  mule_last_ms + 7 days.

An account with is_mule 0 is a non-mule on every Monday. True mules outside their activity are
left out of every ranking; their scores on the early-warning Monday and after activity are
reported apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..tigergraph.executor import EvaluationExecutor, QueryExecutor
from ..tigergraph.queries import READ_ANSWER_KEY, VERSIONS
from ..tigergraph.queries import read_answer_key as answer_key_query
from ..tigergraph.queries.read_answer_key import AccountTruth
from ..tigergraph.queries.rows import DAY_MS, PAGE_LIMIT

# How long after its last payment a mule stays active, and how long before its onset its
# early-warning Monday may fall: a week each.
WEEK_MS = 7 * DAY_MS
# What a true mule or a non-mule is on a Monday.
NON_MULE = "non_mule"
ACTIVE = "active"
EARLY_WARNING = "early_warning"
BEFORE = "before"
AFTER = "after"
STATUSES = (NON_MULE, ACTIVE, EARLY_WARNING, BEFORE, AFTER)
# The answer key's columns, in the order of its rows' fields.
COLUMNS = tuple(field.name for field in fields(AccountTruth))


def active(truth: AccountTruth, t_ms: int) -> bool:
    """Whether a true mule is active on the Monday t: its onset strictly before t, its last
    payment at most 7 days before t."""
    return truth.is_mule == 1 and truth.mule_onset_ms < t_ms <= truth.mule_last_ms + WEEK_MS


def status(truth: AccountTruth, t_ms: int) -> str:
    """What the account is on the Monday t: a non-mule, or a true mule active, on its
    early-warning Monday, before it or after its activity."""
    if truth.is_mule != 1:
        return NON_MULE
    if active(truth, t_ms):
        return ACTIVE
    if truth.mule_onset_ms - WEEK_MS < t_ms <= truth.mule_onset_ms:
        return EARLY_WARNING
    return BEFORE if t_ms <= truth.mule_onset_ms else AFTER


def statuses(
    t_ms: NDArray[np.int64],
    is_mule: NDArray[np.int64],
    onset_ms: NDArray[np.int64],
    last_ms: NDArray[np.int64],
) -> NDArray[np.str_]:
    """status for every row of aligned arrays, as a vectorised form of it."""
    mule = is_mule == 1
    found = np.full(len(t_ms), NON_MULE, dtype=object)
    found[mule & (t_ms <= onset_ms)] = BEFORE
    found[mule & (onset_ms - WEEK_MS < t_ms) & (t_ms <= onset_ms)] = EARLY_WARNING
    found[mule & (onset_ms < t_ms) & (t_ms <= last_ms + WEEK_MS)] = ACTIVE
    found[mule & (t_ms > last_ms + WEEK_MS)] = AFTER
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
        {"status": "ok", "version": VERSIONS[READ_ANSWER_KEY]},
        {"accounts": [{"attributes": row} for row in listed]},
    ]
    return AnswerKey(answer_key_query.parse(response, "", max(1, len(listed))))
