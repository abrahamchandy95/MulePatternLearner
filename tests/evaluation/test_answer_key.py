"""The answer key: read a page at a time through the evaluation executor, which runs nothing
else, and what a true mule is on a Monday (docs/design.md, "read_answer_key" and "Metrics")."""

from __future__ import annotations

import numpy as np
import pytest

from mule_pattern_learner.evaluation.answer_key import (
    ACTIVE,
    AFTER,
    BEFORE,
    EARLY_WARNING,
    NON_MULE,
    WEEK_MS,
    read_answer_key,
    status,
    statuses,
)
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import T
from mule_pattern_learner.tigergraph.executor import QueryNotAllowed
from mule_pattern_learner.tigergraph.queries import READ_ANSWER_KEY, READ_LABELS
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth

D = 86_400_000


def test_every_page_is_read_through_the_evaluation_executor(graph: ReferenceGraph) -> None:
    spy = Spy(graph)
    key = read_answer_key(spy, page=10)
    # 36 internal accounts (the 34 scorable, a closed one and the loan ledger), in pages of
    # 10: the fourth comes back short.
    assert len(key.truths) == 36 and spy.names() == [READ_ANSWER_KEY] * 4
    assert [truth.account_id for truth in key.truths] == sorted(key.by_id)
    assert {truth.account_id for truth in key.mules} == {
        "a_mule",
        "a_collector",
        "a_mule2",
        "a_mule3",
        "a_hr1",
        "a_hr2",
        "a_hr3",
        "a_drop",
        "a_late",
    }
    assert key.by_id["a_hr2"].mule_ring_id == 11 and key.by_id["a_drop"].mule_ring_id == -1


def test_the_evaluation_executor_runs_the_answer_key_alone(graph: ReferenceGraph) -> None:
    from mule_pattern_learner.tigergraph.executor import EvaluationExecutor

    with pytest.raises(QueryNotAllowed):
        EvaluationExecutor(Spy(graph)).run(READ_LABELS, {"after_id": "", "page": 10})


def _mule(onset: int, last: int) -> AccountTruth:
    return AccountTruth("a_m", 1, 3, onset, last, 0, "deposit", True)


def test_a_mules_status_on_each_monday_and_its_bounds() -> None:
    mule = _mule(T - D, T + 3 * D)
    # Active from the Monday after its onset to the Monday at most 7 days after its last.
    assert status(mule, T) == ACTIVE and status(mule, T + WEEK_MS) == ACTIVE
    assert status(mule, T + 2 * WEEK_MS) == AFTER
    # Its early-warning Monday is the one with onset - 7 days < t <= onset.
    assert status(mule, T - WEEK_MS) == EARLY_WARNING
    assert status(mule, T - 2 * WEEK_MS) == BEFORE
    # An onset exactly at t is not in view: t is its early-warning Monday.
    at = _mule(T, T + D)
    assert status(at, T) == EARLY_WARNING and status(at, T + WEEK_MS) == ACTIVE
    # A last payment exactly 7 days before t still makes it active.
    edge = _mule(T - 9 * D, T - WEEK_MS)
    assert status(edge, T) == ACTIVE and status(edge, T + D) == AFTER
    assert status(AccountTruth("a_n", 0, -1, 0, 0, 0, "deposit", True), T) == NON_MULE


def test_statuses_agrees_with_status() -> None:
    truths = [
        _mule(T - D, T + 3 * D),
        _mule(T, T + D),
        _mule(T - 9 * D, T - WEEK_MS),
        AccountTruth("a_n", 0, -1, 0, 0, 0, "deposit", True),
    ]
    mondays = [T + week * WEEK_MS for week in range(-2, 3)]
    pairs = [(truth, t_ms) for truth in truths for t_ms in mondays]
    found = statuses(
        np.array([t_ms for _, t_ms in pairs], dtype=np.int64),
        np.array([truth.is_mule for truth, _ in pairs], dtype=np.int64),
        np.array([truth.mule_onset_ms for truth, _ in pairs], dtype=np.int64),
        np.array([truth.mule_last_ms for truth, _ in pairs], dtype=np.int64),
    )
    assert found.tolist() == [status(truth, t_ms) for truth, t_ms in pairs]
