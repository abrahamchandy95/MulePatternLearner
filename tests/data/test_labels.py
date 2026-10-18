"""The labels: every identified mule of read_labels, their identity, when a mule is active, and
the identified mules never scored on a Monday of their activity (docs/design.md, "Labels")."""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.labels import (
    active,
    active_mules,
    label_identity,
    read_identified,
    unscored,
)
from mule_pattern_learner.testing.calendar import (
    EARLY,
    EDGE,
    GONE,
    JOINED,
    LONG,
    SHORT,
    Spy,
    calendar_world,
)
from mule_pattern_learner.testing.world import DAY_MS, Mule, T
from mule_pattern_learner.tigergraph.queries.read_labels import IdentifiedMule
from mule_pattern_learner.tigergraph.queries.rows import QueryRefused

W = 7 * DAY_MS
IDENTIFIED = {
    "a_long": LONG,
    "a_short": SHORT,
    "a_edge": EDGE,
    "a_gone": GONE,
    "a_early": EARLY,
}


def test_every_identified_mule_is_read_and_no_other() -> None:
    spy = Spy(calendar_world().graph())
    labels = read_identified(spy, page=2)
    expected = tuple(
        IdentifiedMule(account, mule.onset_ms, mule.last_ms)
        for account, mule in sorted(IDENTIFIED.items())
    )
    # a_hidden is a mule no one identified, so read_labels never prints it.
    assert labels.mules == expected
    assert labels.accounts == frozenset(IDENTIFIED)
    # Five mules in pages of 2: 2, 2 and 1.
    assert spy.names() == ["read_labels"] * 3
    assert read_identified(spy) == labels


def test_the_label_identity_is_the_hash_of_the_rows() -> None:
    mules = [IdentifiedMule("a_1", 10, 20), IdentifiedMule("a_2", 30, 30)]
    found = label_identity(mules)
    assert len(found) == 32 and found == label_identity(list(reversed(mules)))
    assert label_identity([IdentifiedMule("a_1", 10, 21), mules[1]]) != found
    assert label_identity(mules[:1]) != found
    # Golden: blake2b, 16 bytes, of [["a_1",10,20],["a_2",30,30]].
    assert found == "dc34298aea9b0ac7dfbedb645a833490"


def test_a_mule_without_a_valid_activity_period_is_refused() -> None:
    w = calendar_world()
    w.customer("bad", JOINED, mule=Mule(T, T - 1, identified=True))
    with pytest.raises(QueryRefused, match="invalid_activity"):
        read_identified(Spy(w.graph()))


def test_a_mule_is_active_from_after_its_onset_to_a_week_after_its_last_payment() -> None:
    mule = IdentifiedMule("a_1", T, T + 3 * DAY_MS)
    # An onset exactly at t is not in view.
    assert not active(mule, T)
    assert active(mule, T + 1) and active(mule, T + W)
    # Its last payment plus 7 days exactly at t is active; a millisecond later is not.
    assert active(mule, T + 3 * DAY_MS + W)
    assert not active(mule, T + 3 * DAY_MS + W + 1)
    assert active_mules([mule, IdentifiedMule("a_2", T + W, T + W)], T + W) == [mule]


def test_mules_never_scored_on_a_monday_of_their_activity_are_reported() -> None:
    mondays = DEFAULT_CONFIG.calendar.mondays()
    mules = [
        IdentifiedMule(account, mule.onset_ms, mule.last_ms)
        for account, mule in sorted(IDENTIFIED.items())
    ]
    found = unscored(mules, mondays, {"a_long": 26, "a_short": 1, "a_edge": 1})
    # a_early's activity ended before the first scored Monday; a_gone was active on T + 1 week
    # and closed, so never scorable then.
    assert found.no_active_monday == ("a_early",)
    assert found.never_scorable == ("a_gone",)
    assert sum(active(mules[2], t_ms) for t_ms in mondays) == 1  # a_gone
