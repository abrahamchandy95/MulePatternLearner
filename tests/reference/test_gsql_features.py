"""The CPU mirror of the context query's messages: timing, selection and associations."""

from typing import Any

import pytest

from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan
from mule_pattern_learner.reference.gsql_features import (
    association_messages,
    payment_features,
    payment_messages,
    root_features,
    stratify,
)


def event(
    seq: int,
    ts: int,
    direction: str = "in",
    peer: str = "peer",
    amount: float = 100,
    rail: str = "zelle",
    currency: str = "USD",
) -> dict[str, Any]:
    return dict(
        event_id=f"e{seq}",
        event_seq=seq,
        event_ts_ms=ts,
        relation=f"zelle_{direction}",
        node_type="Account",
        node_id=peer,
        rail=rail,
        currency=currency,
        amount=amount,
        amount_present=True,
    )


def test_timing_uses_both_clocks_and_marks_censored_receipts():
    root = ContextKey("Account", "root", 10, 10000)
    history = [
        event(1, 1000),
        event(2, 1000, "out", amount=90),
        event(3, 2000),
        event(4, 3000, "out", amount=80, rail="cash"),
        event(5, 9000),
        event(6, 11000, "out"),
        event(10, 9500, "out"),
        event(8, 9200, "out", currency="EUR"),
    ]
    rows = payment_features(history, root)
    first = rows["zelle_in:e1"]
    assert first["flow_present"] and first["flow_delay_seconds"] == 0
    assert first["flow_amount_ratio"] == 0.9
    assert rows["zelle_in:e3"]["flow_delay_seconds"] == 1
    assert not rows["zelle_in:e3"]["flow_same_rail"]
    assert rows["zelle_out:e4"]["flow_delay_seconds"] == 1
    last = rows["zelle_in:e5"]
    assert last["flow_censored"] and not last["flow_present"]
    assert last["flow_observation_seconds"] == 1
    assert rows == payment_features(history[:5], root)
    assert rows["zelle_in:e3"]["pair_prior_count"] == 1
    assert rows["zelle_in:e5"]["pair_first_age_seconds"] == 8


def test_a_missing_amount_is_not_invented():
    day = 86400000
    key = ContextKey("Account", "root", 10, 2 * day)
    e = event(1, day)
    e["amount"], e["amount_present"] = 0, False
    rows = payment_features([e, event(2, day + 1000, "out")], key)
    assert not rows["zelle_in:e1"]["flow_ratio_present"]


def test_rank_and_peer_picks_survive_a_recent_burst_without_duplicate_events():
    pool = PoolPlan(4, 3, 2, 2, 2048)
    old = [event(i, i * 1000, peer=f"p{i}") for i in range(1, 61)]
    burst = [event(i, 61000 + i, peer="burst") for i in range(61, 101)]
    rows = stratify(old + burst, pool)
    # The 4 most recent, the ranks 27, 51 and 75 of 100 (sequences 73, 49 and 25), then
    # the two newest events of peers not yet selected, in the query's order.
    assert [e["event_seq"] for e in rows] == [100, 99, 98, 97, 73, 49, 25, 60, 59]
    assert stratify(list(reversed(old + burst)), pool) == rows
    with pytest.raises(ValueError, match="capacity"):
        stratify(old + burst, PoolPlan(4, 3, 2, 2, 32))


def test_payment_messages_carry_every_field_of_the_selected_events():
    root = ContextKey("Account", "root", 10, 10000)
    peer = {"peer_first_ms": 5, "peer_external": True, "peer_deposit": False}
    history = [{**event(seq, seq * 1000, peer="a"), **peer} for seq in (1, 2, 3)]
    rows = payment_messages(history, root, PoolPlan(2, 0, 0, 0, 2048))
    assert [m["event_seq"] for m in rows] == [3, 2]
    newest = rows[0]
    assert newest["age_ms"] == 7000 and newest["peer_first_ms"] == 5 and newest["peer_external"]
    assert (newest["gap_ms"], newest["pair_prior_count"]) == (1000, 2)
    assert newest == {**newest, **payment_features(history, root)["zelle_in:e3"]}


def test_associations_keep_the_newest_valid_tenures_of_each_relation():
    key = ContextKey("Party", "owner", 100, 10000)

    def link(node: str, valid_from: int, valid_to: int = 0, first_seq: int = 1) -> dict[str, Any]:
        return dict(
            relation="Party_Owns_Account",
            node_type="Account",
            node_id=node,
            valid_from_seq=valid_from,
            valid_to_seq=valid_to,
            first_seen_seq=first_seq,
            first_seen_ts_ms=1000,
            external=node == "b",
            deposit=True,
        )

    # Ended (c), not yet valid at state_seq 99 (d) and first seen after it (e, f) are left
    # out; b and b2 tie on valid_from_seq, so the node id orders them.
    links = [
        link("a", 10),
        link("b", 20),
        link("b2", 20),
        link("c", 30, 50),
        link("d", 100),
        link("e", 5, 0, 100),
        link("f", 1, 0, 200),
    ]
    rows = association_messages(links, key, 2)
    assert [(m["node_id"], m["peer_external"]) for m in rows] == [("b", True), ("b2", False)]
    assert rows[0]["event_seq"] == 100 and rows[0]["event_ts_ms"] == 10000
    assert rows[0]["event_id"] == "" and rows[0]["rail"] == "unknown"
    assert root_features(key, True, True) == {}
    account = ContextKey("Account", "a", 100, 10000)
    assert root_features(account, False, True) == {"is_deposit": 1.0}
