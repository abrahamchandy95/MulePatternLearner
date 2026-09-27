"""The CPU mirror of the context query's timing, decay and strata."""

from typing import Any

import pytest

from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan
from mule_pattern_learner.reference.gsql_features import payment_features, stratify


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
    rows, _ = payment_features(history, root)
    first = rows["zelle_in:e1"]
    assert first["flow_present"] and first["flow_delay_seconds"] == 0
    assert first["flow_amount_ratio"] == 0.9
    assert rows["zelle_in:e3"]["flow_delay_seconds"] == 1
    assert not rows["zelle_in:e3"]["flow_same_rail"]
    assert rows["zelle_out:e4"]["flow_delay_seconds"] == 1
    last = rows["zelle_in:e5"]
    assert last["flow_censored"] and not last["flow_present"]
    assert last["flow_observation_seconds"] == 1
    assert rows == payment_features(history[:5], root)[0]
    assert rows["zelle_in:e3"]["pair_prior_count"] == 1
    assert rows["zelle_in:e5"]["pair_first_age_seconds"] == 8


def test_decay_is_smooth_and_missing_amount_is_not_invented():
    day = 86400000
    key = ContextKey("Account", "root", 10, 2 * day)
    e = event(1, day)
    e["amount"], e["amount_present"] = 0, False
    rows, sums = payment_features([e, event(2, day + 1000, "out")], key)
    assert sums["decay_1d_in_count"] == 0.5
    assert sums["decay_1d_in_amount"] == 0
    assert not rows["zelle_in:e1"]["flow_ratio_present"]
    later = ContextKey("Account", "root", 10, 2 * day + 1000)
    assert 0 < payment_features([e], later)[1]["decay_1d_in_count"] < 0.5


def test_rank_and_peer_strata_survive_a_recent_burst_without_duplicate_events():
    pool = PoolPlan(4, 3, 2, 2, 2048)
    old = [event(i, i * 1000, peer=f"p{i}") for i in range(1, 61)]
    burst = [event(i, 61000 + i, peer="burst") for i in range(61, 101)]
    rows = stratify(old + burst, pool)
    assert len({e["event_id"] for e in rows}) == len(rows) == 9
    assert {e["stratum"] for e in rows} == {"recent", "older", "distinct"}
    assert any(e["event_seq"] < 61 for e in rows)
    assert stratify(list(reversed(old + burst)), pool) == rows
    with pytest.raises(ValueError, match="capacity"):
        stratify(old + burst, PoolPlan(4, 3, 2, 2, 32))
