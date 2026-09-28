"""The CPU mirror of the context queries' timing, decay, strata and account features."""

from typing import Any

import pytest

from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan
from mule_pattern_learner.contract.analytics_features import ANALYTICS_GROUPS
from mule_pattern_learner.reference.gsql_features import (
    MIRRORED_ACCOUNT_GROUPS,
    account_features,
    payment_features,
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


def test_account_features_count_windows_peers_ratios_and_recency() -> None:
    hour, day = 3_600_000, 86_400_000
    cutoff = 40 * day
    key = ContextKey("Account", "root", 100, cutoff)
    token = {**event(3, cutoff - 2 * day, peer="card"), "node_type": "Token"}
    missing = {**event(4, cutoff - 10 * day, "out", peer="shop"), "amount": 0}
    missing["amount_present"] = False
    history = [
        event(1, cutoff - hour // 2, peer="a", amount=300),
        {**event(2, cutoff - 2 * hour, "out", peer="b", amount=90), "relation": "payment_out"},
        token,
        missing,
        event(5, cutoff - 3 * day, peer="a", currency="EUR"),
        event(100, cutoff - 3 * day, peer="late"),
    ]
    values = account_features(history, key, first_seen_ms=cutoff - 30 * day)
    names = {n for group in MIRRORED_ACCOUNT_GROUPS for n in ANALYTICS_GROUPS[group].names}
    assert set(values) <= names
    assert values["age_days"] == 30.0
    # The EUR event and the one at the cutoff's sequence are not visible.
    assert values["visible_event_count"] == 4 and values["history_lt_5_events"] == 1
    assert (values["1h_in_count"], values["1h_in_amount"], values["1h_in_zelle"]) == (1, 300, 1)
    assert "1h_out_count" not in values
    assert (values["1d_out_count"], values["1d_out_zelle"], values["1d_out_unique"]) == (1, 0, 1)
    # A sending Token is no distinct payer; a recipient of either type is a payee.
    assert (values["7d_in_count"], values["7d_in_unique"]) == (2, 1)
    assert (values["30d_out_count"], values["30d_out_missing"], values["30d_out_unique"]) == (
        2,
        1,
        2,
    )
    # Out over in, the incoming amount floored at 1 and the ratio capped at 100.
    assert values["1d_out_in_amount_ratio"] == 0.3
    assert values["7d_out_in_amount_ratio"] == 90 / 400
    assert values["in_recency_days"] == pytest.approx(1 / 48)
    assert values["out_recency_days"] == pytest.approx(1 / 12)
    assert values["in_recency_present"] == values["out_recency_present"] == 1
    assert {k: v for k, v in values.items() if k.startswith("decay_")} == payment_features(
        history, key
    )[1]
    alone = account_features([{**history[1], "amount": 500}], key, first_seen_ms=cutoff - day)
    assert alone["1d_out_in_amount_ratio"] == 100.0 and "in_recency_days" not in alone
