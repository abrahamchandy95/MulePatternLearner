"""Small CPU reference for GSQL parity tests, never a production extraction fallback.

Input events must already respect the experiment's endpoint visibility scope.
Currency filtering matches the USD-only query. Sequence and milliseconds both
constrain visibility; timestamps are never synthesized from sequence numbers.
payment_features and stratify mirror what the training context query computes for
sampled messages; account_features mirrors the account features the analytics context
query computes from the root's payment history.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from ..contract.analytics_features import (
    AMOUNT_RATIO_CAP,
    AMOUNT_RATIO_FLOOR,
    AMOUNT_RATIO_WINDOWS,
    HALF_LIVES,
    WINDOWS,
)
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import PoolPlan

Event = dict[str, Any]
DAY_MS = 86_400_000
# The analytics groups account_features mirrors. association_counts and identity_order
# read the root's associations, which a payment history does not hold.
MIRRORED_ACCOUNT_GROUPS = (
    "entity_age",
    "history_support",
    "rolling_windows",
    "amount_ratios",
    "recency",
    "decayed_activity",
)


def visible_history(events: list[Event], key: ContextKey) -> list[Event]:
    return sorted(
        (
            e
            for e in events
            if 0 < e["event_seq"] < key.cutoff_seq
            and 0 < e["event_ts_ms"] <= key.cutoff_ms
            and e.get("currency", "USD") == "USD"
        ),
        key=lambda e: (e["event_seq"], e["event_id"], e["relation"]),
    )


def stratify(events: list[Event], pool: PoolPlan) -> list[Event]:
    """The candidate pool of one relation's history, as the context query selects it.

    The most recent events, deterministic rank quantiles of the older ones and
    additional distinct peers; no partial history.
    """
    result: list[Event] = []
    groups: defaultdict[str, list[Event]] = defaultdict(list)
    for event in events:
        groups[event["relation"]].append(event)
    for rows in groups.values():
        rows.sort(key=lambda e: (-e["event_seq"], e["event_id"]))
        if len(rows) > pool.max_history:
            raise ValueError("history_capacity_exceeded")
        selected, peers = set(), set()

        def add(index: int, stratum: str) -> None:
            e = rows[index]
            if e["event_id"] not in selected:
                result.append({**e, "stratum": stratum})
                selected.add(e["event_id"])
                peers.add((e["node_type"], e["node_id"]))

        for i in range(min(pool.recent, len(rows))):
            add(i, "recent")
        if len(rows) > pool.recent:
            for j in range(1, pool.older + 1):
                rank = pool.recent + (len(rows) - pool.recent - 1) * j // (pool.older + 1)
                add(rank, "older")
        diverse = 0
        for i, e in enumerate(rows):
            if diverse < pool.distinct and (e["node_type"], e["node_id"]) not in peers:
                add(i, "distinct")
                diverse += 1
    return result


def payment_features(
    events: list[Event], key: ContextKey
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Independent numerical specification for event timing and smooth summaries."""
    history = visible_history(events, key)
    prior: defaultdict[tuple[str, str, str, str], list[Event]] = defaultdict(list)
    features: dict[str, dict[str, Any]] = {}
    summary: dict[str, float] = {}
    for e in history:
        pair = (e["relation"], e["rail"], e["node_type"], e["node_id"])
        earlier = [
            p
            for p in prior[pair]
            if p["event_seq"] < e["event_seq"] and p["event_ts_ms"] <= e["event_ts_ms"]
        ]
        incoming = e["relation"].endswith("_in")
        matching = [
            o
            for o in history
            if (o["relation"].endswith("_out") if incoming else o["relation"].endswith("_in"))
            and (
                o["event_seq"] > e["event_seq"] and o["event_ts_ms"] >= e["event_ts_ms"]
                if incoming
                else o["event_seq"] < e["event_seq"] and o["event_ts_ms"] <= e["event_ts_ms"]
            )
        ]
        other = (matching[0] if incoming else matching[-1]) if matching else None
        amounts = other is not None and e["amount_present"] and other["amount_present"]
        received = e if incoming else other
        sent = other if incoming else e
        features[e["relation"] + ":" + e["event_id"]] = {
            "pair_prior_count": len(earlier),
            "pair_first_age_seconds": (e["event_ts_ms"] - earlier[0]["event_ts_ms"]) / 1000
            if earlier
            else 0,
            "pair_first_present": bool(earlier),
            "gap_ms": e["event_ts_ms"] - earlier[-1]["event_ts_ms"] if earlier else 0,
            "gap_present": bool(earlier),
            "flow_delay_seconds": abs(other["event_ts_ms"] - e["event_ts_ms"]) / 1000
            if other
            else 0,
            "flow_present": other is not None,
            "flow_censored": incoming and other is None,
            "flow_observation_seconds": (key.cutoff_ms - e["event_ts_ms"]) / 1000
            if incoming
            else 0,
            "flow_amount_ratio": min(sent["amount"] / max(received["amount"], 1), 100)
            if amounts and sent is not None and received is not None
            else 0,
            "flow_ratio_present": bool(amounts),
            "flow_same_rail": other is not None and other["rail"] == e["rail"],
        }
        prior[pair].append(e)
        direction = "in" if incoming else "out"
        for name, half in HALF_LIVES.items():
            decay = 2 ** (-(key.cutoff_ms - e["event_ts_ms"]) / half)
            for field, value in (("count", 1), ("amount", e["amount"])):
                label = f"decay_{name}_{direction}_{field}"
                summary[label] = summary.get(label, 0) + value * decay
    assert all(math.isfinite(v) and v >= 0 for v in summary.values())
    return features, summary


def account_features(events: list[Event], key: ContextKey, first_seen_ms: int) -> dict[str, float]:
    """The features of MIRRORED_ACCOUNT_GROUPS, as the analytics context query computes them.

    ``events`` is the root's payment history (each with its counterparty as node_type and
    node_id) and ``first_seen_ms`` the root's first observation. A window counts the
    visible events less than its length before the cutoff; its distinct peers are the
    recipients of outgoing events and the sending Accounts of incoming ones. The query
    prints only what it accumulated, so a feature missing here is zero there.
    """
    history = visible_history(events, key)
    values: dict[str, float] = {"age_days": (key.cutoff_ms - first_seen_ms) / DAY_MS}
    peers: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    last: dict[str, int] = {}

    def add(name: str, value: float) -> None:
        values[name] = values.get(name, 0.0) + value

    for e in history:
        direction = "in" if e["relation"].endswith("_in") else "out"
        last[direction] = max(last.get(direction, 0), e["event_ts_ms"])
        for window, length in WINDOWS.items():
            if key.cutoff_ms - e["event_ts_ms"] < length:
                prefix = f"{window}_{direction}"
                add(f"{prefix}_count", 1)
                add(f"{prefix}_amount", e["amount"])
                add(f"{prefix}_missing", float(not e["amount_present"]))
                add(f"{prefix}_zelle", float(e["relation"].startswith("zelle")))
                if direction == "out" or e["node_type"] == "Account":
                    peers[f"{prefix}_unique"].add((e["node_type"], e["node_id"]))
    values |= {name: float(len(found)) for name, found in peers.items()}
    for window in AMOUNT_RATIO_WINDOWS:
        incoming = max(values.get(f"{window}_in_amount", 0.0), AMOUNT_RATIO_FLOOR)
        outgoing = values.get(f"{window}_out_amount", 0.0)
        values[f"{window}_out_in_amount_ratio"] = min(outgoing / incoming, AMOUNT_RATIO_CAP)
    values["visible_event_count"] = float(len(history))
    if len(history) < 5:
        values["history_lt_5_events"] = 1.0
    for direction, ts in last.items():
        values[f"{direction}_recency_days"] = (key.cutoff_ms - ts) / DAY_MS
        values[f"{direction}_recency_present"] = 1.0
    values |= payment_features(events, key)[1]
    return values
