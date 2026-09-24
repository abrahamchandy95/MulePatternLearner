"""Small CPU reference for GSQL parity tests, never a production extraction fallback.

Input events must already respect the experiment's endpoint visibility scope.
Currency filtering matches the USD-only query. Sequence and milliseconds both
constrain visibility; timestamps are never synthesized from sequence numbers.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from .contract import ContextKey, HALF_LIVES, SamplerPlan

Event = dict[str, Any]


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


def stratify(events: list[Event], sampler: SamplerPlan) -> list[Event]:
    """Deterministic rank quantiles and additional distinct peers; no partial history."""
    result: list[Event] = []
    groups: defaultdict[str, list[Event]] = defaultdict(list)
    for event in events:
        groups[event["relation"]].append(event)
    for rows in groups.values():
        rows.sort(key=lambda e: (-e["event_seq"], e["event_id"]))
        if len(rows) > sampler.max_history:
            raise ValueError("history_capacity_exceeded")
        selected, peers = set(), set()

        def add(index: int, stratum: str) -> None:
            e = rows[index]
            if e["event_id"] not in selected:
                result.append({**e, "stratum": stratum})
                selected.add(e["event_id"])
                peers.add((e["node_type"], e["node_id"]))

        for i in range(min(sampler.recent, len(rows))):
            add(i, "recent")
        if len(rows) > sampler.recent:
            for j in range(1, sampler.older + 1):
                rank = sampler.recent + (len(rows) - sampler.recent - 1) * j // (sampler.older + 1)
                add(rank, "older")
        diverse = 0
        for i, e in enumerate(rows):
            if diverse < sampler.distinct and (e["node_type"], e["node_id"]) not in peers:
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
