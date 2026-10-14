"""Small CPU reference for GSQL parity tests, never a production extraction fallback.

Input events must already respect the experiment's endpoint visibility scope.
Currency filtering matches the USD-only query. Sequence and milliseconds both
constrain visibility; timestamps are never synthesized from sequence numbers.
payment_messages and association_messages mirror every field of the messages the
training context query returns for a root (an Account's payments and any root's
associations), and root_features its node features.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import PoolPlan

Event = dict[str, Any]

# The fields of an association message the query prints as zeros: it is no payment.
_NO_PAYMENT: Event = {
    "rail": "unknown",
    "event_id": "",
    "amount": 0.0,
    "amount_present": False,
    "age_ms": 0,
    "gap_ms": 0,
    "gap_present": False,
    "pair_prior_count": 0,
    "pair_first_age_seconds": 0.0,
    "pair_first_present": False,
    "flow_delay_seconds": 0.0,
    "flow_present": False,
    "flow_censored": False,
    "flow_observation_seconds": 0.0,
    "flow_amount_ratio": 0.0,
    "flow_ratio_present": False,
    "flow_same_rail": False,
}


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

        def add(index: int) -> None:
            e = rows[index]
            if e["event_id"] not in selected:
                result.append(e)
                selected.add(e["event_id"])
                peers.add((e["node_type"], e["node_id"]))

        for i in range(min(pool.recent, len(rows))):
            add(i)
        if len(rows) > pool.recent:
            for j in range(1, pool.older + 1):
                add(pool.recent + (len(rows) - pool.recent - 1) * j // (pool.older + 1))
        diverse = 0
        for i, e in enumerate(rows):
            if diverse < pool.distinct and (e["node_type"], e["node_id"]) not in peers:
                add(i)
                diverse += 1
    return result


def payment_features(events: list[Event], key: ContextKey) -> dict[str, dict[str, Any]]:
    """Independent numerical specification for event timing, by relation:event_id."""
    history = visible_history(events, key)
    prior: defaultdict[tuple[str, str, str, str], list[Event]] = defaultdict(list)
    features: dict[str, dict[str, Any]] = {}
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
    return features


def payment_messages(events: list[Event], key: ContextKey, pool: PoolPlan) -> list[Event]:
    """The payment messages of an Account root, every field the query prints.

    events are the root's payments with their peer (node_type, node_id) and the peer's
    first_seen_ts_ms and flags (peer_first_ms, peer_external, peer_deposit).
    """
    features = payment_features(events, key)
    return [
        {
            "node_type": e["node_type"],
            "node_id": e["node_id"],
            "relation": e["relation"],
            "rail": e["rail"],
            "event_id": e["event_id"],
            "event_seq": e["event_seq"],
            "event_ts_ms": e["event_ts_ms"],
            "amount": e["amount"],
            "amount_present": e["amount_present"],
            "age_ms": key.cutoff_ms - e["event_ts_ms"],
            "peer_first_ms": e["peer_first_ms"],
            "peer_external": e["peer_external"],
            "peer_deposit": e["peer_deposit"],
            **features[e["relation"] + ":" + e["event_id"]],
        }
        for e in stratify(visible_history(events, key), pool)
    ]


def association_messages(links: list[Event], key: ContextKey, associations: int) -> list[Event]:
    """The association messages of a root, every field the query prints.

    links are the root's association edges, each with its relation, its far end
    (node_type, node_id, first_seen_seq, first_seen_ts_ms and, for an Account, its
    external and deposit flags) and its tenure. A link valid at state_seq =
    cutoff_seq - 1, whose far end was first seen by then, is visible; each relation keeps
    its `associations` newest by valid_from_seq, ties by node id. Its message carries the
    context cutoff as its clocks.
    """
    state = key.cutoff_seq - 1
    groups: defaultdict[str, list[Event]] = defaultdict(list)
    for link in links:
        if (
            link["valid_from_seq"] <= state
            and (link["valid_to_seq"] == 0 or state < link["valid_to_seq"])
            and link["first_seen_seq"] <= state
            and link["first_seen_ts_ms"] <= key.cutoff_ms
        ):
            groups[link["relation"]].append(link)
    messages: list[Event] = []
    for relation, rows in groups.items():
        rows.sort(key=lambda link: (-link["valid_from_seq"], link["node_id"]))
        messages += [
            {
                **_NO_PAYMENT,
                "node_type": link["node_type"],
                "node_id": link["node_id"],
                "relation": relation,
                "event_seq": key.cutoff_seq,
                "event_ts_ms": key.cutoff_ms,
                "peer_first_ms": link["first_seen_ts_ms"],
                "peer_external": link["external"],
                "peer_deposit": link["deposit"],
            }
            for link in rows[:associations]
        ]
    return messages


def root_features(key: ContextKey, external: bool, deposit: bool) -> dict[str, float]:
    """The node features the query prints for a root: an Account's set flags, else none."""
    if key.node_type != "Account":
        return {}
    return {
        name: 1.0 for name, flag in (("is_external", external), ("is_deposit", deposit)) if flag
    }
