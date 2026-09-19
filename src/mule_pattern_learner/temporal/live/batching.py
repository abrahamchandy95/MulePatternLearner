"""Assemble two-hop batches without collapsing different times for the same node."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .contract import ContextKey, FEATURE_NAMES, NODE_TYPES, RELATIONS, RAILS
from .source import ContextSource
from .memory import BatchIndex, BatchLimits


def child_key(message: dict[str, Any], parent: ContextKey | None = None) -> ContextKey:
    return ContextKey(
        message["node_type"],
        message["node_id"],
        int(message["event_seq"]),
        int(message["event_ts_ms"]),
        parent.scope_id if parent else "",
        parent.visibility_phase if parent else 3,
    )


def select_messages(context: dict[str, Any], fanout: int) -> list[dict[str, Any]]:
    """Deterministic relation-interleaved recent history; no label-based selection."""
    if fanout < 1:
        raise ValueError("Fanout must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in context["messages"]:
        buckets[message["relation"]].append(message)
    for messages in buckets.values():
        messages.sort(key=lambda m: (-int(m["event_seq"]), m["event_id"], m["node_id"]))
    selected = []
    for position in range(max((len(v) for v in buckets.values()), default=0)):
        for relation in RELATIONS:
            if position < len(buckets[relation]):
                selected.append(buckets[relation][position])
                if len(selected) == fanout:
                    return selected
    return selected


def node_features(context: dict[str, Any]) -> np.ndarray:
    values = {**context["features"], "type_" + context["node_type"]: 1.0}
    unexpected = set(values) - set(FEATURE_NAMES)
    if unexpected:
        raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    x = np.asarray([values.get(name, 0) for name in FEATURE_NAMES], dtype=np.float32)
    # Fixed transforms need no full-dataset fit and do not use validation/test data.
    x[8:] = np.log1p(x[8:])
    return x


def base_features(message: dict[str, Any]) -> np.ndarray:
    key = child_key(message)
    first_ms = int(message["peer_first_ms"])
    if first_ms <= 0 or first_ms > key.cutoff_ms:
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    x = np.zeros(9, dtype=np.float32)
    x[NODE_TYPES.index(key.node_type)] = 1
    x[6:9] = (
        message["peer_external"],
        message["peer_deposit"],
        np.log1p((key.cutoff_ms - first_ms) / 86_400_000),
    )
    return x


def edge_features(context: dict[str, Any], message: dict[str, Any]) -> np.ndarray:
    is_event = bool(message["event_id"])
    x = np.zeros(135, dtype=np.float32)
    x[:7] = (
        np.log1p(message["amount"]),
        message["amount_present"],
        is_event,
        message["gap_present"],
        np.log1p(message["pair_count_1h"]),
        np.log1p(message["pair_count_1d"]),
        np.log1p(message["pair_count_7d"]),
    )
    if is_event:
        event_key = message["relation"] + ":" + message["event_id"]
        x[7:71] = context["age_encoding"][event_key]
        if message["gap_present"]:
            x[71:] = context["gap_encoding"][event_key]
    return x


def make_live_batch(
    store: ContextSource,
    roots: list[ContextKey],
    *,
    fanouts: tuple[int, int] = (8, 4),
    device: str | torch.device = "cpu",
    limits: BatchLimits = BatchLimits(),
) -> dict[str, torch.Tensor]:
    if not roots or len(fanouts) != 2 or min(fanouts) < 1 or max(fanouts) > 64:
        raise ValueError("Nonempty roots and two fanouts in [1,64] are required")
    limits.validate(len(roots), fanouts)
    if len({(key.scope_id, key.visibility_phase) for key in roots}) != 1:
        raise ValueError("A batch must have one visibility scope and phase")
    root_contexts = store.fetch(roots)
    first = [select_messages(row, fanouts[0]) for row in root_contexts]
    lookup = BatchIndex(
        roots
        + [child_key(m, key) for key, messages in zip(roots, first, strict=True) for m in messages],
        capacity=limits.max_contexts,
    )
    unique = lookup.keys
    contexts = store.fetch(unique)
    second = [select_messages(row, fanouts[1]) for row in contexts]
    arrays: dict[str, np.ndarray] = {
        "root_positions": np.asarray([lookup[key] for key in roots], dtype=np.int64),
        "x": np.stack([node_features(row) for row in contexts]),
        "neighbor_positions": np.zeros((len(roots), fanouts[0]), dtype=np.int64),
        "second_x": np.zeros((len(unique), fanouts[1], 9), dtype=np.float32),
    }
    for prefix, rows, messages, fanout in (
        ("first_", root_contexts, first, fanouts[0]),
        ("second_", contexts, second, fanouts[1]),
    ):
        arrays[prefix + "edge"] = np.zeros((len(rows), fanout, 135), dtype=np.float32)
        arrays[prefix + "relation"] = np.zeros((len(rows), fanout), dtype=np.int64)
        arrays[prefix + "rail"] = np.zeros((len(rows), fanout), dtype=np.int64)
        arrays[prefix + "mask"] = np.zeros((len(rows), fanout), dtype=bool)
        for i, (row, neighbors) in enumerate(zip(rows, messages, strict=True)):
            for j, message in enumerate(neighbors):
                arrays[prefix + "edge"][i, j] = edge_features(row, message)
                arrays[prefix + "relation"][i, j] = RELATIONS.index(message["relation"])
                arrays[prefix + "rail"][i, j] = RAILS.index(message["rail"])
                arrays[prefix + "mask"][i, j] = True
                if prefix == "first_":
                    arrays["neighbor_positions"][i, j] = lookup[child_key(message, roots[i])]
                else:
                    arrays["second_x"][i, j] = base_features(message)
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
