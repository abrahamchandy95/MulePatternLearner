"""Scalar node, base and edge features: the per-item reference of batching's matrices.

Tests compare `batching.node_matrix`, `base_matrix` and `edge_block` against these
functions, one context or message at a time. Nothing in training, preparation or
scoring imports this module; the restructure places it under `reference/`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..encoding import fourier64
from .batching import DAY_MS, child_key, pool_activity
from .contract import FEATURE_GROUPS, POOL_GROUPS, FeaturePlan

# Columns kept as they are; every other column gets log1p.
IDENTITY = frozenset(
    {"gap_present"} | {n for spec in FEATURE_GROUPS.values() for n in spec.identity}
)
NODE_NAMES = frozenset(
    n for spec in FEATURE_GROUPS.values() if spec.path in ("node", "summary") for n in spec.names
)


def transform(name: str, value: float) -> float:
    if name in IDENTITY or "_fourier_" in name:
        return value
    return float(np.log1p(value))


def node_features(context: dict[str, Any], plan: FeaturePlan = FeaturePlan()) -> np.ndarray:
    """One context's node and summary columns, pool counts included for pool plans."""
    values = {**context["features"], "type_" + context["node_type"]: 1.0}
    unexpected = set(values) - NODE_NAMES
    if unexpected:
        raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    if set(POOL_GROUPS) & set(plan.groups):
        values |= pool_activity(context)
    return np.asarray([transform(n, values.get(n, 0)) for n in plan.node_names], dtype=np.float32)


def base_features(
    message: dict[str, Any], plan: FeaturePlan = FeaturePlan(), *, history_withheld: bool = False
) -> np.ndarray:
    """Outermost peer features known from the message alone."""
    key = child_key(message)
    first_ms = int(message["peer_first_ms"])
    if first_ms <= 0 or first_ms > key.cutoff_ms:
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    values = {
        "type_" + key.node_type: 1,
        "is_external": message["peer_external"],
        "is_deposit": message["peer_deposit"],
        "age_days": (key.cutoff_ms - first_ms) / DAY_MS,
        "history_withheld": int(history_withheld),
    }
    return np.asarray(
        [transform(n, values.get(n, 0)) for n in plan.names("node")], dtype=np.float32
    )


def edge_features(message: dict[str, Any], plan: FeaturePlan = FeaturePlan()) -> np.ndarray:
    """One message's edge columns; Fourier features come from age_ms and gap_ms."""
    values = {**message, "is_event": bool(message["event_id"])}
    if values["is_event"] and "time_encoding" in plan.groups:
        age = fourier64(np.array([message["age_ms"]], dtype=np.int64))[0]
        values.update({f"age_fourier_{i}": v for i, v in enumerate(age)})
        if message["gap_present"]:
            gap = fourier64(np.array([message["gap_ms"]], dtype=np.int64))[0]
            values.update({f"gap_fourier_{i}": v for i, v in enumerate(gap)})
    missing = [n for n in plan.edge_names if n not in values and "_fourier_" not in n]
    if missing:
        raise ValueError(f"Message lacks required fields: {missing}")
    return np.asarray([transform(n, values.get(n, 0)) for n in plan.edge_names], dtype=np.float32)
