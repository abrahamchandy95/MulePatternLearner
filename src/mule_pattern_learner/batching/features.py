"""Vectorised node, base and edge feature matrices of many contexts or messages.

Tensors are assembled by column gathers; Fourier columns stay zero here and are
computed on the target device from `age_ms` and `gap_ms` (assemble.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..contract.feature_groups import (
    FEATURE_GROUPS,
    POOL_ACTIVITY_FEATURES,
    POOL_GROUPS,
    POOL_INTERNAL_FEATURES,
    FeaturePlan,
)
from ..contract.graph_schema import CHANNELS, RAILS, RELATION_INDEX, STRATA
from .pool_counts import pool_activity

DAY_MS = 86_400_000
# Columns kept as they are; every other column gets log1p.
IDENTITY = frozenset(
    {"gap_present"} | {n for spec in FEATURE_GROUPS.values() for n in spec.identity}
)
# Every column a context's node and summary features may carry.
NODE_NAMES = frozenset(
    n for spec in FEATURE_GROUPS.values() if spec.path in ("node", "summary") for n in spec.names
)
_POOL_NAMES = frozenset(POOL_ACTIVITY_FEATURES + POOL_INTERNAL_FEATURES)
_RAIL = {name: i for i, name in enumerate(RAILS)}
_CHANNEL = {name: i for i, name in enumerate(CHANNELS)}
_STRATUM = {name: i for i, name in enumerate(STRATA)}


def _log_columns(names: Sequence[str]) -> np.ndarray:
    return np.asarray([n not in IDENTITY and "_fourier_" not in n for n in names], dtype=bool)


def _pooled(plan: FeaturePlan) -> bool:
    return any(group in plan.groups for group in POOL_GROUPS)


def _column(items: Sequence[dict[str, Any]], name: str, dtype: Any = np.float64) -> np.ndarray:
    try:
        return np.fromiter((m[name] for m in items), dtype=dtype, count=len(items))
    except KeyError:
        raise ValueError(f"Message lacks required field {name!r}") from None
    except (TypeError, ValueError) as error:  # keep the tuple form for Python 3.12/3.13
        raise ValueError(f"Message field {name!r} is not numeric: {error}") from None


def node_matrix(
    rows: Sequence[dict[str, Any]], plan: FeaturePlan, *, pooled: int | None = None
) -> np.ndarray:
    """The node and summary columns of many contexts (reference.batch_features.node_features).

    Only the first ``pooled`` rows (all when None) get the client-computed pool counts;
    the rest keep zeros there. Split batches pass their roots, which lead the rows,
    since the split model reads the pool columns of roots only.
    """
    names = plan.node_names
    for row in rows:
        unexpected = set(row["features"]) - NODE_NAMES
        if unexpected:
            raise ValueError(f"Unrecognized feature fields: {sorted(unexpected)}")
    pools = [pool_activity(row) for row in rows[:pooled]] if _pooled(plan) else []
    values = np.zeros((len(rows), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if name in _POOL_NAMES:
            values[: len(pools), j] = [pool[name] for pool in pools]
            continue
        values[:, j] = np.fromiter(
            (
                1.0 if name == "type_" + row["node_type"] else row["features"].get(name, 0)
                for row in rows
            ),
            dtype=np.float64,
            count=len(rows),
        )
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    return values.astype(np.float32)


def base_matrix(
    messages: Sequence[dict[str, Any]], withheld: np.ndarray, plan: FeaturePlan
) -> np.ndarray:
    """The base columns of many second-hop messages (reference.batch_features.base_features)."""
    names = plan.names("node")
    cutoff = _column(messages, "event_ts_ms", np.int64)
    first = _column(messages, "peer_first_ms", np.int64)
    if np.any(first <= 0) or np.any(first > cutoff):
        raise ValueError("Peer metadata is not visible at its historical cutoff")
    values = np.zeros((len(messages), len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if name.startswith("type_"):
            values[:, j] = [m["node_type"] == name[5:] for m in messages]
        elif name == "is_external":
            values[:, j] = _column(messages, "peer_external")
        elif name == "is_deposit":
            values[:, j] = _column(messages, "peer_deposit")
        elif name == "age_days":
            values[:, j] = (cutoff - first) / DAY_MS
        elif name == "history_withheld":
            values[:, j] = withheld
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    return values.astype(np.float32)


def _stratum(message: dict[str, Any]) -> int:
    name = message.get("stratum", "recent" if message["event_id"] else "association")
    try:
        return _STRATUM[name]
    except KeyError:
        raise ValueError(f"Unknown sampling stratum {name!r}") from None


def edge_block(messages: Sequence[dict[str, Any]], plan: FeaturePlan) -> dict[str, np.ndarray]:
    """Edge features and categorical codes for many messages, by column.

    Fourier columns stay zero here; `age_ms`/`gap_ms` and their masks are returned
    so the encoding can run on the target device.
    """
    names = plan.edge_names
    count = len(messages)
    event = np.fromiter((bool(m["event_id"]) for m in messages), dtype=bool, count=count)
    values = np.zeros((count, len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        if "_fourier_" not in name:
            values[:, j] = event if name == "is_event" else _column(messages, name)
    log = _log_columns(names)
    values[:, log] = np.log1p(values[:, log])
    try:
        relation = [RELATION_INDEX[m["relation"]] for m in messages]
        rail = [_RAIL[m["rail"]] for m in messages]
    except KeyError as error:
        raise ValueError(f"Unknown relation or rail {error.args[0]!r}") from None
    other = _CHANNEL["other"]
    block = {
        "edge": values.astype(np.float32),
        "relation": np.asarray(relation, dtype=np.int64),
        "rail": np.asarray(rail, dtype=np.int64),
        "channel": np.asarray(
            [_CHANNEL.get(m.get("channel", "unknown"), other) for m in messages], dtype=np.int64
        ),
        "stratum": np.asarray([_stratum(m) for m in messages], dtype=np.int64),
    }
    if "time_encoding" in plan.groups:
        gap_on = event & _column(messages, "gap_present", bool)
        events = [m for m, e in zip(messages, event, strict=True) if e]
        age = np.zeros(count, dtype=np.int64)
        age[event] = _column(events, "age_ms", np.int64)
        gap = np.zeros(count, dtype=np.int64)
        gap[gap_on] = _column(
            [m for m, g in zip(messages, gap_on, strict=True) if g], "gap_ms", np.int64
        )
        if np.any(age < 0) or np.any(gap < 0):
            raise ValueError("Future timestamps cannot be encoded as historical context")
        block |= {"age_ms": age, "age_on": event, "gap_ms": gap, "gap_on": gap_on}
    return block
