"""The temporal_training_context request protocol: requests, bisection and validation.

queries.py renders the query; this module is its client side. It validates every
returned context against its key, clocks and the feature contract, and splits a
multi-key request that TigerGraph times out on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import logging
from typing import Any

import numpy as np

from ..encoding import BASIS_ID, fourier64
from .contract import (
    AMOUNT_RATIO_CAP,
    AMOUNT_RATIO_FEATURES,
    CHANNELS,
    CLIENT_GROUPS,
    CONTRACT_VERSION,
    FEATURE_GROUPS,
    NODE_TYPES,
    RAILS,
    RELATIONS,
    STRATA,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
)
from .executor import CONVERSION_ERRORS, QueryExecutor, ServerTimeoutError, error_summary

LOGGER = logging.getLogger(__name__)

CONTEXT_QUERY = "temporal_training_context"
# Per-request statuses: the query continues with the next request and the
# client receives None for that key. Every other non-ok status is call-level.
PER_REQUEST_STATUSES = frozenset(
    {
        "invalid_request",
        "missing_entity",
        "invisible_entity",
        "history_capacity_exceeded",
        "nonmonotonic_pair_clock",
        "invalid_payment_fields",
        "invalid_event_roles",
    }
)
# Node features TigerGraph may return; client groups (hub_indicator) never come from it.
KNOWN_NODE_FEATURES = frozenset(
    name
    for group, spec in FEATURE_GROUPS.items()
    if spec.path in ("node", "summary") and group not in CLIENT_GROUPS
    for name in spec.names
)


class ContextTimeoutError(RuntimeError):
    """One context keeps timing out on the server, even as a single-key request.

    Fatal on purpose: whether a key times out depends on server load, so
    dropping it would make the training data differ between runs.
    """

    def __init__(self, key: ContextKey, hop: int, cause: str) -> None:
        super().__init__(
            f"TigerGraph timed out on context {key} (hop {hop}) even as a single-key request "
            f"({cause}). The context cannot be skipped without making the data depend on "
            "server load: retry when the server is less busy, or lower the sampler's "
            "max_history for this preparation."
        )
        self.key, self.hop = key, hop


def response_bound(sampler: SamplerPlan, hop: int) -> int:
    """Largest message count a context may contain at this hop."""
    return int(sampler.pool(hop).response_bound)


def validate_context(
    key: ContextKey,
    row: dict[str, Any],
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    *,
    require_encodings: bool = False,
) -> int:
    """Check one context against its key, clocks and the feature contract.

    Encodings are optional. When age_encoding is present (or required for a
    spot-check request), every vector is compared with the shared fourier64
    basis in one vectorized call. Returns the number of messages whose channel
    is outside CHANNELS; those are allowed and mapped to "other" downstream.
    """
    if any(row.get(name) != value for name, value in asdict(key).items()):
        raise ValueError("Returned context differs from requested entity/cutoff")
    if row.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Missing current feature contract; install the current context query")
    if row.get("basis_id") != BASIS_ID:
        raise ValueError("Fourier basis mismatch")
    messages: list[dict[str, Any]] = row["messages"]
    if len(messages) > response_bound(sampler, hop):
        raise ValueError("Query response exceeds the neighborhood bound")
    features: dict[str, Any] = row["features"]
    if "amount_ratios" in plan.groups and any(
        name not in features for name in AMOUNT_RATIO_FEATURES
    ):
        raise ValueError(
            "GSQL response is missing amount ratios; install the current context query"
        )
    if set(features) - KNOWN_NODE_FEATURES:
        raise ValueError("Unknown node feature in response")
    if features and not _finite_nonnegative(list(features.values())):
        raise ValueError("Features must be finite and nonnegative")
    if any(features.get(name, 0) > AMOUNT_RATIO_CAP for name in AMOUNT_RATIO_FEATURES):
        raise ValueError("GSQL amount ratio exceeds the feature contract")
    numeric = [
        (group, name)
        for group in ("flow_timing", "pair_history", "device_ip_context")
        if group in plan.groups
        for name in FEATURE_GROUPS[group].names
    ]
    if numeric and messages:
        table = [[message.get(name) for _, name in numeric] for message in messages]
        if not _finite_nonnegative(table):
            for message in messages:
                for group, name in numeric:
                    if not _finite_nonnegative([message.get(name)]):
                        raise ValueError(f"Missing or invalid {group} field: {name}")
    flow = "flow_timing" in plan.groups
    unknown_channels = 0
    events: list[dict[str, Any]] = []
    for message in messages:
        if message.get("stratum", "recent") not in STRATA:
            raise ValueError("Unknown sampling stratum")
        if message["node_type"] not in NODE_TYPES or message["relation"] not in RELATIONS:
            raise ValueError("Unknown entity or relation in neighborhood")
        if message["rail"] not in RAILS:
            raise ValueError("Unknown payment rail")
        if message.get("channel", "unknown") not in CHANNELS:
            unknown_channels += 1
        if flow:
            if message["flow_present"] and message["flow_censored"]:
                raise ValueError("Observed forward event cannot be censored")
            if not message["flow_present"] and (
                message["flow_delay_seconds"] or message["flow_ratio_present"]
            ):
                raise ValueError("Missing flow event must not have delay/amount evidence")
        if message["event_id"]:
            if not 0 < message["event_seq"] < key.cutoff_seq:
                raise ValueError("Future or invalid event sequence")
            if not 0 < message["event_ts_ms"] <= key.cutoff_ms:
                raise ValueError("Future or invalid event timestamp")
            if message["age_ms"] != key.cutoff_ms - message["event_ts_ms"]:
                raise ValueError("Event age differs from cutoff")
            if message["gap_ms"] < 0:
                raise ValueError("Negative predecessor gap")
            if not message["gap_present"] and message["gap_ms"]:
                raise ValueError("Missing predecessor must not have an encoding or gap")
            events.append(message)
        elif (message["event_seq"], message["event_ts_ms"]) != (key.cutoff_seq, key.cutoff_ms):
            raise ValueError("Association context changed the cutoff")
    age_map: dict[str, Any] = row.get("age_encoding") or {}
    gap_map: dict[str, Any] = row.get("gap_encoding") or {}
    if age_map or gap_map or (require_encodings and "time_encoding" in plan.groups):
        _check_encodings(events, age_map, gap_map)
    return unknown_channels


def _finite_nonnegative(values: list[Any]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except CONVERSION_ERRORS:
        return False
    return bool(np.isfinite(array).all() and (array >= 0).all())


def _check_encodings(
    events: list[dict[str, Any]], age_map: dict[str, Any], gap_map: dict[str, Any]
) -> None:
    keys = [message["relation"] + ":" + message["event_id"] for message in events]
    gap_keys = [k for k, message in zip(keys, events, strict=True) if message["gap_present"]]
    if set(gap_map) - set(gap_keys):
        raise ValueError("Missing predecessor must not have an encoding")
    if set(age_map) != set(keys) or set(gap_map) != set(gap_keys):
        raise ValueError("GSQL time encodings do not cover the returned events")
    if not keys:
        return
    deltas = [int(message["age_ms"]) for message in events] + [
        int(message["gap_ms"]) for message in events if message["gap_present"]
    ]
    try:
        vectors = np.asarray(
            [age_map[k] for k in keys] + [gap_map[k] for k in gap_keys], dtype=np.float64
        )
    except CONVERSION_ERRORS:
        raise ValueError("GSQL time encoding has an invalid shape") from None
    expected = fourier64(np.asarray(deltas, dtype=np.int64))
    if vectors.shape != expected.shape or not np.allclose(vectors, expected, atol=1e-5):
        raise ValueError("GSQL time encoding does not match the shared basis")


def query_context_rows(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
    diagnostics: Counter[str] | None = None,
    timeout_retries: int = 1,
) -> list[dict[str, Any]]:
    """One REST call; validated ok rows or per-request status rows, in key order."""
    if not batch or len({(k.scope_id, k.visibility_phase) for k in batch}) != 1:
        raise ValueError("Query batch must have one visibility scope and phase")
    if len(batch) > 64:
        raise ValueError("A context request carries at most 64 keys")
    params = {
        "node_types": [key.node_type for key in batch],
        "node_ids": [key.node_id for key in batch],
        "cutoff_seqs": [key.cutoff_seq for key in batch],
        "cutoff_times": [key.cutoff_ms for key in batch],
        **sampler.query_params(hop),
        "emit_encodings": emit_encodings,
        **plan.query_flags(hop),
        "scope_id": batch[0].scope_id,
        "visibility_phase": batch[0].visibility_phase,
    }
    result = executor.run(CONTEXT_QUERY, params, timeout_retries=timeout_retries)
    indexed: dict[int, dict[str, Any]] = {}
    for row in result:
        if "request_index" not in row:
            if row.get("status", "ok") != "ok":
                raise ValueError(f"TigerGraph rejected the context call: {row}")
            continue
        index = int(row["request_index"])
        if index in indexed or not 0 <= index < len(batch):
            raise ValueError("Incomplete or duplicated query response")
        indexed[index] = row
    if len(indexed) != len(batch):
        raise ValueError("Incomplete or duplicated query response")
    rows = []
    for index, key in enumerate(batch):
        row = indexed[index]
        status = row.get("status")
        if status == "ok":
            unknown = validate_context(
                key, row, plan, sampler, hop, require_encodings=emit_encodings
            )
            if diagnostics is not None and unknown:
                diagnostics["unknown_channel"] += unknown
        elif status not in PER_REQUEST_STATUSES:
            raise ValueError(f"Unknown per-request status from TigerGraph: {row}")
        rows.append(row)
    if diagnostics is not None and emit_encodings:
        diagnostics["encoding_checks"] += 1
    return rows


def query_context_split(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
    diagnostics: Counter[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """query_context_rows that bisects a block TigerGraph times out on; (rows, calls).

    A multi-key request that exceeds the server timeout is split in two halves
    at once instead of being repeated, which isolates a slow key in about
    log2(block) extra calls. A single key is retried once and then raises
    ContextTimeoutError naming it; it is never mapped to None, because which
    keys time out depends on server load.
    """
    try:
        rows = query_context_rows(
            executor,
            batch,
            plan=plan,
            sampler=sampler,
            hop=hop,
            emit_encodings=emit_encodings,
            diagnostics=diagnostics,
            timeout_retries=0 if len(batch) > 1 else 1,
        )
    except ServerTimeoutError as error:
        if len(batch) == 1:
            raise ContextTimeoutError(batch[0], hop, error_summary(error)) from error
        LOGGER.warning(
            "TigerGraph timed out on a %d-key context request (hop %d); splitting it",
            len(batch),
            hop,
        )
        if diagnostics is not None:
            diagnostics["timeout_splits"] += 1
        middle = len(batch) // 2
        options: dict[str, Any] = {
            "plan": plan,
            "sampler": sampler,
            "hop": hop,
            "emit_encodings": emit_encodings,
            "diagnostics": diagnostics,
        }
        left, left_calls = query_context_split(executor, batch[:middle], **options)
        right, right_calls = query_context_split(executor, batch[middle:], **options)
        return left + right, left_calls + right_calls
    return rows, 1


def query_context_batch(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    plan: FeaturePlan = FeaturePlan(),
    sampler: SamplerPlan = SamplerPlan(),
    hop: int = 1,
    emit_encodings: bool = False,
) -> list[dict[str, Any] | None]:
    """Validated contexts in key order; None where TigerGraph rejected one request."""
    return [
        row if row.get("status") == "ok" else None
        for row in query_context_rows(
            executor, batch, plan=plan, sampler=sampler, hop=hop, emit_encodings=emit_encodings
        )
    ]
