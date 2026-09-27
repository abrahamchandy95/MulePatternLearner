"""The context query's client side: response validation and call-level errors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.reference.batch_features import node_features
from mule_pattern_learner.testing.builders import (
    PLAN,
    SAMPLER,
    context,
    context_row,
    event,
    message,
    query_context_batch,
    root,
)
from mule_pattern_learner.testing.fake_connection import FakeConn, executor
from mule_pattern_learner.testing.fake_graph import Runner
from mule_pattern_learner.tigergraph.context_query import validate_context


def test_validation_errors_from_a_response_are_never_retried() -> None:
    key = root(1)
    bad = {**context_row(key, [], encodings=False), "request_index": 0, "cutoff_seq": 999}
    conn = FakeConn([[bad], [bad]])
    with pytest.raises(ValueError, match="differs"):
        query_context_batch(executor(conn), [key], plan=PLAN, sampler=SAMPLER)
    assert len(conn.calls) == 1


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"status": "scope_not_ready"}], "rejected the context call"),
        ([{"status": "surprise", "request_index": 0}], "Unknown per-request status"),
        ([], "Incomplete"),
        ([{"status": "missing_entity", "request_index": 1}], "Incomplete"),
    ],
)
def test_call_level_errors_and_malformed_responses_raise(
    rows: list[dict[str, Any]], message: str
) -> None:
    fake = Runner(lambda name, params: rows)
    with pytest.raises(ValueError, match=message):
        query_context_batch(fake, [root(0)], plan=PLAN, sampler=SAMPLER)


def test_response_bound_is_per_hop() -> None:
    key = root(0)
    many = [event(key.cutoff_seq - 1 - i, key) for i in range(SAMPLER.children.response_bound + 1)]
    row = context_row(key, many, encodings=False)
    validate_context(key, row, PLAN, SAMPLER, 1)
    with pytest.raises(ValueError, match="bound"):
        validate_context(key, row, PLAN, SAMPLER, 2)


def test_unknown_channel_is_counted_but_unknown_rail_is_rejected() -> None:
    key = root(0)
    row = context_row(key, [event(990, key)], encodings=False)
    row["messages"][0]["channel"] = "carrier_pigeon"
    assert validate_context(key, row, PLAN, SAMPLER) == 1
    row["messages"][0]["rail"] = "carrier_pigeon"
    with pytest.raises(ValueError, match="rail"):
        validate_context(key, row, PLAN, SAMPLER)


def test_future_and_same_event_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    for seq, ts in ((100, 1000), (101, 900), (90, 1001)):
        row = context(key, [message(seq, min(ts, 1000), key)])
        row["messages"][0]["event_ts_ms"] = ts
        with pytest.raises(ValueError):
            validate_context(key, row)


def test_basis_and_clock_corruption_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key, [message(90, 900, key)])
    bad = deepcopy(row)
    bad["age_encoding"]["zelle_out:E90"][0] += 0.1
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["cutoff_seq"] = 101
    with pytest.raises(ValueError, match="differs"):
        validate_context(key, bad)
    bad = deepcopy(row)
    bad["messages"][0]["gap_present"] = False
    with pytest.raises(ValueError, match="Missing predecessor"):
        validate_context(key, bad)


def test_amount_ratios_are_required_from_gsql_and_preserved_by_tensor_conversion() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key)
    row["features"].update({"1d_out_in_amount_ratio": 2.5, "7d_out_in_amount_ratio": 100.0})
    plan = FeaturePlan(("entity_meta", "rolling_windows", "amount_ratios", "message_core"))
    validate_context(key, row, plan)
    features = node_features(row, plan)
    ratio = plan.node_names.index
    assert features[ratio("1d_out_in_amount_ratio")] == pytest.approx(np.log1p(2.5))
    assert features[ratio("7d_out_in_amount_ratio")] == pytest.approx(np.log1p(100.0))
    del row["features"]["1d_out_in_amount_ratio"]
    with pytest.raises(ValueError, match="missing amount ratios"):
        validate_context(key, row, plan)
