"""The context query's client side: response validation and call-level errors."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY
from mule_pattern_learner.testing.builders import (
    CORE_PLAN,
    SMALL_SAMPLER,
    context,
    event,
    message,
    query_context_batch,
    root,
)
from mule_pattern_learner.testing.fake_connection import FakeConn, executor
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.context_query import validate_context


def test_validation_errors_from_a_response_are_never_retried() -> None:
    key = root(1)
    bad = {**context(key, [], encodings=False), "request_index": 0, "cutoff_seq": 999}
    conn = FakeConn([[bad], [bad]])
    with pytest.raises(ValueError, match="differs"):
        query_context_batch(executor(conn), [key], plan=CORE_PLAN, sampler=SMALL_SAMPLER)
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
    fake = FakeTigerGraph(answers={CONTEXT_QUERY: lambda params: rows})
    with pytest.raises(ValueError, match=message):
        query_context_batch(fake, [root(0)], plan=CORE_PLAN, sampler=SMALL_SAMPLER)


def test_response_bound_is_per_hop() -> None:
    key = root(0)
    many = [
        event(key.cutoff_seq - 1 - i, key) for i in range(SMALL_SAMPLER.children.response_bound + 1)
    ]
    row = context(key, many, encodings=False)
    validate_context(key, row, CORE_PLAN, SMALL_SAMPLER, 1)
    with pytest.raises(ValueError, match="bound"):
        validate_context(key, row, CORE_PLAN, SMALL_SAMPLER, 2)


def test_unknown_channel_is_counted_but_unknown_rail_is_rejected() -> None:
    key = root(0)
    row = context(key, [event(990, key)], encodings=False)
    row["messages"][0]["channel"] = "carrier_pigeon"
    assert validate_context(key, row, CORE_PLAN, SMALL_SAMPLER) == 1
    row["messages"][0]["rail"] = "carrier_pigeon"
    with pytest.raises(ValueError, match="rail"):
        validate_context(key, row, CORE_PLAN, SMALL_SAMPLER)


def test_future_and_same_event_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    for seq, ts in ((100, 1000), (101, 900), (90, 1001)):
        row = context(key, [message(seq, min(ts, 1000), key)])
        row["messages"][0]["event_ts_ms"] = ts
        with pytest.raises(ValueError):
            validate_context(key, row, plan=FeaturePlan(), sampler=SamplerPlan())


def test_basis_and_clock_corruption_are_rejected() -> None:
    key = ContextKey("Account", "root", 100, 1000)
    row = context(key, [message(90, 900, key)])
    bad = deepcopy(row)
    bad["age_encoding"]["zelle_out:E90"][0] += 0.1
    with pytest.raises(ValueError, match="encoding"):
        validate_context(key, bad, plan=FeaturePlan(), sampler=SamplerPlan())
    bad = deepcopy(row)
    bad["cutoff_seq"] = 101
    with pytest.raises(ValueError, match="differs"):
        validate_context(key, bad, plan=FeaturePlan(), sampler=SamplerPlan())
    bad = deepcopy(row)
    bad["messages"][0]["gap_present"] = False
    with pytest.raises(ValueError, match="Missing predecessor"):
        validate_context(key, bad, plan=FeaturePlan(), sampler=SamplerPlan())


def test_a_row_with_a_feature_training_does_not_read_is_refused() -> None:
    # The training query computes no analytics group, so such a feature means another
    # query answered.
    key = ContextKey("Account", "root", 100, 1000)
    validate_context(key, context(key), FeaturePlan(), sampler=SamplerPlan())
    for name in ("age_days", "1d_out_count", "1d_out_in_amount_ratio", "visible_event_count"):
        row = context(key, features={"is_deposit": 1, name: 1})
        with pytest.raises(ValueError, match="Unknown node feature"):
            validate_context(key, row, FeaturePlan(), sampler=SamplerPlan())
