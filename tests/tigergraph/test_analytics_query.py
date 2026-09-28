"""The analytics context query's client: its requests and the checks of its rows."""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.contract.analytics_features import ANALYTICS_GROUPS
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import ANALYTICS_CONTEXT_QUERY, ANALYTICS_CONTRACT
from mule_pattern_learner.testing.builders import SMALL_SAMPLER, context, neighbourhood
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.analytics_query import (
    ANALYTICS_NODE_FEATURES,
    TigerGraphAnalyticsFetcher,
    validate_analytics_context,
)
from mule_pattern_learner.tigergraph.context_query import (
    ContextTimeoutError,
    query_context_split,
    validate_context,
)
from mule_pattern_learner.tigergraph.executor import ServerTimeoutError

KEYS = [ContextKey("Account", f"A{i}", 400, 400 * 3_600_000, "s", 3) for i in range(4)]


def account_features(key: ContextKey) -> dict[str, float]:
    return {"visible_event_count": 7.0, "age_days": 12.5, "decay_7d_in_amount": 30.0}


def analytics_calls(graph: FakeTigerGraph) -> list[dict[str, Any]]:
    return [params for name, params in graph.calls if name == ANALYTICS_CONTEXT_QUERY]


def test_a_request_names_the_keys_and_pool_and_every_group_is_computed() -> None:
    graph = FakeTigerGraph(factory=neighbourhood, analytics=account_features)
    rows, calls = TigerGraphAnalyticsFetcher(graph).request(KEYS, sampler=SMALL_SAMPLER)
    assert calls == 1 and [row["node_id"] for row in rows] == [key.node_id for key in KEYS]
    (params,) = analytics_calls(graph)
    # No include flag, so the query computes every group, and no Fourier vectors.
    assert not any(name.startswith("include_") for name in params)
    assert "emit_encodings" not in params
    assert {k: params[k] for k in SMALL_SAMPLER.query_params()} == SMALL_SAMPLER.query_params()
    # The messages the training query samples for the same keys, with the analytics fields.
    training, _ = query_context_split(graph, KEYS, plan=FeaturePlan(), sampler=SMALL_SAMPLER)
    for row, trained in zip(rows, training, strict=True):
        assert row["contract_version"] == ANALYTICS_CONTRACT
        assert row["features"] == {**trained["features"], **account_features(KEYS[0])}
        assert [m["event_id"] for m in row["messages"]] == [
            m["event_id"] for m in trained["messages"]
        ]
        assert all("pair_count_7d" in m and "device_present" in m for m in row["messages"])


def test_rows_are_checked_against_the_analytics_contract_and_features() -> None:
    key = KEYS[0]
    graph = FakeTigerGraph(factory=neighbourhood, analytics=account_features)
    (row,) = TigerGraphAnalyticsFetcher(graph).request([key], sampler=SMALL_SAMPLER)[0]
    assert validate_analytics_context(key, row, SMALL_SAMPLER) == 0
    # A row of the training query is refused, and the analytics row by the training check.
    with pytest.raises(ValueError, match="install the current fetch_analytics_context"):
        validate_analytics_context(key, neighbourhood(key), SMALL_SAMPLER)
    with pytest.raises(ValueError, match="install the current fetch_training_context"):
        validate_context(key, row, FeaturePlan(), SamplerPlan())
    # Every analytics node feature is known; anything else is not.
    for group, spec in ANALYTICS_GROUPS.items():
        if spec.path != "message":
            assert set(spec.names) <= ANALYTICS_NODE_FEATURES, group
    with pytest.raises(ValueError, match="Unknown node feature"):
        validate_analytics_context(key, {**row, "features": {"no_such": 1.0}}, SMALL_SAMPLER)
    negative = {**row, "messages": [{**row["messages"][0], "pair_count_1d": -1}]}
    with pytest.raises(ValueError, match="pair_window_counts field: pair_count_1d"):
        validate_analytics_context(key, negative, SMALL_SAMPLER)


def test_rejected_keys_are_status_rows_and_timeouts_split_the_block() -> None:
    def slow(name: str, params: dict[str, Any]) -> None:
        if len(params["node_ids"]) > 1:
            raise ServerTimeoutError(f"{name} timed out")

    graph = FakeTigerGraph(
        factory=lambda key: context(key, encodings=False),
        statuses={KEYS[2].node_id: "history_capacity_exceeded"},
        before=slow,
    )
    rows, calls = TigerGraphAnalyticsFetcher(graph).request(KEYS, sampler=SMALL_SAMPLER)
    assert calls == 4 and [len(p["node_ids"]) for p in analytics_calls(graph)] == [
        4,
        2,
        1,
        1,
        2,
        1,
        1,
    ]
    assert [row.get("status") for row in rows] == ["ok", "ok", "history_capacity_exceeded", "ok"]

    def always(name: str, params: dict[str, Any]) -> None:
        raise ServerTimeoutError(f"{name} timed out")

    stuck = FakeTigerGraph(before=always)
    with pytest.raises(ContextTimeoutError, match="A0"):
        TigerGraphAnalyticsFetcher(stuck).request(KEYS[:1], sampler=SMALL_SAMPLER)
