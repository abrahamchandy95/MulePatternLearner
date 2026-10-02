"""The installed context query on the graph: its text, time encodings and cutoff boundary.

Read-only. The accounts are the first of the built-in scope's population, chosen
without labels, scored at the test cutoff. `mule check` covers the rest of the
training path on the graph: one batch and one training step. The pair counts are analytics, so they
are checked on the analytics context query, when it is installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.server import (
    ANALYTICS_CONTEXT_QUERY,
    CONTEXT_QUERY,
    PAYMENT_PAIR_QUERY,
    ZELLE_PAIR_QUERY,
)
from mule_pattern_learner.data.splits import resolve_cutoff
from mule_pattern_learner.tigergraph.context_query import validate_context
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor, checked_rows
from mule_pattern_learner.tigergraph.installer import (
    ANALYTICS_QUERY_FILES,
    query_problems,
    verify_sources,
)
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader

pytestmark = pytest.mark.graph

# Accounts of the scope's first page searched for one with outgoing payments.
SEARCHED = 32
PAYMENT_RELATIONS = ("zelle_out", "payment_out", "payment_in")


def context_rows(graph: TigerGraphExecutor, params: dict[str, Any]) -> list[dict[str, Any]]:
    return checked_rows(graph.run(CONTEXT_QUERY, params))


def request(keys: list[ContextKey], per_relation: int, **flags: bool) -> dict[str, Any]:
    """Parameters of an unscoped request; flags and pools not named take the defaults."""
    return {
        "node_types": [key.node_type for key in keys],
        "node_ids": [key.node_id for key in keys],
        "cutoff_seqs": [key.cutoff_seq for key in keys],
        "cutoff_times": [key.cutoff_ms for key in keys],
        "per_relation": per_relation,
        **flags,
    }


@pytest.fixture(scope="module")
def sampled(graph: TigerGraphExecutor) -> tuple[ContextKey, dict[str, Any]]:
    """The first scope account with a sampled outgoing payment, and its context row.

    The row carries the Fourier vectors (emit_encodings), so they are checked against
    numpy's fourier64.
    """
    (date,) = DEFAULT_CONFIG.dataset.dates.test
    cutoff_seq, cutoff_ms = resolve_cutoff(TigerGraphCutoffReader(graph), date)
    pages = TigerGraphScopeReader(graph).population_pages(
        DEFAULT_CONFIG.scope.id, include_observed=False
    )
    accounts = [row["account_id"] for row in next(iter(pages))[:SEARCHED]]
    for account in accounts:
        key = ContextKey("Account", account, cutoff_seq, cutoff_ms)
        (row,) = context_rows(graph, request([key], 2, emit_encodings=True))
        if any(m["relation"] in ("zelle_out", "payment_out") for m in row["messages"]):
            return key, row
    pytest.skip(f"none of the first {len(accounts)} scope accounts has an outgoing payment")


def test_the_label_contract_type_and_the_installed_queries_are_the_repositorys(
    graph: TigerGraphExecutor,
) -> None:
    def read_schema(conn: Any) -> dict[str, Any]:
        return conn.getSchema(force=True)

    schema = graph.call(read_schema, what="getSchema")
    attributes = {
        vertex["Name"]: {
            a["AttributeName"]: a["AttributeType"]["Name"] for a in vertex["Attributes"]
        }
        for vertex in schema["VertexTypes"]
    }
    assert attributes["Account"]["is_mule"] == "INT"
    assert verify_sources(graph)


def test_a_context_row_matches_its_contract_and_numpy_fourier64(
    graph: TigerGraphExecutor, sampled: tuple[ContextKey, dict[str, Any]]
) -> None:
    key, row = sampled
    # The request names no flags or pools beyond per_relation, so it gets the defaults:
    # every flag on, the built-in run's groups.
    validate_context(key, row, FeaturePlan(), SamplerPlan(), require_encodings=True)


def test_the_seed_event_is_excluded_until_the_next_cutoff(
    graph: TigerGraphExecutor, sampled: tuple[ContextKey, dict[str, Any]]
) -> None:
    key, row = sampled
    event = next(m for m in row["messages"] if m["relation"] in ("zelle_out", "payment_out"))
    seq, event_ms = event["event_seq"], event["event_ts_ms"]
    boundary = [
        ContextKey("Account", key.node_id, seq, event_ms),
        ContextKey("Account", key.node_id, seq + 1, event_ms),
    ]
    rows = context_rows(graph, request(boundary, 1))
    before, after = sorted(rows, key=lambda value: value["request_index"])
    assert not any(m["event_id"] == event["event_id"] for m in before["messages"])
    assert any(m["event_id"] == event["event_id"] for m in after["messages"])


def test_pair_gaps_and_counts_match_the_analytics_queries(
    graph: TigerGraphExecutor, sampled: tuple[ContextKey, dict[str, Any]]
) -> None:
    problems = query_problems(graph, ANALYTICS_QUERY_FILES)
    if problems:
        pytest.skip(
            f"the analytics queries are not installed as the repository defines them "
            f"({problems}); install(executor, analytics=True) installs them"
        )
    key, row = sampled
    # The same request of the analytics query samples the same messages, with their pair
    # window counts.
    (counted,) = checked_rows(graph.run(ANALYTICS_CONTEXT_QUERY, request([key], 2)))
    windows = {m["event_id"]: m for m in counted["messages"] if m["event_id"]}
    checked = []
    for relation in PAYMENT_RELATIONS:
        message = next((m for m in row["messages"] if m["relation"] == relation), None)
        if message is None or message["node_type"] != "Account":
            continue
        sender, recipient = (
            (message["node_id"], key.node_id)
            if relation.endswith("_in")
            else (key.node_id, message["node_id"])
        )
        params: dict[str, Any] = {
            "sender": (sender,),
            "recipient_type": "Account",
            "recipient_id": recipient,
            "seed_seq": message["event_seq"] + 1,
            "seed_ts_ms": message["event_ts_ms"],
            "persist": False,
            "max_events": 10000,
        }
        name = ZELLE_PAIR_QUERY if relation.startswith("zelle") else PAYMENT_PAIR_QUERY
        if name == PAYMENT_PAIR_QUERY:
            params["payment_rail"] = message["rail"]
        history = checked_rows(graph.run(name, params))
        event = next(v for v in history if v.get("event_id") == message["event_id"])
        assert event["pair_delta_t_ms"] == message["gap_ms"]
        assert event["pair_delta_t_present"] == message["gap_present"]
        for field, window in (
            ("pair_count_1h", 3600000),
            ("pair_count_1d", 86400000),
            ("pair_count_7d", 604800000),
        ):
            expected = sum(
                1
                for item in history
                if item.get("event_seq", message["event_seq"]) < message["event_seq"]
                and message["event_ts_ms"] - item["event_ts_ms"] < window
            )
            assert windows[message["event_id"]][field] == expected, (relation, field)
        checked.append(relation)
    if not checked:
        pytest.skip("the sampled account's payments have no Account counterparty")
