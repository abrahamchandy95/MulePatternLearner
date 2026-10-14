"""The installed context query on the graph: its text, time encodings and cutoff boundary.

Read-only. The accounts are the first of the graph's split's population, chosen
without labels, scored at the test cutoff. `mule check` covers the rest of the
training path on the graph: one batch and one training step. Given an earlier text of
the query (--baseline-context-query), a last test runs both texts under INTERPRET and
checks that they return the same rows. The earlier text reads the payments' `channel`,
which the current schema no longer declares, so that test runs on a graph loaded with the
earlier schema, as the owner's graph is (it keeps the attribute).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from mule_pattern_learner.batching.assemble import child_key
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.bounds import REQUEST_KEYS
from mule_pattern_learner.contract.feature_groups import FeaturePlan
from mule_pattern_learner.contract.graph_schema import (
    NO_SPLIT,
    NODE_TYPES,
    PHASE_SPLIT,
    ContextKey,
)
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY, CONTEXT_QUERY_FILE
from mule_pattern_learner.data.splits import resolve_cutoff
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_blocks import as_interpreted
from mule_pattern_learner.tigergraph.context_query import batch_params, validate_context
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor, checked_rows
from mule_pattern_learner.tigergraph.installer import verify_sources
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader

pytestmark = pytest.mark.graph

# Accounts of the split's first page searched for one with outgoing payments.
SEARCHED = 32


def context_rows(graph: TigerGraphExecutor, params: dict[str, Any]) -> list[dict[str, Any]]:
    return checked_rows(graph.run(CONTEXT_QUERY, params))


def request(keys: list[ContextKey], per_relation: int, **options: bool) -> dict[str, Any]:
    """Parameters of a request without the split (phase 0): the defaults where not named."""
    return {
        "node_types": [key.node_type for key in keys],
        "node_ids": [key.node_id for key in keys],
        "cutoff_seqs": [key.cutoff_seq for key in keys],
        "cutoff_times": [key.cutoff_ms for key in keys],
        "per_relation": per_relation,
        **options,
    }


@pytest.fixture(scope="module")
def sampled(graph: TigerGraphExecutor) -> tuple[ContextKey, dict[str, Any]]:
    """The split's first account with a sampled outgoing payment, and its context row.

    The row carries the Fourier vectors (emit_encodings), so they are checked against
    numpy's fourier64.
    """
    (date,) = DEFAULT_CONFIG.dataset.dates.test
    cutoff_seq, cutoff_ms = resolve_cutoff(TigerGraphCutoffReader(graph), date)
    pages = TigerGraphScopeReader(graph).population_pages()
    accounts = [row["account_id"] for row in next(iter(pages))[:SEARCHED]]
    for account in accounts:
        key = ContextKey("Account", account, cutoff_seq, cutoff_ms)
        (row,) = context_rows(graph, request([key], 2, emit_encodings=True))
        if any(m["relation"] in ("zelle_out", "payment_out") for m in row["messages"]):
            return key, row
    pytest.skip(f"none of the split's first {len(accounts)} accounts has an outgoing payment")


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
    # The request names no pool beyond per_relation, so it gets the defaults; the query
    # computes every group, so the built-in run's plan checks them all.
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


# The candidate pools of the baseline comparison: the built-in run's roots and children,
# and a root pool that also takes older and distinct events.
BASELINE_POOLS = (
    DEFAULT_CONFIG.sampler.roots,
    DEFAULT_CONFIG.sampler.children,
    PoolPlan(recent=4, older=3, distinct=2, associations=2),
)
# Accounts of the split's first page the comparison starts from, and the most of their
# children of each node type it requests per phase.
COMPARED_ACCOUNTS = 16
COMPARED_CHILDREN = 16
# The node types of an account's children, each of which must return an ok row: the
# counterparties, the Zelle tokens (their prior-pair scan), the owners and the devices.
CHILD_TYPES = ("Account", "Token", "Party", "Device")
# What a row of either text says about its key and the model's inputs; the contract
# version names the text, and an earlier text may print fields the model never read.
COMPARED_FIELDS = (
    "status",
    "request_index",
    "node_type",
    "node_id",
    "cutoff_seq",
    "cutoff_ms",
    "visibility_phase",
    "basis_id",
    "features",
    "age_encoding",
    "gap_encoding",
)
# Every field of a message of the current text, which the earlier text's messages must
# hold with the same values; an earlier text may print more.
MESSAGE_FIELDS = (
    "node_type",
    "node_id",
    "relation",
    "rail",
    "event_id",
    "event_seq",
    "event_ts_ms",
    "amount",
    "amount_present",
    "age_ms",
    "gap_ms",
    "gap_present",
    "peer_first_ms",
    "peer_external",
    "peer_deposit",
    "pair_prior_count",
    "pair_first_age_seconds",
    "pair_first_present",
    "flow_delay_seconds",
    "flow_present",
    "flow_censored",
    "flow_observation_seconds",
    "flow_amount_ratio",
    "flow_ratio_present",
    "flow_same_rail",
)


@pytest.fixture(scope="module")
def baseline(request: pytest.FixtureRequest) -> str:
    """The earlier query text --baseline-context-query names, as an INTERPRET query."""
    path = request.config.getoption("--baseline-context-query")
    if not path:
        pytest.skip("pass --baseline-context-query PATH to compare with an earlier query text")
    return as_interpreted(Path(path).read_text())


def compared(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each row with COMPARED_FIELDS and its messages' MESSAGE_FIELDS."""
    kept = []
    for row in rows:
        value = {name: row[name] for name in COMPARED_FIELDS if name in row}
        if "messages" in row:
            value["messages"] = [{k: m[k] for k in MESSAGE_FIELDS} for m in row["messages"]]
        kept.append(value)
    return kept


def request_rows(
    graph: TigerGraphExecutor, text: str, params: dict[str, Any]
) -> list[dict[str, Any]]:
    """An INTERPRET query's row of every request, in key order: ok or a per-request status."""
    rows = graph.client.conn.runInterpretedQuery(text, params)
    answered = sorted(
        (row for row in rows if "request_index" in row), key=lambda row: row["request_index"]
    )
    assert [row["request_index"] for row in answered] == list(range(len(params["node_ids"]))), rows
    return answered


def same_rows(
    graph: TigerGraphExecutor, texts: tuple[str, str], keys: list[ContextKey], pool: PoolPlan
) -> list[dict[str, Any]]:
    """Both texts' rows of the keys, which must be equal; the current text's rows, in key order."""
    current_rows: list[dict[str, Any]] = []
    for start in range(0, len(keys), REQUEST_KEYS.high):
        params = {
            **batch_params(keys[start : start + REQUEST_KEYS.high]),
            **pool.query_params(),
            "emit_encodings": True,
        }
        earlier, current = (request_rows(graph, text, params) for text in texts)
        messages = [m for row in current if row.get("status") == "ok" for m in row["messages"]]
        assert all(set(m) == set(MESSAGE_FIELDS) for m in messages), "message fields differ"
        assert compared(current) == compared(earlier), (pool, params)
        current_rows += current
    return current_rows


def test_the_current_text_returns_the_rows_of_the_baseline_text(
    graph: TigerGraphExecutor, baseline: str
) -> None:
    # Every status, node feature, message field and Fourier vector, messages in order, of
    # the split's first accounts and of their children of each type, in every phase and
    # pool.
    current = as_interpreted((GSQL_DIR / CONTEXT_QUERY_FILE).read_text())
    (date,) = DEFAULT_CONFIG.dataset.dates.test
    cutoff_seq, cutoff_ms = resolve_cutoff(TigerGraphCutoffReader(graph), date)
    pages = TigerGraphScopeReader(graph).population_pages()
    accounts = [row["account_id"] for row in next(iter(pages))[:COMPARED_ACCOUNTS]]
    statuses: Counter[str] = Counter()
    answered: set[str] = set()
    for phase in (NO_SPLIT, *PHASE_SPLIT):
        roots = [ContextKey("Account", a, cutoff_seq, cutoff_ms, phase) for a in accounts]
        children: set[ContextKey] = set()
        for pool in BASELINE_POOLS:
            rows = same_rows(graph, (baseline, current), roots, pool)
            for key, row in zip(roots, rows, strict=True):
                statuses[row["status"]] += 1
                children |= {child_key(m, key) for m in row.get("messages", [])}
        chosen = [
            key
            for node_type in NODE_TYPES
            for key in sorted(k for k in children if k.node_type == node_type)[:COMPARED_CHILDREN]
        ]
        for pool in BASELINE_POOLS:
            rows = same_rows(graph, (baseline, current), chosen, pool)
            for key, row in zip(chosen, rows, strict=True):
                statuses[row["status"]] += 1
                if row["status"] == "ok":
                    answered.add(key.node_type)
    assert statuses["ok"] > 0, statuses
    assert set(CHILD_TYPES) <= answered, answered
