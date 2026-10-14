"""The context query's rows against an independent reference, on the graph.

Read-only. The accounts are the first of the graph's split's population, chosen
without labels, and the parties that own them. Their raw payments, associations and
flags are read with interpreted audit queries, without the split on purpose
(tests/integration/test_scope_isolation.py checks it), and reference.gsql_features
recomputes every field of every message and every node feature the training context
query returned: the model's inputs. The query's text is checked installed and, through
INTERPRET, as it is in the repository.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import CORE_GROUPS, FeaturePlan
from mule_pattern_learner.contract.graph_schema import (
    ASSOCIATION_TARGETS,
    ASSOCIATIONS,
    ContextKey,
)
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY, CONTEXT_QUERY_FILE, GRAPH_NAME
from mule_pattern_learner.data.splits import resolve_cutoff
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.reference.gsql_features import (
    association_messages,
    payment_messages,
    root_features,
)
from mule_pattern_learner.testing.gsql_blocks import as_interpreted
from mule_pattern_learner.tigergraph.context_query import validate_context
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffReader
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor, checked_rows
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader

pytestmark = pytest.mark.graph

# Accounts of the split's first page whose rows are recomputed, with their owners.
ACCOUNTS = 3
# The groups TigerGraph computes for training, which the reference recomputes.
PLAN = FeaturePlan(CORE_GROUPS, "tgat")
SAMPLER = SamplerPlan(roots=PoolPlan(recent=4, older=3, distinct=2))
# The message fields compared within a tolerance; every other field must be equal.
FLOATS = frozenset(
    {
        "amount",
        "pair_first_age_seconds",
        "flow_delay_seconds",
        "flow_observation_seconds",
        "flow_amount_ratio",
    }
)
# The primary id attribute of each association target type.
ID_ATTRIBUTE = {"Token": "token_id", "Address": "address_id"}


def audit_history(graph: TigerGraphExecutor, key: ContextKey) -> list[dict[str, Any]]:
    """Every payment of the account before the cutoff, with its peer, read without the query."""
    parts = [
        f"""INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} SYNTAX V2 {{
    TYPEDEF TUPLE<STRING event_id, UINT event_seq, UINT event_ts_ms, STRING relation,
                  STRING node_type, STRING node_id, STRING rail, STRING currency,
                  DOUBLE amount, BOOL amount_present, UINT peer_first_ms,
                  BOOL peer_external, BOOL peer_deposit> Event;
    ListAccum<Event> @@history;
    SumAccum<INT> @@degree;
    VERTEX seed_vertex = to_vertex({json.dumps(key.node_id)}, "Account");
    Root = {{seed_vertex}};
    Guard = SELECT a FROM Root:a ACCUM @@degree += a.outdegree("Account_Sent_Zelle_Transfer")
      + a.outdegree("Account_Received_Zelle_Transfer") + a.outdegree("Account_Initiated_Transaction")
      + a.outdegree("Account_Received_Transaction");
    IF @@degree > 8192 THEN PRINT "audit_degree_exceeded" AS status; RETURN; END;
    """
    ]
    for prefix, vertex, pid, rail, stem, edges in (
        (
            "Transfer",
            "Zelle_Transfer",
            "transfer_id",
            '"zelle"',
            "zelle",
            ("Account_Sent_Zelle_Transfer", "Account_Received_Zelle_Transfer"),
        ),
        (
            "Transaction",
            "Payment_Transaction",
            "transaction_id",
            "t.payment_rail",
            "payment",
            ("Account_Initiated_Transaction", "Account_Received_Transaction"),
        ),
    ):
        for direction, edge in zip(("out", "in"), edges, strict=True):
            role = "To" if direction == "out" else "From"
            parts.append(f"""Events_{stem}_{direction} = SELECT t FROM Root:a -({edge}>:e)- {vertex}:t
              WHERE t.event_seq < {key.cutoff_seq} AND t.event_ts_ms <= {key.cutoff_ms};""")
            for typ, attr in (("Account", "id"), ("Token", "token_id")):
                fallback = (
                    f' AND t.outdegree("{prefix}_{role}_Account") == 0' if typ == "Token" else ""
                )
                flags = (
                    'p.is_external, p.account_type == "deposit"'
                    if typ == "Account"
                    else "FALSE, FALSE"
                )
                parts.append(f'''Peers_{stem}_{direction}_{typ} = SELECT t
                  FROM Events_{stem}_{direction}:t -({prefix}_{role}_{typ}>:e)- {typ}:p
                  WHERE e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms{fallback}
                  ACCUM @@history += Event(t.{pid}, t.event_seq, t.event_ts_ms, "{stem}_{direction}",
                    "{typ}", p.{attr}, {rail}, t.currency, t.amount, t.amount_present,
                    p.first_seen_ts_ms, {flags});''')
    parts.append('PRINT "ok" AS status, @@history AS history; }')
    rows = graph.client.conn.runInterpretedQuery("\n".join(parts))
    return checked_rows(rows)[0]["history"]


def association_edges(node_type: str) -> list[tuple[str, str]]:
    """The association edges from a root of this type, each with its target type."""
    edges = []
    for (forward, reverse), (to_forward, to_reverse) in zip(
        ASSOCIATIONS, ASSOCIATION_TARGETS, strict=True
    ):
        if to_reverse == node_type:
            edges.append((forward, to_forward))
        if to_forward == node_type:
            edges.append((reverse, to_reverse))
    return edges


def audit_root(graph: TigerGraphExecutor, key: ContextKey) -> dict[str, Any]:
    """The root's flags and association edges, with their far ends, read without the query."""
    primary = ID_ATTRIBUTE.get(key.node_type, "id")
    parts = [
        f"""INTERPRET QUERY () FOR GRAPH {GRAPH_NAME} SYNTAX V2 {{
    TYPEDEF TUPLE<STRING relation, STRING node_type, STRING node_id, UINT valid_from_seq,
                  UINT valid_to_seq, UINT first_seen_seq, UINT first_seen_ts_ms,
                  BOOL external, BOOL deposit> Link;
    ListAccum<Link> @@links;
    OrAccum @@external, @@deposit;
    SetAccum<STRING> @@ids, @@found;
    @@ids += {json.dumps(key.node_id)};
    Root = to_vertex_set(@@ids, {json.dumps(key.node_type)});
    Found = SELECT s FROM Root:s ACCUM @@found += s.{primary};
    """
    ]
    if key.node_type == "Account":
        parts.append(
            "Flags = SELECT s FROM Root:s ACCUM @@external += s.is_external, "
            '@@deposit += s.account_type == "deposit";'
        )
    for index, (edge, target) in enumerate(association_edges(key.node_type)):
        attr = ID_ATTRIBUTE.get(target, "id")
        flags = (
            't.is_external, t.account_type == "deposit"' if target == "Account" else "FALSE, FALSE"
        )
        parts.append(f"""Links_{index} = SELECT t FROM Root:s -({edge}>:e)- {target}:t
          ACCUM @@links += Link("{edge}", "{target}", t.{attr}, e.valid_from_seq, e.valid_to_seq,
            t.first_seen_seq, t.first_seen_ts_ms, {flags});""")
    parts.append(
        'PRINT "ok" AS status, @@found AS found, @@external AS external, @@deposit AS deposit, '
        "@@links AS links; }"
    )
    row = checked_rows(graph.client.conn.runInterpretedQuery("\n".join(parts)))[0]
    assert row["found"] == [key.node_id], key
    return row


@pytest.fixture(scope="module")
def keys(graph: TigerGraphExecutor) -> list[ContextKey]:
    """The first accounts of the split at the test cutoff, without the split (phase 0)."""
    pages = TigerGraphScopeReader(graph).population_pages()
    population = next(iter(pages))[:ACCOUNTS]
    (date,) = DEFAULT_CONFIG.dataset.dates.test
    seq, ms = resolve_cutoff(TigerGraphCutoffReader(graph), date)
    return [ContextKey("Account", row["account_id"], seq, ms) for row in population]


def request(key: ContextKey) -> dict[str, Any]:
    return {
        "node_types": [key.node_type],
        "node_ids": [key.node_id],
        "cutoff_seqs": [key.cutoff_seq],
        "cutoff_times": [key.cutoff_ms],
        **SAMPLER.query_params(),
    }


def context_row(graph: TigerGraphExecutor, params: dict[str, Any], text: str) -> dict[str, Any]:
    """The installed query's row, or that of the repository text run under INTERPRET."""
    if text == "interpreted":
        # The repository text runs under INTERPRET with only its header swapped.
        query = as_interpreted((GSQL_DIR / CONTEXT_QUERY_FILE).read_text())
        return checked_rows(graph.client.conn.runInterpretedQuery(query, params))[0]
    return checked_rows(graph.run(CONTEXT_QUERY, params))[0]


def message_differences(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> list[str]:
    """Every field of every message that differs, messages matched by relation and item."""

    def item(message: dict[str, Any]) -> tuple[str, str, str]:
        return message["relation"], message["event_id"], message["node_id"]

    have = sorted(actual, key=item)
    want = sorted(expected, key=item)
    if [item(m) for m in have] != [item(m) for m in want]:
        return [f"messages {sorted(map(item, have))} != {sorted(map(item, want))}"]
    problems = []
    for got, wanted in zip(have, want, strict=True):
        if set(got) != set(wanted):
            problems.append(f"{item(got)}: fields {sorted(set(got) ^ set(wanted))}")
            continue
        for name, value in wanted.items():
            same = (
                bool(np.isclose(got[name], value, rtol=1e-5, atol=1e-6))
                if name in FLOATS
                else got[name] == value
            )
            if not same:
                problems.append(f"{item(got)}: {name} {got[name]!r} != {value!r}")
    return problems


@pytest.mark.parametrize("text", ["installed", "interpreted"])
def test_context_rows_match_the_reference(
    graph: TigerGraphExecutor, keys: list[ContextKey], text: str
) -> None:
    # Each account, then the parties its row names: they read the Account end of an
    # association, with its flags.
    pending, checked, parties = list(keys), 0, 0
    while pending:
        key = pending.pop(0)
        root = audit_root(graph, key)
        expected = association_messages(root["links"], key, SAMPLER.roots.associations)
        if key.node_type == "Account":
            expected += payment_messages(audit_history(graph, key), key, SAMPLER.roots)
        # Print the Fourier vectors so they are compared with numpy fourier64.
        row = context_row(graph, {**request(key), "emit_encodings": True}, text)
        assert row["status"] == "ok", (key, row)
        validate_context(key, row, PLAN, SAMPLER, require_encodings=True)
        assert row["features"] == root_features(key, root["external"], root["deposit"]), key
        problems = message_differences(row["messages"], expected)
        assert not problems, (key, problems)
        checked += 1
        if key.node_type == "Account":
            owners = [m for m in row["messages"] if m["relation"] == "Account_Owned_By_Party"]
            pending += [
                ContextKey("Party", m["node_id"], key.cutoff_seq, key.cutoff_ms) for m in owners
            ]
        else:
            parties += 1
    assert checked == len(keys) + parties
