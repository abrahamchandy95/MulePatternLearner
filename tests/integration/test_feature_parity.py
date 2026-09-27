"""The context query's features against an independent reference, on the graph.

Read-only. The accounts are the first of the built-in scope's population, chosen
without labels. Their raw payment history is read with an interpreted audit query,
unscoped on purpose (tests/integration/test_scope_isolation.py checks the scope), and
reference.gsql_features recomputes the pair, flow and decayed features and the sampler
strata the context query returned. The repository's query text is checked installed
and, through INTERPRET, as it is in the repository.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.feature_groups import FEATURE_GROUPS, FeaturePlan
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.contract.server import CONTEXT_QUERY_FILE
from mule_pattern_learner.data.splits import resolve_cutoff
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.reference.gsql_features import payment_features, stratify, visible_history
from mule_pattern_learner.tigergraph.context_query import CONTEXT_QUERY, validate_context
from mule_pattern_learner.tigergraph.cutoffs import TigerGraphCutoffs
from mule_pattern_learner.tigergraph.executor import TigerGraphExecutor, checked_rows
from mule_pattern_learner.tigergraph.render import as_interpreted
from mule_pattern_learner.tigergraph.scope import TigerGraphScope

pytestmark = pytest.mark.graph

# Accounts of the scope's first page whose features are recomputed.
ACCOUNTS = 3
# Every group the reference recomputes: all but the window counts, the ratios, the pair
# window counts and the device and IP context.
GROUPS = tuple(
    g
    for g in FEATURE_GROUPS
    if g not in ("rolling_windows", "amount_ratios", "pair_window_counts", "device_ip_context")
)
PLAN = FeaturePlan(GROUPS, "tgat")
SAMPLER = SamplerPlan(roots=PoolPlan(recent=4, older=3, distinct=2))


def audit_history(graph: TigerGraphExecutor, key: ContextKey) -> list[dict[str, Any]]:
    """Every payment of the account before the cutoff, read without the context query."""
    parts = [
        f"""INTERPRET QUERY () FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {{
    TYPEDEF TUPLE<STRING event_id, UINT event_seq, UINT event_ts_ms, STRING relation,
                  STRING node_type, STRING node_id, STRING rail, STRING currency,
                  DOUBLE amount, BOOL amount_present> Event;
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
                parts.append(f'''Peers_{stem}_{direction}_{typ} = SELECT t
                  FROM Events_{stem}_{direction}:t -({prefix}_{role}_{typ}>:e)- {typ}:p
                  WHERE e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms{fallback}
                  ACCUM @@history += Event(t.{pid}, t.event_seq, t.event_ts_ms, "{stem}_{direction}",
                    "{typ}", p.{attr}, {rail}, t.currency, t.amount, t.amount_present);''')
    parts.append('PRINT "ok" AS status, @@history AS history; }')
    rows = graph.client.conn.runInterpretedQuery("\n".join(parts))
    return checked_rows(rows)[0]["history"]


@pytest.fixture(scope="module")
def keys(graph: TigerGraphExecutor) -> list[ContextKey]:
    """The first scope accounts at the test cutoff, unscoped."""
    (date,) = DEFAULT_CONFIG.dataset.dates.test
    seq, ms = resolve_cutoff(TigerGraphCutoffs(graph), date)
    pages = TigerGraphScope(graph).population_pages(DEFAULT_CONFIG.scope.id, include_observed=False)
    return [
        ContextKey("Account", row["account_id"], seq, ms) for row in next(iter(pages))[:ACCOUNTS]
    ]


@pytest.mark.parametrize("text", ["installed", "interpreted"])
def test_context_features_match_the_reference(
    graph: TigerGraphExecutor, keys: list[ContextKey], text: str
) -> None:
    for key in keys:
        history = audit_history(graph, key)
        expected, decay = payment_features(history, key)
        params: dict[str, Any] = {
            "node_types": [key.node_type],
            "node_ids": [key.node_id],
            "cutoff_seqs": [key.cutoff_seq],
            "cutoff_times": [key.cutoff_ms],
            **SAMPLER.query_params(),
            # Print the Fourier vectors so they are compared with numpy fourier64.
            "emit_encodings": True,
            **PLAN.query_flags(),
        }
        if text == "interpreted":
            # The repository text runs under INTERPRET with only its header swapped.
            query = as_interpreted((GSQL_DIR / CONTEXT_QUERY_FILE).read_text())
            row = checked_rows(graph.client.conn.runInterpretedQuery(query, params))[0]
        else:
            row = checked_rows(graph.run(CONTEXT_QUERY, params))[0]
        validate_context(key, row, PLAN, SAMPLER, require_encodings=True)
        actual = [m for m in row["messages"] if m["event_id"]]
        selected = stratify(visible_history(history, key), SAMPLER.roots)

        def identity(m: dict[str, Any]) -> tuple[str, str, str]:
            return m["relation"], m["event_id"], m["stratum"]

        assert {identity(m) for m in actual} == {identity(m) for m in selected}, key
        for m in actual:
            for name, value in expected[m["relation"] + ":" + m["event_id"]].items():
                assert np.isclose(m[name], value, rtol=1e-5, atol=1e-6), (name, m[name], value)
        for name, value in decay.items():
            have = row["features"][name]
            assert np.isclose(have, value, rtol=1e-5, atol=1e-6), (name, have, value)
