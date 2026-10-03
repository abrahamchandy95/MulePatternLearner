"""Offline checks of the rendered context queries (no server contact)."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

import pytest

from mule_pattern_learner.contract.analytics_features import ANALYTICS_GROUPS
from mule_pattern_learner.contract.feature_groups import (
    CORE_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.contract.server import (
    ANALYTICS_CONTEXT_FILE,
    ANALYTICS_CONTEXT_QUERY,
    ANALYTICS_CONTRACT,
    CONTEXT_CONTRACT,
    CONTEXT_QUERY_FILE,
    FOURIER_QUERY,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph import render
from mule_pattern_learner.tigergraph.render import (
    ANALYTICS_QUERY_GROUPS,
    TRAINING_GROUPS,
    analytics_contract,
    as_interpreted,
    context_contract,
    render_analytics_query,
    render_context_query,
)

header = render._header  # pyright: ignore[reportPrivateUsage]
# What the analytics groups name in a query's text: their flags and their features.
ANALYTICS_NAMES = (
    *(f"include_{group}" for group in ANALYTICS_GROUPS),
    "age_days",
    "_out_count",
    "out_in_amount_ratio",
    "recency_days",
    "_active",
    "decay_",
    "_last10",
    "visible_event_count",
    "pair_count_",
    "device_",
    "ip_age",
    "@@distinct",
    "@@last_ten",
)


@pytest.fixture(scope="module")
def analytics() -> str:
    return render_analytics_query()


@pytest.fixture(params=["training", "analytics"])
def either(request: pytest.FixtureRequest, text: str, analytics: str) -> str:
    """Each generated query in turn: the checks both must pass."""
    return text if request.param == "training" else analytics


def test_query_renderer_matches_reviewed_source_and_uses_no_labels(
    text: str, analytics: str
) -> None:
    for rendered, name in ((text, CONTEXT_QUERY_FILE), (analytics, ANALYTICS_CONTEXT_FILE)):
        assert re.sub(r"\s+", "", rendered) == re.sub(r"\s+", "", (GSQL_DIR / name).read_text())
        for field in ("is_mule", "fraud_label", "pu_label", "ring_id", "pair_time_encoding"):
            assert field not in rendered
        assert FOURIER_QUERY in rendered
        assert "e.valid_from_seq <= state_seq" in rendered
        assert "state_seq < e.valid_to_seq" in rendered


def test_renderer_guards_currency_and_does_not_use_a_global_reference_date(either: str) -> None:
    assert '"excluded_non_usd_events"' in either
    assert '"history_capacity_exceeded"' in either
    assert '"nonmonotonic_pair_clock"' in either
    assert 't.currency != "USD"),\n' not in either
    assert "now()" not in either.lower()
    for flag in FeaturePlan(CORE_GROUPS, "tgat").query_flags():
        assert f"BOOL {flag}" in either


def signature(query: str) -> str:
    return query.split("CREATE OR REPLACE QUERY", 1)[1].split("SYNTAX V2", 1)[0]


def test_rendered_files_are_byte_identical(text: str, analytics: str) -> None:
    assert (GSQL_DIR / CONTEXT_QUERY_FILE).read_text() == text
    assert (GSQL_DIR / ANALYTICS_CONTEXT_FILE).read_text() == analytics


@pytest.mark.parametrize(
    ("name", "recorded", "derive"),
    [
        ("CONTEXT_QUERY", CONTEXT_CONTRACT, context_contract),
        ("ANALYTICS_CONTEXT_QUERY", ANALYTICS_CONTRACT, analytics_contract),
    ],
)
def test_the_contract_is_derived_from_the_rendered_query(
    name: str, recorded: str, derive: Callable[[], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A changed query needs a new contract; the failure prints the value to set.
    assert recorded == derive(), derive()
    rendered = render_context_query() if name == "CONTEXT_QUERY" else render_analytics_query()
    assert rendered.count(f'"{recorded}" AS contract_version') == 1

    # Comments and whitespace do not count; any other change names a new contract.
    def commented(*args: Any) -> str:
        return "/* a comment */ " + header(*args)

    monkeypatch.setattr(render, "_header", commented)
    assert derive() == recorded
    monkeypatch.setattr(render, name, "fetch_other_context")
    assert derive() != recorded


def test_the_training_query_computes_only_the_groups_training_reads(
    text: str, analytics: str
) -> None:
    # The owner's decision (docs/architecture.md): the training query has a flag for each group
    # TigerGraph computes of the built-in run, and nothing of the analytics groups.
    flags = re.findall(r"BOOL (include_\w+) =", signature(text))
    assert flags == list(FeaturePlan().query_flags()) == [f"include_{g}" for g in TRAINING_GROUPS]
    assert "include_hub_indicator" not in text
    for hop in (1, 2):
        assert set(FeaturePlan(CORE_GROUPS, "tgat").query_flags(hop)) == set(flags)
    assert [name for name in ANALYTICS_NAMES if name in text] == []
    # The analytics query computes every group, training's among them.
    everything = re.findall(r"BOOL (include_\w+) =", signature(analytics))
    assert everything == [f"include_{g}" for g in ANALYTICS_QUERY_GROUPS]
    assert [name for name in ANALYTICS_NAMES if name not in analytics] == []
    assert f"CREATE OR REPLACE QUERY {ANALYTICS_CONTEXT_QUERY}(" in analytics
    # Every flag defaults to TRUE: the built-in run, and every analytics feature.
    for query in (text, analytics):
        assert "= FALSE" not in signature(query).split("emit_encodings = FALSE", 1)[1]


def test_signature_order_and_bounds(either: str) -> None:
    params = signature(either)
    assert params.index("INT max_history = 2048") < params.index("BOOL emit_encodings = FALSE")
    assert params.index("BOOL emit_encodings = FALSE") < params.index("BOOL include_")
    for bound in (
        "@@request_ids.size() > 64",
        "per_relation < 1 OR per_relation > 32",
        "k_old < 0 OR k_old > 16",
        "k_div < 0 OR k_div > 16",
        "k_assoc < 0 OR k_assoc > 8",
        "max_history < 32 OR max_history > 4096",
    ):
        assert bound in either


def test_ids_resolve_without_runtime_errors(either: str) -> None:
    for node_type in ("Account", "Token", "Party", "Device", "IP", "Address"):
        assert f'to_vertex_set(@@requested_{node_type}, "{node_type}")' in either
    # to_vertex is used once, after the typed lookup has proven the ID exists.
    assert either.count("to_vertex(") == 1
    assert either.index('"missing_entity"') < either.index("root = to_vertex(root_id, root_type)")


def test_interpreter_compatible_text(either: str) -> None:
    assert "getAttr" not in either
    body = either.split("SYNTAX V2 {", 1)[1]
    declarations, statements = body.split("@@request_ids = node_ids;", 1)
    scalars = set(
        re.findall(
            r"^\s+(?:INT|UINT|DOUBLE|BOOL|STRING|VERTEX|PairKey)\s+(\w+)", declarations, re.M
        )
    )
    globals_ = set(re.findall(r"@@(\w+)", either))
    params = set(re.findall(r"(?:STRING|INT|UINT|BOOL|LIST<\w+>)\s+(\w+)", signature(either)))
    assert "recipient_key" in scalars
    assert not scalars & globals_
    assert not params & globals_
    # A query declares exactly the accumulators it uses, so dropping a group's code
    # left no use without its declaration and no declaration without a use.
    assert set(re.findall(r"@@(\w+)", declarations)) == globals_
    assert set(re.findall(r"@@(\w+)", "@@request_ids = node_ids;" + statements)) == globals_
    interpreted = as_interpreted(either)
    assert interpreted.startswith("INTERPRET QUERY (\n  LIST<STRING> node_types")
    assert "CREATE" not in interpreted


def test_no_per_event_point_selects(either: str) -> None:
    for pattern in ("to_vertex(item.", "Event = {", "Peer = {", "@@sender_vertices"):
        assert pattern not in either
    per_event = either.split("FOREACH item IN @@events DO")[1:]
    assert per_event
    for loop in per_event:
        assert " SELECT " not in loop.split("\n    END;\n", 1)[0]


def test_encodings_only_on_request(either: str) -> None:
    lines = either.splitlines()
    calls = [i for i, line in enumerate(lines) if f"{FOURIER_QUERY}(" in line]
    assert len(calls) == 2
    for i in calls:
        assert lines[i - 1].strip() == "IF include_time_encoding AND emit_encodings THEN"
    assert "@@age_encoding AS age_encoding, @@gap_encoding AS gap_encoding" in either


RELATION_EDGES = (
    "Account_Sent_Zelle_Transfer>|Token_Sent_Transfer>",
    "Account_Received_Zelle_Transfer>|Token_Received_Transfer>",
    "Account_Initiated_Transaction>|Token_Sent_Transaction>",
    "Account_Received_Transaction>|Token_Received_Transaction>",
)


def test_each_relation_is_scanned_once_unless_float_sums_need_v4_order(
    text: str, analytics: str
) -> None:
    # The training query computes no floating sum, so each relation is scanned once.
    for edge in RELATION_EDGES:
        (scan,) = [m.start() for m in re.finditer(re.escape(f"Visible:s -(({edge}):e)-"), text)]
        assert text[:scan].rsplit("\n", 1)[1].lstrip().startswith("Candidates_")
    assert "include_rolling_windows OR include_decayed_activity" not in text
    # The analytics query scans again, in the v4 order, when it sums.
    guard = "      IF include_rolling_windows OR include_decayed_activity THEN\n"
    for edge in RELATION_EDGES:
        pattern = re.escape(f"Visible:s -(({edge}):e)-")
        scans = [m.start() for m in re.finditer(pattern, analytics)]
        assert len(scans) == 2
        candidates, ordered = scans
        assert analytics[:candidates].rstrip().endswith("= SELECT t FROM")
        assert analytics[:candidates].rsplit("\n", 1)[1].lstrip().startswith("Candidates_")
        # The second traversal only runs in the branch that keeps the v4 summation order.
        branch = analytics[:ordered].rsplit(guard, 1)
        assert len(branch) == 2 and "      ELSE\n" not in branch[1]
    assert analytics.count(guard) == 4


def test_pair_history_paths(text: str, analytics: str) -> None:
    # Path A (chronology) for Account roots and path B (the prior-pair scan) for the
    # others; the analytics query takes path B whenever pair window counts are asked for.
    assert (
        'use_chronology = root_type == "Account" AND (include_time_encoding OR '
        "include_pair_history);"
    ) in text
    assert (
        'use_prior_scan = root_type != "Account" AND (include_time_encoding OR '
        "include_pair_history);"
    ) in text
    assert "AND NOT include_pair_window_counts;" in analytics
    assert (
        "IF include_pair_window_counts OR ((include_time_encoding OR include_pair_history)"
        in analytics
    )
