"""Offline checks of the rendered context query (no server contact)."""

from __future__ import annotations

import re

from mule_pattern_learner.contract.feature_groups import (
    CLIENT_GROUPS,
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph.render import as_interpreted, render_context_query

GSQL = GSQL_DIR / "queries"


def test_query_renderer_matches_reviewed_source_and_uses_no_labels() -> None:
    text = render_context_query()
    assert re.sub(r"\s+", "", text) == re.sub(
        r"\s+", "", (GSQL / "training_context.gsql").read_text()
    )
    for field in ("is_mule", "fraud_label", "pu_label", "ring_id", "pair_time_encoding"):
        assert field not in text
    assert "temporal_fourier64_values" in text
    assert "e.valid_from_seq <= state_seq" in text
    assert "state_seq < e.valid_to_seq" in text


def test_renderer_guards_currency_and_does_not_use_a_global_reference_date():
    query = render_context_query()
    assert '"excluded_non_usd_events"' in query
    assert '"history_capacity_exceeded"' in query
    assert '"nonmonotonic_pair_clock"' in query
    assert 't.currency != "USD"),\n' not in query
    assert "now()" not in query.lower()
    assert "IF include_pair_window_counts OR" in query
    for flag in FeaturePlan(DEFAULT_GROUPS, "split").query_flags():
        assert f"BOOL {flag}" in query


def signature(query: str) -> str:
    return query.split("CREATE OR REPLACE QUERY", 1)[1].split("SYNTAX V2", 1)[0]


def test_rendered_file_is_byte_identical(text: str) -> None:
    assert (GSQL / "training_context.gsql").read_text() == text


def test_flags_are_exactly_the_non_client_groups(text: str) -> None:
    flags = re.findall(r"BOOL (include_\w+) =", signature(text))
    expected = [
        "include_" + name
        for name, spec in FEATURE_GROUPS.items()
        if spec.path != "categorical" and name != "message_core" and name not in CLIENT_GROUPS
    ]
    assert flags == expected == list(FeaturePlan().query_flags())
    assert "include_hub_indicator" not in text
    for hop in (1, 2):
        assert set(FeaturePlan(DEFAULT_GROUPS, "split").query_flags(hop)) == set(flags)


def test_signature_order_and_bounds(text: str) -> None:
    params = signature(text)
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
        assert bound in text


def test_ids_resolve_without_runtime_errors(text: str) -> None:
    for node_type in ("Account", "Token", "Party", "Device", "IP", "Address"):
        assert f'to_vertex_set(@@requested_{node_type}, "{node_type}")' in text
    # to_vertex is used once, after the typed lookup has proven the ID exists.
    assert text.count("to_vertex(") == 1
    assert text.index('"missing_entity"') < text.index("root = to_vertex(root_id, root_type)")


def test_interpreter_compatible_text(text: str) -> None:
    assert "getAttr" not in text
    body = text.split("SYNTAX V2 {", 1)[1]
    declarations = body.split("@@request_ids = node_ids;", 1)[0]
    scalars = set(
        re.findall(
            r"^\s+(?:INT|UINT|DOUBLE|BOOL|STRING|VERTEX|PairKey)\s+(\w+)", declarations, re.M
        )
    )
    globals_ = set(re.findall(r"@@(\w+)", text))
    params = set(re.findall(r"(?:STRING|INT|UINT|BOOL|LIST<\w+>)\s+(\w+)", signature(text)))
    assert "recipient_key" in scalars
    assert not scalars & globals_
    assert not params & globals_
    interpreted = as_interpreted(text)
    assert interpreted.startswith("INTERPRET QUERY (\n  LIST<STRING> node_types")
    assert "CREATE" not in interpreted


def test_no_per_event_point_selects(text: str) -> None:
    for pattern in ("to_vertex(item.", "Event = {", "Peer = {", "@@sender_vertices"):
        assert pattern not in text
    per_event = text.split("FOREACH item IN @@events DO")[1:]
    assert per_event
    for loop in per_event:
        assert " SELECT " not in loop.split("\n    END;\n", 1)[0]


def test_encodings_only_on_request(text: str) -> None:
    lines = text.splitlines()
    calls = [i for i, line in enumerate(lines) if "temporal_fourier64_values(" in line]
    assert len(calls) == 2
    for i in calls:
        assert lines[i - 1].strip() == "IF include_time_encoding AND emit_encodings THEN"
    assert "@@age_encoding AS age_encoding, @@gap_encoding AS gap_encoding" in text


def test_each_relation_is_scanned_once_unless_float_sums_need_v4_order(text: str) -> None:
    guard = "      IF include_rolling_windows OR include_decayed_activity THEN\n"
    for edge in (
        "Account_Sent_Zelle_Transfer>|Token_Sent_Transfer>",
        "Account_Received_Zelle_Transfer>|Token_Received_Transfer>",
        "Account_Initiated_Transaction>|Token_Sent_Transaction>",
        "Account_Received_Transaction>|Token_Received_Transaction>",
    ):
        scans = [m.start() for m in re.finditer(re.escape(f"Visible:s -(({edge}):e)-"), text)]
        assert len(scans) == 2
        candidates, ordered = scans
        assert text[:candidates].rstrip().endswith("= SELECT t FROM")
        assert text[:candidates].rsplit("\n", 1)[1].lstrip().startswith("Candidates_")
        # The second traversal only runs in the branch that keeps the v4 summation order.
        branch = text[:ordered].rsplit(guard, 1)
        assert len(branch) == 2 and "      ELSE\n" not in branch[1]
    assert text.count(guard) == 4


def test_chronology_skipped_when_prior_scan_overrides_it(text: str) -> None:
    assert "AND NOT include_pair_window_counts;" in text
    assert (
        "IF include_pair_window_counts OR ((include_time_encoding OR include_pair_history)" in text
    )
