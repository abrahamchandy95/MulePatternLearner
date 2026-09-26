"""Offline checks of the v5 TigerGraph query texts (no server contact)."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from mule_pattern_learner.temporal.live.contract import (
    CLIENT_GROUPS,
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.temporal.live.installation import definitions, parameter_names
from mule_pattern_learner.temporal.live.queries import as_interpreted, render_context_query

ROOT = Path(__file__).resolve().parents[2]
GSQL = ROOT / "gsql/temporal"
ORACLE = (
    "is_mule",
    "is_mule_masked",
    "pu_label",
    "mule_ring_id",
    "mule_label",
    "fraud_label",
    "label_known",
    "label_available",
)
PER_REQUEST = (
    "invalid_request",
    "missing_entity",
    "invisible_entity",
    "history_capacity_exceeded",
    "nonmonotonic_pair_clock",
    "invalid_payment_fields",
    "invalid_event_roles",
)
CALL_LEVEL = ("invalid_parameters", "invalid_visibility_phase", "scope_not_ready")
POPULATION_QUERIES = ("temporal_training_population", "temporal_scope_population")


@pytest.fixture(scope="module")
def text() -> str:
    return render_context_query()


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


def test_per_request_failures_continue_and_call_errors_return(text: str) -> None:
    for code in PER_REQUEST:
        prints = [
            m.end() for m in re.finditer(rf'PRINT "{code}" AS status, i AS request_index', text)
        ]
        assert prints, code
        for end in prints:
            following = text[end:].split(";", 2)
            assert following[1].strip() == "CONTINUE", code
    for code in CALL_LEVEL:
        found = re.search(rf'PRINT "{code}" AS status;', text)
        assert found is not None, code
        end = found.end()
        assert text[end:].lstrip().startswith("RETURN;"), code
    # The only RETURNs are the three call-level errors.
    assert text.count("RETURN;") == len(CALL_LEVEL)


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


def code(query: str) -> str:
    """Query text without block comments."""
    return re.sub(r"/\*.*?\*/", "", query, flags=re.S)


def query_texts(name: str) -> dict[str, str]:
    """Comment-free definitions of one repository file, by query name."""
    return {key: code(value) for key, value in definitions((GSQL / name).read_text()).items()}


def select_block(query: str, variable: str) -> str:
    """One `Variable = SELECT ...;` statement."""
    match = re.search(rf"\b{variable} = SELECT .*?;", query, flags=re.S)
    assert match is not None, variable
    return match[0]


def test_training_queries_read_no_oracle_attributes(text: str) -> None:
    assert not [field for field in ORACLE if field in text]
    checked = set()
    for name in ("training_cutoffs.gsql", "hub_registry.gsql", "training_scope.gsql"):
        for query_name, query in query_texts(name).items():
            if query_name in POPULATION_QUERIES:
                continue
            assert not [field for field in ORACLE if field in query], query_name
            checked.add(query_name)
    assert {
        "temporal_hub_registry",
        "temporal_create_training_scope",
        "temporal_finalize_training_scope",
        "temporal_scope_policy",
    } <= checked


def test_population_queries_default_to_no_observed_labels() -> None:
    for name, query in population_queries().items():
        assert "BOOL include_observed = FALSE" in query, name
        assert "include_observed = TRUE" not in query, name
        # Oracle attributes are only touched behind the explicit flag.
        before, guarded = query.split("IF include_observed THEN", 1)
        guarded, after = guarded.split("\n  END;\n", 1)
        assert not [field for field in ORACLE if field in before + after], name
        assert [field for field in ORACLE if field in guarded], name


def population_queries() -> dict[str, str]:
    texts = {**query_texts("training_population.gsql"), **query_texts("training_scope.gsql")}
    return {name: texts[name] for name in POPULATION_QUERIES}


CONTRACT_ROWS = {
    # name: (pu_label, is_mule, mule_label_known, is_mule_masked), label contract states
    "revealed": (1, 1, True, False),
    "hidden": (0, 1, True, True),
    "labeled_non_mule": (0, 0, True, False),
    "unknown": (0, 0, False, True),
    "unknown_mule": (0, 1, False, True),
    "inconsistent_masked": (1, 1, True, True),
    "inconsistent_unknown": (1, 1, False, False),
}


def observed_rule(query: str) -> tuple[str, list[str]]:
    guarded = query.split("IF include_observed THEN", 1)[1].split("\n  END;\n", 1)[0]
    match = re.fullmatch(
        r"\s*\w+ = SELECT a FROM Accounts:a\s+WHERE (.*?)\s+ACCUM (.*?);\s*", guarded, flags=re.S
    )
    assert match is not None, guarded
    return " ".join(match[1].split()), [" ".join(part.split()) for part in match[2].split(",")]


def evaluate(predicate: str, row: tuple[int, int, bool, bool]) -> bool:
    """The GSQL predicate over one Account row, as Python boolean logic."""
    names = ("pu_label", "is_mule", "mule_label_known", "is_mule_masked")
    expression = predicate.replace("a.", "").replace(" AND ", " and ").replace("NOT ", "not ")
    assert set(re.findall(r"[a-z_]+", expression)) <= {*names, "and", "not"}, expression
    return bool(eval(expression, {"__builtins__": {}}, dict(zip(names, row, strict=True))))


def test_observed_positive_is_the_revealed_contract_positive() -> None:
    expected = "a.pu_label == 1 AND a.is_mule == 1 AND a.mule_label_known AND NOT a.is_mule_masked"
    for name, query in population_queries().items():
        predicate, updates = observed_rule(query)
        assert predicate == expected, name
        # The discovery clock is emitted only for the accounts that pass the predicate.
        assert len(updates) == 2 and updates[0].endswith("+= TRUE"), name
        assert updates[1].endswith("+= a.mule_label_available_ts_ms"), name
        assert query.count("mule_label_available_ts_ms") == 1, name
        positives = {row for row, state in CONTRACT_ROWS.items() if evaluate(predicate, state)}
        assert positives == {"revealed"}, name


def test_cutoffs_report_every_requested_key() -> None:
    query = (GSQL / "training_cutoffs.gsql").read_text()
    init = query.index("@@sequences += (cutoff -> 0);")
    assert init < query.index("ScanEvents")


HUB_RELATIONS = (
    "Account_Initiated_Transaction",
    "Account_Received_Transaction",
    "Account_Sent_Zelle_Transfer",
    "Account_Received_Zelle_Transfer",
)
ROLE_EDGES = {
    "Payment_Transaction": "(Transaction_From_Account>|Transaction_To_Account>)",
    "Zelle_Transfer": "(Transfer_From_Account>|Transfer_To_Account>)",
}


def hub_query() -> str:
    return query_texts("hub_registry.gsql")["temporal_hub_registry"]


def test_hub_registry_contract() -> None:
    query = hub_query()
    assert (
        "CREATE OR REPLACE QUERY temporal_hub_registry(\n"
        '  LIST<UINT> cutoff_seqs, UINT threshold = 2048, STRING scope_id = ""\n)'
    ) in query
    assert parameter_names(query) == {"cutoff_seqs", "threshold", "scope_id"}
    for write in ("INSERT", "UPDATE", "DELETE"):
        assert write not in query
    for edge in HUB_RELATIONS:
        assert f'a.outdegree("{edge}") > threshold' in query
        assert f"-({edge}>:e)-" in query
    assert "@@requested.size() > 24" in query
    assert query.count("IF e.event_seq < cutoff THEN") == 4
    assert 'PRINT "invalid_parameters" AS status; RETURN;' in query
    # Scan cost is gone: all-time degree is informational and never decides a row.
    assert "scan" not in query.lower()
    assert query.count('"visible_history"') == 1 and query.count("HubRow(") == 1
    assert not re.search(r"@max_degree\s*[<>]", query)
    rows = select_block(query, "Rows")
    assert rows.count("@@hubs +=") == 1
    condition = rows.split("IF ", 1)[1].split(" THEN", 1)[0]
    assert "a.@visible.get(cutoff).get(p) > threshold" in condition
    assert "max_degree" not in condition
    assert "HubRow(a.id, cutoff, p, a.@visible.get(cutoff).get(p), a.@max_degree," in rows
    assert re.search(
        r"TUPLE<STRING account_id, UINT cutoff_seq, INT visibility_phase, UINT max_visible,"
        r"\s+UINT max_degree, STRING reason> HubRow",
        query,
    )
    assert "scope_id AS scope_id" in query


def test_hub_registry_unscoped_calls_count_everything_at_phase_three() -> None:
    query = hub_query()
    unscoped, scoped = query.split('IF scope_id == "" THEN', 1)[1].split("ELSE", 1)
    assert unscoped.strip() == "@@phases += 3;"
    for phase in (1, 2, 3):
        assert f"@@phases += {phase};" in scoped.split("END;\n  Accounts =", 1)[0]
    # Every count and every row filter falls back to "all events" without a scope.
    for relation in ("initiated", "received", "sent_zelle", "received_zelle"):
        assert re.search(
            r'IF scope_id == "" OR t\.@event_phase <= p THEN\s+'
            rf"a\.@{relation} \+= \(cutoff -> \(p -> 1\)\)",
            query,
        ), relation
    assert '(scope_id == "" OR (a.@in_scope AND a.@partition >= 1 AND a.@partition <= p))' in (
        select_block(query, "Rows")
    )


def test_hub_registry_scoped_counts_use_the_context_endpoint_rule(text: str) -> None:
    query = hub_query()
    ready = "ReadyScope = SELECT r FROM Scopes:r WHERE r.scope_id == scope_id AND r.ready;"
    assert ready in query
    assert re.search(
        r'IF ReadyScope.size\(\) != 1 THEN PRINT "scope_not_ready" AS status; RETURN; END;', query
    )
    # Each member Account's partition is read once.
    assert query.count("Training_Scope_Has_Entity") == 1
    members = select_block(query, "Members")
    assert "ReadyScope:r -(Training_Scope_Has_Entity>:m)- Account:a" in members
    assert "a.@in_scope += TRUE, a.@partition += m.partition" in members
    # The same Account role edges the context query uses to block events.
    context_roles = set(
        re.findall(r"Blocked_\w+ = SELECT t FROM \w+:t -\((\(.*?\)):role\)- Account:a", text)
    )
    assert context_roles == set(ROLE_EDGES.values())
    assert "WHERE NOT a.@scope_allowed ACCUM t.@scope_blocked += TRUE" in text
    assert (
        "membership.partition >= 1\n            AND membership.partition <= visibility_phase"
        in (text)
    )
    for variable, (event, roles) in zip(
        ("PaymentPhases", "TransferPhases"), ROLE_EDGES.items(), strict=True
    ):
        phases = select_block(query, variable)
        assert f"-({roles}:role)- Account:a" in phases, variable
        # Visible in p exactly when every endpoint has 1 <= partition <= p (4: never).
        assert re.search(
            r"ACCUM IF a\.@in_scope AND a\.@partition >= 1 AND a\.@partition <= 3 THEN\s+"
            r"t\.@event_phase \+= a\.@partition\s+ELSE\s+t\.@event_phase \+= 4\s+END;",
            phases,
        ), variable
        source = select_block(query, "Payments" if event == "Payment_Transaction" else "Transfers")
        assert f"- {event}:t" in source and "WHERE e.event_seq < @@last_cutoff" in source
    assert query.count("t.@event_phase += 4") == 2
    # The hub itself must be allowed in some phase to be a candidate at all.
    candidates = select_block(query, "Candidates")
    assert '(scope_id == "" OR (a.@in_scope AND a.@partition >= 1 AND a.@partition <= 3))' in (
        candidates
    )
    scoped = query.split('IF scope_id != "" THEN', 1)[1].split("\n  END;\n", 1)[0]
    for variable in ("Payments", "Transfers", "PaymentPhases", "TransferPhases"):
        assert f"{variable} = SELECT" in scoped


def scope_queries() -> dict[str, str]:
    return query_texts("training_scope.gsql")


def test_scope_unowned_policy_keeps_party_partitions() -> None:
    create = scope_queries()["temporal_create_training_scope"]
    params = create.split("CREATE OR REPLACE QUERY temporal_create_training_scope(", 1)[1]
    params = params.split(")", 1)[0]
    assert params.rstrip().endswith('STRING unowned_policy = "independent"')
    assert "shared_unowned" not in create
    assert (
        'OR (unowned_policy != "independent" AND unowned_policy != "shared"\n'
        '         AND unowned_policy != "linked") THEN\n'
        '    PRINT "invalid_parameters" AS status; RETURN;'
    ) in create
    branch = create.split("ACCUM IF s.@shared THEN", 1)
    assert len(branch) == 2
    shared, hashed = branch[1].split("      ELSE\n", 1)
    assert 'scope_id Temporal_Training_Scope, 1, "shared:" + to_string(s.@component))' in shared
    # The hash rule appears once, unchanged, in the branch taken by Party components.
    rule = "(((s.@component % 2147483647) * 1103515245 + split_seed) % 2147483647) % 10000"
    assert create.count(rule) == 2 and hashed.count(rule) == 2
    assert "< 7000 THEN 1" in hashed and "< 8500 THEN 2" in hashed
    assert "to_string(s.@component))" in hashed
    # Only unowned EXTERNAL accounts and unowned bank ledger ("gl") accounts are shared,
    # and never under "independent".
    assert create.count("@shared +=") == 1
    policy = create.split('IF unowned_policy != "independent" THEN', 1)[1].split("\n  END;", 1)[0]
    assert "SELECT p FROM Parties:p ACCUM @@party_components += p.@component" in policy
    assert (
        "SELECT a FROM Accounts:a\n"
        '      WHERE (a.is_external OR a.account_type == "gl")\n'
        "        AND NOT @@party_components.contains(a.@component)\n"
        "      ACCUM a.@shared += TRUE"
    ) in policy
    # Components are final before any policy runs; only three statements ever write them.
    assert create.index("WHILE @@changed") < create.index('IF unowned_policy != "independent"')
    writes = re.findall(r"\w\.@component (?:=|\+=) [^,;\n]+", create)
    assert writes == [
        "s.@component = getvid(s)",
        "t.@component += s.@component",
        "a.@component = a.@link_component",
    ]
    assert create.index("a.@component = a.@link_component") < create.index("INSERT INTO")


def test_scope_linked_policy_attaches_only_single_counterparty_internal_accounts() -> None:
    create = scope_queries()["temporal_create_training_scope"]
    linked = create.split('IF unowned_policy == "linked" THEN', 1)[1].split("\n  END;", 1)[0]
    assert create.index("WHILE @@changed") < create.index('IF unowned_policy == "linked"')
    # Ledger accounts are shared, so only unowned internal customer accounts can be linked.
    unowned_internal = (
        'NOT a.is_external AND a.account_type != "gl"\n'
        "        AND NOT @@party_components.contains(a.@component)"
    )
    assert unowned_internal in select_block(linked, "Linkable")
    holder = (
        'd.account_type == "deposit" AND NOT d.is_external\n'
        "        AND @@party_components.contains(d.@component)"
    )
    for variable, relations, roles in (
        (
            "Payment",
            "Account_Initiated_Transaction>|Account_Received_Transaction>",
            ROLE_EDGES["Payment_Transaction"],
        ),
        (
            "Transfer",
            "Account_Sent_Zelle_Transfer>|Account_Received_Zelle_Transfer>",
            ROLE_EDGES["Zelle_Transfer"],
        ),
    ):
        events = select_block(linked, f"Link{variable}s")
        assert f"Linkable:a\n      -(({relations}):e)-" in events
        holders = select_block(linked, f"{variable}Holders")
        assert f"Link{variable}s:t\n      -({roles}:role)- Account:d" in holders
        assert holder in holders
        assert "t.@link_min += getvid(d), t.@link_max += getvid(d)" in holders
        links = select_block(linked, f"{variable}Links")
        assert f"{variable}Holders:t\n      -({roles}:role)- Account:a" in links
        assert unowned_internal in links
        assert "a.@link_min += t.@link_min, a.@link_max += t.@link_max" in links
    # Exactly one distinct owned deposit counterparty: min == max over all of them.
    final = select_block(linked, "Linked")
    assert "FROM Linkable:a" in final
    assert "WHERE a.@link_found AND a.@link_min == a.@link_max" in final
    assert "POST-ACCUM a.@component = a.@link_component" in final
    assert "event_seq" not in linked and "ts_ms" not in linked


def test_scope_policy_query_classifies_unowned_accounts() -> None:
    queries = scope_queries()
    policy = queries["temporal_scope_policy"]
    assert "CREATE OR REPLACE QUERY temporal_scope_policy(STRING scope_id)\n" in policy
    for write in ("INSERT", "UPDATE", "DELETE", ".ready ="):
        assert write not in policy
    assert re.search(r'PRINT "scope_not_ready" AS status; RETURN;', policy)
    assert "WHERE r.scope_id == scope_id AND r.ready;" in policy
    unowned = select_block(policy, "Unowned")
    assert 'WHERE a.outdegree("Account_Owned_By_Party") == 0' in unowned
    assert "a.@group += e.group_id" in unowned
    assert 'WHERE a.@group LIKE "shared:%"' in select_block(policy, "Shared")
    assert "WHERE a.@group == to_string(getvid(a))" in select_block(policy, "Independent")
    assert "WHERE NOT a.@classified" in select_block(policy, "Linked")
    printed = policy.split('PRINT "ok" AS status', 1)[1]
    for side in ("internal", "external"):
        for kind in ("shared", "independent", "linked"):
            assert f"@@{kind}_{side} AS {kind}_{side}" in printed
            assert f"@@{kind}_{side} += 1" in policy
    assert "@@members AS members" in printed
    # The classes match what the create query writes for a singleton unowned component.
    create = queries["temporal_create_training_scope"]
    assert "s.@component = getvid(s)" in create
    assert '"shared:" + to_string(s.@component)' in create
    assert "          to_string(s.@component))" in create


@pytest.mark.parametrize(
    "name",
    [
        "training_context.gsql",
        "hub_registry.gsql",
        "training_scope.gsql",
        "training_population.gsql",
        "training_cutoffs.gsql",
    ],
)
def test_brackets_balance(name: str) -> None:
    query = re.sub(r'"[^"\n]*"', '""', (GSQL / name).read_text())
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.S)
    for opening, closing in ("()", "{}", "[]"):
        assert query.count(opening) == query.count(closing), (name, opening)
