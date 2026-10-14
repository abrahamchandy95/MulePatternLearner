"""Offline checks of the repository query files and the parameters the client sends."""

from __future__ import annotations

import re

import pytest

from mule_pattern_learner.contract.feature_groups import ARCHITECTURES, CORE_GROUPS, FeaturePlan
from mule_pattern_learner.contract.server import (
    CONTEXT_QUERY,
    CREATE_SCOPE_QUERY,
    FINALIZE_SCOPE_QUERY,
    HUB_QUERY,
    POPULATION_QUERY,
)
from mule_pattern_learner.data.contexts import ContextSource
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.builders import (
    CORE_PLAN,
    SMALL_SAMPLER,
    hub_rows,
    payments_context,
    root,
)
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.testing.gsql_blocks import flat, split_block
from mule_pattern_learner.tigergraph import gsql_text
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher
from mule_pattern_learner.tigergraph.gsql_text import definitions, parameter_names
from mule_pattern_learner.tigergraph.hubs import TigerGraphHubReader

GSQL = GSQL_DIR / "queries"
ORACLE = (
    "is_mule",
    "pu_label",
    "mule_ring_id",
    "mule_label",
    "fraud_label",
    "label_known",
    "label_available",
)
PER_REQUEST = ("missing_entity", "invisible_entity", "history_capacity_exceeded")
CALL_LEVEL = ("invalid_parameters", "scope_not_ready")
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


def test_sent_parameters_match_the_repository_query_signatures() -> None:
    def signature(path: str, name: str) -> set[str]:
        text = gsql_text.definitions((GSQL_DIR / path).read_text())[name]
        return gsql_text.parameter_names(text)

    context = signature("queries/training_context.gsql", CONTEXT_QUERY)
    architectures = (FeaturePlan(CORE_GROUPS, a) for a in ARCHITECTURES)
    for plan in (CORE_PLAN, *architectures):
        server = FakeTigerGraph(factory=payments_context)
        store = ContextSource(TigerGraphContextFetcher(server), plan=plan, sampler=SMALL_SAMPLER)
        store.fetch([root(0)], hop=1)
        store.fetch([root(0)], hop=2)
        store.close()
        assert len(server.calls) == 2 and all(set(call) == context for _, call in server.calls)
    hubs = FakeTigerGraph(answers={HUB_QUERY: lambda params: hub_rows([1000, 2000])})
    TigerGraphHubReader(hubs).hub_registry([1000, 2000], threshold=1024)
    assert set(hubs.calls[0][1]) == signature("queries/hub_accounts.gsql", HUB_QUERY)
    assert set(hubs.calls[0][1]) == {"cutoff_seqs", "threshold", "by_phase"}
    # The graph holds one split: no GSQL file names a scope id or a source id.
    creation = signature("queries/training_scope.gsql", CREATE_SCOPE_QUERY)
    assert creation == {"train_share", "validation_share", "test_share", "split_seed"}
    files = sorted(GSQL_DIR.rglob("*.gsql"))
    assert len(files) > 8
    for path in files:
        assert not re.search(r"scope_id|source_id", path.read_text()), path


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
    # The only RETURNs are the call-level errors, and no other status is printed.
    assert text.count("RETURN;") == len(CALL_LEVEL)
    statuses = set(re.findall(r'PRINT "(\w+)" AS status', text))
    assert statuses == {*PER_REQUEST, *CALL_LEVEL, "ok"}


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
    for name in ("split_cutoffs.gsql", "hub_accounts.gsql", "training_scope.gsql"):
        for query_name, query in query_texts(name).items():
            if query_name == POPULATION_QUERY:
                continue
            assert not [field for field in ORACLE if field in query], query_name
            checked.add(query_name)
    assert {HUB_QUERY, CREATE_SCOPE_QUERY, FINALIZE_SCOPE_QUERY} <= checked


def test_the_population_query_reports_only_the_revealed_label() -> None:
    query = query_texts("training_scope.gsql")[POPULATION_QUERY]
    labels = select_block(query, "Labels")
    # The revealed positive (pu_label 1, which the reveal writes only for a revealed
    # mule) and its discovery clock; no other label attribute is read.
    assert flat(labels) == (
        "Labels = SELECT a FROM Accounts:a WHERE a.pu_label == 1 "
        "ACCUM a.@positive += TRUE, a.@known_ms += a.mule_label_available_ts_ms;"
    )
    assert not [field for field in ORACLE if field in query.replace(labels, "")]


def test_cutoffs_report_every_requested_key() -> None:
    query = (GSQL / "split_cutoffs.gsql").read_text()
    init = query.index("@@sequences += (cutoff -> 0);")
    assert init < query.index("ScanEvents")


def hub_query() -> str:
    return query_texts("hub_accounts.gsql")[HUB_QUERY]


def test_hub_registry_contract() -> None:
    query = hub_query()
    assert (
        f"CREATE OR REPLACE QUERY {HUB_QUERY}( "
        "LIST<UINT> cutoff_seqs, UINT threshold = 2048, BOOL by_phase = FALSE )"
    ) in flat(query)
    assert parameter_names(query) == {"cutoff_seqs", "threshold", "by_phase"}
    for write in ("INSERT", "UPDATE", "DELETE"):
        assert write not in query
    for edge in HUB_RELATIONS:
        assert f'a.outdegree("{edge}") > threshold' in query
    # One count over the four relations, kept apart by edge type.
    counted = flat(select_block(query, "Counted"))
    relations = "|".join(f"{edge}>" for edge in HUB_RELATIONS)
    assert f"Candidates:a -(({relations}):e)- :t" in counted
    assert "a.@events += (e.type -> (cutoff -> (p -> 1)))" in counted
    assert query.count("IF e.event_seq < cutoff THEN") == 1
    assert "@@requested.size() > 24" in query
    assert 'PRINT "invalid_parameters" AS status; RETURN;' in query
    # The largest relation's count decides a row, which prints only it beside its key.
    visible = flat(select_block(query, "Visible"))
    assert "FOREACH (relation, counts) IN a.@events DO" in visible
    assert "a.@visible += (cutoff -> (p -> counts.get(cutoff).get(p)))" in visible
    assert "MapAccum<UINT, MapAccum<INT, MaxAccum<UINT>>> @visible;" in query
    assert (
        "TUPLE< STRING account_id, UINT cutoff_seq, INT visibility_phase, UINT max_visible > HubRow"
        in flat(query)
    )
    rows = select_block(query, "Rows")
    assert rows.count("@@hubs +=") == 1 and query.count("HubRow(") == 1
    condition = rows.split("IF ", 1)[1].split(" THEN", 1)[0]
    assert "a.@visible.get(cutoff).get(p) > threshold" in condition
    assert "HubRow(a.id, cutoff, p, a.@visible.get(cutoff).get(p))" in rows
    assert 'PRINT "ok" AS status, @@hubs AS hubs;' in query
    assert "max_degree" not in query and "@@candidates" not in query


def test_hub_registry_calls_without_phases_count_everything_at_phase_zero() -> None:
    query = hub_query()
    unsplit, split = flat(split_block(query, "IF NOT by_phase THEN")[1]).split(" ELSE ", 1)
    assert unsplit == "@@phases += 0;"
    for phase in (1, 2, 3):
        assert f"@@phases += {phase};" in split
    # Every count and every row filter falls back to "all events" without phases.
    assert re.search(
        r"IF NOT by_phase OR t\.@event_phase <= p THEN\s+"
        r"a\.@events \+= \(e\.type -> \(cutoff -> \(p -> 1\)\)\)",
        query,
    )
    assert "(NOT by_phase OR a.@partition <= p)" in flat(select_block(query, "Rows"))


def test_hub_registry_counts_by_phase_use_the_context_endpoint_rule(text: str) -> None:
    query = hub_query()
    # By phase, the counts are those of the graph's one ready split.
    ready = "ReadyScope = SELECT r FROM Scopes:r WHERE r.ready;"
    assert ready in query
    assert re.search(
        r'IF ReadyScope.size\(\) != 1 THEN PRINT "scope_not_ready" AS status; RETURN; END;', query
    )
    # Each member Account's partition is read once; finalize_training_scope gave every
    # Account one from 1 to 3, so no other value is tested for.
    assert query.count("Training_Scope_Has_Entity") == 1
    members = select_block(query, "Members")
    assert "ReadyScope:r -(Training_Scope_Has_Entity>:m)- Account:a" in members
    assert "ACCUM a.@partition += m.partition;" in members
    assert "@in_scope" not in query and "+= 4" not in query
    # The same Account role edges the context query uses to block events: one pass over
    # the events of both types, of the root and of the prior-pair scan.
    context_roles = re.findall(
        r"(?:Prior)?Blocked = SELECT t FROM \w+:t -\(\((.*?)\):role\)- Account:a", text
    )
    both = "|".join(edges.strip("()") for edges in reversed(ROLE_EDGES.values()))
    assert context_roles == [both, both]
    assert "WHERE NOT a.@scope_allowed ACCUM t.@scope_blocked += TRUE" in text
    assert "membership.partition >= 1 AND membership.partition <= visibility_phase" in flat(text)
    for variable, (event, roles) in zip(
        ("PaymentPhases", "TransferPhases"), ROLE_EDGES.items(), strict=True
    ):
        phases = select_block(query, variable)
        assert f"-({roles}:role)- Account:a" in phases, variable
        # Visible in p exactly when every endpoint has partition <= p.
        assert "ACCUM t.@event_phase += a.@partition;" in phases, variable
        source = select_block(query, "Payments" if event == "Payment_Transaction" else "Transfers")
        assert f"- {event}:t" in source and "WHERE e.event_seq < @@last_cutoff" in source
    by_phase = split_block(query, "IF by_phase THEN")[1]
    for variable in ("Payments", "Transfers", "PaymentPhases", "TransferPhases"):
        assert f"{variable} = SELECT" in by_phase


def scope_queries() -> dict[str, str]:
    return query_texts("training_scope.gsql")


def test_scope_partitions_hash_ownership_groups_and_share_unowned_ledgers() -> None:
    create = scope_queries()[CREATE_SCOPE_QUERY]
    params = create.split(f"CREATE OR REPLACE QUERY {CREATE_SCOPE_QUERY}(", 1)[1]
    assert flat(params.split(")", 1)[0]) == (
        "DOUBLE train_share, DOUBLE validation_share, DOUBLE test_share, UINT split_seed = 42"
    )
    assert "WHILE @@changed AND iteration < 100 DO" in create
    membership = flat(select_block(create, "Membership"))
    # Shared accounts are in partition 1 (visible in every phase); every other entity's
    # partition is its ownership group's hash against the shares.
    rule = "(((s.@component % 2147483647) * 1103515245 + split_seed) % 2147483647) % 10000"
    assert create.count(rule) == 2
    assert (
        '"split" Temporal_Training_Scope, CASE WHEN s.@shared THEN 1 '
        f"WHEN ({rule} + 0.5) / 10000 < train_share THEN 1 "
        f"WHEN ({rule} + 0.5) / 10000 < train_share + validation_share THEN 2 "
        "ELSE 3 END);"
    ) in membership
    assert "7000" not in create and "8500" not in create
    # Only unowned EXTERNAL accounts and unowned bank ledger ("gl") accounts are shared.
    assert create.count("@shared +=") == 1
    assert "SELECT p FROM Parties:p ACCUM @@party_components += p.@component" in flat(create)
    assert flat(select_block(create, "SharedAccounts")) == (
        "SharedAccounts = SELECT a FROM Accounts:a "
        'WHERE (a.is_external OR a.account_type == "gl") '
        "AND NOT @@party_components.contains(a.@component) "
        "ACCUM a.@shared += TRUE;"
    )
    # Components are final before any account is shared or linked; only three
    # statements ever write them.
    assert create.index("WHILE @@changed") < create.index("Parties = {Party.*};")
    writes = re.findall(r"\w\.@component (?:=|\+=) [^,;\n]+", create)
    assert writes == [
        "s.@component = getvid(s)",
        "t.@component += s.@component",
        "a.@component = a.@link_component",
    ]
    assert create.index("a.@component = a.@link_component") < create.index("INSERT INTO")


def test_scope_links_only_single_counterparty_internal_accounts() -> None:
    create = scope_queries()[CREATE_SCOPE_QUERY]
    linked = flat(create[create.index("Linkable = SELECT") : create.index("INSERT INTO")])
    assert create.index("WHILE @@changed") < create.index("Linkable = SELECT")
    # Ledger accounts are shared, so only unowned internal customer accounts can be linked.
    unowned_internal = (
        'NOT a.is_external AND a.account_type != "gl" '
        "AND NOT @@party_components.contains(a.@component)"
    )
    assert unowned_internal in select_block(linked, "Linkable")
    holder = (
        'd.account_type == "deposit" AND NOT d.is_external '
        "AND @@party_components.contains(d.@component)"
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
        assert f"Linkable:a -(({relations}):e)-" in events
        holders = select_block(linked, f"{variable}Holders")
        assert f"Link{variable}s:t -({roles}:role)- Account:d" in holders
        assert holder in holders
        assert "t.@link_min += getvid(d), t.@link_max += getvid(d)" in holders
        links = select_block(linked, f"{variable}Links")
        assert f"{variable}Holders:t -({roles}:role)- Account:a" in links
        assert unowned_internal in links
        assert "a.@link_min += t.@link_min, a.@link_max += t.@link_max" in links
    # Exactly one distinct owned deposit counterparty: min == max over all of them.
    final = select_block(linked, "Linked")
    assert "FROM Linkable:a" in final
    assert "WHERE a.@link_found AND a.@link_min == a.@link_max" in final
    assert "POST-ACCUM a.@component = a.@link_component" in final
    assert "event_seq" not in linked and "ts_ms" not in linked


@pytest.mark.parametrize(
    "name",
    [
        "queries/training_context.gsql",
        "queries/hub_accounts.gsql",
        "queries/training_scope.gsql",
        "queries/split_cutoffs.gsql",
    ],
)
def test_brackets_balance(name: str) -> None:
    query = re.sub(r'"[^"\n]*"', '""', (GSQL_DIR / name).read_text())
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.S)
    for opening, closing in ("()", "{}", "[]"):
        assert query.count(opening) == query.count(closing), (name, opening)
