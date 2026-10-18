"""The six infrastructure queries, read as text (docs/design.md, "The queries"): list_accounts,
read_labels, read_answer_key, read_hubs, read_world and profile_hubs.

Each passes every rule of testing.gsql_rules, takes the parameters its Python builder gives and
prints the fields of the dataclasses its parser reads, each from what it is named for. Beside
those rules, each query's own rules are held here clause by clause, since no reference graph
runs them yet: its declarations, its refusals, the scored population, the label readers, the
hub and crowd measures and the world's identity. gsql-lsp
checks every file against the schema in tests/gsql/test_schema.py, and the live compile test
compiles them once the workspace has the schema.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
import re
from typing import Any

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract import LABEL_COLUMNS, TABLES, Kind
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    Select,
    Step,
    common_problems,
    field_rule,
    parameter_rule,
    projection_fields,
    projection_items,
    scopes,
    source_rule,
    statements,
    status_fields,
    status_items,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    LIST_ACCOUNTS,
    PROFILE_HUBS,
    QUERY_FILES,
    READ_ANSWER_KEY,
    READ_HUBS,
    READ_LABELS,
    READ_WORLD,
    REGISTRY,
    VERSIONS,
    written_files,
)
from mule_pattern_learner.tigergraph.queries import read_answer_key as answer_key
from mule_pattern_learner.tigergraph.queries.list_accounts import ScorableAccount
from mule_pattern_learner.tigergraph.queries.profile_hubs import HubProfile
from mule_pattern_learner.tigergraph.queries.read_hubs import HubAccount, Leaf
from mule_pattern_learner.tigergraph.queries.read_labels import IdentifiedMule
from mule_pattern_learner.tigergraph.queries.read_world import SECONDS, World
from mule_pattern_learner.tigergraph.queries.rows import (
    LOOKBACK_MS,
    MONDAY_MS,
    PAGE_LIMIT,
    WEEK_MS,
)

INFRASTRUCTURE = (LIST_ACCOUNTS, READ_LABELS, READ_ANSWER_KEY, READ_HUBS, READ_WORLD, PROFILE_HUBS)
TEXTS = {name: (GSQL_DIR / QUERY_FILES[name]).read_text() for name in INFRASTRUCTURE}
DEFINITIONS = {name: definitions(text)[name] for name, text in TEXTS.items()}
SUBGRAPH = DEFAULT_CONFIG.subgraph
# A Monday after the lookback, for the builders.
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# Each query's arguments for its builder, and the dataclass of each printed set.
ARGUMENTS: dict[str, tuple[Any, ...]] = {
    LIST_ACCOUNTS: (T_MS, "", PAGE_LIMIT),
    READ_LABELS: ("", PAGE_LIMIT),
    READ_ANSWER_KEY: ("", PAGE_LIMIT),
    READ_HUBS: (T_MS,),
    READ_WORLD: (),
    PROFILE_HUBS: (T_MS,),
}
PROJECTIONS: list[tuple[str, str, type]] = [
    (LIST_ACCOUNTS, "Page", ScorableAccount),
    (READ_LABELS, "Identified", IdentifiedMule),
    (READ_ANSWER_KEY, "Page", answer_key.AccountTruth),
    (READ_HUBS, "HubAccounts", HubAccount),
    (READ_HUBS, "DeviceLeaves", Leaf),
    (READ_HUBS, "IpLeaves", Leaf),
]
HEADS: list[tuple[str, type]] = [(READ_WORLD, World), (PROFILE_HUBS, HubProfile)]
PAGED = (LIST_ACCOUNTS, READ_LABELS, READ_ANSWER_KEY)
AS_OF = (LIST_ACCOUNTS, READ_HUBS, PROFILE_HUBS)


def _builder(name: str) -> Callable[..., dict[str, Any]]:
    """The builder of a query's parameters: the registry's, or the answer key's own module's."""
    built = answer_key.params if name == READ_ANSWER_KEY else REGISTRY[name].params
    assert built is not None, name
    return built


def _select(name: str, target: str) -> Select:
    """The SELECT of a query that assigns target."""
    for statement, _, select in scopes(DEFINITIONS[name]):
        if select is not None and statement[:2] == [target, "="]:
            return select
    raise AssertionError(f"{name} assigns no {target} by a SELECT")


def _code(name: str, target: str) -> str:
    """The text of the statement that assigns target, its tokens joined by spaces."""
    return " ".join(_select(name, target).statement)


def _edges(select: Select) -> list[frozenset[str]]:
    """The edge types of each step of a SELECT's pattern, in order."""
    return [edge.types for edge in select.edges]


def _visible(alias: str) -> set[str]:
    """The two conjuncts of a link visible at t."""
    return {
        f"{alias}.valid_from_ms < t_ms",
        f"( {alias}.valid_to_ms == 0 OR {alias}.valid_to_ms > t_ms - lookback_ms )",
    }


WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms"}
SENT_AND_RECEIVED = frozenset(
    {
        "Account_Initiated_Transaction",
        "Account_Received_Transaction",
        "Account_Sent_Zelle_Transfer",
        "Account_Received_Zelle_Transfer",
    }
)
PAYMENT_ENDS = frozenset(
    {"Transaction_From_Account", "Transaction_To_Account", "Transfer_From_Account"}
    | {"Transfer_To_Account"}
)
SEEN_ON = frozenset(
    {
        "Device_Used_By_Transaction",
        "IP_Observed_In_Transaction",
        "Device_Used_By_Transfer",
        "IP_Used_By_Transfer",
    }
)
SENDERS = frozenset({"Transaction_From_Account", "Transfer_From_Account"})
OWNER = frozenset({"Account_Owned_By_Party"})


# Every rule, and the Python above each query.


@pytest.mark.parametrize("name", INFRASTRUCTURE)
def test_each_query_passes_every_rule_and_prints_its_version(name: str) -> None:
    assert QUERY_FILES[name] in written_files().values()
    assert common_problems(name, TEXTS[name]) == []
    assert version_rule(name, DEFINITIONS[name], VERSIONS[name]) == []


@pytest.mark.parametrize("name", INFRASTRUCTURE)
def test_each_query_takes_the_parameters_its_builder_gives(name: str) -> None:
    built = _builder(name)(*ARGUMENTS[name])
    assert parameter_rule(name, DEFINITIONS[name], built) == []


@pytest.mark.parametrize(("name", "vertex_set", "kind"), PROJECTIONS)
def test_each_printed_set_has_the_fields_of_its_dataclass(
    name: str, vertex_set: str, kind: type
) -> None:
    printed = projection_fields(DEFINITIONS[name], vertex_set)
    assert field_rule(f"{name} {vertex_set}", printed, kind) == []
    items = projection_items(DEFINITIONS[name], vertex_set)
    assert source_rule(f"{name} {vertex_set}", items) == []


@pytest.mark.parametrize(("name", "kind"), HEADS)
def test_each_printed_status_row_has_the_fields_of_its_dataclass(name: str, kind: type) -> None:
    assert field_rule(f"{name} status", status_fields(DEFINITIONS[name]), kind) == []
    assert source_rule(f"{name} status", status_items(DEFINITIONS[name])) == []


def test_the_registry_holds_each_querys_builder_and_rows_but_the_answer_keys() -> None:
    for name, kind in [*((n, k) for n, _, k in PROJECTIONS), *HEADS]:
        if name != READ_ANSWER_KEY:
            assert kind in REGISTRY[name].rows, (name, kind)
    # The answer key's parser is imported by evaluation and testing only.
    assert REGISTRY[READ_ANSWER_KEY].params is None and REGISTRY[READ_ANSWER_KEY].rows == ()


# What each query declares, exactly, in order. The type of an accumulator decides what a
# measure counts: a SetAccum of customers counts each customer once where a ListAccum would
# count each payment; an OrAccum flag is set when any write is true where an AndAccum would
# need all; a SumAccum map adds where a MaxAccum keeps the largest value.
DECLARATIONS: dict[str, list[str]] = {
    LIST_ACCOUNTS: ["MaxAccum<STRING> @owner_id", f"UINT lookback_ms = {LOOKBACK_MS}"],
    READ_LABELS: ["SetAccum<STRING> @@invalid"],
    READ_ANSWER_KEY: ["OrAccum @owned"],
    READ_HUBS: [
        "SetAccum<VERTEX<Party>> @hub_customers, @crowd_customers, @crowd_enrolled",
        "SumAccum<INT> @customers_90d",
        "OrAccum @observed_only, @hub",
        f"UINT lookback_ms = {LOOKBACK_MS}",
        f"UINT crowd_threshold = {SUBGRAPH.crowd_threshold}",
        f"UINT hub_threshold = {SUBGRAPH.hub_threshold}",
    ],
    READ_WORLD: [
        "MapAccum<STRING, SumAccum<INT>> @out_degree, @in_degree",
        "MapAccum<STRING, SumAccum<INT>> @@vertices, @@edges, @@account_types, @@payment_rails, "
        "@@seconds, @@cents, @@squared_degrees, @@squared_target_degrees, @@degree_products",
        "SumAccum<INT> @@external_accounts",
        "MinAccum<UINT> @@first_joined_ms, @@first_event_ms",
        "MaxAccum<UINT> @@last_event_ms",
    ],
    PROFILE_HUBS: [
        "SetAccum<VERTEX<Party>> @hub_customers, @crowd_customers, @crowd_enrolled",
        "MapAccum<INT, SumAccum<INT>> @@accounts, @@registered_devices, @@observed_devices, "
        "@@registered_ips, @@observed_ips",
        f"UINT lookback_ms = {LOOKBACK_MS}",
    ],
}
DECLARES = re.compile(r"\w+Accum\b|TYPEDEF |(?:INT|UINT|FLOAT|DOUBLE|STRING|BOOL|DATETIME) ")


def _declarations(name: str) -> list[str]:
    """The statements of a query's body that declare an accumulator, a tuple or a local
    scalar, in order, each with its whitespace made single spaces."""
    code = without_comments(DEFINITIONS[name])
    body = code[code.index("{") + 1 : code.rindex("}")]
    found = [" ".join(part.split()) for part in body.split(";")]
    return [text for text in found if DECLARES.match(text)]


@pytest.mark.parametrize("name", INFRASTRUCTURE)
def test_each_query_declares_exactly_its_accumulators_and_settings(name: str) -> None:
    assert _declarations(name) == DECLARATIONS[name]


# Parameters every query checks alike.


def _refusal(name: str) -> str:
    """The condition of the query's first IF, which refuses invalid parameters."""
    first = next(s for s in statements(DEFINITIONS[name]) if s[0] == "IF")
    text = " ".join(first)
    assert text.endswith('THEN PRINT "invalid_parameters" AS status'), text
    return text[len("IF ") : -len(' THEN PRINT "invalid_parameters" AS status')]


# The status each query refuses with, in order; read_world takes no parameter.
REFUSALS: dict[str, list[str]] = {
    LIST_ACCOUNTS: ["invalid_parameters"],
    READ_LABELS: ["invalid_parameters", "invalid_activity"],
    READ_ANSWER_KEY: ["invalid_parameters"],
    READ_HUBS: ["invalid_parameters"],
    READ_WORLD: [],
    PROFILE_HUBS: ["invalid_parameters"],
}


@pytest.mark.parametrize("name", INFRASTRUCTURE)
def test_every_refusal_returns_at_once(name: str) -> None:
    """A refusal that went on would read with t at or below the lookback, where
    t_ms - lookback_ms wraps in UINT, or print a page after its refusal."""
    found = statements(DEFINITIONS[name])
    refused: list[str] = []
    for index, statement in enumerate(found):
        if "PRINT" not in statement:
            continue
        at = statement.index("PRINT")
        word = statement[at + 1]
        if statement[at + 2 : at + 4] == ["AS", "status"] and word != '"ok"':
            refused.append(word.strip('"'))
            assert statement[0] == "IF", statement
            assert found[index + 1 : index + 3] == [["RETURN"], ["END"]], statement
    assert refused == REFUSALS[name]


@pytest.mark.parametrize("name", AS_OF)
def test_a_query_with_t_refuses_a_t_other_than_a_monday_after_the_lookback(name: str) -> None:
    conditions = _refusal(name).split(" OR ")
    assert "t_ms <= lookback_ms" in conditions
    assert f"t_ms % {WEEK_MS} != {MONDAY_MS}" in conditions
    assert re.search(rf"UINT lookback_ms = {LOOKBACK_MS};", DEFINITIONS[name])


@pytest.mark.parametrize("name", PAGED)
def test_a_paged_query_pages_by_strictly_increasing_id(name: str) -> None:
    conditions = _refusal(name).split(" OR ")
    assert {"page < 1", f"page > {PAGE_LIMIT}"} <= set(conditions)
    target = "Identified" if name == READ_LABELS else "Page"
    select = _select(name, target)
    assert "a.id > after_id" in select.conjuncts
    assert _code(name, target).endswith("ORDER BY a.id ASC LIMIT page")


# The label readers.


def _named_labels(name: str) -> set[str]:
    return set(re.findall(r"\w+", without_comments(DEFINITIONS[name]))) & set(LABEL_COLUMNS)


def test_read_labels_and_read_answer_key_alone_name_label_columns() -> None:
    for name in written_files():
        definition = definitions((GSQL_DIR / QUERY_FILES[name]).read_text())[name]
        named = set(re.findall(r"\w+", without_comments(definition))) & set(LABEL_COLUMNS)
        if name not in (READ_LABELS, READ_ANSWER_KEY):
            assert named == set(), name
    assert _named_labels(READ_LABELS) == {"pu_label", "mule_onset_ms", "mule_last_ms"}
    # The answer key reads the truth and every mule's dates, never pu_label.
    assert _named_labels(READ_ANSWER_KEY) == {
        "is_mule",
        "mule_ring_id",
        "mule_onset_ms",
        "mule_last_ms",
    }


def test_read_labels_reads_dates_only_of_the_accounts_its_pu_label_selection_keeps() -> None:
    definition = DEFINITIONS[READ_LABELS]
    identified = _select(READ_LABELS, "Identified")
    assert [step.types for step in identified.vertices] == [frozenset({"Account"})]
    assert "a.pu_label == 1" in identified.conjuncts
    # Every set read after it descends from it, and every statement naming a date reads one.
    filtered = {"Identified"}
    for statement, _, select in scopes(definition):
        names_dates = {"mule_onset_ms", "mule_last_ms"} & {
            item.rsplit(".", 1)[-1] for item in statement
        }
        if select is not None and statement[:2] != ["Identified", "="]:
            sources = {item for item in statement[statement.index("FROM") + 1 :][:1]}
            assert sources <= filtered, statement
            filtered.add(statement[0])
        if names_dates and statement[0] == "PRINT":
            assert statement[1] in filtered, statement
    assert filtered == {"Identified", "Invalid"}


def test_read_labels_refuses_a_page_with_an_activity_period_out_of_order() -> None:
    invalid = _select(READ_LABELS, "Invalid")
    assert invalid.conjuncts == {"a.mule_onset_ms == 0 OR a.mule_onset_ms > a.mule_last_ms"}
    assert "@@invalid += a.id" in " ".join(invalid.accumulations)
    found = [" ".join(s) for s in statements(DEFINITIONS[READ_LABELS])]
    refusal = found.index(
        'IF Invalid.size ( ) > 0 THEN PRINT "invalid_activity" AS status , @@invalid AS accounts'
    )
    assert found[refusal + 1] == "RETURN"
    assert refusal < next(i for i, s in enumerate(found) if s.startswith('PRINT "ok"'))


def test_read_answer_key_reads_every_internal_account_and_whether_it_has_an_owner() -> None:
    page = _select(READ_ANSWER_KEY, "Page")
    assert page.conjuncts == {"a.id > after_id", "NOT a.is_external"}
    owned = _select(READ_ANSWER_KEY, "Owned")
    assert [s.alias for s in owned.vertices] == ["a", "q"] and _edges(owned) == [OWNER]
    # Any ownership link at all: the answer key has no t.
    assert owned.conjuncts == frozenset()
    assert " ".join(owned.accumulations) == "a.@owned += TRUE"
    assert "OrAccum @owned" in DEFINITIONS[READ_ANSWER_KEY]


# The scored population.


def test_list_accounts_lists_internal_deposit_accounts_open_and_owned_at_t() -> None:
    page = _select(LIST_ACCOUNTS, "Page")
    assert [(s.alias, s.types) for s in page.vertices] == [
        ("a", frozenset({"Account"})),
        ("q", frozenset({"Party"})),
    ]
    assert _edges(page) == [OWNER]
    assert page.conjuncts == {
        "a.id > after_id",
        "NOT a.is_external",
        'a.account_type == "deposit"',
        "a.opened_ms < t_ms",
        "( a.closed_ms == 0 OR a.closed_ms >= t_ms )",
        *_visible("o"),
    }
    assert " ".join(page.accumulations) == "a.@owner_id += q.id"


def test_list_accounts_prints_no_date_of_record() -> None:
    printed = projection_fields(DEFINITIONS[LIST_ACCOUNTS], "Page")
    assert printed == ["account_id", "owner_id"]
    prints = [s for s in statements(DEFINITIONS[LIST_ACCOUNTS]) if s[0] == "PRINT"]
    assert not any(re.search(r"opened_ms|closed_ms|joined_ms", " ".join(s)) for s in prints)


# The hubs and crowds.


def test_an_account_hub_counts_the_owners_of_its_payments_other_accounts() -> None:
    busy = _select(READ_HUBS, "Busy")
    assert busy.conjuncts == {
        "a.opened_ms < t_ms",
        " + ".join(f'a.outdegree ( "{edge}" )' for edge in sorted(SENT_AND_RECEIVED, key=_order))
        + " > hub_threshold",
    }
    counted = _select(READ_HUBS, "Counted")
    assert [s.alias for s in counted.vertices] == ["a", "p", "b", "q"]
    assert _edges(counted) == [SENT_AND_RECEIVED, PAYMENT_ENDS, OWNER]
    # A self-payment has no other account, and an ownerless counterparty is never counted.
    assert counted.conjuncts == {*WINDOW, "b != a", "b.opened_ms < t_ms", *_visible("o")}
    assert " ".join(counted.accumulations) == "a.@hub_customers += q"
    hubs = _select(READ_HUBS, "HubAccounts")
    assert hubs.conjuncts == {"a.@hub_customers.size ( ) > hub_threshold"}
    assert " ".join(hubs.accumulations) == "a.@customers_90d += a.@hub_customers.size ( )"


def _order(edge: str) -> int:
    """The order in which read_hubs' prefilter sums an account's payment edges."""
    return [
        "Account_Initiated_Transaction",
        "Account_Received_Transaction",
        "Account_Sent_Zelle_Transfer",
        "Account_Received_Zelle_Transfer",
    ].index(edge)


def test_a_crowd_counts_enrolled_customers_and_the_owners_of_deposit_senders_seen_on_it() -> None:
    enrolled = _select(READ_HUBS, "Enrolled")
    assert _edges(enrolled) == [frozenset({"Device_Used_By_Party", "IP_Used_By_Party"})]
    assert enrolled.conjuncts == _visible("l")
    assert " ".join(enrolled.accumulations) == "d.@crowd_customers += q , d.@crowd_enrolled += q"
    seen = _select(READ_HUBS, "Seen")
    assert [s.alias for s in seen.vertices] == ["d", "p", "b", "q"]
    # The sender's owner alone: a receiver is not seen on the payment's device or IP.
    assert _edges(seen) == [SEEN_ON, SENDERS, OWNER]
    assert seen.conjuncts == {
        *WINDOW,
        'b.account_type == "deposit"',
        "b.opened_ms < t_ms",
        *_visible("o"),
    }
    assert " ".join(seen.accumulations) == "d.@crowd_customers += q"


def test_a_leaf_is_printed_with_its_measure_observed_only_and_hub() -> None:
    leaves = _select(READ_HUBS, "Leaves")
    assert leaves.conjuncts == {"d.@crowd_customers.size ( ) > crowd_threshold"}
    assert " ".join(leaves.accumulations) == (
        "d.@customers_90d += d.@crowd_customers.size ( ) , "
        "d.@observed_only += d.@crowd_enrolled.size ( ) == 0 , "
        "d.@hub += d.@crowd_customers.size ( ) > hub_threshold"
    )
    assert _select(READ_HUBS, "DeviceLeaves").conjuncts == {'d.type == "Device"'}
    assert _select(READ_HUBS, "IpLeaves").conjuncts == {'d.type == "IP"'}
    for setting in ("crowd_threshold", "hub_threshold", "lookback_ms"):
        assert setting in DEFINITIONS[READ_HUBS]
    assert f"UINT crowd_threshold = {SUBGRAPH.crowd_threshold};" in DEFINITIONS[READ_HUBS]
    assert f"UINT hub_threshold = {SUBGRAPH.hub_threshold};" in DEFINITIONS[READ_HUBS]


def test_profile_hubs_measures_exactly_as_read_hubs_without_the_prefilter() -> None:
    assert _select(PROFILE_HUBS, "Existing").conjuncts == {"a.opened_ms < t_ms"}
    hubs, profile = _code(READ_HUBS, "Counted"), _code(PROFILE_HUBS, "Counted")
    assert profile == hubs.replace("FROM Busy :", "FROM Existing :")
    for target in ("Enrolled", "Seen"):
        assert _code(PROFILE_HUBS, target) == _code(READ_HUBS, target)
    accounts = _select(PROFILE_HUBS, "AccountProfiles")
    assert [s.alias for s in accounts.vertices] == ["a"] and accounts.vertices[0].chosen
    assert " ".join(accounts.accumulations) == "@@accounts += ( a.@hub_customers.size ( ) -> 1 )"
    # One SELECT per type, since the live type check refuses a second d.type test in one IF
    # chain: registered with a visible enrolment link, observed-only otherwise.
    for target, kind, registered, observed in (
        ("DeviceProfiles", "Device", "@@registered_devices", "@@observed_devices"),
        ("IpProfiles", "IP", "@@registered_ips", "@@observed_ips"),
    ):
        nodes = _select(PROFILE_HUBS, target)
        assert nodes.vertices == [Step("d", frozenset({"Device", "IP"}), False)]
        assert nodes.conjuncts == {"d.@crowd_customers.size ( ) > 0", f'd.type == "{kind}"'}
        assert " ".join(nodes.accumulations) == (
            f"IF d.@crowd_enrolled.size ( ) > 0 THEN {registered} += "
            "( d.@crowd_customers.size ( ) -> 1 ) "
            f"ELSE {observed} += ( d.@crowd_customers.size ( ) -> 1 ) END"
        )
    assert "outdegree" not in DEFINITIONS[PROFILE_HUBS]


# The world's identity.


FORWARD = {table.name for table in TABLES if table.kind is not Kind.VERTEX}
LINKS = {table.name for table in TABLES if table.kind is Kind.LINK}
VERTICES = {table.name for table in TABLES if table.kind is Kind.VERTEX}


def test_read_world_counts_every_vertex_type_once_under_its_own_name() -> None:
    counted: list[str] = []
    for statement, _, select in scopes(DEFINITIONS[READ_WORLD]):
        if select is not None and not select.edges and not select.vertices[0].chosen:
            (step,) = select.vertices
            (kind,) = step.types
            text = " ".join(select.accumulations)
            # One count, keyed by the type the SELECT reads: by name, or by its alias's type.
            keys = re.findall(r"@@vertices \+= \( (\S+) -> 1 \)", text)
            assert text.count("@@vertices") == 1, statement
            assert keys in ([f'"{kind}"'], [f"{step.alias}.type"]), statement
            counted.append(kind)
    assert sorted(counted) == sorted(VERTICES)


def test_read_world_steps_along_every_forward_edge_type_once_per_pass() -> None:
    for suffix in ("Ends", "Products"):
        stepped: list[str] = []
        for source in ("Party", "Token", "Transaction", "Transfer"):
            select = _select(READ_WORLD, f"{source}{suffix}")
            assert len(select.edges) == 1 and select.vertices[0].chosen
            stepped += select.edges[0].types
        assert sorted(stepped) == sorted(FORWARD), suffix


def test_read_world_tallies_both_ends_of_each_edge_and_multiplies_them() -> None:
    for source, alias in (("Party", "q"), ("Token", "k"), ("Transaction", "p"), ("Transfer", "p")):
        ends = " ".join(_select(READ_WORLD, f"{source}Ends").accumulations)
        assert ends.startswith(
            f"{alias}.@out_degree += ( e.type -> 1 ) , x.@in_degree += ( e.type -> 1 )"
        ), ends
        products = " ".join(_select(READ_WORLD, f"{source}Products").accumulations)
        assert products == (
            f"@@degree_products += ( e.type -> {alias}.@out_degree.get ( e.type ) * "
            "x.@in_degree.get ( e.type ) )"
        )
    sources = " ".join(_select(READ_WORLD, "SourceDegrees").accumulations)
    assert (
        "FOREACH ( kind , degree ) IN v.@out_degree DO @@edges += ( kind -> degree ) , "
        "@@squared_degrees += ( kind -> degree * degree ) END" == sources
    )
    targets = " ".join(_select(READ_WORLD, "TargetDegrees").accumulations)
    assert targets == (
        "FOREACH ( kind , degree ) IN v.@in_degree DO "
        "@@squared_target_degrees += ( kind -> degree * degree ) END"
    )
    found = {" ".join(s) for s in statements(DEFINITIONS[READ_WORLD])}
    assert "Sources = Parties UNION Tokens UNION Transactions UNION Transfers" in found
    assert "Targets = Accounts UNION Tokens UNION Devices UNION Ips" in found


def test_read_world_sums_every_identity_time_in_whole_seconds() -> None:
    code = " ".join(" ".join(s) for s in statements(DEFINITIONS[READ_WORLD]))
    named = re.findall(r'@@seconds \+= \( "(\w+\.\w+)" -> floor \( (\w)\.(\w+) / 1000 \) \)', code)
    assert {key for key, _, _ in named} | {
        f"{link}.{end}" for link in LINKS for end in ("valid_from_ms", "valid_to_ms")
    } == set(SECONDS)
    assert all(key.endswith(f".{column}") for key, _, column in named)
    # Each link type's start and end, through the steps along the five link types.
    linked: set[str] = set()
    for source in ("Party", "Token"):
        ends = _select(READ_WORLD, f"{source}Ends")
        text = " ".join(ends.accumulations)
        for end in ("valid_from_ms", "valid_to_ms"):
            assert f'@@seconds += ( e.type + ".{end}" -> floor ( e.{end} / 1000 ) )' in text
        linked |= ends.edges[0].types
    assert linked == LINKS
    for source in ("Transaction", "Transfer"):
        assert "@@seconds" not in " ".join(_select(READ_WORLD, f"{source}Ends").accumulations)


def test_read_world_sums_cents_and_bounds_the_data() -> None:
    for target, kind in (("Transactions", "Payment_Transaction"), ("Transfers", "Zelle_Transfer")):
        text = " ".join(_select(READ_WORLD, target).accumulations)
        assert f'@@cents += ( "{kind}" -> floor ( p.amount * 100 + 0.5 ) )' in text
        assert "@@first_event_ms += p.event_ts_ms , @@last_event_ms += p.event_ts_ms" in text
    parties = " ".join(_select(READ_WORLD, "Parties").accumulations)
    assert "@@first_joined_ms += q.joined_ms" in parties
    accounts = " ".join(_select(READ_WORLD, "Accounts").accumulations)
    assert "@@account_types += ( a.account_type -> 1 )" in accounts
    assert "IF a.is_external THEN @@external_accounts += 1 END" in accounts
    transactions = " ".join(_select(READ_WORLD, "Transactions").accumulations)
    assert "@@payment_rails += ( p.payment_rail -> 1 )" in transactions
    declared = DEFINITIONS[READ_WORLD]
    assert "MinAccum<UINT> @@first_joined_ms, @@first_event_ms;" in declared
    assert "MaxAccum<UINT> @@last_event_ms;" in declared


def test_read_world_prints_every_field_of_the_world() -> None:
    assert status_fields(DEFINITIONS[READ_WORLD]) == [field.name for field in fields(World)]
