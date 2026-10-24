"""read_shared_identity, B5, read as text (docs/design.md, "read_shared_identity (B5): the
device and IP graph", and docs/features.md, "What one example's subgraph holds", items 3 and 5
to 9, and the customer, device, IP, enrolment and seen tables).

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclasses its parser reads. Beside those rules, each part
is held here clause by clause, since no reference graph runs it yet: B4's accounts found again
by B4's own rules; the leaves by read_hubs' own crowd rule; the known-start test on every start;
the sharing rule; the caps and their orders; the co-users and the account each brings; the
edges around the scored account's own nodes; and the customer and node rows. gsql-lsp checks
the file against the schema in tests/gsql/test_schema.py, and the live compile test compiles
it.
"""

from __future__ import annotations

from dataclasses import fields
import re

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    Select,
    accumulation_text,
    common_problems,
    dataflow_problems,
    field_rule,
    moved_select,
    normal,
    parameter_rule,
    parameters,
    projection_fields,
    projection_items,
    select_flows,
    select_of,
    source_rule,
    statements,
    status_fields,
    tuples,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    QUERY_FILES,
    READ_HUBS,
    READ_PAYMENT_EVENTS,
    READ_SHARED_IDENTITY,
    REGISTRY,
    VERSIONS,
    read_shared_identity,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_shared_identity import (
    CustomerRow,
    Enrolled,
    IdentityAccount,
    NodeRow,
    OwnerEnrolled,
    Seen,
)
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS


def _definition(name: str) -> str:
    return definitions((GSQL_DIR / QUERY_FILES[name]).read_text())[name]


TEXT = (GSQL_DIR / QUERY_FILES[READ_SHARED_IDENTITY]).read_text()
DEFINITION = definitions(TEXT)[READ_SHARED_IDENTITY]
PAYMENT_EVENTS = _definition(READ_PAYMENT_EVENTS)
HUBS = _definition(READ_HUBS)
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The caps of docs/design.md, "Time": B4's 32, 16 and 8; 8 devices and 8 IPs per customer and
# per account; 48 co-users per scored account. The crowd threshold of "Shared rules", and the
# windows: 30 days for sharing and new, 7 days for new customers.
SCORED = SUBGRAPH.scored_payments
OTHER = SUBGRAPH.counterparty_payments
SIBLINGS = SUBGRAPH.own_accounts
NODES = SUBGRAPH.devices_ips
CO_USERS = SUBGRAPH.co_users
CROWD = SUBGRAPH.crowd_threshold
DAY_MS = 86_400_000
WINDOW_7D_MS = 7 * DAY_MS  # 604,800,000
WINDOW_30D_MS = SUBGRAPH.sharing_days * DAY_MS  # 2,592,000,000


def _source(statement: tuple[str, ...] | list[str]) -> str:
    """The vertex set a SELECT starts from."""
    return statement[statement.index("FROM") + 1]


def _code(definition: str, target: str) -> str:
    """The statement that assigns target, its tokens joined by spaces."""
    return " ".join(select_of(definition, target).statement)


def _shape(select: Select) -> tuple[object, ...]:
    """A SELECT's pattern and WHERE clause: each step's alias and types, and its conjuncts."""
    return (
        [(step.alias, step.types, step.chosen) for step in select.vertices],
        [(edge.alias, edge.types) for edge in select.edges],
        select.conjuncts,
    )


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_SHARED_IDENTITY] in written_files().values()
    assert common_problems(READ_SHARED_IDENTITY, TEXT) == []
    assert version_rule(READ_SHARED_IDENTITY, DEFINITION, VERSIONS[READ_SHARED_IDENTITY]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    """Like B4, since it finds B4's accounts again: an expanded hop-1 account is one that is no
    hub, so the hubs come as the same parameter."""
    built = read_shared_identity.params(T_MS, ["a_01", "a_02"], ["h_01"])
    assert parameter_rule(READ_SHARED_IDENTITY, DEFINITION, built) == []
    assert parameters(DEFINITION) == parameters(PAYMENT_EVENTS)
    assert built["hub_accounts"] == [{"id": "h_01"}]


PRINTED_SETS = {
    "CustomersListed": CustomerRow,
    "DevicesListed": NodeRow,
    "IpsListed": NodeRow,
    "Accounts": IdentityAccount,
}
PRINTED_TUPLES = {"Enrolled": Enrolled, "OwnerEnrolled": OwnerEnrolled, "Seen": Seen}


def test_each_printed_row_has_the_fields_of_its_dataclass() -> None:
    for vertex_set, kind in PRINTED_SETS.items():
        what = f"read_shared_identity {vertex_set}"
        assert field_rule(what, projection_fields(DEFINITION, vertex_set), kind) == []
        assert source_rule(what, projection_items(DEFINITION, vertex_set)) == []
    for name, kind in PRINTED_TUPLES.items():
        assert tuples(DEFINITION)[name] == [field.name for field in fields(kind)], name
    # The status row prints the status and the version alone; then the rows, and per scored
    # account its customers, nodes and co-users' accounts and its three kinds of edge.
    assert status_fields(DEFINITION) == []
    printed = [" ".join(s) for s in statements(DEFINITION) if s[0] == "PRINT"]
    assert [text.split(" [")[0] for text in printed[-10:]] == [
        "PRINT CustomersListed",
        "PRINT DevicesListed",
        "PRINT IpsListed",
        "PRINT Accounts",
        "PRINT @@subgraph_customers AS subgraph_customers",
        "PRINT @@subgraph_nodes AS subgraph_nodes",
        "PRINT @@co_user_accounts AS co_user_accounts",
        "PRINT @@enrolled AS enrolled",
        "PRINT @@owner_enrolled AS owner_enrolled",
        "PRINT @@seen AS seen",
    ]
    assert [text.rsplit(" AS ", 1)[-1] for text in printed[-10:-6]] == [
        "customers",
        "devices",
        "ips",
        "accounts",
    ]


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_SHARED_IDENTITY]
    assert query.params is read_shared_identity.params
    assert query.rows == (CustomerRow, NodeRow, IdentityAccount, Enrolled, OwnerEnrolled, Seen)


# What the query declares, exactly, in order. Every heap has its cap as its capacity and keeps
# the newest, ties by id; every time is UINT and every count INT.
DECLARATIONS = [
    "TYPEDEF TUPLE<INT sec, STRING id, STRING other> Picked",
    "TYPEDEF TUPLE<INT sec, STRING id> Sibling",
    "TYPEDEF TUPLE<UINT ts, STRING id> Ranked",
    "TYPEDEF TUPLE< STRING customer_id, STRING node_id, BOOL since_start_present, "
    "UINT since_start_ms, BOOL since_last_seen_present, UINT since_last_seen_ms > Enrolled",
    "TYPEDEF TUPLE< STRING account_id, STRING node_id, BOOL since_start_present, "
    "UINT since_start_ms, BOOL since_last_seen_present, UINT since_last_seen_ms > OwnerEnrolled",
    "TYPEDEF TUPLE< STRING account_id, STRING node_id, INT uses_90d, "
    "BOOL since_first_use_present, UINT since_first_use_ms, UINT since_last_use_ms > Seen",
    "OrAccum @hub, @expanding",
    f"HeapAccum<Picked>({SCORED}, sec DESC, id ASC) @scored_out, @scored_in",
    f"HeapAccum<Picked>({OTHER}, sec DESC, id ASC) @other_out, @other_in",
    "MaxAccum<UINT> @last_active",
    f"HeapAccum<Sibling>({SIBLINGS}, sec DESC, id ASC) @siblings",
    f"HeapAccum<Ranked>({NODES}, ts DESC, id ASC) @kept_devices, @kept_ips, @kept_seen_devices, "
    "@kept_seen_ips",
    "MapAccum<STRING, MaxAccum<UINT>> @candidate_evidence",
    f"HeapAccum<Ranked>({CO_USERS}, ts DESC, id ASC) @kept_co_users",
    "MaxAccum<STRING> @sender_id, @owner_id, @user_id, @receiver_id",
    "OrAccum @deposit_sender",
    "SetAccum<STRING> @seen_on",
    "SetAccum<VERTEX<Party>> @crowd_customers, @crowd_enrolled, @crowd_reached",
    "OrAccum @leaf",
    "SetAccum<STRING> @seen_accounts",
    "MapAccum<STRING, MinAccum<UINT>> @pair_start",
    "SetAccum<STRING> @device_co_users, @ip_co_users",
    "SumAccum<INT> @accounts_owned, @devices_30d, @ips_30d, @new_devices_30d, @new_ips_30d, "
    "@co_users, @new_co_users_30d, @co_users_both",
    "SumAccum<INT> @customers_30d, @customers_90d, @new_customers_7d, @new_customers_30d, "
    "@accounts_seen_30d, @payments_seen_30d",
    "OrAccum @observed_only",
    "MapAccum<STRING, SetAccum<STRING>> @@subgraph_accounts, @@reached_by, @@expanded_of, "
    "@@customer_accounts",
    "SetAccum<STRING> @@hop1_ids, @@hub_ids",
    "MapAccum<STRING, MaxAccum<STRING>> @@owner_of",
    "MapAccum<STRING, MapAccum<STRING, SumAccum<INT>>> @@uses",
    "MapAccum<STRING, MapAccum<STRING, MinAccum<UINT>>> @@first_use",
    "MapAccum<STRING, MapAccum<STRING, MaxAccum<UINT>>> @@last_use",
    "MapAccum<STRING, SetAccum<STRING>> @@first_use_known, @@node_senders",
    "SetAccum<STRING> @@seen_account_ids",
    "MapAccum<STRING, MapAccum<STRING, MinAccum<UINT>>> @@link_start, @@known_link_start, "
    "@@deposit_first_use, @@share_start",
    "MapAccum<STRING, MapAccum<STRING, MaxAccum<UINT>>> @@last_seen, @@deposit_last_use",
    "MapAccum<STRING, SetAccum<STRING>> @@link_current, @@shares",
    "MapAccum<STRING, MinAccum<UINT>> @@first_own",
    "MapAccum<STRING, SetAccum<STRING>> @@own_seen",
    "MapAccum<STRING, MapAccum<STRING, MinAccum<UINT>>> @@first_unseen",
    "SetAccum<STRING> @@ip_ids, @@leaf_ids, @@shared_ids, @@scored_node_ids",
    "MapAccum<STRING, SetAccum<STRING>> @@candidates, @@node_users",
    "MapAccum<STRING, MapAccum<STRING, MaxAccum<STRING>>> @@node_link_accounts",
    "MapAccum<STRING, SetAccum<STRING>> @@deposit_accounts, @@kept_links, @@kept_seen, "
    "@@open_nodes, @@kept_co_users",
    "SetAccum<STRING> @@co_user_deposit_ids",
    "MapAccum<STRING, MaxAccum<UINT>> @@last_active",
    "MapAccum<STRING, SetAccum<STRING>> @@subgraph_customers, @@subgraph_nodes, "
    "@@co_user_accounts, @@edge_accounts",
    "MapAccum<STRING, SetAccum<Enrolled>> @@enrolled",
    "MapAccum<STRING, SetAccum<OwnerEnrolled>> @@owner_enrolled",
    "MapAccum<STRING, SetAccum<Seen>> @@seen",
    "SetAccum<STRING> @@printed_customers, @@printed_nodes, @@printed_accounts",
    "SetAccum<STRING> @@ids",
    "SumAccum<INT> @@given",
    f"UINT lookback_ms = {LOOKBACK_MS}",
    f"UINT crowd_threshold = {CROWD}",
]
DECLARES = re.compile(r"\w+Accum\b|TYPEDEF |(?:INT|UINT|FLOAT|DOUBLE|STRING|BOOL|DATETIME) ")


def test_it_declares_exactly_its_tuples_accumulators_and_settings() -> None:
    code = without_comments(DEFINITION)
    body = code[code.index("{") + 1 : code.rindex("}")]
    found = [" ".join(part.split()) for part in body.split(";")]
    assert [text for text in found if DECLARES.match(text)] == DECLARATIONS
    assert (SCORED, OTHER, SIBLINGS, NODES, CO_USERS, CROWD) == (32, 16, 8, 8, 48, 8)


# The refusals and the hubs, as B4's.


def test_it_refuses_and_checks_exactly_as_b4_before_any_payment_is_read() -> None:
    """t, then every account, then every hub, each refusal returning at once; the statements
    up to the hubs' marks are B4's, and the mark also lists the hubs, which no edge joins."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    theirs = [" ".join(statement) for statement in statements(PAYMENT_EVENTS)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    their_start = next(at for at, text in enumerate(theirs) if text.startswith("IF "))
    assert found[start : start + 25] == theirs[their_start : their_start + 25]
    assert found[start + 25] == (
        "Marked = SELECT h FROM HubAccounts : h POST-ACCUM h.@hub += TRUE , @@hub_ids += h.id"
    )
    refused = [
        statement[statement.index("PRINT") + 1].strip('"')
        for statement in statements(DEFINITION)
        if "PRINT" in statement
        and statement[statement.index("PRINT") + 2 :][:2] == ["AS", "status"]
    ]
    assert refused == [
        "invalid_parameters",
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
        "ok",
    ]
    assert normal(DEFINITION).count("FROM Requested") == 1


def test_the_hub_parameter_is_the_only_hub_input() -> None:
    """A hop-1 account that is a hub is not expanded, and a hub account other than the scored
    account carries no edge; nothing else reads a hub, and no account hub is counted here."""
    code = normal(DEFINITION)
    assert code.count("hub_accounts") == 2
    assert re.findall(r"\S*@hub\b\S*", code) == ["@hub", "h.@hub", "c.@hub"]
    assert re.findall(r"@@hub_ids\S*", code) == [
        "@@hub_ids",
        "@@hub_ids",
        "@@hub_ids.contains",
        "@@hub_ids.contains",
    ]
    assert not re.search(r"outdegree|hub_threshold", code)


# The order of the SELECTs, pinned whole, and the rule that explains it.
SELECT_ORDER = [
    "Scored",
    "HubAccounts",
    "Marked",
    "ScoredSent",
    "ScoredReceived",
    "ScoredPicked",
    "Hop1",
    "Hop1Sent",
    "Hop1Received",
    "Hop1Picked",
    "Siblings",
    "Active",
    "SiblingsRanked",
    "SiblingsPicked",
    "CustomerAccounts",
    "ExpandingMarked",
    "Owners",
    "ExpandingSent",
    "ExpandingSeen",
    "OwnerLinks",
    "EnrolledCrowd",
    "SeenCrowd",
    "Leaves",
    "LeafIds",
    "ScoredNodes",
    "CandidateNodes",
    "LinkedCandidates",
    "SeenCandidates",
    "RowLinks",
    "RowAccounts",
    "RowSent",
    "RowSeen",
    "SharedReached",
    "UntestedEnrolledCrowd",
    "UntestedSeenCrowd",
    "UntestedLeaves",
    "UntestedLeafIds",
    "Open",
    "OpenReceivers",
    "LinkUsers",
    "SeenUsers",
    "OpenActivity",
    "LinkedAccounts",
    "OtherSent",
    "OtherSeen",
    "RowUnseen",
    "OtherUnseen",
    "Starts",
    "FirstUses",
    "KeptLinks",
    "KeptLinksListed",
    "KeptSeen",
    "KeptSeenListed",
    "ScoredOpen",
    "CoUserEvidence",
    "CoUsersPicked",
    "CoUsersListed",
    "CoUserActive",
    "CoUserAccounts",
    "ScoredCustomers",
    "ScoredNodesKept",
    "ScoredAccounts",
    "Printed",
    "EnrolledEdges",
    "OwnerEnrolledEdges",
    "SeenEdges",
    "Pairs",
    "CustomerRows",
    "NodeRows",
    "Accounts",
]
SETS = [
    'Requested = to_vertex_set ( @@ids , "Account" )',
    "Hubs = { hub_accounts }",
    'Hop1Listed = to_vertex_set ( @@hop1_ids , "Account" )',
    "Expanding = Scored UNION Hop1",
    "Discovered = ExpandingSeen UNION OwnerLinks",
    "Candidates = LinkedCandidates UNION SeenCandidates",
    "RowCustomers = Owners UNION Candidates",
    "NewRows = Candidates MINUS Owners",
    "Reached = RowLinks UNION RowSeen",
    "Untested = SharedReached MINUS Discovered",
    "Tested = Discovered UNION Untested",
    "Users = LinkUsers UNION SeenUsers",
    "OtherUsers = Users MINUS RowCustomers",
    "AllUsers = RowCustomers UNION Users",
    'SeenAccounts = to_vertex_set ( @@seen_account_ids , "Account" )',
    'CoUserDeposits = to_vertex_set ( @@co_user_deposit_ids , "Account" )',
    'CustomersListed = to_vertex_set ( @@printed_customers , "Party" )',
    'DevicesListed = to_vertex_set ( @@printed_nodes , "Device" )',
    'IpsListed = to_vertex_set ( @@printed_nodes , "IP" )',
    "NodesListed = DevicesListed UNION IpsListed",
    'AccountsListed = to_vertex_set ( @@printed_accounts , "Account" )',
]


def test_the_selects_and_sets_run_in_the_order_the_parts_need() -> None:
    assert [target for target, _, _ in select_flows(DEFINITION)] == SELECT_ORDER
    assert [
        " ".join(statement)
        for statement in statements(DEFINITION)
        if statement[1:2] == ["="] and "SELECT" not in statement
    ] == SETS


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    assert dataflow_problems(DEFINITION) == []
    reads = {target: read for target, _, read in select_flows(DEFINITION) if read}
    assert reads == {
        "ScoredPicked": {"scored_out", "scored_in"},
        "Hop1Picked": {"other_out", "other_in"},
        "SiblingsRanked": {"last_active"},
        "SiblingsPicked": {"siblings"},
        "ExpandingSeen": {"sender_id"},
        "Leaves": {"crowd_customers"},
        "RowSeen": {"owner_id", "deposit_sender"},
        "UntestedLeaves": {"crowd_reached"},
        "OpenActivity": {"expanding"},
        "OtherSeen": {"user_id"},
        "RowUnseen": {"owner_id", "seen_on"},
        "OtherUnseen": {"user_id", "seen_on"},
        "KeptLinksListed": {"kept_devices", "kept_ips"},
        "KeptSeenListed": {"kept_seen_devices", "kept_seen_ips"},
        "CoUsersPicked": {"candidate_evidence"},
        "CoUsersListed": {"kept_co_users"},
        "CustomerRows": {"pair_start", "device_co_users", "ip_co_users"},
        "NodeRows": {"crowd_customers", "crowd_enrolled", "seen_accounts"},
    }


@pytest.mark.parametrize(
    ("target", "after", "problem"),
    [
        ("ExpandingSent", "ExpandingSeen", "ExpandingSeen reads @sender_id, written by"),
        ("SeenCrowd", "Leaves", "Leaves reads @crowd_customers, written by EnrolledCrowd, Seen"),
        ("RowSent", "RowSeen", "RowSeen reads @owner_id, written by RowSent"),
        ("OtherSeen", "OtherUnseen", "OtherUnseen reads @seen_on, written by RowSeen, OtherSeen"),
        ("ExpandingMarked", "OpenActivity", "OpenActivity reads @expanding, written by"),
        ("CoUserEvidence", "CoUsersPicked", "CoUsersPicked reads @candidate_evidence"),
        ("Pairs", "CustomerRows", "CustomerRows reads @device_co_users, written by Pairs"),
        ("OpenActivity", "NodeRows", "NodeRows reads @seen_accounts, written by OpenActivity"),
    ],
)
def test_a_select_moved_before_the_one_it_reads_from_breaks_the_rule(
    target: str, after: str, problem: str
) -> None:
    moved = moved_select(DEFINITION, target, after)
    assert moved != DEFINITION
    assert any(problem in found for found in dataflow_problems(moved))


# B4's accounts, found again by B4's rules.


@pytest.mark.parametrize(
    "target",
    ["ScoredSent", "ScoredReceived", "Hop1Sent", "Hop1Received"],
)
def test_the_capped_payments_are_b4s_without_the_kind_of_payment(target: str) -> None:
    """The heaps keep B4's payments: same pattern, bounds, order and caps; only whether a
    payment is a Zelle transfer, which this query never reads, is left out."""
    assert _code(DEFINITION, target) == _code(PAYMENT_EVENTS, target).replace(
        ' , p.type == "Zelle_Transfer"', ""
    )


@pytest.mark.parametrize("target", ["Hop1", "Siblings", "Active", "SiblingsRanked"])
def test_the_expanded_accounts_and_the_customers_other_accounts_are_b4s(target: str) -> None:
    assert _code(DEFINITION, target) == _code(PAYMENT_EVENTS, target)


def test_every_subgraph_holds_b4s_accounts() -> None:
    """The scored account, its hop-1 accounts, their hop-2 accounts if they are expanded, and
    the other accounts of its customer; and, of those, the ones its customer owns, through
    B4's own ownership test. Each expanded hop-1 account is listed per scored account."""
    assert _shape(select_of(DEFINITION, "CustomerAccounts")) == _shape(
        select_of(PAYMENT_EVENTS, "CustomerAccounts")
    )
    assert accumulation_text(DEFINITION, "CustomerAccounts") == normal(
        """IF @@subgraph_accounts.get(a.id).contains(b.id) THEN
            @@customer_accounts += (a.id -> b.id)
        END"""
    )
    picked = """@@subgraph_accounts += (a.id -> x.other),
        @@hop1_ids += x.other,
        @@reached_by += (x.other -> a.id)"""
    assert accumulation_text(DEFINITION, "ScoredPicked") == normal(
        f"""@@subgraph_accounts += (a.id -> a.id),
        FOREACH x IN a.@scored_out DO {picked} END,
        FOREACH x IN a.@scored_in DO {picked} END"""
    )
    hop2 = "FOREACH i IN @@reached_by.get(c.id) DO @@subgraph_accounts += (i -> x.other) END"
    assert accumulation_text(DEFINITION, "Hop1Picked") == normal(
        f"""FOREACH i IN @@reached_by.get(c.id) DO @@expanded_of += (i -> c.id) END,
        FOREACH x IN c.@other_out DO {hop2} END,
        FOREACH x IN c.@other_in DO {hop2} END"""
    )
    assert accumulation_text(DEFINITION, "SiblingsPicked") == normal(
        "FOREACH x IN a.@siblings DO @@subgraph_accounts += (a.id -> x.id) END"
    )


VISIBLE_O = {
    "o.valid_from_ms < t_ms",
    "( o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms )",
}
VISIBLE_W = {
    "w.valid_from_ms < t_ms",
    "( w.valid_to_ms == 0 OR w.valid_to_ms > t_ms - lookback_ms )",
}
VISIBLE_L = {
    "l.valid_from_ms < t_ms",
    "( l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms )",
}
WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms"}
PAYMENTS = frozenset({"Payment_Transaction", "Zelle_Transfer"})
NODE_TYPES = frozenset({"Device", "IP"})
SENT_FROM = frozenset({"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"})
SENT_TO = frozenset({"Transaction_To_Account", "Transfer_To_Account"})
FROM = frozenset({"Transaction_From_Account", "Transfer_From_Account"})
USED = frozenset(
    {"Transaction_Used_Device", "Transaction_Used_IP", "Transfer_Used_Device", "Transfer_Used_IP"}
)
USED_BY = frozenset(
    {
        "Device_Used_By_Transaction",
        "IP_Observed_In_Transaction",
        "Device_Used_By_Transfer",
        "IP_Used_By_Transfer",
    }
)
LINKS = frozenset({"Party_Uses_Device", "Party_Uses_IP"})
LINKED_BY = frozenset({"Device_Used_By_Party", "IP_Used_By_Party"})


def test_the_expanded_accounts_and_their_customers() -> None:
    """The expanded accounts are the scored ones and the hop-1 accounts that are no hub; each
    has its customer through a visible ownership link, or none."""
    owners = select_of(DEFINITION, "Owners")
    assert [(step.alias, step.chosen) for step in owners.vertices] == [("a", True), ("q", False)]
    assert owners.conjuncts == VISIBLE_O
    assert accumulation_text(DEFINITION, "Owners") == "@@owner_of += ( a.id -> q.id )"
    assert accumulation_text(DEFINITION, "ExpandingMarked") == "a.@expanding += TRUE"


# The nodes and the leaves.


def test_the_expanded_accounts_payments_give_their_seen_nodes() -> None:
    """Each expanded account's payments sent in the lookback, but self-payments, and per node
    seen on them: how many, the first and the last. By hand: a_01 sent p_01 on 2024-03-01 and
    p_02 on 2024-03-20 from phone d_01, and p_03 to itself from d_02: d_01 has 2 uses, first
    2024-03-01, last 2024-03-20, and d_02 is no seen node of a_01."""
    sent = select_of(DEFINITION, "ExpandingSent")
    assert sent.selected == "p"
    assert [(step.alias, step.types) for step in sent.vertices] == [
        ("a", frozenset({"Account"})),
        ("p", PAYMENTS),
        ("b", frozenset({"Account"})),
    ]
    assert [edge.types for edge in sent.edges] == [SENT_FROM, SENT_TO]
    assert sent.conjuncts == WINDOW | {"b != a"}
    assert accumulation_text(DEFINITION, "ExpandingSent") == "p.@sender_id += a.id"
    seen = select_of(DEFINITION, "ExpandingSeen")
    assert [(step.alias, step.types, step.chosen) for step in seen.vertices] == [
        ("p", PAYMENTS, True),
        ("n", NODE_TYPES, False),
    ]
    assert [edge.types for edge in seen.edges] == [USED]
    assert not seen.conjuncts
    assert accumulation_text(DEFINITION, "ExpandingSeen") == normal(
        """@@uses += (p.@sender_id -> (n.id -> 1)),
        @@first_use += (p.@sender_id -> (n.id -> p.event_ts_ms)),
        @@last_use += (p.@sender_id -> (n.id -> p.event_ts_ms)),
        @@node_senders += (n.id -> p.@sender_id),
        @@seen_account_ids += p.@sender_id,
        IF n.type == "IP" THEN @@ip_ids += n.id END"""
    )


LINK_FACTS = f"""@@link_start += (q.id -> (n.id -> l.valid_from_ms)),
    IF l.valid_from_ms < t_ms
      AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - {WINDOW_30D_MS}) THEN
        @@link_current += (q.id -> n.id),
        @@shares += (q.id -> n.id),
        @@shared_ids += n.id
    END,
    IF n.type == "IP" THEN @@ip_ids += n.id END"""


@pytest.mark.parametrize(("target", "source"), [("OwnerLinks", "Owners"), ("RowLinks", "NewRows")])
def test_a_customers_visible_links_give_its_nodes(target: str, source: str) -> None:
    """Every visible link, its start, and whether it is held in the last 30 days, the current
    test, which also shares the node; the links of one node are one link with the earliest
    start. By hand: links to d_01 from 2023-12-01 to 2024-01-10 and from 2024-02-20, open,
    start on 2023-12-01 and are current; a link ended on 2024-03-01, 31 days before t, is
    visible but not current."""
    select = select_of(DEFINITION, target)
    assert _source(select.statement) == source and select.selected == "n"
    assert [(step.alias, step.types) for step in select.vertices] == [
        ("q", frozenset({"Party"})),
        ("n", NODE_TYPES),
    ]
    assert [(edge.alias, edge.types) for edge in select.edges] == [("l", LINKS)]
    assert select.conjuncts == VISIBLE_L
    assert accumulation_text(DEFINITION, target) == normal(LINK_FACTS)
    assert "MapAccum<STRING, MapAccum<STRING, MinAccum<UINT>>> @@link_start" in " ".join(
        DECLARATIONS
    )


@pytest.mark.parametrize(
    ("enrolled", "seen", "source", "crowd"),
    [
        ("EnrolledCrowd", "SeenCrowd", "Discovered", "@crowd_customers"),
        ("UntestedEnrolledCrowd", "UntestedSeenCrowd", "Untested", "@crowd_reached"),
    ],
)
def test_the_crowd_count_is_read_hubs_own(
    enrolled: str, seen: str, source: str, crowd: str
) -> None:
    """A node's customers are counted exactly as read_hubs counts them, so the inline leaf
    flag equals read_hubs' crowd flag; payments are read only for a node that 8 enrolled
    customers or fewer do not already make a leaf, which changes the cost and never the
    flag."""
    theirs_enrolled = _code(HUBS, "Enrolled").replace("FROM Nodes", f"FROM {source}")
    theirs_seen = _code(HUBS, "Seen").replace("FROM Nodes", f"FROM {source}")
    if crowd != "@crowd_customers":
        theirs_enrolled = theirs_enrolled.replace(
            "d.@crowd_customers += q , d.@crowd_enrolled += q", f"d.{crowd} += q"
        )
        theirs_seen = theirs_seen.replace("d.@crowd_customers", f"d.{crowd}")
    assert _code(DEFINITION, enrolled) == theirs_enrolled.replace("Enrolled =", f"{enrolled} =")
    assert _code(DEFINITION, seen) == theirs_seen.replace("Seen =", f"{seen} =").replace(
        "WHERE p.event_ts_ms", f"WHERE d.{crowd}.size ( ) <= crowd_threshold AND p.event_ts_ms"
    )


def test_a_leaf_is_a_node_with_more_customers_than_the_crowd_threshold() -> None:
    for leaves, source, crowd in (
        ("Leaves", "Discovered", "@crowd_customers"),
        ("UntestedLeaves", "Untested", "@crowd_reached"),
    ):
        assert _code(DEFINITION, leaves) == (
            f"{leaves} = SELECT d FROM {source} : d POST-ACCUM d.@leaf += "
            f"d.{crowd}.size ( ) > crowd_threshold"
        )
    for ids, source in (("LeafIds", "Discovered"), ("UntestedLeafIds", "Untested")):
        assert _code(DEFINITION, ids) == (
            f"{ids} = SELECT d FROM {source} : d WHERE d.@leaf POST-ACCUM @@leaf_ids += d.id"
        )
    # The nodes reached first through a customer of a row are those it shares that were not
    # tested yet; every node tested that is no leaf is open.
    assert select_of(DEFINITION, "SharedReached").conjuncts == {"@@shared_ids.contains ( n.id )"}
    assert select_of(DEFINITION, "Open").conjuncts == {"NOT d.@leaf"}
    assert _source(select_of(DEFINITION, "Open").statement) == "Tested"


# The co-user candidates.


def test_each_scored_accounts_nodes_before_the_caps_and_their_candidates() -> None:
    """A scored account's nodes are its customer's links and the nodes seen on its own
    payments; on those that are no leaf, the candidates are the customers of the crowd rule,
    read_hubs' measure, whose deposit account's payment may be one to itself."""
    assert accumulation_text(DEFINITION, "ScoredNodes") == normal(
        """FOREACH (n, last_ms) IN @@last_use.get(a.id) DO @@scored_node_ids += n END,
        FOREACH (n, start_ms) IN @@link_start.get(@@owner_of.get(a.id)) DO
            @@scored_node_ids += n
        END"""
    )
    assert select_of(DEFINITION, "CandidateNodes").conjuncts == {
        "NOT d.@leaf",
        "@@scored_node_ids.contains ( d.id )",
    }
    linked = select_of(DEFINITION, "LinkedCandidates")
    assert [edge.types for edge in linked.edges] == [LINKED_BY]
    assert linked.conjuncts == VISIBLE_L | {"NOT d.@leaf"}
    seen = select_of(DEFINITION, "SeenCandidates")
    assert _shape(seen)[1] == _shape(select_of(HUBS, "Seen"))[1]
    assert seen.conjuncts == select_of(HUBS, "Seen").conjuncts | {"NOT d.@leaf"}
    for target in ("LinkedCandidates", "SeenCandidates"):
        assert _source(select_of(DEFINITION, target).statement) == "CandidateNodes"
        assert accumulation_text(DEFINITION, target) == "@@candidates += ( d.id -> q.id )"


# The customers of a row, and every user's own payments for the known-start test.


def test_a_customer_of_a_rows_accounts_and_deposit_accounts() -> None:
    """Every account that exists at t with a visible ownership link, open or closed, counts;
    the deposit ones may be brought by a co-user."""
    select = select_of(DEFINITION, "RowAccounts")
    assert _source(select.statement) == "RowCustomers"
    assert select.conjuncts == VISIBLE_W | {"b.opened_ms < t_ms"}
    assert accumulation_text(DEFINITION, "RowAccounts") == normal(
        """q.@accounts_owned += 1,
        IF b.account_type == "deposit" THEN @@deposit_accounts += (q.id -> b.id) END"""
    )


OWN_SENT = VISIBLE_W | WINDOW | {"b.opened_ms < t_ms", "r != b"}


def test_a_customers_own_payments_but_self_payments_with_the_nodes_each_was_seen_on() -> None:
    """Every payment sent in the lookback by an account the customer owns through a visible
    ownership link, but self-payments: its earliest, and per node seen on it the latest from
    any such account, and the first and latest from a deposit account, which also shares it
    when in the last 30 days. By hand: a customer whose savings account paid its card on
    2024-03-25 from d_01 and whose deposit account paid a friend on 2024-02-01 from d_01 has
    its earliest payment on 2024-02-01, its latest on d_01 on 2024-03-25, and its first and
    latest from a deposit account on d_01 on 2024-02-01, more than 30 days before t, so it
    does not share d_01 through payments."""
    for target, source in (("RowSent", "RowCustomers"), ("OtherSent", "OtherUsers")):
        select = select_of(DEFINITION, target)
        assert _source(select.statement) == source and select.selected == "p"
        assert [step.alias for step in select.vertices] == ["q", "b", "p", "r"]
        assert [edge.types for edge in select.edges] == [
            frozenset({"Party_Owns_Account"}),
            SENT_FROM,
            SENT_TO,
        ]
        assert select.conjuncts == OWN_SENT
    assert accumulation_text(DEFINITION, "RowSent") == normal(
        """p.@owner_id += q.id,
        p.@deposit_sender += b.account_type == "deposit",
        @@first_own += (q.id -> p.event_ts_ms)"""
    )
    assert accumulation_text(DEFINITION, "OtherSent") == normal(
        "p.@user_id += q.id, @@first_own += (q.id -> p.event_ts_ms)"
    )
    assert accumulation_text(DEFINITION, "RowSeen") == normal(
        f"""p.@seen_on += n.id,
        @@own_seen += (p.@owner_id -> n.id),
        @@last_seen += (p.@owner_id -> (n.id -> p.event_ts_ms)),
        IF p.@deposit_sender THEN
            @@deposit_first_use += (p.@owner_id -> (n.id -> p.event_ts_ms)),
            @@deposit_last_use += (p.@owner_id -> (n.id -> p.event_ts_ms)),
            IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
                @@shares += (p.@owner_id -> n.id),
                @@shared_ids += n.id
            END
        END,
        IF n.type == "IP" THEN @@ip_ids += n.id END"""
    )
    assert accumulation_text(DEFINITION, "OtherSeen") == normal(
        "p.@seen_on += n.id, @@own_seen += (p.@user_id -> n.id)"
    )
    for target, source in (("RowSeen", "RowSent"), ("OtherSeen", "OtherSent")):
        select = select_of(DEFINITION, target)
        assert _source(select.statement) == source
        assert [edge.types for edge in select.edges] == [USED]
        assert not select.conjuncts


def test_each_customers_earliest_payment_not_seen_on_each_node() -> None:
    """For each node its payments were seen on, a customer's earliest own payment not seen on
    it; for any other node, its earliest own payment. By hand: payments on 2024-02-01 from
    d_01 and on 2024-02-10 from d_02 give d_01 2024-02-10 and d_02 2024-02-01."""
    for target, owner in (("RowUnseen", "p.@owner_id"), ("OtherUnseen", "p.@user_id")):
        assert accumulation_text(DEFINITION, target) == normal(
            f"""FOREACH n IN @@own_seen.get({owner}) DO
                IF NOT p.@seen_on.contains(n) THEN
                    @@first_unseen += ({owner} -> (n -> p.event_ts_ms))
                END
            END"""
        )


# The open nodes: their users, activity and accounts.


def test_an_open_nodes_users_through_links_and_payments_but_self_payments() -> None:
    """Each payment seen on an open node knows its receiver, so a payment of an account to
    itself is read nowhere. A user through a visible link has the link's facts; a user through
    a payment from an account it owns has its latest there, and from a deposit account its
    first, latest and recent ones, and is one of the node's users."""
    receivers = select_of(DEFINITION, "OpenReceivers")
    assert [edge.types for edge in receivers.edges] == [USED_BY, SENT_TO]
    assert receivers.conjuncts == WINDOW | {"NOT d.@leaf"}
    assert accumulation_text(DEFINITION, "OpenReceivers") == "p.@receiver_id += r.id"
    links = select_of(DEFINITION, "LinkUsers")
    assert links.conjuncts == VISIBLE_L | {"NOT d.@leaf"}
    assert accumulation_text(DEFINITION, "LinkUsers") == normal(
        f"""@@link_start += (q.id -> (d.id -> l.valid_from_ms)),
        IF l.valid_from_ms < t_ms
          AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - {WINDOW_30D_MS}) THEN
            @@link_current += (q.id -> d.id),
            @@shares += (q.id -> d.id)
        END,
        @@node_users += (d.id -> q.id)"""
    )
    seen = select_of(DEFINITION, "SeenUsers")
    assert [edge.types for edge in seen.edges] == [
        USED_BY,
        FROM,
        frozenset({"Account_Owned_By_Party"}),
    ]
    assert seen.conjuncts == WINDOW | VISIBLE_O | {
        "NOT d.@leaf",
        "b.opened_ms < t_ms",
        "b.id != p.@receiver_id",
    }
    assert accumulation_text(DEFINITION, "SeenUsers") == normal(
        f"""@@last_seen += (q.id -> (d.id -> p.event_ts_ms)),
        IF b.account_type == "deposit" THEN
            @@deposit_first_use += (q.id -> (d.id -> p.event_ts_ms)),
            @@deposit_last_use += (q.id -> (d.id -> p.event_ts_ms)),
            @@node_users += (d.id -> q.id),
            IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN @@shares += (q.id -> d.id) END
        END"""
    )


def test_an_open_nodes_activity_and_the_accounts_seen_on_a_scored_accounts_node() -> None:
    """Every payment seen on it in the last 30 days, but self-payments, and the accounts that
    sent them, whether or not they have an owner; on a scored account's node, each account
    seen on it in the lookback that is not expanded, whose own payments give an expanded
    account's, with its uses as there."""
    select = select_of(DEFINITION, "OpenActivity")
    assert [edge.types for edge in select.edges] == [USED_BY, FROM]
    assert select.conjuncts == WINDOW | {"NOT d.@leaf", "b.id != p.@receiver_id"}
    assert accumulation_text(DEFINITION, "OpenActivity") == normal(
        f"""IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
            d.@seen_accounts += b.id,
            d.@payments_seen_30d += 1
        END,
        IF NOT b.@expanding AND @@scored_node_ids.contains(d.id) THEN
            @@uses += (b.id -> (d.id -> 1)),
            @@first_use += (b.id -> (d.id -> p.event_ts_ms)),
            @@last_use += (b.id -> (d.id -> p.event_ts_ms)),
            @@node_senders += (d.id -> b.id),
            @@seen_account_ids += b.id
        END"""
    )
    linked = select_of(DEFINITION, "LinkedAccounts")
    assert _source(linked.statement) == "CandidateNodes"
    assert [edge.types for edge in linked.edges] == [
        LINKED_BY,
        frozenset({"Party_Owns_Account"}),
    ]
    assert linked.conjuncts == VISIBLE_L | VISIBLE_W | {"NOT d.@leaf", "b.opened_ms < t_ms"}
    assert accumulation_text(DEFINITION, "LinkedAccounts") == normal(
        "@@node_link_accounts += (d.id -> (b.id -> q.id))"
    )


# The known-start test, on every start.


def _known(start: str, joined: str, customer: str) -> str:
    """The known-start test of a start of a customer on node n."""
    return f"""IF @@own_seen.get({customer}).contains(n) THEN
        found = @@first_unseen.get({customer}).containsKey(n),
        unseen_ms = @@first_unseen.get({customer}).get(n)
    ELSE
        found = @@first_own.containsKey({customer}),
        unseen_ms = @@first_own.get({customer})
    END,
    IF {start} > t_ms - lookback_ms AND {start} > {joined}
      AND found AND unseen_ms < {start} THEN"""


def test_a_link_start_or_a_first_use_counts_only_when_it_is_known() -> None:
    """A start is known after the lookback's start, after the customer joined, and after a
    payment of its own not seen on the node; an unknown start counts as t - 90 days in a
    sharing start and is no since_start, no new link and no since_first_use. By hand, on
    t = 2024-04-01: a customer that joined on 2024-03-10 and first paid from its new phone
    d_01 that evening has a link from 2024-03-10 21:00 that is not known (no earlier own
    payment), so it shares d_01 from 2024-01-02, t - 90 days; one that paid from d_02 on
    2024-02-01 and enrolled d_01 on 2024-03-20 has a known start there, 12 days before t."""
    assert accumulation_text(DEFINITION, "Starts") == normal(
        f"""UINT unseen_ms = 0, BOOL found = FALSE,
        FOREACH (n, start_ms) IN @@link_start.get(q.id) DO
            {_known("start_ms", "q.joined_ms", "q.id")}
                @@known_link_start += (q.id -> (n -> start_ms)),
                @@share_start += (q.id -> (n -> start_ms))
            ELSE
                @@share_start += (q.id -> (n -> t_ms - lookback_ms))
            END
        END,
        FOREACH (n, start_ms) IN @@deposit_first_use.get(q.id) DO
            {_known("start_ms", "q.joined_ms", "q.id")}
                @@share_start += (q.id -> (n -> start_ms))
            ELSE
                @@share_start += (q.id -> (n -> t_ms - lookback_ms))
            END
        END"""
    )
    assert _source(select_of(DEFINITION, "Starts").statement) == "AllUsers"
    first = select_of(DEFINITION, "FirstUses")
    assert [(step.alias, step.types) for step in first.vertices] == [
        ("a", frozenset({"Account"})),
        ("q", frozenset({"Party"})),
    ]
    assert first.conjuncts == VISIBLE_O | {"a.opened_ms < t_ms"}
    assert accumulation_text(DEFINITION, "FirstUses") == normal(
        f"""UINT unseen_ms = 0, BOOL found = FALSE,
        FOREACH (n, start_ms) IN @@first_use.get(a.id) DO
            {_known("start_ms", "q.joined_ms", "q.id")}
                @@first_use_known += (a.id -> n)
            END
        END"""
    )
    # joined_ms is read in these three tests alone.
    assert len(re.findall(r"\w+\.joined_ms", without_comments(DEFINITION))) == 3


# The caps.


def test_each_customer_keeps_its_newest_8_links_of_each_kind() -> None:
    """By the later of the link's known start and the latest payment seen on the node from an
    account the customer owns; a link with neither has 0 and comes last; ties by node id. By
    hand: a link known from 2024-03-20 to d_01 and an old one to d_02 last used on 2024-03-25
    keep d_02 first; an old link never used comes after both."""
    assert accumulation_text(DEFINITION, "KeptLinks") == normal(
        """UINT evidence = 0,
        FOREACH (n, start_ms) IN @@link_start.get(q.id) DO
            evidence = 0,
            IF @@known_link_start.get(q.id).containsKey(n) THEN
                evidence = @@known_link_start.get(q.id).get(n)
            END,
            IF @@last_seen.get(q.id).containsKey(n)
              AND @@last_seen.get(q.id).get(n) > evidence THEN
                evidence = @@last_seen.get(q.id).get(n)
            END,
            IF @@ip_ids.contains(n) THEN
                q.@kept_ips += Ranked(evidence, n)
            ELSE
                q.@kept_devices += Ranked(evidence, n)
            END
        END"""
    )
    assert accumulation_text(DEFINITION, "KeptLinksListed") == normal(
        """FOREACH x IN q.@kept_devices DO @@kept_links += (q.id -> x.id) END,
        FOREACH x IN q.@kept_ips DO @@kept_links += (q.id -> x.id) END"""
    )
    assert _source(select_of(DEFINITION, "KeptLinks").statement) == "Owners"


def test_each_expanded_account_keeps_its_8_latest_seen_nodes_of_each_kind() -> None:
    assert accumulation_text(DEFINITION, "KeptSeen") == normal(
        """FOREACH (n, last_ms) IN @@last_use.get(a.id) DO
            IF @@ip_ids.contains(n) THEN
                a.@kept_seen_ips += Ranked(last_ms, n)
            ELSE
                a.@kept_seen_devices += Ranked(last_ms, n)
            END
        END"""
    )
    assert accumulation_text(DEFINITION, "KeptSeenListed") == normal(
        """FOREACH x IN a.@kept_seen_devices DO @@kept_seen += (a.id -> x.id) END,
        FOREACH x IN a.@kept_seen_ips DO @@kept_seen += (a.id -> x.id) END"""
    )
    assert _source(select_of(DEFINITION, "KeptSeen").statement) == "Expanding"


def test_the_newest_48_co_users_by_their_evidence_on_the_scored_accounts_nodes() -> None:
    """On each kept node of the scored account that is no leaf, every candidate but C_s, by
    the later of its known link start there and its latest use there from a deposit account,
    t - 90 days with neither, the newest over the nodes; then the newest 48, ties by customer
    id."""
    assert accumulation_text(DEFINITION, "ScoredOpen") == normal(
        """FOREACH n IN @@kept_links.get(@@owner_of.get(a.id)) DO
            IF NOT @@leaf_ids.contains(n) THEN @@open_nodes += (a.id -> n) END
        END,
        FOREACH n IN @@kept_seen.get(a.id) DO
            IF NOT @@leaf_ids.contains(n) THEN @@open_nodes += (a.id -> n) END
        END"""
    )
    assert accumulation_text(DEFINITION, "CoUserEvidence") == normal(
        """UINT evidence = 0,
        FOREACH n IN @@open_nodes.get(a.id) DO
            FOREACH c IN @@candidates.get(n) DO
                IF c != @@owner_of.get(a.id) THEN
                    evidence = t_ms - lookback_ms,
                    IF @@known_link_start.get(c).containsKey(n)
                      AND @@known_link_start.get(c).get(n) > evidence THEN
                        evidence = @@known_link_start.get(c).get(n)
                    END,
                    IF @@deposit_last_use.get(c).containsKey(n)
                      AND @@deposit_last_use.get(c).get(n) > evidence THEN
                        evidence = @@deposit_last_use.get(c).get(n)
                    END,
                    a.@candidate_evidence += (c -> evidence)
                END
            END
        END"""
    )
    assert accumulation_text(DEFINITION, "CoUsersPicked") == normal(
        """FOREACH (c, evidence) IN a.@candidate_evidence DO
            a.@kept_co_users += Ranked(evidence, c)
        END"""
    )
    assert accumulation_text(DEFINITION, "CoUsersListed") == normal(
        """FOREACH x IN a.@kept_co_users DO
            @@kept_co_users += (a.id -> x.id),
            FOREACH b IN @@deposit_accounts.get(x.id) DO @@co_user_deposit_ids += b END
        END"""
    )


def test_each_co_user_brings_the_deposit_account_seen_last_on_the_nodes() -> None:
    """Else the one with the latest payment in the lookback, either way but self-payments,
    else one with none; ties by account id. By hand: deposit accounts b_01 seen on d_01 on
    2024-03-10 and b_02 seen there on 2024-03-28 bring b_02; neither seen, b_01 last paid on
    2024-03-30 and b_02 on 2024-02-01 bring b_01; neither paid, b_01 before b_02 by id."""
    active = select_of(DEFINITION, "CoUserActive")
    assert _shape(active)[1:] == _shape(select_of(PAYMENT_EVENTS, "Active"))[1:]
    assert accumulation_text(DEFINITION, "CoUserActive") == normal(
        "@@last_active += (b.id -> p.event_ts_ms)"
    )
    assert accumulation_text(DEFINITION, "CoUserAccounts") == normal(
        """STRING best = "", UINT best_seen = 0, UINT best_active = 0, UINT seen_ms = 0,
        UINT active_ms = 0,
        FOREACH c IN @@kept_co_users.get(a.id) DO
            best = "", best_seen = 0, best_active = 0,
            FOREACH b IN @@deposit_accounts.get(c) DO
                seen_ms = 0, active_ms = 0,
                FOREACH n IN @@open_nodes.get(a.id) DO
                    IF @@last_use.get(b).containsKey(n) AND @@last_use.get(b).get(n) > seen_ms THEN
                        seen_ms = @@last_use.get(b).get(n)
                    END
                END,
                IF @@last_active.containsKey(b) THEN active_ms = @@last_active.get(b) END,
                IF best == "" OR seen_ms > best_seen
                  OR (seen_ms == best_seen AND seen_ms == 0 AND active_ms > best_active)
                  OR (seen_ms == best_seen AND (seen_ms > 0 OR active_ms == best_active)
                    AND b < best) THEN
                    best = b, best_seen = seen_ms, best_active = active_ms
                END
            END,
            IF best != "" THEN
                @@co_user_accounts += (a.id -> best),
                @@printed_accounts += best
            END
        END"""
    )


# Each subgraph's customers, nodes and accounts.


def test_each_subgraphs_customers_nodes_and_accounts_that_may_carry_an_edge() -> None:
    """Its customers: C_s, each expanded hop-1 account's and each co-user with a visible link
    to one of the scored account's open nodes. Its nodes: D_s, S_s and each expanded hop-1
    account's. Its accounts that may carry an edge: B4's and the co-users', but hubs other
    than the scored account."""
    assert accumulation_text(DEFINITION, "ScoredCustomers") == normal(
        """IF @@owner_of.containsKey(a.id) THEN
            @@subgraph_customers += (a.id -> @@owner_of.get(a.id))
        END,
        FOREACH e IN @@expanded_of.get(a.id) DO
            IF @@owner_of.containsKey(e) THEN
                @@subgraph_customers += (a.id -> @@owner_of.get(e))
            END
        END,
        FOREACH c IN @@kept_co_users.get(a.id) DO
            FOREACH n IN @@open_nodes.get(a.id) DO
                IF @@link_start.get(c).containsKey(n) THEN
                    @@subgraph_customers += (a.id -> c)
                END
            END
        END"""
    )
    assert accumulation_text(DEFINITION, "ScoredNodesKept") == normal(
        """FOREACH n IN @@kept_links.get(@@owner_of.get(a.id)) DO
            @@subgraph_nodes += (a.id -> n)
        END,
        FOREACH n IN @@kept_seen.get(a.id) DO @@subgraph_nodes += (a.id -> n) END,
        FOREACH e IN @@expanded_of.get(a.id) DO
            FOREACH n IN @@kept_links.get(@@owner_of.get(e)) DO
                @@subgraph_nodes += (a.id -> n)
            END,
            FOREACH n IN @@kept_seen.get(e) DO @@subgraph_nodes += (a.id -> n) END
        END"""
    )
    assert accumulation_text(DEFINITION, "ScoredAccounts") == normal(
        """FOREACH b IN @@subgraph_accounts.get(a.id) DO
            IF b == a.id OR NOT @@hub_ids.contains(b) THEN @@edge_accounts += (a.id -> b) END
        END,
        FOREACH b IN @@co_user_accounts.get(a.id) DO
            IF b == a.id OR NOT @@hub_ids.contains(b) THEN @@edge_accounts += (a.id -> b) END
        END"""
    )
    assert accumulation_text(DEFINITION, "Printed") == normal(
        """FOREACH c IN @@subgraph_customers.get(a.id) DO @@printed_customers += c END,
        FOREACH n IN @@subgraph_nodes.get(a.id) DO @@printed_nodes += n END"""
    )


# The edges.


def _link(customer: str) -> str:
    """The fields of a customer's link to node n, into the edge's locals."""
    return f"""start_present = @@known_link_start.get({customer}).containsKey(n),
    start_ms = 0,
    IF start_present THEN start_ms = t_ms - @@known_link_start.get({customer}).get(n) END,
    seen_present = @@last_seen.get({customer}).containsKey(n),
    seen_ms = 0,
    IF seen_present THEN seen_ms = t_ms - @@last_seen.get({customer}).get(n) END,"""


def test_enrolment_edges_from_c_s_and_around_the_scored_accounts_open_nodes() -> None:
    """C_s to each of its kept links, a leaf among them; and every customer of the subgraph
    with a visible link to an open node of the scored account. Each carries the time since
    its known start and since its latest payment seen there. By hand: C_s known on d_01 from
    2024-03-20 and last paying from it on 2024-03-31 12:00 has since_start 12 days and
    since_last_seen 12 hours."""
    assert accumulation_text(DEFINITION, "EnrolledEdges") == normal(
        f"""STRING owner = @@owner_of.get(a.id), BOOL start_present = FALSE, UINT start_ms = 0,
        BOOL seen_present = FALSE, UINT seen_ms = 0,
        FOREACH n IN @@kept_links.get(owner) DO
            {_link("owner")}
            @@enrolled += (a.id -> Enrolled(owner, n, start_present, start_ms, seen_present,
              seen_ms))
        END,
        FOREACH n IN @@open_nodes.get(a.id) DO
            FOREACH c IN @@subgraph_customers.get(a.id) DO
                IF @@link_start.get(c).containsKey(n) THEN
                    {_link("c")}
                    @@enrolled += (a.id -> Enrolled(c, n, start_present, start_ms,
                      seen_present, seen_ms))
                END
            END
        END"""
    )


def test_owner_enrolment_edges_copy_the_customers_link_onto_its_accounts() -> None:
    """Every account of C_s in the subgraph to C_s's kept links; each expanded hop-1 account
    to its customer's; and every account of the subgraph whose customer has a visible link to
    an open node of the scored account, to it. A hub other than the scored account has none.
    A hop-2 account has none to its own customer's nodes, only to the scored account's."""
    assert accumulation_text(DEFINITION, "OwnerEnrolledEdges") == normal(
        f"""STRING owner = @@owner_of.get(a.id), STRING customer = "", BOOL start_present = FALSE,
        UINT start_ms = 0, BOOL seen_present = FALSE, UINT seen_ms = 0,
        FOREACH b IN @@customer_accounts.get(a.id) DO
            IF @@edge_accounts.get(a.id).contains(b) THEN
                FOREACH n IN @@kept_links.get(owner) DO
                    {_link("owner")}
                    @@owner_enrolled += (a.id -> OwnerEnrolled(b, n, start_present, start_ms,
                      seen_present, seen_ms))
                END
            END
        END,
        FOREACH e IN @@expanded_of.get(a.id) DO
            customer = @@owner_of.get(e),
            FOREACH n IN @@kept_links.get(customer) DO
                {_link("customer")}
                @@owner_enrolled += (a.id -> OwnerEnrolled(e, n, start_present, start_ms,
                  seen_present, seen_ms))
            END
        END,
        FOREACH n IN @@open_nodes.get(a.id) DO
            FOREACH (b, linked) IN @@node_link_accounts.get(n) DO
                IF @@edge_accounts.get(a.id).contains(b) THEN
                    {_link("linked")}
                    @@owner_enrolled += (a.id -> OwnerEnrolled(b, n, start_present, start_ms,
                      seen_present, seen_ms))
                END
            END
        END"""
    )


def _seen(account: str) -> str:
    """The fields of an account's uses of node n, and its seen edge."""
    return f"""first_present = @@first_use_known.get({account}).contains(n),
    first_ms = 0,
    IF first_present THEN first_ms = t_ms - @@first_use.get({account}).get(n) END,
    @@seen += (a.id -> Seen({account}, n, @@uses.get({account}).get(n), first_present,
      first_ms, t_ms - @@last_use.get({account}).get(n)))"""


def test_seen_edges_from_the_expanded_accounts_and_around_the_open_nodes() -> None:
    """The scored account to its kept seen nodes, each expanded hop-1 account to its own, and
    every account of the subgraph seen on an open node of the scored account to it; each with
    its uses there in the lookback, since its first when that is a known start, and since its
    last. By hand: 2 uses on 2024-03-01, a known first use, and 2024-03-20 give uses_90d 2,
    since_first_use 31 days and since_last_use 12 days."""
    assert accumulation_text(DEFINITION, "SeenEdges") == normal(
        f"""BOOL first_present = FALSE, UINT first_ms = 0,
        FOREACH n IN @@kept_seen.get(a.id) DO {_seen("a.id")} END,
        FOREACH e IN @@expanded_of.get(a.id) DO
            FOREACH n IN @@kept_seen.get(e) DO {_seen("e")} END
        END,
        FOREACH n IN @@open_nodes.get(a.id) DO
            FOREACH b IN @@node_senders.get(n) DO
                IF @@edge_accounts.get(a.id).contains(b) THEN {_seen("b")} END
            END
        END"""
    )


# The rows.


def test_a_customers_co_users_share_a_node_with_it_that_is_no_leaf() -> None:
    """On every node it shares, by a link held in the last 30 days or a recent payment from a
    deposit account, each other customer sharing it is a co-user; the pair began sharing at
    the later of the two starts, and at the earliest over the nodes. By hand: q_01 shares
    phone d_01 from 2024-01-02 (unknown) and IP i_01 from 2024-03-20 (known) with q_02, which
    started on both on 2024-03-25: one co-user, from 2024-03-25 on d_01 and on i_01, so new
    over 30 days, sharing a device and an IP."""
    assert accumulation_text(DEFINITION, "Pairs") == normal(
        """UINT own_ms = 0, UINT other_ms = 0,
        FOREACH n IN @@shares.get(q.id) DO
            IF NOT @@leaf_ids.contains(n) THEN
                FOREACH c IN @@node_users.get(n) DO
                    IF c != q.id AND @@shares.get(c).contains(n) THEN
                        own_ms = @@share_start.get(q.id).get(n),
                        other_ms = @@share_start.get(c).get(n),
                        q.@pair_start += (c -> CASE WHEN own_ms > other_ms THEN own_ms
                          ELSE other_ms END),
                        IF @@ip_ids.contains(n) THEN
                            q.@ip_co_users += c
                        ELSE
                            q.@device_co_users += c
                        END
                    END
                END
            END
        END"""
    )
    assert accumulation_text(DEFINITION, "CustomerRows") == normal(
        f"""FOREACH n IN @@link_current.get(q.id) DO
            IF @@ip_ids.contains(n) THEN q.@ips_30d += 1 ELSE q.@devices_30d += 1 END
        END,
        FOREACH (n, start_ms) IN @@known_link_start.get(q.id) DO
            IF start_ms >= t_ms - {WINDOW_30D_MS} THEN
                IF @@ip_ids.contains(n) THEN q.@new_ips_30d += 1 ELSE q.@new_devices_30d += 1 END
            END
        END,
        FOREACH (c, start_ms) IN q.@pair_start DO
            q.@co_users += 1,
            IF start_ms >= t_ms - {WINDOW_30D_MS} THEN q.@new_co_users_30d += 1 END,
            IF q.@device_co_users.contains(c) AND q.@ip_co_users.contains(c) THEN
                q.@co_users_both += 1
            END
        END"""
    )


def test_a_nodes_row_counts_its_sharers_and_its_new_customers() -> None:
    """A node that is no leaf: its users sharing it, those whose sharing start is in the last
    7 or 30 days, read_hubs' measure, its accounts and payments seen in the last 30 days, and
    whether no customer has a visible link to it. A leaf's row is its flag alone."""
    rows = select_of(DEFINITION, "NodeRows")
    assert _source(rows.statement) == "NodesListed" and rows.conjuncts == {"NOT d.@leaf"}
    assert accumulation_text(DEFINITION, "NodeRows") == normal(
        f"""FOREACH c IN @@node_users.get(d.id) DO
            IF @@shares.get(c).contains(d.id) THEN d.@customers_30d += 1 END,
            IF @@share_start.get(c).get(d.id) >= t_ms - {WINDOW_7D_MS} THEN
                d.@new_customers_7d += 1
            END,
            IF @@share_start.get(c).get(d.id) >= t_ms - {WINDOW_30D_MS} THEN
                d.@new_customers_30d += 1
            END
        END,
        d.@customers_90d += d.@crowd_customers.size(),
        d.@accounts_seen_30d += d.@seen_accounts.size(),
        d.@observed_only += d.@crowd_enrolled.size() == 0"""
    )
    assert select_of(DEFINITION, "Accounts").conjuncts == {"a.opened_ms < t_ms"}


# Bounds, link ends, dates of record and labels.


def test_no_bound_at_t_takes_t_itself_and_every_window_lies_in_the_lookback() -> None:
    code = normal(DEFINITION)
    assert not re.search(r"<= t_ms\b|(?<![\w.@])t_ms >(?!= lookback)(?! lookback)", code)
    assert set(re.findall(r"t_ms - (\d+)", code)) == {str(WINDOW_7D_MS), str(WINDOW_30D_MS)}
    # Every payment read lies in the lookback, by its own window in the 15 SELECTs that step
    # to a payment, or as one chosen by them.
    assert code.count("p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms") == 15


def test_a_link_end_is_read_only_by_the_visible_and_current_tests() -> None:
    """valid_to_ms is never a value, an order or a test against t but those two: 23 visible
    tests, one per link a pattern steps along, and 3 current ones."""
    code = normal(DEFINITION)
    visible = re.findall(
        r"(\w+)\.valid_from_ms < t_ms AND \( \1\.valid_to_ms == 0 OR \1\.valid_to_ms > "
        r"t_ms - lookback_ms \)",
        code,
    )
    current = re.findall(
        r"(\w+)\.valid_from_ms < t_ms AND \( \1\.valid_to_ms == 0 OR \1\.valid_to_ms > "
        rf"t_ms - {WINDOW_30D_MS} \)",
        code,
    )
    assert (len(visible), len(current)) == (23, 3)
    assert code.count("valid_to_ms") == 2 * (len(visible) + len(current))


def test_it_reads_no_label_and_dates_of_record_only_as_existence_and_known_starts() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(r"closed_ms|Token|is_mule|pu_label|mule_", code)
    assert set(re.findall(r"\w+\.opened_ms", code)) == {
        "a.opened_ms",
        "b.opened_ms",
        "c.opened_ms",
        "h.opened_ms",
        "s.opened_ms",
    }
    assert set(re.findall(r'\.account_type == "(\w+)"', code)) == {"deposit"}
    assert re.findall(r"\w+\.is_external", code) == ["Accounts.is_external"]
