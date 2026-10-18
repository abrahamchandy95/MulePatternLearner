"""read_community, B6, read as text (docs/design.md, "read_community (B6): how the mule network
formed", and docs/features.md, "The community block").

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclass its parser reads. Beside those rules, each part is
held here clause by clause, since no reference graph runs it yet: the owner as round 0; each
customer's sharing read as read_shared_identity (B5) reads it, its known starts included; the
leaves by read_hubs' own crowd rule; the two sharing rounds and the payment round; the heap
that keeps each round's members newest first until the community holds 64; the five passes of
the loop that runs them; and the block. gsql-lsp checks the file against the schema in
tests/gsql/test_schema.py, and the live compile test compiles it.
"""

from __future__ import annotations

from dataclasses import fields
import re

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    Select,
    common_problems,
    dataflow_problems,
    field_rule,
    normal,
    parameter_rule,
    parameters,
    projection_fields,
    projection_items,
    scopes,
    select_flows,
    select_of,
    source_rule,
    statements,
    status_fields,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    QUERY_FILES,
    READ_COMMUNITY,
    READ_COUNTERPARTIES,
    READ_HUBS,
    READ_SHARED_IDENTITY,
    REGISTRY,
    VERSIONS,
    read_community,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_community import Community
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS


def _definition(name: str) -> str:
    return definitions((GSQL_DIR / QUERY_FILES[name]).read_text())[name]


TEXT = (GSQL_DIR / QUERY_FILES[READ_COMMUNITY]).read_text()
DEFINITION = definitions(TEXT)[READ_COMMUNITY]
COUNTERPARTIES = _definition(READ_COUNTERPARTIES)
SHARED_IDENTITY = _definition(READ_SHARED_IDENTITY)
HUBS = _definition(READ_HUBS)
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The community's cap of docs/design.md, "Time", and the crowd threshold of "Shared rules";
# sharing and the payment round read the last 30 days.
CAP = SUBGRAPH.community
CROWD = SUBGRAPH.crowd_threshold
DAY_MS = 86_400_000
WINDOW_30D_MS = SUBGRAPH.sharing_days * DAY_MS  # 2,592,000,000

# A statement that starts a block (WHILE ... DO, IF ... THEN, ELSE) carries the block's head
# before the name it assigns.
HEAD = re.compile(r"^(?:(?:WHILE|IF) .*? (?:DO|THEN) |ELSE )?(\w+) (?:\( \w+ \) )?= ")


def _target(statement: tuple[str, ...] | list[str]) -> str | None:
    """The name a statement assigns, whether or not it starts a block."""
    match = HEAD.match(" ".join(statement))
    return match[1] if match else None


def _select(target: str) -> Select:
    """The SELECT that assigns target, in a block or not."""
    for _, _, select in scopes(DEFINITION):
        if select is not None and _target(select.statement) == target:
            return select
    raise AssertionError(f"no SELECT assigns {target}")


def _code(target: str) -> str:
    """The statement that assigns target, from its name on, its tokens joined by spaces."""
    text = " ".join(_select(target).statement)
    return text[text.index(f"{target} ") :]


def _accumulation(target: str) -> str:
    return " ".join(_select(target).accumulations)


def _source(select: Select) -> str:
    """The vertex set a SELECT starts from."""
    return select.statement[select.statement.index("FROM") + 1]


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_COMMUNITY] in written_files().values()
    assert common_problems(READ_COMMUNITY, TEXT) == []
    assert version_rule(READ_COMMUNITY, DEFINITION, VERSIONS[READ_COMMUNITY]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    """Like B2, B4 and B5: the account hubs come as a parameter, since a member's accounts and
    the payment round's counterparties are no hub."""
    built = read_community.params(T_MS, ["a_01", "a_02"], ["h_01"])
    assert parameter_rule(READ_COMMUNITY, DEFINITION, built) == []
    assert parameters(DEFINITION) == parameters(COUNTERPARTIES) == parameters(SHARED_IDENTITY)


def test_it_prints_the_fields_of_its_dataclass() -> None:
    """One block per scored account, and nothing else: the status row prints the status and
    the version alone."""
    assert field_rule("read_community", projection_fields(DEFINITION, "Blocks"), Community) == []
    assert source_rule("read_community", projection_items(DEFINITION, "Blocks")) == []
    assert status_fields(DEFINITION) == []
    printed = [" ".join(s) for s in statements(DEFINITION) if s[0] == "PRINT"]
    assert printed[-1].startswith("PRINT Blocks [") and printed[-1].endswith("] AS accounts")
    assert [field.name for field in fields(Community)][1:] == [
        "members",
        "capped",
        "shared_nodes",
        "density",
        "new_link_share_30d",
        "internal_amount_30d",
        "isolation",
        "member_payments_30d",
    ]


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_COMMUNITY]
    assert query.params is read_community.params
    assert query.rows == (Community,)


# What the query declares, exactly, in order. The heap has the community's cap as its capacity
# and keeps each round before the next, newest first, ties by customer id; every time is UINT
# and every count and cent INT.
DECLARATIONS = [
    "TYPEDEF TUPLE<INT round_number, UINT ts, STRING id> Member",
    "OrAccum @hub, @read, @tested",
    "MaxAccum<STRING> @owner_id, @receiver_id",
    "OrAccum @deposit_sender",
    "SetAccum<STRING> @seen_on",
    "SetAccum<VERTEX<Party>> @crowd_customers",
    "OrAccum @leaf",
    f"HeapAccum<Member>({CAP}, round_number ASC, ts DESC, id ASC) @community",
    "SumAccum<INT> @proposed",
    "MapAccum<STRING, SumAccum<INT>> @node_links, @node_new_links",
    "SetAccum<STRING> @communities",
    "MaxAccum<STRING> @member_owner",
    "SumAccum<INT> @members, @shared_nodes, @internal_amount_30d, @member_payments_30d",
    "OrAccum @capped",
    "SumAccum<DOUBLE> @density, @new_link_share_30d, @isolation",
    "MapAccum<STRING, MapAccum<STRING, MinAccum<UINT>>> @@link_start, @@first_unseen, "
    "@@dep_first, @@share_start",
    "MapAccum<STRING, MinAccum<UINT>> @@first_own",
    "MapAccum<STRING, SetAccum<STRING>> @@own_seen, @@shares, @@sharers",
    "SetAccum<STRING> @@leaf_ids",
    "MapAccum<STRING, SetAccum<STRING>> @@members, @@frontier, @@member_of",
    "MapAccum<STRING, MapAccum<STRING, MaxAccum<UINT>>> @@candidate_time",
    "SetAccum<STRING> @@member_ids, @@frontier_ids, @@frontier_node_ids",
    "MapAccum<STRING, SumAccum<INT>> @@internal, @@internal_cents, @@external",
    "SetAccum<STRING> @@ids",
    "SumAccum<INT> @@given",
    f"UINT lookback_ms = {LOOKBACK_MS}",
    f"UINT crowd_threshold = {CROWD}",
    "INT community_round = 0",
]
DECLARES = re.compile(r"\w+Accum\b|TYPEDEF |(?:INT|UINT|FLOAT|DOUBLE|STRING|BOOL|DATETIME) ")


def test_it_declares_exactly_its_tuples_accumulators_and_settings() -> None:
    code = without_comments(DEFINITION)
    body = code[code.index("{") + 1 : code.rindex("}")]
    found = [" ".join(part.split()) for part in body.split(";")]
    assert [text for text in found if DECLARES.match(text)] == DECLARATIONS
    assert (CAP, CROWD) == (64, 8)


# The refusals and the hubs, as B2's.


def test_it_refuses_and_checks_exactly_as_b2_before_any_payment_is_read() -> None:
    """t, then every account, then every hub, each refusal returning at once: the statements
    from the test of t to the hubs' marks are B2's."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    theirs = [" ".join(statement) for statement in statements(COUNTERPARTIES)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    their_start = next(at for at, text in enumerate(theirs) if text.startswith("IF "))
    marked = found.index("Marked = SELECT h FROM HubAccounts : h POST-ACCUM h.@hub += TRUE")
    assert found[start : marked + 1] == theirs[their_start : their_start + marked + 1 - start]
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
    """A member's account and a counterparty of the payment round are no hub; nothing else
    reads a hub, and no account hub is counted here."""
    code = normal(DEFINITION)
    assert code.count("hub_accounts") == 2
    assert re.findall(r"\S*@hub\b\S*", code) == [
        "@hub",
        "h.@hub",
        "b.@hub",
        "r.@hub",
        "b.@hub",
        "r.@hub",
        "b.@hub",
    ]
    assert not re.search(r"outdegree|hub_threshold|@@hub", code)


# The order of the statements, pinned whole, and the rule that explains it.
SELECT_ORDER = [
    "Scored",
    "HubAccounts",
    "Marked",
    "Owners",
    "Fresh",
    "FreshLinks",
    "FreshSent",
    "FreshSeen",
    "FreshUnseen",
    "FreshStarts",
    "Untested",
    "EnrolledCrowd",
    "SeenCrowd",
    "Leaves",
    "LeafIds",
    "Proposed",
    "Payees",
    "Payers",
    "Ranked",
    "Kept",
    "FrontierReceivers",
    "LinkUsers",
    "SeenUsers",
    "MemberAccounts",
    "MemberSent",
    "MemberReceived",
    "NodeLinks",
    "Blocks",
]
# Every other statement from the owners on, in order: the loop, its blocks, the clears, the
# vertex sets, and the step of the round number.
OTHERS = [
    "Pending ( ANY ) = Owners",
    "WHILE community_round < 5 DO @@candidate_time.clear ( )",
    "FreshNodes = FreshLinks UNION FreshSeen",
    "END",
    'IF community_round == 3 THEN PayingMembers = to_vertex_set ( @@member_ids , "Party" )',
    "END",
    "@@frontier.clear ( )",
    "@@frontier_ids.clear ( )",
    "IF community_round < 2 THEN @@frontier_node_ids.clear ( )",
    "FOREACH m IN @@frontier_ids DO FOREACH n IN @@shares.get ( m ) DO "
    "IF NOT @@leaf_ids.contains ( n ) THEN @@frontier_node_ids += n",
    "END",
    "END",
    "END",
    'FrontierDevices = to_vertex_set ( @@frontier_node_ids , "Device" )',
    'FrontierIps = to_vertex_set ( @@frontier_node_ids , "IP" )',
    "FrontierNodes = FrontierDevices UNION FrontierIps",
    "Pending = LinkUsers UNION SeenUsers",
    'ELSE Pending = to_vertex_set ( @@frontier_ids , "Party" )',
    "END",
    "community_round = community_round + 1",
    "END",
    'AllMembers = to_vertex_set ( @@member_ids , "Party" )',
]


def test_the_statements_run_in_the_order_the_rounds_need() -> None:
    """The loop of five passes holds the reading of each customer handed to it, the leaf test
    of the nodes it shares, the round's candidates (rounds 1 and 2 in passes 1 and 2, the
    payment round in pass 3), their keeping, and what the next pass reads; the block comes
    after it."""
    assert [_target(select.statement) for select in _selects()] == SELECT_ORDER
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    owners = next(at for at, text in enumerate(found) if text.startswith("Owners ="))
    blocks = next(at for at, text in enumerate(found) if text.startswith("Blocks ="))
    assert [
        text for text in found[owners + 1 : blocks] if "SELECT" not in text.split(" ")
    ] == OTHERS
    # The rounds 1 and 2 propose in passes 1 and 2 alone.
    assert " ".join(_select("Proposed").statement).startswith(
        "IF community_round == 1 OR community_round == 2 THEN Proposed = SELECT"
    )


def _selects() -> list[Select]:
    return [select for _, _, select in scopes(DEFINITION) if select is not None]


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    """Each pass reads, of the vertex accumulators, only what an earlier statement of the same
    pass, or of the request before the loop, wrote."""
    assert dataflow_problems(DEFINITION) == []
    reads = {
        _target(select.statement): read
        for select, (_, _, read) in zip(_selects(), select_flows(DEFINITION), strict=True)
        if read
    }
    assert reads == {
        "FreshSeen": {"owner_id", "deposit_sender"},
        "FreshUnseen": {"owner_id", "seen_on"},
        "Leaves": {"crowd_customers"},
        "Kept": {"community"},
        "MemberSent": {"communities", "member_owner"},
        "MemberReceived": {"communities"},
        "NodeLinks": {"community"},
        "Blocks": {"node_links", "node_new_links", "community", "proposed"},
    }


# Round 0, and the reading of each customer's sharing, as B5 reads it.

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
CURRENT_L = {
    "l.valid_from_ms < t_ms",
    f"( l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - {WINDOW_30D_MS} )",
}
WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms"}
WINDOW_30D = {f"p.event_ts_ms >= t_ms - {WINDOW_30D_MS}", "p.event_ts_ms < t_ms"}
ACCOUNT = frozenset({"Account"})
PARTY = frozenset({"Party"})
PAYMENTS = frozenset({"Payment_Transaction", "Zelle_Transfer"})
NODE_TYPES = frozenset({"Device", "IP"})
OWNS = frozenset({"Party_Owns_Account"})
OWNED_BY = frozenset({"Account_Owned_By_Party"})
SENT_FROM = frozenset({"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"})
RECEIVED_BY = frozenset({"Account_Received_Transaction", "Account_Received_Zelle_Transfer"})
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


def test_round_0_is_the_scored_accounts_owner() -> None:
    """Each scored account's owner, through a visible ownership link, goes first into its
    heap, as round 0 at time 0, and counts as proposed; an account with none has no member.
    The owners are the first customers read."""
    owners = _select("Owners")
    assert [(step.alias, step.types, step.chosen) for step in owners.vertices] == [
        ("a", ACCOUNT, True),
        ("q", PARTY, False),
    ]
    assert [edge.types for edge in owners.edges] == [OWNED_BY]
    assert owners.conjuncts == VISIBLE_O
    assert _accumulation("Owners") == normal("a.@community += Member(0, 0, q.id), a.@proposed += 1")
    assert _source(owners) == "Scored"


def test_each_pass_reads_a_customer_once() -> None:
    """A customer's sharing is the same whichever scored account or pass reaches it, so a
    pass reads only the customers handed to it that no pass has read."""
    assert _code("Fresh") == "Fresh = SELECT q FROM Pending : q WHERE NOT q.@read POST-ACCUM " + (
        "q.@read += TRUE"
    )


def test_a_customers_visible_links_give_its_nodes_and_those_it_shares() -> None:
    """Every visible link and its start, the earliest per node; a link held in the last 30
    days, the current test, shares the node. By hand, on t = 2024-04-01: links to d_01 from
    2023-12-01 to 2024-01-10 and from 2024-02-20, open, start on 2023-12-01 and share d_01; a
    link to d_02 ended on 2024-03-01, 31 days before t, is visible and shares nothing."""
    links = _select("FreshLinks")
    assert _source(links) == "Fresh" and links.selected == "n"
    assert [(step.alias, step.types) for step in links.vertices] == [
        ("q", PARTY),
        ("n", NODE_TYPES),
    ]
    assert [(edge.alias, edge.types) for edge in links.edges] == [("l", LINKS)]
    assert links.conjuncts == VISIBLE_L
    assert _accumulation("FreshLinks") == normal(
        f"""@@link_start += (q.id -> (n.id -> l.valid_from_ms)),
        IF l.valid_from_ms < t_ms
          AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - {WINDOW_30D_MS}) THEN
            @@shares += (q.id -> n.id),
            @@sharers += (n.id -> q.id)
        END"""
    )


def test_a_customers_own_payments_are_read_as_b5_reads_them() -> None:
    """Every payment sent in the lookback, but self-payments, by an account the customer owns
    through a visible ownership link and that exists at t: whether its sender is a deposit
    account, the customer's earliest, and, per node seen on its payments, its earliest payment
    not seen there; B5's own reading of a customer of a row."""
    sent = _select("FreshSent")
    theirs = select_of(SHARED_IDENTITY, "RowSent")
    assert _source(sent) == "Fresh" and sent.selected == "p"
    assert sent.paths == theirs.paths
    assert sent.conjuncts == theirs.conjuncts
    assert _accumulation("FreshSent") == " ".join(theirs.accumulations)
    assert _source(_select("FreshUnseen")) == "FreshSent"
    assert _accumulation("FreshUnseen") == " ".join(
        select_of(SHARED_IDENTITY, "RowUnseen").accumulations
    )


def test_the_nodes_seen_on_them_and_those_a_deposit_account_shares() -> None:
    """Per node seen on a customer's own payments: the node is one of its own payments', and
    its first use from a deposit account starts its sharing there; a use from a deposit
    account in the last 30 days shares the node. By hand: a customer whose savings account
    paid a friend on 2024-03-25 from d_01 and whose card account paid on 2024-03-28 from d_02
    shares d_01 from 2024-03-25 and never shares d_02."""
    seen = _select("FreshSeen")
    assert _source(seen) == "FreshSent" and seen.selected == "n"
    assert [(step.alias, step.types, step.chosen) for step in seen.vertices] == [
        ("p", PAYMENTS, True),
        ("n", NODE_TYPES, False),
    ]
    assert [edge.types for edge in seen.edges] == [USED]
    assert not seen.conjuncts
    assert _accumulation("FreshSeen") == normal(
        f"""p.@seen_on += n.id,
        @@own_seen += (p.@owner_id -> n.id),
        IF p.@deposit_sender THEN
            @@dep_first += (p.@owner_id -> (n.id -> p.event_ts_ms)),
            IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
                @@shares += (p.@owner_id -> n.id),
                @@sharers += (n.id -> p.@owner_id)
            END
        END"""
    )


def test_a_sharing_start_counts_only_when_it_is_known_as_in_b5() -> None:
    """A link start or a first use from a deposit account is known after the lookback's start,
    after the customer joined, and after a payment of its own not seen on the node; each is
    t - 90 days when not known, and the sharing start is the earliest. B5's own test, without
    its known link starts, which no number of the block reads. By hand, on t = 2024-04-01: a
    customer that joined on 2024-03-10 and first paid from its new phone d_01 that evening
    shares d_01 from 2024-01-02, t - 90 days; one that paid from d_02 on 2024-02-01 and
    enrolled d_01 on 2024-03-20 shares d_01 from 2024-03-20."""
    theirs = " ".join(select_of(SHARED_IDENTITY, "Starts").accumulations)
    known_link = "@@known_link += ( q.id -> ( n -> start_ms ) ) , "
    assert theirs.count(known_link) == 1
    assert _accumulation("FreshStarts") == theirs.replace(known_link, "")
    assert _source(_select("FreshStarts")) == "Fresh"
    # joined_ms is read in these two tests alone.
    assert len(re.findall(r"\w+\.joined_ms", without_comments(DEFINITION))) == 2


@pytest.mark.parametrize(("ours", "theirs"), [("EnrolledCrowd", "Enrolled"), ("SeenCrowd", "Seen")])
def test_each_node_shared_is_tested_once_by_read_hubs_crowd_rule(ours: str, theirs: str) -> None:
    """A node's customers are counted exactly as read_hubs counts them, so the inline leaf
    flag equals read_hubs' crowd flag; payments are read only for a node that 8 enrolled
    customers or fewer do not already make a leaf, which changes the cost and never the
    flag. Only the nodes some customer shares are tested, each once."""
    expected = (
        " ".join(select_of(HUBS, theirs).statement)
        .replace(f"{theirs} =", f"{ours} =")
        .replace("FROM Nodes", "FROM Untested")
        .replace(" , d.@crowd_enrolled += q", "")
        .replace(
            "WHERE p.event_ts_ms",
            "WHERE d.@crowd_customers.size ( ) <= crowd_threshold AND p.event_ts_ms",
        )
    )
    assert _code(ours) == expected
    assert _code("Untested") == (
        "Untested = SELECT n FROM FreshNodes : n WHERE @@sharers.containsKey ( n.id ) AND NOT "
        "n.@tested POST-ACCUM n.@tested += TRUE"
    )
    assert _code("Leaves") == (
        "Leaves = SELECT d FROM Untested : d POST-ACCUM d.@leaf += "
        "d.@crowd_customers.size ( ) > crowd_threshold"
    )
    assert _code("LeafIds") == (
        "LeafIds = SELECT d FROM Untested : d WHERE d.@leaf POST-ACCUM @@leaf_ids += d.id"
    )


# The rounds and the heap.


def test_rounds_1_and_2_propose_the_sharers_of_the_nodes_of_the_members_kept_before() -> None:
    """On each node that is no leaf shared by a member kept in the round before, every
    customer sharing it that is no member yet, at the later of the two sharing starts, its
    latest over those nodes and members. By hand, on t = 2024-04-01: the owner shares the
    ring phone d_01 from 2024-01-02 (unknown start) and the ring IP i_01 from 2024-03-11;
    c_02 shares d_01 from 2024-03-12 and i_01 from 2024-01-02: its time is the latest of
    max(2024-01-02, 2024-03-12) and max(2024-03-11, 2024-01-02), 2024-03-12."""
    proposed = _select("Proposed")
    assert _source(proposed) == "Scored" and not proposed.edges and not proposed.conjuncts
    assert _accumulation("Proposed") == normal(
        """UINT own_ms = 0, UINT other_ms = 0,
        FOREACH m IN @@frontier.get(s.id) DO
            FOREACH n IN @@shares.get(m) DO
                IF NOT @@leaf_ids.contains(n) THEN
                    FOREACH c IN @@sharers.get(n) DO
                        IF NOT @@members.get(s.id).contains(c) THEN
                            own_ms = @@share_start.get(m).get(n),
                            other_ms = @@share_start.get(c).get(n),
                            @@candidate_time += (s.id -> (c -> CASE
                              WHEN own_ms > other_ms THEN own_ms ELSE other_ms END))
                        END
                    END
                END
            END
        END"""
    )


@pytest.mark.parametrize(
    ("target", "way", "end"),
    [("Payees", SENT_FROM, SENT_TO), ("Payers", RECEIVED_BY, FROM)],
)
def test_the_payment_round_proposes_the_owners_of_the_members_counterparties(
    target: str, way: frozenset[str], end: frozenset[str]
) -> None:
    """From every account of a member kept in rounds 0 to 2 that is no hub and exists at t
    through a visible ownership link, each payment of the last 30 days, but self-payments,
    with an internal account that exists at t and is no hub: its owner through a visible
    ownership link, when no member yet, at its latest such payment. By hand, on
    t = 2024-04-01: a member's account paid c_06's deposit account on 2024-03-05 and
    2024-03-20, and received from c_07's card account on 2024-03-25 and from a payroll
    account of another bank: c_06 at 2024-03-20 and c_07 at 2024-03-25; the payroll's owner
    is no candidate."""
    select = _select(target)
    assert _source(select) == "PayingMembers" and select.selected == "c"
    assert [(step.alias, step.types, step.chosen) for step in select.vertices] == [
        ("q", PARTY, False),
        ("b", ACCOUNT, False),
        ("p", PAYMENTS, False),
        ("r", ACCOUNT, False),
        ("c", PARTY, False),
    ]
    assert [edge.types for edge in select.edges] == [OWNS, way, end, OWNED_BY]
    assert select.conjuncts == VISIBLE_W | VISIBLE_O | WINDOW_30D | {
        "b.opened_ms < t_ms",
        "NOT b.@hub",
        "r != b",
        "NOT r.is_external",
        "NOT r.@hub",
        "r.opened_ms < t_ms",
    }
    assert _accumulation(target) == normal(
        """FOREACH s IN @@member_of.get(q.id) DO
            IF NOT @@members.get(s).contains(c.id) THEN
                @@candidate_time += (s -> (c.id -> p.event_ts_ms))
            END
        END"""
    )


def test_the_heap_keeps_each_round_newest_first_until_the_community_holds_64() -> None:
    """Every candidate of the pass's round goes into the scored account's heap, which orders
    by round, then newest first, then by customer id, and keeps 64: a later round never
    displaces an earlier one, and within a round the oldest go first. Then every member is
    listed, and those of this round are the round's kept members. By hand: with 62 members
    after round 1 and round 2's candidates c_10 at 2024-03-20, c_11 and c_12 at 2024-03-25,
    the heap keeps c_11 and c_12 and leaves c_10 out, so the community is capped."""
    assert _accumulation("Ranked") == normal(
        """FOREACH (customer, candidate_ms) IN @@candidate_time.get(s.id) DO
            s.@community += Member(community_round, candidate_ms, customer),
            s.@proposed += 1
        END"""
    )
    assert _accumulation("Kept") == normal(
        """FOREACH x IN s.@community DO
            @@members += (s.id -> x.id),
            @@member_of += (x.id -> s.id),
            @@member_ids += x.id,
            IF x.round_number == community_round THEN
                @@frontier += (s.id -> x.id),
                @@frontier_ids += x.id
            END
        END"""
    )
    for target in ("Ranked", "Kept"):
        assert _source(_select(target)) == "Scored"


def test_passes_0_and_1_hand_on_the_sharers_of_the_kept_members_nodes() -> None:
    """The nodes that are no leaf of the members just kept, and every customer sharing one:
    through a link held in the last 30 days, or a payment of the last 30 days, but
    self-payments, from a deposit account it owns that exists at t through a visible
    ownership link. Passes 2 and 3 hand on the members they kept."""
    receivers = _select("FrontierReceivers")
    assert _source(receivers) == "FrontierNodes" and receivers.selected == "p"
    assert [edge.types for edge in receivers.edges] == [USED_BY, SENT_TO]
    assert receivers.conjuncts == WINDOW_30D | {"NOT d.@leaf"}
    assert _accumulation("FrontierReceivers") == "p.@receiver_id += r.id"
    links = _select("LinkUsers")
    assert _source(links) == "FrontierNodes" and links.selected == "q"
    assert [(edge.alias, edge.types) for edge in links.edges] == [("l", LINKED_BY)]
    assert links.conjuncts == CURRENT_L | {"NOT d.@leaf"}
    assert not links.accumulations
    seen = _select("SeenUsers")
    assert _source(seen) == "FrontierNodes" and seen.selected == "q"
    assert [edge.types for edge in seen.edges] == [USED_BY, FROM, OWNED_BY]
    assert seen.conjuncts == WINDOW_30D | VISIBLE_O | {
        "NOT d.@leaf",
        'b.account_type == "deposit"',
        "b.opened_ms < t_ms",
        "b.id != p.@receiver_id",
    }
    assert not seen.accumulations


# The block.


def test_the_members_accounts_and_their_payments_of_the_last_30_days() -> None:
    """A member's accounts exist at t through a visible ownership link and are no hub. A
    payment, but self-payments, between accounts of two members is internal, counted once
    from its sender with its cents; between two accounts of one member it is neither; with
    any other account it is external, counted from the member's side. By hand: a member
    sending USD 662.00 to another member's account adds 1 internal payment of 66,200 cents;
    one sending USD 40.50 to a biller, a hub, adds 1 external payment."""
    accounts = _select("MemberAccounts")
    assert _source(accounts) == "AllMembers" and accounts.selected == "b"
    assert [edge.types for edge in accounts.edges] == [OWNS]
    assert accounts.conjuncts == VISIBLE_W | {"b.opened_ms < t_ms", "NOT b.@hub"}
    assert _accumulation("MemberAccounts") == normal(
        """b.@member_owner += q.id,
        FOREACH s IN @@member_of.get(q.id) DO b.@communities += s END"""
    )
    sent = _select("MemberSent")
    assert [edge.types for edge in sent.edges] == [SENT_FROM, SENT_TO]
    assert sent.conjuncts == WINDOW_30D | {"r != b"}
    assert _accumulation("MemberSent") == normal(
        """INT cents = floor(p.amount * 100 + 0.5),
        FOREACH s IN b.@communities DO
            IF r.@communities.contains(s) THEN
                IF r.@member_owner != b.@member_owner THEN
                    @@internal += (s -> 1),
                    @@internal_cents += (s -> cents)
                END
            ELSE
                @@external += (s -> 1)
            END
        END"""
    )
    received = _select("MemberReceived")
    assert [edge.types for edge in received.edges] == [RECEIVED_BY, FROM]
    assert received.conjuncts == WINDOW_30D | {"r != b"}
    assert _accumulation("MemberReceived") == normal(
        """FOREACH s IN b.@communities DO
            IF NOT r.@communities.contains(s) THEN @@external += (s -> 1) END
        END"""
    )
    for target in ("MemberSent", "MemberReceived"):
        assert _source(_select(target)) == "MemberAccounts"


def test_the_block_counts_links_on_nodes_two_members_share() -> None:
    """A link is a member sharing a node that is no leaf; a node two members or more share is
    a shared node. By hand, for 6 members: d_01 shared by 4 of them, 3 since the last 30
    days, i_01 by 3, all new, i_02 by 2, both old, and i_03 by 1: 3 shared nodes and 9 links,
    density 9 / (6 x 3) = 0.5 and new_link_share_30d 6 / 9; with 4 internal payments of
    264,800 cents and 12 external ones, isolation 4 / 16 = 0.25 and 16 member payments."""
    assert _accumulation("NodeLinks") == normal(
        f"""FOREACH x IN s.@community DO
            FOREACH n IN @@shares.get(x.id) DO
                IF NOT @@leaf_ids.contains(n) THEN
                    s.@node_links += (n -> 1),
                    IF @@share_start.get(x.id).get(n) >= t_ms - {WINDOW_30D_MS} THEN
                        s.@node_new_links += (n -> 1)
                    END
                END
            END
        END"""
    )
    assert _accumulation("Blocks") == normal(
        """INT nodes = 0, INT links = 0, INT new_links = 0, INT internal = 0, INT payments = 0,
        FOREACH (n, sharers) IN s.@node_links DO
            IF sharers >= 2 THEN
                nodes = nodes + 1,
                links = links + sharers,
                new_links = new_links + s.@node_new_links.get(n)
            END
        END,
        internal = @@internal.get(s.id),
        payments = @@internal.get(s.id) + @@external.get(s.id),
        s.@members += s.@community.size(),
        s.@capped += s.@proposed > s.@community.size(),
        s.@shared_nodes += nodes,
        IF nodes > 0 THEN
            s.@density += links * 1.0 / (s.@community.size() * nodes)
        END,
        IF links > 0 THEN s.@new_link_share_30d += new_links * 1.0 / links END,
        s.@internal_amount_30d += @@internal_cents.get(s.id),
        IF payments > 0 THEN s.@isolation += internal * 1.0 / payments END,
        s.@member_payments_30d += payments"""
    )
    for target in ("NodeLinks", "Blocks"):
        assert _source(_select(target)) == "Scored"


# Bounds, link ends, dates of record, labels and algorithms.


def test_no_bound_at_t_takes_t_itself_and_every_window_lies_in_the_lookback() -> None:
    code = normal(DEFINITION)
    assert not re.search(r"<= t_ms\b|(?<![\w.@])t_ms >(?!= lookback)(?! lookback)", code)
    assert set(re.findall(r"t_ms - (\d+)", code)) == {str(WINDOW_30D_MS)}
    # Every payment read lies in the lookback: by its own window in the 8 SELECTs that step to
    # a payment, or as one chosen by them. Customers are read over the whole lookback; sharing,
    # the payment round and the block over its last 30 days.
    assert code.count("p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms") == 2
    assert code.count(f"p.event_ts_ms >= t_ms - {WINDOW_30D_MS} AND p.event_ts_ms < t_ms") == 6


def test_a_link_end_is_read_only_by_the_visible_and_current_tests() -> None:
    """valid_to_ms is never a value, an order or a test against t but those two: 11 visible
    tests, one per link a pattern steps along, and 2 current ones."""
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
    assert (len(visible), len(current)) == (11, 2)
    assert code.count("valid_to_ms") == 2 * (len(visible) + len(current))


def test_it_reads_no_label_and_dates_of_record_only_as_existence_and_known_starts() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(r"closed_ms|Token|is_mule|pu_label|mule_", code)
    assert set(re.findall(r"\w+\.opened_ms", code)) == {
        "b.opened_ms",
        "h.opened_ms",
        "r.opened_ms",
        "s.opened_ms",
    }
    assert set(re.findall(r'\.account_type == "(\w+)"', code)) == {"deposit"}
    assert re.findall(r"\w+\.is_external", code) == ["r.is_external", "r.is_external"]


def test_no_library_algorithm_and_no_other_query_builds_it() -> None:
    """Connected components, Louvain and label propagation run on the whole graph with no
    as-of time (docs/design.md, "read_community (B6)"); the community is built here, round by
    round."""
    code = without_comments(DEFINITION).lower()
    assert not re.search(r"\b(?:tg_\w+|louvain|label_prop\w*|wcc|scc|conn_comp\w*)\b", code)
    assert not re.search(r"\brun\s+query\b|\bdemo_", code)
    assert len(re.findall(r"\bquery\b", code)) == 1


# The rules read the statements inside the loop and its blocks: a slip there is caught as
# anywhere else.
SLIPS = [
    (
        "WHERE NOT d.@leaf AND l.valid_from_ms < t_ms",
        "WHERE l.valid_from_ms < t_ms",
        "steps between d and its users without the leaf test",
    ),
    (
        "AND p.event_ts_ms < t_ms AND r != b\n                AND NOT r.is_external",
        "AND p.event_ts_ms <= t_ms AND r != b\n                AND NOT r.is_external",
        "no allowed form at t",
    ),
    (
        "AND NOT r.is_external AND NOT r.@hub AND r.opened_ms < t_ms\n"
        "                AND o.valid_from_ms < t_ms\n"
        "                AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms)\n"
        "              ACCUM\n"
        "                FOREACH s IN @@member_of.get(q.id) DO\n"
        "                    IF NOT @@members.get(s).contains(c.id) THEN\n"
        "                        @@candidate_time += (s -> (c.id -> p.event_ts_ms))\n"
        "                    END\n"
        "                END;\n"
        "            Payers",
        "AND NOT r.is_external AND NOT r.@hub\n"
        "                AND o.valid_from_ms < t_ms\n"
        "                AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms)\n"
        "              ACCUM\n"
        "                FOREACH s IN @@member_of.get(q.id) DO\n"
        "                    IF NOT @@members.get(s).contains(c.id) THEN\n"
        "                        @@candidate_time += (s -> (c.id -> p.event_ts_ms))\n"
        "                    END\n"
        "                END;\n"
        "            Payers",
        "reaches an account that may not exist at t",
    ),
    (
        "AND (w.valid_to_ms == 0 OR w.valid_to_ms > t_ms - lookback_ms)\n"
        "            AND b.opened_ms < t_ms\n"
        "            AND p.event_ts_ms >= t_ms - lookback_ms",
        "AND b.opened_ms < t_ms\n            AND p.event_ts_ms >= t_ms - lookback_ms",
        "steps along a link that is not visible or current",
    ),
]


@pytest.mark.parametrize(("old", "new", "problem"), SLIPS, ids=["leaf", "at t", "exists", "link"])
def test_the_rules_read_the_statements_inside_the_loop(old: str, new: str, problem: str) -> None:
    assert old in TEXT
    found = common_problems(READ_COMMUNITY, TEXT.replace(old, new, 1))
    assert any(problem in item for item in found), found
