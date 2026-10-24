"""read_payment_events, B4, read as text (docs/design.md, "read_payment_events (B4): the payment
graph", and docs/features.md, "What one example's subgraph holds", items 1 to 5 and 10, and
"Payment edges").

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclasses its parser reads. Beside those rules, each part
is held here clause by clause, since no reference graph runs it yet: the capped payments of the
scored account and of each expanded hop-1 account, newest first with ties by id; the hop-1
accounts that are hubs left unexpanded; the other accounts of the scored account's customer;
the ownership edges; and each payment's row, its cents, rail and age, its pair and its gaps,
absent at a hub end. gsql-lsp checks the file against the schema in tests/gsql/test_schema.py,
and the live compile test compiles it.
"""

from __future__ import annotations

from dataclasses import fields
import re

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    accumulation_text,
    common_problems,
    dataflow_problems,
    declared_locals,
    field_rule,
    moved_select,
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
    tuples,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    QUERY_FILES,
    READ_PAYMENT_EVENTS,
    REGISTRY,
    VERSIONS,
    read_payment_events,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_payment_events import (
    EventAccount,
    OwnedBy,
    PaymentEvent,
)
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, MONDAY_MS, WEEK_MS

TEXT = (GSQL_DIR / QUERY_FILES[READ_PAYMENT_EVENTS]).read_text()
DEFINITION = definitions(TEXT)[READ_PAYMENT_EVENTS]
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The caps of docs/design.md, "Time": the scored account's newest 32 payments each way, each
# expanded counterparty's newest 16, and 8 other accounts of the scored account's customer.
SCORED = SUBGRAPH.scored_payments
OTHER = SUBGRAPH.counterparty_payments
SIBLINGS = SUBGRAPH.own_accounts


def _source(statement: tuple[str, ...]) -> str:
    """The vertex set a SELECT starts from."""
    return statement[statement.index("FROM") + 1]


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_PAYMENT_EVENTS] in written_files().values()
    assert common_problems(READ_PAYMENT_EVENTS, TEXT) == []
    assert version_rule(READ_PAYMENT_EVENTS, DEFINITION, VERSIONS[READ_PAYMENT_EVENTS]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    built = read_payment_events.params(T_MS, ["a_01", "a_02"], ["h_01"])
    assert parameter_rule(READ_PAYMENT_EVENTS, DEFINITION, built) == []
    # The accounts as ids, which the query checks itself; the account hubs as a set of
    # vertices, of which an id the graph lacks can be no counterparty.
    assert parameters(DEFINITION) == {
        "t_ms": ["UINT"],
        "account_ids": ["LIST", "<", "STRING", ">"],
        "hub_accounts": ["SET", "<", "VERTEX", "<", "Account", ">", ">"],
    }
    assert built["hub_accounts"] == [{"id": "h_01"}]


def test_each_printed_row_has_the_fields_of_its_dataclass() -> None:
    for vertex_set, kind in (("Accounts", EventAccount), ("Rows", PaymentEvent)):
        what = f"read_payment_events {vertex_set}"
        assert field_rule(what, projection_fields(DEFINITION, vertex_set), kind) == []
        assert source_rule(what, projection_items(DEFINITION, vertex_set)) == []
    # The ownership edges are tuples, printed in a map by scored account.
    assert tuples(DEFINITION)["OwnedBy"] == [field.name for field in fields(OwnedBy)]
    # The status row prints the status and the version alone; then the accounts, the payments
    # and per scored account its accounts, its payments and its ownership edges, each by the
    # name of what it prints.
    assert status_fields(DEFINITION) == []
    printed = [" ".join(s) for s in statements(DEFINITION) if s[0] == "PRINT"]
    assert [text.split(" [")[0] for text in printed[-5:]] == [
        "PRINT Accounts",
        "PRINT Rows",
        "PRINT @@subgraph_accounts AS subgraph_accounts",
        "PRINT @@subgraph_payments AS subgraph_payments",
        "PRINT @@owned_by AS owned_by",
    ]
    assert printed[-5].endswith("] AS accounts") and printed[-4].endswith("] AS payments")


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_PAYMENT_EVENTS]
    assert query.params is read_payment_events.params
    assert query.rows == (EventAccount, PaymentEvent, OwnedBy)


# What the query declares, exactly, in order. Every heap has its cap as its capacity and keeps
# the newest by second, ties by id; every cent is INT, never mixed with UINT.
DECLARATIONS = [
    "TYPEDEF TUPLE<INT sec, STRING id, STRING other, BOOL zelle> Picked",
    "TYPEDEF TUPLE<INT sec, STRING id> Sibling",
    "TYPEDEF TUPLE<STRING account_id, STRING customer_id> OwnedBy",
    "OrAccum @hub",
    f"HeapAccum<Picked>({SCORED}, sec DESC, id ASC) @scored_out, @scored_in",
    f"HeapAccum<Picked>({OTHER}, sec DESC, id ASC) @other_out, @other_in",
    "MaxAccum<UINT> @last_active",
    f"HeapAccum<Sibling>({SIBLINGS}, sec DESC, id ASC) @siblings",
    "SumAccum<INT> @amount, @pair_prior_count",
    "MaxAccum<STRING> @rail, @sender_id, @receiver_id",
    "OrAccum @sender_hub, @receiver_hub, @pair_gap_present, @pair_age_present, "
    "@sender_gap_present, @receiver_gap_present",
    "MinAccum<UINT> @pair_first",
    "MaxAccum<UINT> @pair_last, @sender_last, @receiver_last",
    "SumAccum<UINT> @age_ms, @pair_gap_ms, @pair_age_ms, @sender_gap_ms, @receiver_gap_ms",
    "MapAccum<STRING, SetAccum<STRING>> @@subgraph_accounts, @@subgraph_payments, @@reached_by",
    "MapAccum<STRING, SetAccum<OwnedBy>> @@owned_by",
    "SetAccum<STRING> @@hop1_ids, @@account_ids, @@transaction_ids, @@transfer_ids",
    "SetAccum<STRING> @@ids",
    "SumAccum<INT> @@given",
    f"UINT lookback_ms = {LOOKBACK_MS}",
]
DECLARES = re.compile(r"\w+Accum\b|TYPEDEF |(?:INT|UINT|FLOAT|DOUBLE|STRING|BOOL|DATETIME) ")


def test_it_declares_exactly_its_tuples_accumulators_and_settings() -> None:
    code = without_comments(DEFINITION)
    body = code[code.index("{") + 1 : code.rindex("}")]
    found = [" ".join(part.split()) for part in body.split(";")]
    assert [text for text in found if DECLARES.match(text)] == DECLARATIONS
    assert (SCORED, OTHER, SIBLINGS) == (32, 16, 8)


# The refusals.


def test_every_refusal_returns_at_once() -> None:
    """A refusal that went on would read with t at or below the lookback, where
    t_ms - lookback_ms wraps in UINT, answer a request of no account, of an account twice or
    of an account the graph lacks with fewer subgraphs than it asked for, read accounts that
    do not exist at t, or read as hubs accounts that were none at t."""
    found = statements(DEFINITION)
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
    assert refused == [
        "invalid_parameters",
        "invalid_parameters",
        "duplicate_account",
        "unknown_account",
        "invisible_account",
        "invisible_hub",
    ]


def test_t_then_every_account_then_every_hub_is_checked_before_any_payment_is_read() -> None:
    """t is checked first; then the ids: at least one, none twice (a list keeps a repeat, a
    set would drop it), each an account of the graph (to_vertex_set drops an id it lacks) and
    each existing at t; then the hubs, each existing at t, before they are marked. Only the
    accounts that passed every check are scored."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    assert found[start : start + 26] == [
        f"IF t_ms <= lookback_ms OR t_ms % {WEEK_MS} != {MONDAY_MS} THEN "
        'PRINT "invalid_parameters" AS status',
        "RETURN",
        "END",
        "FOREACH account_id IN account_ids DO @@ids += account_id",
        "@@given += 1",
        "END",
        'IF @@given == 0 THEN PRINT "invalid_parameters" AS status',
        "RETURN",
        "END",
        'IF @@ids.size ( ) != @@given THEN PRINT "duplicate_account" AS status',
        "RETURN",
        "END",
        'Requested = to_vertex_set ( @@ids , "Account" )',
        'IF Requested.size ( ) != @@ids.size ( ) THEN PRINT "unknown_account" AS status',
        "RETURN",
        "END",
        "Scored = SELECT s FROM Requested : s WHERE s.opened_ms < t_ms",
        'IF Scored.size ( ) != Requested.size ( ) THEN PRINT "invisible_account" AS status',
        "RETURN",
        "END",
        "Hubs = { hub_accounts }",
        "HubAccounts = SELECT h FROM Hubs : h WHERE h.opened_ms < t_ms",
        'IF HubAccounts.size ( ) != Hubs.size ( ) THEN PRINT "invisible_hub" AS status',
        "RETURN",
        "END",
        "Marked = SELECT h FROM HubAccounts : h POST-ACCUM h.@hub += TRUE",
    ]
    # Every SELECT that starts from the scored accounts starts from the checked ones.
    assert normal(DEFINITION).count("FROM Requested") == 1
    starts = {
        select.statement[0]: select.vertices[0]
        for _, _, select in scopes(DEFINITION)
        if select is not None and _source(select.statement) == "Scored"
    }
    assert list(starts) == [
        "ScoredSent",
        "ScoredReceived",
        "ScoredPicked",
        "Siblings",
        "SiblingsRanked",
        "SiblingsPicked",
        "CustomerAccounts",
    ]
    assert {(step.alias, step.chosen) for step in starts.values()} == {("a", True)}


# The account hubs.


def test_the_hub_parameter_is_the_only_hub_input() -> None:
    """A hub is an account of hub_accounts, read_hubs' account hubs of the same t: the query
    counts no customer, reads no degree and has no threshold of its own. The flag is set once,
    on the hubs that exist at t, and read three times: a hop-1 account that is a hub is not
    expanded, and each end of a payment says whether it is one."""
    code = normal(DEFINITION)
    assert code.count("hub_accounts") == 2  # the parameter and Hubs = {hub_accounts}
    assert re.findall(r"\S*@hub\b\S*", code) == ["@hub", "h.@hub", "c.@hub", "a.@hub", "b.@hub"]
    assert "h.@hub += TRUE" in accumulation_text(DEFINITION, "Marked")
    assert "c.@hub" in " ".join(select_of(DEFINITION, "Hop1").conjuncts)
    assert accumulation_text(DEFINITION, "Senders").endswith("p.@sender_hub += a.@hub")
    assert accumulation_text(DEFINITION, "Receivers").endswith("p.@receiver_hub += b.@hub")
    assert not re.search(r"outdegree|threshold|crowd|leaf|customers", code)


# The order of the SELECTs. A value a step reads is read in a later SELECT than the one that
# wrote it, so the order is pinned whole, and the rule that explains it is checked on the text
# and on text moves.
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
    "ExpandedOwners",
    "CustomerAccounts",
    "Accounts",
    "Transactions",
    "Transfers",
    "Senders",
    "Receivers",
    "ReceiverHistory",
    "SenderHistory",
    "Rows",
]


def test_the_selects_run_in_the_order_the_parts_need() -> None:
    assert [target for target, _, _ in select_flows(DEFINITION)] == SELECT_ORDER
    # The sets built from ids, and the payments of both kinds as one set.
    sets = [
        " ".join(statement)
        for statement in statements(DEFINITION)
        if statement[1:2] == ["="] and "SELECT" not in statement
    ]
    assert sets == [
        'Requested = to_vertex_set ( @@ids , "Account" )',
        "Hubs = { hub_accounts }",
        'Hop1Listed = to_vertex_set ( @@hop1_ids , "Account" )',
        'AccountsListed = to_vertex_set ( @@account_ids , "Account" )',
        'TransactionsListed = to_vertex_set ( @@transaction_ids , "Payment_Transaction" )',
        'TransfersListed = to_vertex_set ( @@transfer_ids , "Zelle_Transfer" )',
        "Payments = Transactions UNION Transfers",
    ]


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    assert dataflow_problems(DEFINITION) == []
    reads = {target: read for target, _, read in select_flows(DEFINITION) if read}
    assert reads == {
        "ScoredPicked": {"scored_out", "scored_in"},
        "Hop1Picked": {"other_out", "other_in"},
        "SiblingsRanked": {"last_active"},
        "SiblingsPicked": {"siblings"},
        "Senders": {"hub"},
        "Receivers": {"hub"},
        "ReceiverHistory": {"sender_id"},
        "SenderHistory": {"sender_hub", "receiver_hub", "receiver_id"},
        "Rows": {"pair_prior_count", "pair_first", "pair_last", "sender_last", "receiver_last"},
    }
    # The WHERE clauses read the hub flags of a hop-1 account and of a payment's ends.
    assert "NOT c.@hub" in select_of(DEFINITION, "Hop1").conjuncts
    assert "NOT p.@receiver_hub" in select_of(DEFINITION, "ReceiverHistory").conjuncts
    assert (
        "( NOT p.@sender_hub OR p.@receiver_hub )"
        in select_of(DEFINITION, "SenderHistory").conjuncts
    )


@pytest.mark.parametrize(
    ("target", "after", "problem"),
    [
        # The heaps read before they are filled: every subgraph holds its scored account alone.
        ("ScoredSent", "ScoredPicked", "ScoredPicked reads @scored_out, written by ScoredSent"),
        ("Hop1Received", "Hop1Picked", "Hop1Picked reads @other_in, written by Hop1Received"),
        # The other accounts picked before their latest payments are read: by id alone.
        ("Active", "SiblingsRanked", "SiblingsRanked reads @last_active, written by Active"),
        # Hubs marked after the ends are read: no gap is ever absent at a hub end.
        ("Marked", "Senders", "Senders reads @hub, written by Marked"),
        # The ends read after the history: no pair is ever found.
        ("Senders", "ReceiverHistory", "ReceiverHistory reads @sender_id, written by Senders"),
        # The rows before the history: every gap is absent.
        ("SenderHistory", "Rows", "Rows reads @sender_last, written by SenderHistory"),
    ],
)
def test_a_select_moved_before_the_one_it_reads_from_breaks_the_rule(
    target: str, after: str, problem: str
) -> None:
    moved = moved_select(DEFINITION, target, after)
    assert moved != DEFINITION
    assert problem in dataflow_problems(moved)


# The capped payments.


WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms"}
PAYMENTS = {"Payment_Transaction", "Zelle_Transfer"}
SENT = [
    {"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"},
    {"Transaction_To_Account", "Transfer_To_Account"},
]
RECEIVED = [
    {"Account_Received_Transaction", "Account_Received_Zelle_Transfer"},
    {"Transaction_From_Account", "Transfer_From_Account"},
]


@pytest.mark.parametrize(
    ("target", "alias", "edges", "heap"),
    [
        ("ScoredSent", "a", SENT, "@scored_out"),
        ("ScoredReceived", "a", RECEIVED, "@scored_in"),
        ("Hop1Sent", "c", SENT, "@other_out"),
        ("Hop1Received", "c", RECEIVED, "@other_in"),
    ],
)
def test_each_capped_list_reads_the_lookbacks_payments_but_self_payments(
    target: str, alias: str, edges: list[set[str]], heap: str
) -> None:
    """Every payment of the lookback, of either kind, one way, goes into the account's heap
    with its second, its id, the account at its other end and whether it is a Zelle transfer.
    A payment whose other end is the account itself is read nowhere."""
    select = select_of(DEFINITION, target)
    assert [step.alias for step in select.vertices] == [alias, "p", "b"]
    assert select.vertices[0].chosen and select.vertices[1].types == PAYMENTS
    assert select.vertices[2].types == {"Account"}
    assert [set(edge.types) for edge in select.edges] == edges
    assert select.conjuncts == WINDOW | {f"b != {alias}"}
    assert accumulation_text(DEFINITION, target) == normal(
        f"""{alias}.{heap} += Picked(floor(p.event_ts_ms / 1000), p.id, b.id,
          p.type == "Zelle_Transfer")"""
    )


def test_the_caps_keep_the_newest_by_second_and_break_ties_by_id() -> None:
    """Every heap orders by the second of the payment, newest first, and then by id, so of
    two payments of one second only the id decides which is kept: a pseudonym, never the
    row order. By hand: of 33 payments sent in the lookback, the oldest two in one second,
    2024-03-01 09:00:00, with ids p_a and p_b, the heap of 32 keeps p_a and drops p_b; with a
    millisecond apart in that second they are still ordered by id."""
    heaps = [text for text in DECLARATIONS if text.startswith("HeapAccum")]
    assert len(heaps) == 3
    assert all("sec DESC, id ASC)" in text for text in heaps)
    # The second is the payment's milliseconds rounded down, and the only order of a heap.
    code = normal(DEFINITION)
    assert code.count("Picked ( floor ( p.event_ts_ms / 1000 ) , p.id ,") == 4
    assert "Sibling ( floor ( b.@last_active / 1000 ) , b.id )" in code


# The hop-1 and hop-2 accounts.


def test_the_scored_accounts_payments_join_its_subgraph_with_their_ends() -> None:
    """The scored account, each of its kept payments and the account at the other end of
    each join its subgraph; each such account is a hop-1 account of it, and the payment is a
    transaction or a transfer by its kind. By hand: a scored a_01 that paid c_01 twice and was
    paid by c_02 has the accounts a_01, c_01 and c_02, three payments, and is the scored
    account c_01 and c_02 were reached by."""
    picked = """@@subgraph_payments += (a.id -> x.id),
        @@subgraph_accounts += (a.id -> x.other),
        @@account_ids += x.other,
        @@hop1_ids += x.other,
        @@reached_by += (x.other -> a.id),
        IF x.zelle THEN @@transfer_ids += x.id ELSE @@transaction_ids += x.id END"""
    assert accumulation_text(DEFINITION, "ScoredPicked") == normal(
        f"""@@subgraph_accounts += (a.id -> a.id),
        @@account_ids += a.id,
        FOREACH x IN a.@scored_out DO {picked} END,
        FOREACH x IN a.@scored_in DO {picked} END"""
    )


def test_only_hop1_accounts_that_are_no_hub_are_expanded() -> None:
    """A hop-1 account hub is in the subgraph with the payments it shares with the scored
    account and nothing more; every other hop-1 account adds its own capped payments and their
    other ends, the hop-2 accounts, to the subgraph of every scored account it is a hop-1
    account of. By hand: c_01, a hop-1 account of a_01 and a_02, adds its 16 newest payments
    sent and their payees to both subgraphs; the biller h_01, a hub paid by a_01, adds
    nothing."""
    hop1 = select_of(DEFINITION, "Hop1")
    assert hop1.conjuncts == {"c.opened_ms < t_ms", "NOT c.@hub"}
    assert [step.alias for step in hop1.vertices] == ["c"]
    picked = """FOREACH i IN @@reached_by.get(c.id) DO
            @@subgraph_payments += (i -> x.id),
            @@subgraph_accounts += (i -> x.other)
        END,
        @@account_ids += x.other,
        IF x.zelle THEN @@transfer_ids += x.id ELSE @@transaction_ids += x.id END"""
    assert accumulation_text(DEFINITION, "Hop1Picked") == normal(
        f"""FOREACH x IN c.@other_out DO {picked} END,
        FOREACH x IN c.@other_in DO {picked} END"""
    )
    # The expanded accounts are the only ones but the scored accounts that a SELECT starts
    # from, to read their payments and their owners; the hop-2 accounts are never expanded.
    assert [
        select.statement[0]
        for _, _, select in scopes(DEFINITION)
        if select is not None and _source(select.statement) == "Hop1"
    ] == ["Hop1Sent", "Hop1Received", "Hop1Picked", "ExpandedOwners"]


# The other accounts of the scored account's customer.


VISIBLE = {
    "o.valid_from_ms < t_ms",
    "( o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms )",
}
BOTH_VISIBLE = VISIBLE | {
    "w.valid_from_ms < t_ms",
    "( w.valid_to_ms == 0 OR w.valid_to_ms > t_ms - lookback_ms )",
}
OWNED = ["Account_Owned_By_Party", "Party_Owns_Account"]


@pytest.mark.parametrize("target", ["Siblings", "SiblingsRanked", "CustomerAccounts"])
def test_a_customers_accounts_are_read_through_visible_ownership(target: str) -> None:
    """The scored account's customer owns it through a visible ownership link, and the
    customer's accounts are those it owns through one too that exist at t, open or closed:
    an ownership that ended more than 90 days before t, or began at t, is none."""
    select = select_of(DEFINITION, target)
    assert [step.alias for step in select.vertices] == ["a", "q", "b"]
    assert [step.types for step in select.vertices] == [{"Account"}, {"Party"}, {"Account"}]
    assert [sorted(edge.types) for edge in select.edges] == [[name] for name in OWNED]
    assert [edge.alias for edge in select.edges] == ["o", "w"]
    others = {"b != a"} if target != "CustomerAccounts" else set()
    assert select.conjuncts == BOTH_VISIBLE | {"b.opened_ms < t_ms"} | others


def test_the_newest_8_other_accounts_by_their_latest_payment_then_by_id() -> None:
    """Each other account's latest payment in the lookback, either way, but self-payments,
    orders it; one with no payment there has 0 and comes after every one with one, ties by
    id. By hand: a customer with accounts a_01 (scored), b_02 last paying on 2024-03-30, b_03
    last paid on 2024-01-15 and b_04 with no payment in the lookback picks b_02, b_03, b_04
    in that order, and with 9 accounts besides a_01 drops the one of them with no payment and
    the highest id."""
    active = select_of(DEFINITION, "Active")
    assert [step.alias for step in active.vertices] == ["b", "p", "x"]
    assert active.vertices[0].chosen
    assert [set(edge.types) for edge in active.edges] == [
        {
            "Account_Initiated_Transaction",
            "Account_Received_Transaction",
            "Account_Sent_Zelle_Transfer",
            "Account_Received_Zelle_Transfer",
        },
        {
            "Transaction_From_Account",
            "Transaction_To_Account",
            "Transfer_From_Account",
            "Transfer_To_Account",
        },
    ]
    assert active.conjuncts == WINDOW | {"x != b"}
    assert accumulation_text(DEFINITION, "Active") == "b.@last_active += p.event_ts_ms"
    assert accumulation_text(DEFINITION, "SiblingsRanked") == normal(
        "a.@siblings += Sibling(floor(b.@last_active / 1000), b.id)"
    )
    assert accumulation_text(DEFINITION, "SiblingsPicked") == normal(
        """FOREACH x IN a.@siblings DO
            @@subgraph_accounts += (a.id -> x.id),
            @@account_ids += x.id
        END"""
    )


# The ownership edges.


def test_ownership_edges_join_expanded_accounts_and_the_customers_accounts() -> None:
    """Each expanded hop-1 account joins its customer in every subgraph it is a hop-1 account
    of, and every account of a subgraph that the scored account's customer owns, the scored
    account among them, joins that customer; no other ownership edge is fetched. By hand: a
    scored a_01 of customer q_01 that paid its own savings account b_02 and c_01 of customer
    q_09 has the edges (a_01, q_01), (b_02, q_01) and (c_01, q_09); b_02 is both a hop-1
    account and an other account of q_01, and has one edge."""
    owners = select_of(DEFINITION, "ExpandedOwners")
    assert [step.alias for step in owners.vertices] == ["c", "q"]
    assert owners.vertices[0].chosen
    assert [sorted(edge.types) for edge in owners.edges] == [["Account_Owned_By_Party"]]
    assert owners.conjuncts == VISIBLE
    assert accumulation_text(DEFINITION, "ExpandedOwners") == normal(
        """FOREACH i IN @@reached_by.get(c.id) DO
            @@owned_by += (i -> OwnedBy(c.id, q.id))
        END"""
    )
    assert accumulation_text(DEFINITION, "CustomerAccounts") == normal(
        """IF @@subgraph_accounts.get(a.id).contains(b.id) THEN
            @@owned_by += (a.id -> OwnedBy(b.id, q.id))
        END"""
    )
    # Every subgraph is complete before its ownership edges are read from it.
    order = SELECT_ORDER
    assert order.index("CustomerAccounts") > max(
        order.index(name) for name in ("ScoredPicked", "Hop1Picked", "SiblingsPicked")
    )


# The accounts and the payments printed.


def test_every_account_and_payment_printed_exists_at_t_and_lies_in_the_lookback() -> None:
    assert select_of(DEFINITION, "Accounts").conjuncts == {"a.opened_ms < t_ms"}
    for target in ("Transactions", "Transfers"):
        assert select_of(DEFINITION, target).conjuncts == WINDOW, target


def test_each_payment_has_its_cents_rail_and_age() -> None:
    """Cents are floor(amount x 100 + 0.5) in INT; the rail is the payment's, or zelle for a
    Zelle transfer; the age is t minus its time. By hand: USD 662.005 at 2024-03-31 21:00 is
    66,201 cents with an age of 3 hours, 10,800,000 ms."""
    for target, rail in (("Transactions", "p.payment_rail"), ("Transfers", '"zelle"')):
        assert accumulation_text(DEFINITION, target) == normal(
            f"""INT cents = floor(p.amount * 100 + 0.5),
            p.@amount += cents,
            p.@rail += {rail},
            p.@age_ms += t_ms - p.event_ts_ms"""
        ), target
        assert declared_locals(DEFINITION, target) == [("INT", "cents")]
    assert select_of(DEFINITION, "Transactions").vertices[0].types == {"Payment_Transaction"}
    assert select_of(DEFINITION, "Transfers").vertices[0].types == {"Zelle_Transfer"}


@pytest.mark.parametrize(
    ("target", "edge", "end"),
    [
        ("Senders", {"Transaction_From_Account", "Transfer_From_Account"}, "a"),
        ("Receivers", {"Transaction_To_Account", "Transfer_To_Account"}, "b"),
    ],
)
def test_each_payment_knows_its_ends_and_whether_each_is_a_hub(
    target: str, edge: set[str], end: str
) -> None:
    select = select_of(DEFINITION, target)
    assert [step.alias for step in select.vertices] == ["p", end]
    assert [set(step.types) for step in select.edges] == [edge]
    assert not select.conjuncts
    role = "sender" if end == "a" else "receiver"
    assert accumulation_text(DEFINITION, target) == normal(
        f"p.@{role}_id += {end}.id, p.@{role}_hub += {end}.@hub"
    )


def test_the_receivers_earlier_payments_give_its_gap_and_the_pairs() -> None:
    """Unless the receiver is a hub, every payment it received in [t - 90 days, ts), but
    self-payments, gives its latest, and those from the same sender count the pair. Strictly
    earlier, so a payment does not count itself. By hand: a mule a_01 paid by v_01 at 10:00
    and 10:20 and by v_02 at 10:05 on 2024-03-30: the 10:20 payment has a receiver gap of 15
    minutes, 900,000 ms, one earlier pair payment, and a pair gap and pair age of 20 minutes,
    1,200,000 ms."""
    select = select_of(DEFINITION, "ReceiverHistory")
    assert [step.alias for step in select.vertices] == ["p", "b", "q", "a"]
    assert select.vertices[0].chosen and select.vertices[2].types == PAYMENTS
    assert [set(edge.types) for edge in select.edges] == [
        {"Transaction_To_Account", "Transfer_To_Account"},
        {"Account_Received_Transaction", "Account_Received_Zelle_Transfer"},
        {"Transaction_From_Account", "Transfer_From_Account"},
    ]
    assert select.conjuncts == {
        "NOT p.@receiver_hub",
        "q.event_ts_ms >= t_ms - lookback_ms",
        "q.event_ts_ms < p.event_ts_ms",
        "a != b",
    }
    assert accumulation_text(DEFINITION, "ReceiverHistory") == normal(
        """p.@receiver_last += q.event_ts_ms,
        IF a.id == p.@sender_id THEN
            p.@pair_prior_count += 1,
            p.@pair_first += q.event_ts_ms,
            p.@pair_last += q.event_ts_ms
        END"""
    )


def test_the_senders_earlier_payments_give_its_gap_and_the_pairs_of_a_hub_receiver() -> None:
    """Unless the sender is a hub, every payment it sent in [t - 90 days, ts), but
    self-payments, gives its latest; when the receiver is a hub, whose own 90 days are not
    read, those to it count the pair, even from a hub sender. By hand: a_01 paid the biller
    h_01 on 2024-02-01 and 2024-03-01 and a friend on 2024-03-10: the payment of 2024-03-01
    has a sender gap of 29 days and one earlier pair payment, 29 days before it."""
    select = select_of(DEFINITION, "SenderHistory")
    assert [step.alias for step in select.vertices] == ["p", "a", "q", "b"]
    assert select.vertices[0].chosen and select.vertices[2].types == PAYMENTS
    assert [set(edge.types) for edge in select.edges] == [
        {"Transaction_From_Account", "Transfer_From_Account"},
        {"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"},
        {"Transaction_To_Account", "Transfer_To_Account"},
    ]
    assert select.conjuncts == {
        "( NOT p.@sender_hub OR p.@receiver_hub )",
        "q.event_ts_ms >= t_ms - lookback_ms",
        "q.event_ts_ms < p.event_ts_ms",
        "b != a",
    }
    assert accumulation_text(DEFINITION, "SenderHistory") == normal(
        """IF NOT p.@sender_hub THEN p.@sender_last += q.event_ts_ms END,
        IF p.@receiver_hub AND b.id == p.@receiver_id THEN
            p.@pair_prior_count += 1,
            p.@pair_first += q.event_ts_ms,
            p.@pair_last += q.event_ts_ms
        END"""
    )


def test_each_gap_is_absent_with_no_earlier_payment() -> None:
    """A gap is the payment's time minus the latest earlier one, and the pair's age minus the
    earliest; each is present only with one. An earlier payment lies in the lookback, so after
    the epoch, and 0 says there was none."""
    assert accumulation_text(DEFINITION, "Rows") == normal(
        """IF p.@pair_prior_count > 0 THEN
            p.@pair_gap_present += TRUE,
            p.@pair_gap_ms += p.event_ts_ms - p.@pair_last,
            p.@pair_age_present += TRUE,
            p.@pair_age_ms += p.event_ts_ms - p.@pair_first
        END,
        IF p.@sender_last > 0 THEN
            p.@sender_gap_present += TRUE,
            p.@sender_gap_ms += p.event_ts_ms - p.@sender_last
        END,
        IF p.@receiver_last > 0 THEN
            p.@receiver_gap_present += TRUE,
            p.@receiver_gap_ms += p.event_ts_ms - p.@receiver_last
        END"""
    )


def test_no_bound_at_t_takes_t_itself() -> None:
    code = normal(DEFINITION)
    # t itself is no bound: no `<= t_ms`, `t_ms >=` or `t_ms >` but the test of t.
    assert not re.search(r"<= t_ms\b|(?<![\w.@])t_ms >(?!= lookback)(?! lookback)", code)
    assert code.count("p.event_ts_ms < t_ms") == 7
    assert code.count("p.event_ts_ms >= t_ms - lookback_ms") == 7
    assert code.count("q.event_ts_ms < p.event_ts_ms") == 2
    assert code.count("q.event_ts_ms >= t_ms - lookback_ms") == 2
    # Every window is the lookback: no 30-day or 7-day window.
    assert not re.search(r"t_ms - \d", code)


def test_it_reads_no_label_no_device_and_no_link_end_but_the_visible_test() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(r"joined_ms|closed_ms|account_type|Device|IP\b|Token", code)
    assert len(re.findall(r"\w+\.valid_to_ms", code)) == 2 * 7
    assert sorted(set(re.findall(r"\w+\.opened_ms", code))) == [
        "a.opened_ms",
        "b.opened_ms",
        "c.opened_ms",
        "h.opened_ms",
        "s.opened_ms",
    ]
    assert re.findall(r"\w+\.is_external", code) == ["Accounts.is_external"]
