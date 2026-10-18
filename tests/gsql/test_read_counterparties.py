"""read_counterparties, B2, read as text (docs/design.md, "read_counterparties (B2): who pays
it and whom it pays", and docs/features.md, "Account node", columns 22 to 33).

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclass its parser reads. Beside those rules, each column
is held here clause by clause, since no reference graph runs it yet: the payments it reads,
the payers and payees and when each first paid, the split by is_external, the shares, the hub
parameter as the only reading of hubs, and the whole-second cut and window of
max_new_payers_24h. gsql-lsp checks the file against the schema in tests/gsql/test_schema.py,
and the live compile test compiles it.
"""

from __future__ import annotations

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
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    QUERY_FILES,
    READ_COUNTERPARTIES,
    REGISTRY,
    VERSIONS,
    read_counterparties,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_counterparties import Counterparties
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, MONDAY_MS, WEEK_MS

TEXT = (GSQL_DIR / QUERY_FILES[READ_COUNTERPARTIES]).read_text()
DEFINITION = definitions(TEXT)[READ_COUNTERPARTIES]
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The windows of docs/features.md, columns 22 to 33, in milliseconds: new over 7 and 30 days,
# the 30 days of the counts and shares, the 90-day lookback (lookback_ms), and the 24 hours of
# max_new_payers_24h.
DAY_MS = 86_400_000
WINDOW_7D_MS = 7 * DAY_MS  # 604,800,000
WINDOW_30D_MS = 30 * DAY_MS  # 2,592,000,000
WINDOW_24H_MS = DAY_MS  # 86,400,000
# max_new_payers_24h keeps the newest recent_payments (256) first-payment times, and its heap
# holds one more, the newest left out.
LIST = SUBGRAPH.recent_payments
HEAP = LIST + 1


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_COUNTERPARTIES] in written_files().values()
    assert common_problems(READ_COUNTERPARTIES, TEXT) == []
    assert version_rule(READ_COUNTERPARTIES, DEFINITION, VERSIONS[READ_COUNTERPARTIES]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    built = read_counterparties.params(T_MS, ["a_01", "a_02"], ["h_01"])
    assert parameter_rule(READ_COUNTERPARTIES, DEFINITION, built) == []
    # The accounts as ids, which the query checks itself; the account hubs as a set of
    # vertices, of which an id the graph lacks can be no payee.
    assert parameters(DEFINITION) == {
        "t_ms": ["UINT"],
        "account_ids": ["LIST", "<", "STRING", ">"],
        "hub_accounts": ["SET", "<", "VERTEX", "<", "Account", ">", ">"],
    }
    assert built["hub_accounts"] == [{"id": "h_01"}]


def test_each_printed_row_has_the_fields_of_its_dataclass() -> None:
    assert (
        field_rule(
            "read_counterparties Rows", projection_fields(DEFINITION, "Rows"), Counterparties
        )
        == []
    )
    assert source_rule("read_counterparties Rows", projection_items(DEFINITION, "Rows")) == []
    # The status row prints the status and the version alone, and the rows are its only other
    # PRINT.
    assert status_fields(DEFINITION) == []
    printed = [" ".join(s[:3]) for s in statements(DEFINITION) if s[0] == "PRINT"]
    assert printed[-2:] == ['PRINT "ok" AS', "PRINT Rows ["]


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_COUNTERPARTIES]
    assert query.params is read_counterparties.params
    assert query.rows == (Counterparties,)


# What the query declares, exactly, in order. Every cent is INT, never mixed with UINT; the
# heap's capacity is one more than the time-ordered cap, and its order keeps the newest.
DECLARATIONS = [
    "TYPEDEF TUPLE<UINT ts, STRING id> NewPayer",
    "OrAccum @hub",
    "MapAccum<STRING, MinAccum<UINT>> @payer_first, @payee_first",
    "MapAccum<STRING, SumAccum<INT>> @payer_cents, @payee_cents",
    "SetAccum<STRING> @external_payers",
    "SumAccum<INT> @received_30d, @sent_30d, @sent_to_hubs_30d",
    f"HeapAccum<NewPayer>({HEAP}, ts DESC, id DESC) @new_payers",
    "SumAccum<INT> @payers_30d, @payees_30d, @new_internal_payers_7d, @new_internal_payers_30d, "
    "@new_external_payers_7d, @new_external_payers_30d, @new_payees_7d, @new_payees_30d, "
    "@max_new_payers_24h",
    "SumAccum<DOUBLE> @new_payer_share_30d, @top_payee_share_30d, @hub_payee_share_30d",
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
    assert (HEAP, LIST) == (257, 256)


# The refusals.


def test_every_refusal_returns_at_once() -> None:
    """A refusal that went on would read with t at or below the lookback, where
    t_ms - lookback_ms wraps in UINT, answer a request of no account, of an account twice or
    of an account the graph lacks with fewer rows than it asked for, read accounts that do not
    exist at t, or read as hubs accounts that were none at t."""
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
    accounts that passed every check are read."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    assert found[start : start + 25] == [
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
    ]
    assert found[start + 25] == "Marked = SELECT h FROM HubAccounts : h POST-ACCUM h.@hub += TRUE"
    # Every SELECT that reads payments or writes the row starts from the checked accounts.
    starts = [
        select.vertices[0]
        for _, _, select in scopes(DEFINITION)
        if select is not None and select.statement[0] in ("Received", "Sent", "Counted", "Rows")
    ]
    assert {(step.alias, step.chosen) for step in starts} == {("a", True)}
    assert normal(DEFINITION).count("FROM Scored : a") == 4


# The account hubs.


def test_the_hub_parameter_is_the_only_hub_input() -> None:
    """A hub is an account of hub_accounts, read_hubs' account hubs of the same t: the query
    counts no customer, reads no degree and has no threshold of its own. The flag is set once,
    on the hubs that exist at t, and read once, on the payee of a payment of the last 30
    days."""
    code = normal(DEFINITION)
    assert code.count("hub_accounts") == 2  # the parameter and Hubs = {hub_accounts}
    assert re.findall(r"\S*@hub\b\S*", code) == ["@hub", "h.@hub", "b.@hub"]
    assert "h.@hub += TRUE" in accumulation_text(DEFINITION, "Marked")
    assert "IF b.@hub THEN a.@sent_to_hubs_30d += cents END" in accumulation_text(
        DEFINITION, "Sent"
    )
    assert not re.search(r"outdegree|threshold|crowd|leaf|Party|customers", code)


# The order of the SELECTs. A value a step reads is read in a later SELECT than the one that
# wrote it, so the order is pinned whole, and the rule that explains it is checked on the text
# and on text moves.
SELECT_ORDER = ["Scored", "HubAccounts", "Marked", "Received", "Sent", "Counted", "Rows"]


def test_the_selects_run_in_the_order_the_columns_need() -> None:
    assert [target for target, _, _ in select_flows(DEFINITION)] == SELECT_ORDER


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    assert dataflow_problems(DEFINITION) == []
    reads = {target: read for target, _, read in select_flows(DEFINITION)}
    assert reads["Sent"] == {"hub"}
    assert reads["Counted"] == {
        "payer_first",
        "payee_first",
        "payer_cents",
        "payee_cents",
        "external_payers",
        "received_30d",
        "sent_30d",
        "sent_to_hubs_30d",
    }
    assert reads["Rows"] == {"new_payers"}


@pytest.mark.parametrize(
    ("target", "after", "problem"),
    [
        # Hubs marked after the payments are read: hub_payee_share_30d reads 0 everywhere.
        ("Marked", "Sent", "Sent reads @hub, written by Marked"),
        # The counts before the payments: every count and share reads empty maps.
        ("Received", "Counted", "Counted reads @payer_first, written by Received"),
        # The window before the heap is filled: max_new_payers_24h reads 0 everywhere.
        ("Counted", "Rows", "Rows reads @new_payers, written by Counted"),
    ],
)
def test_a_select_moved_before_the_one_it_reads_from_breaks_the_rule(
    target: str, after: str, problem: str
) -> None:
    moved = moved_select(DEFINITION, target, after)
    assert moved != DEFINITION
    assert problem in dataflow_problems(moved)


# The payments it reads.


WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms", "b != a"}
PAYMENTS = {"Payment_Transaction", "Zelle_Transfer"}


@pytest.mark.parametrize(
    ("target", "edges", "accumulations"),
    [
        (
            "Received",
            [
                {"Account_Received_Transaction", "Account_Received_Zelle_Transfer"},
                {"Transaction_From_Account", "Transfer_From_Account"},
            ],
            f"""INT cents = floor(p.amount * 100 + 0.5),
            a.@payer_first += (b.id -> p.event_ts_ms),
            IF b.is_external THEN a.@external_payers += b.id END,
            IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
                a.@payer_cents += (b.id -> cents),
                a.@received_30d += cents
            END""",
        ),
        (
            "Sent",
            [
                {"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"},
                {"Transaction_To_Account", "Transfer_To_Account"},
            ],
            f"""INT cents = floor(p.amount * 100 + 0.5),
            a.@payee_first += (b.id -> p.event_ts_ms),
            IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
                a.@payee_cents += (b.id -> cents),
                a.@sent_30d += cents,
                IF b.@hub THEN a.@sent_to_hubs_30d += cents END
            END""",
        ),
    ],
)
def test_it_reads_the_lookbacks_payments_each_way_but_self_payments(
    target: str, edges: list[set[str]], accumulations: str
) -> None:
    """Every payment of the lookback, of either kind, dates its other end's first payment with
    the account; those of the last 30 days count the other end and its cents. A payment whose
    other end is the account itself is read nowhere."""
    select = select_of(DEFINITION, target)
    assert [step.alias for step in select.vertices] == ["a", "p", "b"]
    assert select.vertices[0].chosen and select.vertices[1].types == PAYMENTS
    assert select.vertices[2].types == {"Account"}
    assert [set(edge.types) for edge in select.edges] == edges
    assert select.conjuncts == WINDOW
    assert accumulation_text(DEFINITION, target) == normal(accumulations)


# The columns.


def test_new_payers_and_payees_date_their_first_payment_in_the_lookback() -> None:
    """A payer is new over w when its earliest payment in the lookback is at or after t - w,
    and it splits by its is_external. By hand, at t = 2024-04-01: a payer whose payments fell
    on 2024-01-20 and 2024-03-28 first paid 72 days before t, inside the lookback but before
    t - 30 days, so it is no new payer though it paid in the last 7 days; one that first paid
    on 2024-03-28, 4 days before t, is new over 7 and over 30 days; one whose first payment is
    exactly at t - 30 days, 2024-03-02 00:00, is new over 30 days and not over 7."""
    code = accumulation_text(DEFINITION, "Counted")
    assert code.startswith(normal("INT new_cents = 0, INT top_cents = 0,"))
    payers = f"""FOREACH (payer, first_ms) IN a.@payer_first DO
        IF first_ms >= t_ms - {WINDOW_30D_MS} THEN
            a.@new_payers += NewPayer(first_ms, payer),
            new_cents = new_cents + a.@payer_cents.get(payer),
            IF a.@external_payers.contains(payer) THEN
                a.@new_external_payers_30d += 1,
                IF first_ms >= t_ms - {WINDOW_7D_MS} THEN a.@new_external_payers_7d += 1 END
            ELSE
                a.@new_internal_payers_30d += 1,
                IF first_ms >= t_ms - {WINDOW_7D_MS} THEN a.@new_internal_payers_7d += 1 END
            END
        END
    END,"""
    payees = f"""FOREACH (payee, first_ms) IN a.@payee_first DO
        IF first_ms >= t_ms - {WINDOW_30D_MS} THEN a.@new_payees_30d += 1 END,
        IF first_ms >= t_ms - {WINDOW_7D_MS} THEN a.@new_payees_7d += 1 END
    END,"""
    assert normal(payers) in code
    assert normal(payees) in code
    # The first payment is a minimum over the lookback, by the payer's id.
    assert "MapAccum<STRING, MinAccum<UINT>> @payer_first, @payee_first" in DECLARATIONS


def test_the_counts_and_shares_of_the_last_30_days() -> None:
    """payers_30d and payees_30d count the keys of the 30-day cent maps; each share divides
    two sums of cents once, and is 0 with an empty denominator. By hand: 10,000 cents received
    in the last 30 days, 7,500 of them from two payers new over 30 days, give
    new_payer_share_30d 7,500 / 10,000 = 0.75; 66,200 cents sent to one payee and 3,800 to a
    hub biller give top_payee_share_30d 66,200 / 70,000 = 0.9457... and hub_payee_share_30d
    3,800 / 70,000 = 0.0542...."""
    expected = """FOREACH (payee, cents) IN a.@payee_cents DO
        IF cents > top_cents THEN top_cents = cents END
    END,
    a.@payers_30d += a.@payer_cents.size(),
    a.@payees_30d += a.@payee_cents.size(),
    IF a.@received_30d > 0 THEN
        a.@new_payer_share_30d += new_cents * 1.0 / a.@received_30d
    END,
    IF a.@sent_30d > 0 THEN
        a.@top_payee_share_30d += top_cents * 1.0 / a.@sent_30d,
        a.@hub_payee_share_30d += a.@sent_to_hubs_30d * 1.0 / a.@sent_30d
    END"""
    assert accumulation_text(DEFINITION, "Counted").endswith(normal(expected))


def test_max_new_payers_cuts_whole_seconds_and_counts_every_pair() -> None:
    """With more than 256 new payers the heap holds 257, and every first-payment time at or
    before the second of its oldest is dropped; the most kept times in [x, x + 24 hours) with
    x one of them, every pair compared, so no order of the heap's walk decides it. By hand:
    first payments at 10:00:00, 10:00:00 (two payers of one second), 21:00:00 and 10:00:00 the
    next day give 3 at x = 10:00:00 (the next day's 10:00:00 is exactly 24 hours later and
    out); 258 new payers, the two oldest in one second b, keep the 256 after b."""
    assert accumulation_text(DEFINITION, "Rows") == normal(
        f"""INT cut = -1, INT within = 0, INT most = 0,
        IF a.@new_payers.size() > {LIST} THEN
            FOREACH x IN a.@new_payers DO
                IF cut < 0 OR floor(x.ts / 1000) < cut THEN cut = floor(x.ts / 1000) END
            END
        END,
        FOREACH x IN a.@new_payers DO
            IF floor(x.ts / 1000) > cut THEN
                within = 0,
                FOREACH y IN a.@new_payers DO
                    IF floor(y.ts / 1000) > cut AND y.ts >= x.ts AND y.ts < x.ts + {WINDOW_24H_MS}
                    THEN
                        within = within + 1
                    END
                END,
                IF within > most THEN most = within END
            END
        END,
        a.@max_new_payers_24h += most"""
    )


def test_no_value_reads_the_order_of_a_heaps_walk() -> None:
    """The heap is filled once, its size read once, and walked only by the three FOREACH of
    Rows, which read its times as a set: no index counts its steps and no list copies it."""
    code = normal(DEFINITION)
    assert re.findall(r"a\.@new_payers\S*(?: \S+){0,2}", code) == [
        "a.@new_payers += NewPayer",
        "a.@new_payers.size ( )",
        "a.@new_payers DO IF",
        "a.@new_payers DO IF",
        "a.@new_payers DO IF",
    ]
    assert "ListAccum" not in code


def test_every_cent_and_count_of_the_query_is_int() -> None:
    """Mixing INT with UINT turns arithmetic into UINT, where a value below 0 wraps; every
    local is INT but the times the heap and the maps hold, which stay UINT."""
    for target in ("Received", "Sent"):
        assert declared_locals(DEFINITION, target) == [("INT", "cents")], target
    assert declared_locals(DEFINITION, "Counted") == [("INT", "new_cents"), ("INT", "top_cents")]
    assert declared_locals(DEFINITION, "Rows") == [
        ("INT", "cut"),
        ("INT", "within"),
        ("INT", "most"),
    ]


def test_no_bound_at_t_takes_t_itself() -> None:
    code = normal(DEFINITION)
    # t itself is no bound: no `<= t_ms`, `t_ms >=` or `t_ms >` but the test of t.
    assert not re.search(r"<= t_ms\b|(?<![\w.@])t_ms >(?!= lookback)(?! lookback)", code)
    assert code.count("p.event_ts_ms < t_ms") == 2
    assert code.count("p.event_ts_ms >= t_ms - lookback_ms") == 2


def test_it_reads_no_label_and_no_link() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(r"valid_from_ms|valid_to_ms|joined_ms|closed_ms|account_type", code)
    assert re.findall(r"\w+\.opened_ms", code) == ["s.opened_ms", "h.opened_ms"]
    assert re.findall(r"\w+\.is_external", code) == ["b.is_external"]
