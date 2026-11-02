"""read_activity_change, B3, read as text (docs/design.md, "read_activity_change (B3): the
account against its own past", and docs/features.md, "Account node", columns 34 to 46).

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclass its parser reads. Beside those rules, each column
is held here clause by clause, since no reference graph runs it yet: the payments it reads and
their counts, the whole-second cut of each time-ordered list, the bursts and the shortest gap
over every pair of kept times, and the time since the latest payment each way. gsql-lsp checks
the file against the schema in tests/gsql/test_schema.py, and the live compile test compiles
it.
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
    READ_ACTIVITY_CHANGE,
    REGISTRY,
    VERSIONS,
    read_activity_change,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_activity_change import ActivityChange
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS

TEXT = (GSQL_DIR / QUERY_FILES[READ_ACTIVITY_CHANGE]).read_text()
DEFINITION = definitions(TEXT)[READ_ACTIVITY_CHANGE]
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The windows of docs/features.md, columns 34 to 46, in milliseconds: counts over 7, 30 and
# 90 days (the 90 days are lookback_ms), and bursts in an hour and in 24 hours.
DAY_MS = 86_400_000
WINDOW_7D_MS = 7 * DAY_MS  # 604,800,000
WINDOW_30D_MS = 30 * DAY_MS  # 2,592,000,000
WINDOW_1H_MS = 60 * 60 * 1000  # 3,600,000
WINDOW_24H_MS = DAY_MS  # 86,400,000
# Each list keeps the newest recent_payments (256), and its heap holds one more.
LIST = SUBGRAPH.recent_payments
HEAP = LIST + 1


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_ACTIVITY_CHANGE] in written_files().values()
    assert common_problems(READ_ACTIVITY_CHANGE, TEXT) == []
    assert version_rule(READ_ACTIVITY_CHANGE, DEFINITION, VERSIONS[READ_ACTIVITY_CHANGE]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    built = read_activity_change.params(T_MS, ["a_01", "a_02"])
    assert parameter_rule(READ_ACTIVITY_CHANGE, DEFINITION, built) == []
    # Ids, not a SET<VERTEX>, from which an installed query drops an unknown id unseen.
    assert parameters(DEFINITION) == {"t_ms": ["UINT"], "account_ids": ["LIST", "<", "STRING", ">"]}


def test_each_printed_row_has_the_fields_of_its_dataclass() -> None:
    printed = projection_fields(DEFINITION, "Rows")
    assert field_rule("read_activity_change Rows", printed, ActivityChange) == []
    assert source_rule("read_activity_change Rows", projection_items(DEFINITION, "Rows")) == []
    # The status row prints the status and the version alone, and the rows are its only other
    # PRINT.
    assert status_fields(DEFINITION) == []
    printed = [" ".join(s[:3]) for s in statements(DEFINITION) if s[0] == "PRINT"]
    assert printed[-2:] == ['PRINT "ok" AS', "PRINT Rows ["]


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_ACTIVITY_CHANGE]
    assert query.params is read_activity_change.params
    assert query.rows == (ActivityChange,)


# What the query declares, exactly, in order. Counts are INT; times and the durations taken
# from them stay UINT, as the graph holds them; each heap holds one more than its list.
DECLARATIONS = [
    "TYPEDEF TUPLE<UINT ts, STRING id> Paid",
    "SumAccum<INT> @in_7d, @in_30d, @in_90d, @out_7d, @out_30d, @out_90d",
    "MaxAccum<UINT> @last_in, @last_out",
    f"HeapAccum<Paid>({HEAP}, ts DESC, id DESC) @inflows, @outflows",
    "SumAccum<INT> @burst_in_1h, @burst_in_24h, @burst_out_1h, @burst_out_24h",
    "OrAccum @shortest_in_gap_present, @since_last_in_present, @since_last_out_present",
    "SumAccum<UINT> @shortest_in_gap_ms, @since_last_in_ms, @since_last_out_ms",
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
    of an account the graph lacks with fewer rows than it asked for, or read accounts that do
    not exist at t."""
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
    ]


def test_t_is_a_whole_second_after_the_lookback_and_each_account_is_known_once_and_exists() -> None:
    """t is checked first; then the ids: at least one, none twice (a list keeps a repeat, a
    set would drop it), each an account of the graph (to_vertex_set drops an id it lacks) and
    each existing at t. Only the accounts that passed every check are read."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    assert found[start : start + 18] == [
        'IF t_ms <= lookback_ms OR t_ms % 1000 != 0 THEN PRINT "invalid_parameters" AS status',
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
    ]
    assert select_of(DEFINITION, "Scored").conjuncts == {"s.opened_ms < t_ms"}
    # Every later SELECT starts from the checked accounts.
    starts = [
        select.vertices[0]
        for _, _, select in scopes(DEFINITION)
        if select is not None and select.statement[0] != "Scored"
    ]
    assert len(starts) == 3
    assert {(step.alias, step.chosen) for step in starts} == {("a", True)}
    assert normal(DEFINITION).count("FROM Scored : a") == 3


# The order of the SELECTs. A value a step reads is read in a later SELECT than the one that
# wrote it, so the order is pinned whole, and the rule that explains it is checked on the text
# and on text moves.
SELECT_ORDER = ["Scored", "Received", "Sent", "Rows"]


def test_the_selects_run_in_the_order_the_columns_need() -> None:
    assert [target for target, _, _ in select_flows(DEFINITION)] == SELECT_ORDER


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    assert dataflow_problems(DEFINITION) == []
    reads = {target: read for target, _, read in select_flows(DEFINITION)}
    assert reads["Received"] == reads["Sent"] == set()
    assert reads["Rows"] == {"inflows", "outflows", "in_90d", "out_90d", "last_in", "last_out"}


@pytest.mark.parametrize(
    ("target", "after", "problem"),
    [
        # The row before the inflows are read: every in burst is 0, no gap, no time since.
        ("Received", "Rows", "Rows reads @inflows, written by Received"),
        # The row before the outflows are read: every out burst is 0 and no time since.
        ("Sent", "Rows", "Rows reads @last_out, written by Sent"),
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
    ("target", "edges", "way", "heap"),
    [
        (
            "Received",
            [
                {"Account_Received_Transaction", "Account_Received_Zelle_Transfer"},
                {"Transaction_From_Account", "Transfer_From_Account"},
            ],
            "in",
            "inflows",
        ),
        (
            "Sent",
            [
                {"Account_Initiated_Transaction", "Account_Sent_Zelle_Transfer"},
                {"Transaction_To_Account", "Transfer_To_Account"},
            ],
            "out",
            "outflows",
        ),
    ],
)
def test_it_counts_the_lookbacks_payments_each_way_but_self_payments(
    target: str, edges: list[set[str]], way: str, heap: str
) -> None:
    """Every payment of the lookback, of either kind and whatever its cents, counts over 90
    days and dates the latest each way; those of the last 30 days count over 30 days and go to
    the list; those of the last 7 days count over 7. A payment whose other end is the account
    itself is read nowhere."""
    select = select_of(DEFINITION, target)
    assert [step.alias for step in select.vertices] == ["a", "p", "b"]
    assert select.vertices[0].chosen and select.vertices[1].types == PAYMENTS
    assert select.vertices[2].types == {"Account"}
    assert [set(edge.types) for edge in select.edges] == edges
    assert select.conjuncts == WINDOW
    assert accumulation_text(DEFINITION, target) == normal(
        f"""a.@{way}_90d += 1,
        a.@last_{way} += p.event_ts_ms,
        IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
            a.@{way}_30d += 1,
            a.@{heap} += Paid(p.event_ts_ms, p.id)
        END,
        IF p.event_ts_ms >= t_ms - {WINDOW_7D_MS} THEN a.@{way}_7d += 1 END"""
    )


# The row.


def test_each_lists_cut_is_the_oldest_second_of_its_full_heap() -> None:
    """A heap holds the newest 257 of its list. When it is full, b is the second of its
    oldest, the 257th, and every payment of the list at or before the second b is dropped;
    otherwise the cut is -1, below every second, and every payment is kept. Each list is cut
    on its own. By hand: 258 inflows of the last 30 days, the two oldest at 09:00:00.250 and
    09:00:00.900 of one day: the heap holds the newest 257, the oldest of them at
    09:00:00.900, so b is the second 09:00:00 and the 256 inflows after it are kept."""
    assert accumulation_text(DEFINITION, "Rows").startswith(
        normal(
            f"""INT cut_in = -1, INT cut_out = -1, INT within_1h = 0, INT within_24h = 0,
            INT most_1h = 0, INT most_24h = 0, UINT shortest = 0, BOOL gap_found = FALSE,
            IF a.@inflows.size() > {LIST} THEN
                FOREACH x IN a.@inflows DO
                    IF cut_in < 0 OR floor(x.ts / 1000) < cut_in THEN
                        cut_in = floor(x.ts / 1000)
                    END
                END
            END,
            IF a.@outflows.size() > {LIST} THEN
                FOREACH x IN a.@outflows DO
                    IF cut_out < 0 OR floor(x.ts / 1000) < cut_out THEN
                        cut_out = floor(x.ts / 1000)
                    END
                END
            END,"""
        )
    )


def _bursts(heap: str, cut: str, gap: str) -> str:
    """The walk of one list: for each kept payment x, the kept payments in an hour and in 24
    hours from it, and, for the inflows, the gap to every kept payment strictly before it."""
    return normal(
        f"""FOREACH x IN a.@{heap} DO
            IF floor(x.ts / 1000) > {cut} THEN
                within_1h = 0, within_24h = 0,
                FOREACH y IN a.@{heap} DO
                    IF floor(y.ts / 1000) > {cut} THEN
                        IF y.ts >= x.ts AND y.ts < x.ts + {WINDOW_1H_MS} THEN
                            within_1h = within_1h + 1
                        END,
                        IF y.ts >= x.ts AND y.ts < x.ts + {WINDOW_24H_MS} THEN
                            within_24h = within_24h + 1
                        END{gap}
                    END
                END,
                IF within_1h > most_1h THEN most_1h = within_1h END,
                IF within_24h > most_24h THEN most_24h = within_24h END
            END
        END,"""
    )


GAP = """,
    IF y.ts < x.ts THEN
        IF NOT gap_found OR x.ts - y.ts < shortest THEN
            shortest = x.ts - y.ts, gap_found = TRUE
        END
    END"""


def test_the_bursts_count_every_kept_payment_in_a_window_from_one_of_them() -> None:
    """The most kept payments in [x, x + w) with x one of their times: every pair compared,
    so payments of one time all count at it, and no order of the heap's walk decides a value.
    By hand: inflows at 10:00:00, 10:00:00, 10:59:59 and 11:00:00 give burst_in_1h 3 at
    x = 10:00:00 (11:00:00 is exactly an hour later and out) and burst_in_24h 4."""
    code = accumulation_text(DEFINITION, "Rows")
    inflows = _bursts("inflows", "cut_in", GAP)
    outflows = _bursts("outflows", "cut_out", "")
    assert inflows in code and outflows in code
    assert code.index(inflows) < code.index(outflows)
    # Each list's most starts at 0: the inflows' at their declaration, the outflows' reset.
    assert (
        normal(
            "a.@burst_in_1h += most_1h, a.@burst_in_24h += most_24h, "
            "IF gap_found THEN a.@shortest_in_gap_present += TRUE, "
            "a.@shortest_in_gap_ms += shortest END, most_1h = 0, most_24h = 0,"
        )
        in code
    )
    assert normal("a.@burst_out_1h += most_1h, a.@burst_out_24h += most_24h,") in code


def test_the_shortest_gap_is_to_the_latest_kept_inflow_strictly_before() -> None:
    """The smallest x - x' over kept inflows x and x' with x' strictly before x is the
    smallest gap from an inflow to the latest one strictly before it. Same-time inflows have
    no gap between them, and the subtraction runs only once x' < x, so it never wraps in UINT.
    By hand: inflows at 10:00:00, 10:00:00 and 10:07:30 give 450,000 ms; two inflows of one
    time alone give none."""
    assert _bursts("inflows", "cut_in", GAP) in accumulation_text(DEFINITION, "Rows")
    assert accumulation_text(DEFINITION, "Rows").count("shortest = x.ts - y.ts") == 1
    assert accumulation_text(DEFINITION, "Rows").count("gap_found") == 4


def test_the_time_since_the_latest_payment_each_way() -> None:
    """t minus the latest payment of the lookback, present only when there is one, so the
    UINT subtraction never wraps: the latest is before t. By hand: the latest inflow at
    2024-03-28 18:00:00, 3 days and 6 hours before t, gives 3 x 86,400,000 + 6 x 3,600,000
    = 280,800,000 ms."""
    assert accumulation_text(DEFINITION, "Rows").endswith(
        normal(
            """IF a.@in_90d > 0 THEN
                a.@since_last_in_present += TRUE, a.@since_last_in_ms += t_ms - a.@last_in
            END,
            IF a.@out_90d > 0 THEN
                a.@since_last_out_present += TRUE, a.@since_last_out_ms += t_ms - a.@last_out
            END"""
        )
    )


def test_no_value_reads_the_order_of_a_heaps_walk() -> None:
    """Each heap is filled once, its size read once, and walked only by FOREACH that read its
    times as a set: the cut, and the outer and inner walks of the bursts. No index counts a
    step and no list copies a heap."""
    code = normal(DEFINITION)
    for heap in ("inflows", "outflows"):
        assert re.findall(rf"a\.@{heap}\S*(?: \S+){{0,2}}", code) == [
            f"a.@{heap} += Paid",
            f"a.@{heap}.size ( )",
            f"a.@{heap} DO IF",
            f"a.@{heap} DO IF",
            f"a.@{heap} DO IF",
        ], heap
    assert "ListAccum" not in code


def test_every_count_is_int_and_every_time_uint() -> None:
    """Counts and cut seconds are INT; the shortest gap holds a difference of two UINT times,
    taken only when it is positive."""
    assert declared_locals(DEFINITION, "Rows") == [
        ("INT", "cut_in"),
        ("INT", "cut_out"),
        ("INT", "within_1h"),
        ("INT", "within_24h"),
        ("INT", "most_1h"),
        ("INT", "most_24h"),
        ("UINT", "shortest"),
        ("BOOL", "gap_found"),
    ]
    for target in ("Received", "Sent"):
        assert declared_locals(DEFINITION, target) == [], target


def test_no_bound_at_t_takes_t_itself() -> None:
    code = normal(DEFINITION)
    # t itself is no bound: no `<= t_ms`, `t_ms >=` or `t_ms >` but the test of t.
    assert not re.search(r"<= t_ms\b|(?<![\w.@])t_ms >(?!= lookback)(?! lookback)", code)
    assert code.count("p.event_ts_ms < t_ms") == 2
    assert code.count("p.event_ts_ms >= t_ms - lookback_ms") == 2


def test_it_reads_no_hub_no_label_no_link_and_no_amount() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(
        r"hub|leaf|crowd|valid_from_ms|valid_to_ms|joined_ms|closed_ms|amount|is_external", code
    )
    assert re.findall(r"\w+\.opened_ms", code) == ["s.opened_ms"]
