"""read_money_flow, B1, read as text (docs/design.md, "read_money_flow (B1): velocity", and
B1's walk in docs/features.md, "Account node").

The query passes every rule of testing.gsql_rules, takes the parameters its Python builder
gives and prints the fields of the dataclasses its parser reads. Beside those rules, the walk
is held here step by step, since no reference graph runs it yet: the payments it reads, the
lists and their cut, the span, the lots, the levels, the suffix-minimum stop, the portions and
their windows, the rails, the median and the shares, and the run-time check of heap order.
gsql-lsp checks the file against the schema in tests/gsql/test_schema.py, and the live compile
test compiles it once the workspace has the schema.
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
    field_rule,
    parameter_rule,
    parameters,
    projection_fields,
    projection_items,
    scopes,
    source_rule,
    statements,
    status_fields,
    tokens,
    tuples,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    QUERY_FILES,
    READ_MONEY_FLOW,
    REGISTRY,
    VERSIONS,
    read_money_flow,
    written_files,
)
from mule_pattern_learner.tigergraph.queries.read_money_flow import MoneyFlow, TrackedInflow
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS, MONDAY_MS, WEEK_MS

TEXT = (GSQL_DIR / QUERY_FILES[READ_MONEY_FLOW]).read_text()
DEFINITION = definitions(TEXT)[READ_MONEY_FLOW]
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC, the first scored Monday

# The windows of the walk, from docs/features.md, columns 4 to 21: amounts over 7, 30 and 90
# days in milliseconds (the 90 days are lookback_ms); dwell windows in whole seconds.
DAY_MS = 86_400_000
WINDOW_7D_MS = 7 * DAY_MS  # 604,800,000
WINDOW_30D_MS = 30 * DAY_MS  # 2,592,000,000
DWELL_WINDOWS_S = {
    "15m": 15 * 60,  # 900
    "1h": 60 * 60,  # 3,600
    "6h": 6 * 60 * 60,  # 21,600
    "1d": 24 * 60 * 60,  # 86,400
    "3d": 3 * 24 * 60 * 60,  # 259,200
    "7d": 7 * 24 * 60 * 60,  # 604,800
}
# Each list keeps the newest recent_payments (256) and holds one more, the newest left out.
LIST = SUBGRAPH.recent_payments
HEAP = LIST + 1
# Matching makes at most one portion per lot and one per outflow, n + m <= 2 x 256.
PORTIONS = 2 * LIST


def _normal(text: str) -> str:
    """A GSQL text as the rules read it: its tokens joined by single spaces."""
    return " ".join(tokens(text))


def _select(target: str) -> Select:
    """The SELECT that assigns target."""
    for statement, _, select in scopes(DEFINITION):
        if select is not None and statement[:2] == [target, "="]:
            return select
    raise AssertionError(f"read_money_flow assigns no {target} by a SELECT")


def _accumulations(target: str) -> str:
    """The ACCUM and POST-ACCUM of the SELECT that assigns target, as the rules read them."""
    return " ".join(_select(target).accumulations)


def _has(target: str, *snippets: str) -> None:
    """Each snippet, written as GSQL, is in the clauses of the SELECT that assigns target."""
    code = _accumulations(target)
    for snippet in snippets:
        assert _normal(snippet) in code, snippet


def _declared_locals(target: str) -> list[tuple[str, str]]:
    """The local variables a SELECT's clauses declare, with their types, in order."""
    found = re.findall(
        r"(?:^|, )(INT|UINT|STRING|BOOL|DOUBLE|FLOAT) (\w+) =", _accumulations(target)
    )
    return [(kind, name) for kind, name in found]


# Every rule, and the Python above the query.


def test_it_passes_every_rule_and_prints_its_version() -> None:
    assert QUERY_FILES[READ_MONEY_FLOW] in written_files().values()
    assert common_problems(READ_MONEY_FLOW, TEXT) == []
    assert version_rule(READ_MONEY_FLOW, DEFINITION, VERSIONS[READ_MONEY_FLOW]) == []


def test_it_takes_the_parameters_its_builder_gives() -> None:
    built = read_money_flow.params(T_MS, ["a_01", "a_02"])
    assert parameter_rule(READ_MONEY_FLOW, DEFINITION, built) == []
    # Ids, not a SET<VERTEX>, from which an installed query drops an unknown id unseen.
    assert parameters(DEFINITION) == {"t_ms": ["UINT"], "account_ids": ["LIST", "<", "STRING", ">"]}


def test_each_printed_row_has_the_fields_of_its_dataclass() -> None:
    printed = projection_fields(DEFINITION, "Rows")
    assert field_rule("read_money_flow Rows", printed, MoneyFlow) == []
    assert source_rule("read_money_flow Rows", projection_items(DEFINITION, "Rows")) == []
    # The tracked inflows are tuples, printed whole under their accumulator's name.
    assert tuples(DEFINITION)["TrackedInflow"] == [field.name for field in fields(TrackedInflow)]
    assert "PRINT @@tracked_inflows AS tracked_inflows;" in DEFINITION
    # The status row prints the status and the version alone.
    assert status_fields(DEFINITION) == []


def test_the_registry_holds_its_builder_and_rows() -> None:
    query = REGISTRY[READ_MONEY_FLOW]
    assert query.params is read_money_flow.params
    assert query.rows == (MoneyFlow, TrackedInflow)


# What the query declares, exactly, in order. The type of an accumulator decides what a value
# is: every cent and level is INT, never mixed with UINT; a heap's capacity and order decide
# which payments a list keeps and in which order the walk reads them; a MaxAccum keeps the
# latest second left out of either list.
DECLARATIONS = [
    "TYPEDEF TUPLE<UINT ts, STRING id, INT cents, STRING rail> Paid",
    "TYPEDEF TUPLE<INT dwell, INT cents> Portion",
    "TYPEDEF TUPLE< STRING account_id, STRING payment_id, DOUBLE left_24h, BOOL dwell_present, "
    "INT dwell_ms > TrackedInflow",
    "SumAccum<INT> @in_amount_7d, @in_amount_30d, @in_amount_90d, @out_amount_7d, "
    "@out_amount_30d, @out_amount_90d",
    f"HeapAccum<Paid>({HEAP}, ts DESC, rail DESC, id DESC) @inflows, @outflows",
    "ListAccum<INT> @in_s, @in_cents, @out_s, @out_cents",
    "ListAccum<STRING> @in_id, @out_rail",
    "MaxAccum<INT> @cut_s",
    "ListAccum<INT> @lot_s, @lot_cents, @lot_base, @lot_next, @lot_first, @lot_count, "
    "@out_level, @out_lowest",
    "SumAccum<INT> @kept_in",
    f"HeapAccum<Portion>({PORTIONS}, dwell ASC) @portions",
    "SumAccum<INT> @received, @matched, @within_15m, @within_1h, @within_6h, @within_1d, "
    "@within_3d, @within_7d, @by_transfer, @by_card, @by_cash, @scan_steps",
    "OrAccum @flow_present, @median_dwell_present",
    "SumAccum<INT> @median_dwell_ms",
    "SumAccum<DOUBLE> @left_15m, @left_1h, @left_6h, @left_1d, @left_3d, @left_7d, @held_share, "
    "@left_by_transfer, @left_by_card, @left_by_cash",
    "ListAccum<TrackedInflow> @@tracked_inflows",
    "OrAccum @@unsorted",
    "SetAccum<STRING> @@unsorted_accounts",
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


def test_its_heaps_hold_one_more_than_a_list_and_every_portion() -> None:
    """Each payment heap holds the newest 256 of its list and the newest payment left out,
    whose second cuts the span; the walk keeps the first 256 a heap gives and reads the 257th
    as the cut. Matching makes at most one portion per lot and one per outflow."""
    assert (HEAP, PORTIONS) == (257, 512)
    for way in ("in", "out"):
        heap = f"a.@{way}flows"
        loop = _accumulations("Listed").split(f"FOREACH x IN {heap} DO")[1]
        branch = loop.split(" , previous_ts = x.ts")[0]
        assert f"IF k < {LIST} THEN a.@{way}_s += floor ( x.ts / 1000 )" in branch, way
        assert branch.endswith("ELSE a.@cut_s += floor ( x.ts / 1000 ) END"), way


# The refusals.


def test_every_refusal_returns_at_once() -> None:
    """A refusal that went on would read with t at or below the lookback, where
    t_ms - lookback_ms wraps in UINT, answer a request of no account, of an account twice or
    of an account the graph lacks with fewer rows than it asked for, read accounts that do not
    exist at t, or print rows walked from a heap out of order."""
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
        "unsorted_heap",
        "unsorted_heap",
    ]


def test_t_is_a_monday_after_the_lookback_and_every_account_is_known_once_and_exists() -> None:
    """t is checked first; then the ids: at least one, none twice (a list keeps a repeat, a
    set would drop it), each an account of the graph (to_vertex_set drops an id it lacks) and
    each existing at t. Only the accounts that passed every check are read."""
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    start = next(at for at, text in enumerate(found) if text.startswith("IF "))
    assert found[start : start + 18] == [
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
    ]
    assert _select("Scored").conjuncts == {"s.opened_ms < t_ms"}
    # Every later SELECT starts from the checked accounts.
    starts = [
        select.vertices[0]
        for _, _, select in scopes(DEFINITION)
        if select is not None and select.statement[0] != "Scored"
    ]
    assert len(starts) == 8
    assert {(step.alias, step.chosen) for step in starts} == {("a", True)}
    assert _normal(DEFINITION).count("FROM Scored : a") == 8


# The order of the SELECTs. A value a walk reads later is read in a later SELECT than the one
# that wrote it: POST-ACCUM clauses of one SELECT run in parallel, and a SELECT moved before
# the one that fills what it reads walks empty lists and still compiles, so the order is pinned
# whole, and the rule that explains it is checked on the text and on three text moves.
SELECT_ORDER = [
    "Scored",
    "Received",
    "Sent",
    "SentByZelle",
    "Listed",
    "Levels",
    "Lowest",
    "Matched",
    "Rows",
]
VERTEX_ACCUMULATOR = re.compile(r"\w+\.@(\w+)(\.\w+)?")


def _flows(definition: str) -> list[tuple[str, set[str], set[str]]]:
    """Each assigned SELECT of a query in file order, with the vertex accumulators (`a.@name`)
    its ACCUM and POST-ACCUM clauses write (`+=` or `=` on the accumulator itself) and read
    (any other use, as in `a.@in_s.get(n)` or `FOREACH x IN a.@inflows`)."""
    found: list[tuple[str, set[str], set[str]]] = []
    for _, _, select in scopes(definition):
        if select is None:
            continue
        items = list(select.accumulations)
        written: set[str] = set()
        read: set[str] = set()
        for at, item in enumerate(items):
            match = VERTEX_ACCUMULATOR.fullmatch(item)
            if match is None:
                continue
            if match[2] is None and items[at + 1 : at + 2] in (["+="], ["="]):
                written.add(match[1])
            else:
                read.add(match[1])
        found.append((select.statement[0], written, read))
    return found


def dataflow_problems(definition: str) -> list[str]:
    """Every accumulator a SELECT reads, with a writer in the query, is written only by SELECTs
    strictly before it, and has at least one."""
    flows = _flows(definition)
    problems: list[str] = []
    for index, (target, _, read) in enumerate(flows):
        for name in sorted(read):
            writers = [at for at, (_, written, _) in enumerate(flows) if name in written]
            if not writers or max(writers) >= index:
                by = [flows[at][0] for at in writers] or ["no SELECT"]
                problems.append(f"{target} reads @{name}, written by {', '.join(by)}")
    return problems


def _moved(target: str, after: str) -> str:
    """The query's text with the SELECT that assigns target moved to follow the one that
    assigns after."""

    def statement(name: str) -> re.Pattern[str]:
        return re.compile(rf"^    {name} = SELECT .*?;\n", re.MULTILINE | re.DOTALL)

    moved = statement(target).search(DEFINITION)
    stays = statement(after).search(DEFINITION)
    assert moved is not None and stays is not None
    without = DEFINITION.replace(moved[0], "", 1)
    end = statement(after).search(without)
    assert end is not None
    return without[: end.end()] + moved[0] + without[end.end() :]


def test_the_selects_run_in_the_order_the_walk_needs() -> None:
    assert [target for target, _, _ in _flows(DEFINITION)] == SELECT_ORDER


def test_every_value_a_select_reads_is_written_by_an_earlier_select() -> None:
    assert dataflow_problems(DEFINITION) == []
    # The rule sees the reads that matter: the lists a walk reads, the cut, the levels and the
    # lowest levels, the lots and the portions.
    reads = {target: read for target, _, read in _flows(DEFINITION)}
    assert {"inflows", "outflows"} <= reads["Listed"]
    assert {"in_s", "in_cents", "out_s", "out_cents", "cut_s"} <= reads["Levels"]
    assert reads["Lowest"] == {"out_level"}
    assert {"out_lowest", "out_level", "lot_base", "lot_next", "in_id"} <= reads["Matched"]
    assert {"portions", "received", "matched", "within_1d", "by_card"} <= reads["Rows"]


@pytest.mark.parametrize(
    ("target", "after", "problem"),
    [
        # Lowest after Matched removes the suffix-minimum stop: out_lowest reads as empty.
        ("Lowest", "Matched", "Matched reads @out_lowest, written by Lowest"),
        # Levels before Listed: every list is empty, so every lot is empty and every velocity
        # value is 0.
        ("Listed", "Levels", "Levels reads @in_s, written by Listed"),
        # Rows before Matched: the median and every share read an empty sum.
        ("Matched", "Rows", "Rows reads @received, written by Matched"),
    ],
)
def test_a_select_moved_before_the_one_it_reads_from_breaks_the_rule(
    target: str, after: str, problem: str
) -> None:
    moved = _moved(target, after)
    assert moved != DEFINITION
    assert problem in dataflow_problems(moved)


def test_same_second_outflows_are_walked_by_rail_name_then_id() -> None:
    """Payments of one second have no order, so no value may follow an id. The two outflows of
    one second that take different rails split the cents a lot holds by rail, and which of them
    takes the newest cents decides the split, so the heap orders them by rail name before the
    id, and the walk reads each list from its end, oldest first: ach, card, cash, check,
    internal, unknown, zelle. By hand: a lot A of 10,000 cents at s - 2 days, a lot B of 10,000
    cents at s - 1 hour, and at second s an outflow p1 of 10,000 cents by card and an outflow
    p2 of 10,000 cents by ach. Ach is walked first and takes B (dwell 3,600 s, under a day),
    then card takes A (dwell 172,800 s): left_by_transfer is 10,000 / 10,000 = 1,
    left_by_card 0 and left_1d 10,000 / 20,000 = 0.5, whichever of p1 and p2 has the smaller
    id. Ordered by id alone, the smaller id took B, and swapping the ids swapped the rails."""
    assert re.findall(r"HeapAccum<Paid>\((\d+), ([^)]*)\)", DEFINITION) == [
        (str(HEAP), "ts DESC, rail DESC, id DESC")
    ]
    code = _normal(DEFINITION)
    # Each list is read at m - 1 - j, from its end: the lot test and the dwell read the second,
    # the portion's rail the rail.
    assert code.count("a.@out_s.get ( m - 1 - j )") == 2
    assert code.count("a.@out_rail.get ( m - 1 - j )") == 1
    # The only comparison of two payment ids is the check of each heap's order.
    assert len(re.findall(r"\.id [<>]", code)) == 2


def test_a_heap_walked_out_of_order_stops_the_query() -> None:
    """FOREACH is taken to walk a heap in its sort order, which the documentation does not
    promise: each payment must sort after the one before, (ts DESC, rail DESC, id DESC), and
    each portion after the one before, dwell ASC. A heap out of order names its account and the
    query stops, after the lists are walked and again after the portions are, before any
    row."""
    out_of_order = (
        "IF k > 0 AND (x.ts > previous_ts OR (x.ts == previous_ts AND (x.rail > previous_rail "
        "OR (x.rail == previous_rail AND x.id > previous_id)))) THEN "
        "@@unsorted += TRUE, @@unsorted_accounts += a.id END"
    )
    assert _accumulations("Listed").count(_normal(out_of_order)) == 2
    _has(
        "Listed",
        "previous_ts = x.ts, previous_rail = x.rail, previous_id = x.id, k = k + 1 END",
    )
    _has(
        "Listed",
        'k = 0, previous_ts = 0, previous_rail = "", previous_id = "", FOREACH x IN a.@outflows DO',
    )
    assert _accumulations("Listed").startswith(
        _normal(
            'INT k = 0, UINT previous_ts = 0, STRING previous_rail = "", '
            'STRING previous_id = "", FOREACH x IN a.@inflows'
        )
    )
    _has(
        "Rows",
        "FOREACH x IN a.@portions DO IF started AND x.dwell < previous THEN @@unsorted += TRUE, "
        "@@unsorted_accounts += a.id END, started = TRUE, previous = x.dwell,",
    )
    found = [" ".join(statement) for statement in statements(DEFINITION)]
    stop = 'IF @@unsorted THEN PRINT "unsorted_heap" AS status , @@unsorted_accounts AS accounts'
    after = [found[index - 1].split(" = ")[0] for index, text in enumerate(found) if text == stop]
    assert after == ["Listed", "Rows"]


# The payments it reads.


WINDOW = {"p.event_ts_ms >= t_ms - lookback_ms", "p.event_ts_ms < t_ms", "b != a"}
RECEIVED = frozenset({"Account_Received_Transaction", "Account_Received_Zelle_Transfer"})
SENDERS = frozenset({"Transaction_From_Account", "Transfer_From_Account"})


@pytest.mark.parametrize(
    ("target", "edges", "payments", "rail"),
    [
        ("Received", [RECEIVED, SENDERS], {"Payment_Transaction", "Zelle_Transfer"}, None),
        (
            "Sent",
            [{"Account_Initiated_Transaction"}, {"Transaction_To_Account"}],
            {"Payment_Transaction"},
            "p.payment_rail",
        ),
        (
            "SentByZelle",
            [{"Account_Sent_Zelle_Transfer"}, {"Transfer_To_Account"}],
            {"Zelle_Transfer"},
            '"zelle"',
        ),
    ],
)
def test_it_reads_the_lookbacks_payments_each_way_but_self_payments(
    target: str, edges: list[set[str]], payments: set[str], rail: str | None
) -> None:
    """Every payment of the lookback counts in the amounts, and those of the last 30 days in a
    list: inflows with positive cents, outflows whatever their cents, each with its rail, a
    Zelle transfer's being zelle. A payment whose other end is the account itself is read
    nowhere."""
    select = _select(target)
    assert [step.alias for step in select.vertices] == ["a", "p", "b"]
    assert select.vertices[0].chosen and select.vertices[1].types == payments
    assert [set(edge.types) for edge in select.edges] == edges
    assert select.conjuncts == WINDOW
    way = "in" if rail is None else "out"
    listed = (
        'IF cents > 0 THEN a.@inflows += Paid(p.event_ts_ms, p.id, cents, "") END'
        if rail is None
        else f"a.@outflows += Paid(p.event_ts_ms, p.id, cents, {rail})"
    )
    assert _accumulations(target) == _normal(
        f"""INT cents = floor(p.amount * 100 + 0.5),
        a.@{way}_amount_90d += cents,
        IF p.event_ts_ms >= t_ms - {WINDOW_30D_MS} THEN
            a.@{way}_amount_30d += cents,
            {listed}
        END,
        IF p.event_ts_ms >= t_ms - {WINDOW_7D_MS} THEN a.@{way}_amount_7d += cents END"""
    )


# The walk.


def test_the_span_drops_every_payment_at_or_before_the_cut_second() -> None:
    """With no list cut, the cut second is MaxAccum's least INT, below every second; otherwise
    every payment of either list at or before it is dropped, a prefix of each list newest
    first being kept."""
    assert _accumulations("Levels").startswith(
        _normal(
            "INT cut = a.@cut_s, INT n = 0, INT m = 0, INT i = 0, INT j = 0, INT level = 0, "
            "INT lot_s = 0, INT lot = 0, INT first_in = 0, "
            "WHILE n < a.@in_s.size() AND a.@in_s.get(n) > cut DO n = n + 1 END, "
            "WHILE m < a.@out_s.size() AND a.@out_s.get(m) > cut DO m = m + 1 END, "
            "WHILE i < n OR j < m DO"
        )
    )
    assert _accumulations("Levels").endswith(_normal("j = j + 1 END END, a.@kept_in += n"))


def test_a_lot_holds_the_inflows_of_one_second_and_comes_before_its_outflows() -> None:
    """The walk reads the lists from their ends, oldest first. The next event is a lot while
    its second is at or before the next outflow's; it takes every inflow of that second, and
    records its second, cents, the level before it, the first outflow after it and its
    inflows."""
    _has(
        "Levels",
        "IF i < n AND (j >= m OR a.@in_s.get(n - 1 - i) <= a.@out_s.get(m - 1 - j)) THEN "
        "lot_s = a.@in_s.get(n - 1 - i), lot = 0, first_in = i, "
        "WHILE i < n AND a.@in_s.get(n - 1 - i) == lot_s DO "
        "lot = lot + a.@in_cents.get(n - 1 - i), i = i + 1 END, "
        "a.@lot_s += lot_s, a.@lot_cents += lot, a.@lot_base += level, "
        "a.@lot_next += j, a.@lot_first += first_in, a.@lot_count += i - first_in, "
        "level = level + lot ELSE",
    )


def test_an_outflow_lowers_the_level_never_below_zero() -> None:
    """An outflow larger than the tracked cents spent money from before the span."""
    _has(
        "Levels",
        "ELSE level = CASE WHEN level > a.@out_cents.get(m - 1 - j) "
        "THEN level - a.@out_cents.get(m - 1 - j) ELSE 0 END, "
        "a.@out_level += level, j = j + 1 END",
    )


def test_the_suffix_minima_stop_a_lots_scan() -> None:
    """The lowest level from each outflow to the last, appended from the last back; a lot's
    scan goes on only while it holds cents and a later outflow falls below its lowest level."""
    assert _accumulations("Lowest") == _normal(
        """INT m = a.@out_level.size(), INT j = m - 1, INT low = 0,
        IF m > 0 THEN low = a.@out_level.get(m - 1) END,
        WHILE j >= 0 DO
            IF a.@out_level.get(j) < low THEN low = a.@out_level.get(j) END,
            a.@out_lowest += low,
            j = j - 1
        END"""
    )
    _has(
        "Matched",
        "FOREACH k IN RANGE[0, a.@lot_s.size() - 1] DO "
        "c = a.@lot_cents.get(k), base = a.@lot_base.get(k), low = base + c, "
        "gone = 0, within_24h = 0, half = -1, j = a.@lot_next.get(k), "
        "WHILE j < m AND low > base AND a.@out_lowest.get(m - 1 - j) < low DO "
        "IF a.@out_level.get(j) < low THEN low = a.@out_level.get(j),",
        "steps = steps + 1, j = j + 1 END, a.@received += c, a.@matched += gone,",
    )
    assert _accumulations("Matched").startswith(
        _normal(
            "INT m = a.@out_level.size(), INT n = a.@kept_in, INT j = 0, INT c = 0, "
            "INT base = 0, INT low = 0, INT gone = 0, INT taken = 0, INT dwell = 0, "
            'INT within_24h = 0, INT half = -1, INT steps = 0, INT q = 0, STRING rail = "", '
            "IF a.@lot_s.size() > 0 THEN"
        )
    )
    assert _accumulations("Matched").endswith(_normal("END END, a.@scan_steps += steps"))


def test_each_fall_inside_a_lots_band_is_a_portion_timed_in_whole_seconds() -> None:
    """A lot still holds max(0, lowest level - base) of its c cents, so each fall of its lowest
    level takes the cents above the new lowest level that it had not lost yet; the dwell is the
    outflow's second minus the lot's."""
    _has(
        "Matched",
        "taken = c - (CASE WHEN low > base THEN low - base ELSE 0 END) - gone, "
        "gone = gone + taken, "
        "dwell = a.@out_s.get(m - 1 - j) - a.@lot_s.get(k), "
        "rail = a.@out_rail.get(m - 1 - j), "
        "a.@portions += Portion(dwell, taken),",
    )


def test_a_window_holds_the_portions_with_a_dwell_under_it() -> None:
    windows = [
        f"IF dwell < {DWELL_WINDOWS_S[w]} THEN a.@within_{w} += taken END"
        for w in ("15m", "1h", "6h")
    ]
    _has("Matched", ", ".join(windows) + f", IF dwell < {DWELL_WINDOWS_S['1d']} THEN")
    windows = [
        f"IF dwell < {DWELL_WINDOWS_S[w]} THEN a.@within_{w} += taken END" for w in ("3d", "7d")
    ]
    _has("Matched", "END END, " + ", ".join(windows) + ", IF half < 0")


def test_the_cents_that_left_within_a_day_split_by_rail() -> None:
    """Of the cents that left within a day: a lot's share left within 24 hours, and the
    shares by transfer (ach, check, internal and zelle), card and cash; unknown counts in the
    denominator alone."""
    _has(
        "Matched",
        f"IF dwell < {DWELL_WINDOWS_S['1d']} THEN a.@within_1d += taken, within_24h = gone, "
        'IF rail == "card" THEN a.@by_card += taken '
        'ELSE IF rail == "cash" THEN a.@by_cash += taken '
        'ELSE IF rail == "ach" OR rail == "check" OR rail == "internal" OR rail == "zelle" '
        "THEN a.@by_transfer += taken END END,",
    )


def test_a_lots_left_24h_and_dwell_go_to_each_of_its_inflows() -> None:
    """A lot's dwell is the dwell of the portion at which its matched cents first reach half
    its cents, absent (half -1) when less than half left before t; each inflow of the lot, the
    k-th oldest kept at index n - 1 - k, gets the lot's values."""
    _has(
        "Matched",
        "IF half < 0 AND 2 * gone >= c THEN half = dwell END END,",
        "q = a.@lot_first.get(k), "
        "WHILE q < a.@lot_first.get(k) + a.@lot_count.get(k) DO "
        "@@tracked_inflows += TrackedInflow(a.id, a.@in_id.get(n - 1 - q), "
        "within_24h * 1.0 / c, half >= 0, CASE WHEN half >= 0 THEN half * 1000 ELSE 0 END), "
        "q = q + 1 END",
    )


def test_the_median_and_the_shares_divide_sums_of_cents_once() -> None:
    """The median dwell is that of the portion, by ascending dwell, at which the matched cents
    first reach half of R, the cents received in the span; absent when less than half left.
    Every share is 0 with an empty denominator."""
    _has(
        "Rows",
        "cumulative = cumulative + x.cents, "
        "IF median < 0 AND 2 * cumulative >= a.@received THEN median = x.dwell END END, "
        "IF median >= 0 THEN a.@median_dwell_present += TRUE, "
        "a.@median_dwell_ms += median * 1000 END,",
    )
    assert _accumulations("Rows").startswith(
        _normal("INT cumulative = 0, INT previous = 0, BOOL started = FALSE, INT median = -1,")
    )
    shares = ", ".join(f"a.@left_{w} += a.@within_{w} * 1.0 / a.@received" for w in DWELL_WINDOWS_S)
    rails = ", ".join(
        f"a.@left_by_{rail} += a.@by_{rail} * 1.0 / a.@within_1d"
        for rail in ("transfer", "card", "cash")
    )
    assert _accumulations("Rows").endswith(
        _normal(
            f"IF a.@received > 0 THEN a.@flow_present += TRUE, {shares}, "
            "a.@held_share += (a.@received - a.@matched) * 1.0 / a.@received END, "
            f"IF a.@within_1d > 0 THEN {rails} END"
        )
    )


def test_every_level_and_cent_of_the_walk_is_int() -> None:
    """Mixing INT with UINT turns arithmetic into UINT, where a level or cent below 0 wraps.
    The walk's locals are INT (seconds included), but the heap's millisecond time it compares
    the next payment's with, and its rail and flags."""
    assert _declared_locals("Listed") == [
        ("INT", "k"),
        ("UINT", "previous_ts"),
        ("STRING", "previous_rail"),
        ("STRING", "previous_id"),
    ]
    for target in ("Levels", "Lowest"):
        assert {kind for kind, _ in _declared_locals(target)} == {"INT"}, target
    assert [name for kind, name in _declared_locals("Matched") if kind != "INT"] == ["rail"] and (
        "STRING",
        "rail",
    ) in _declared_locals("Matched")
    assert _declared_locals("Rows") == [
        ("INT", "cumulative"),
        ("INT", "previous"),
        ("BOOL", "started"),
        ("INT", "median"),
    ]
    for target in ("Received", "Sent", "SentByZelle"):
        assert _declared_locals(target) == [("INT", "cents")], target


def test_no_bound_at_t_takes_t_itself() -> None:
    code = _normal(DEFINITION)
    assert not re.search(r"<= t_ms\b|t_ms >=|t_ms > (?!lookback)", code)
    assert code.count("p.event_ts_ms < t_ms") == 3


def test_it_reads_no_hub_no_label_and_no_link() -> None:
    code = without_comments(DEFINITION)
    assert not re.search(r"hub|leaf|crowd|valid_from_ms|valid_to_ms|joined_ms|closed_ms", code)
    assert re.findall(r"\w+\.opened_ms", code) == ["s.opened_ms"]
