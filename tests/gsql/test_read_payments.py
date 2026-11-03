"""read_payments' gaps and pairs against the lookback, read as text (docs/design.md, "Time",
"Marks of a scoring time" and "How its payment rows are read", and docs/features.md, "Payment
edges"): no payment before t - 90 days reaches a payment's row.

read_payments reads stamps that two mark queries write. mark_payment_gaps runs once per world:
it stamps each payment with the time since its sender's and its receiver's latest strictly
earlier payment, kept when that payment is at most 90 days before it, and read_payments reads a
stamp only when that earlier payment lies at or after t - 90 days. mark_pair_history runs at a
scoring time: it stamps each payment of [t - 90 days, t) with its pair's earlier payments of
that window alone, and records its t on the "pairs" Mark, which read_payments checks beside the
hubs'.

Each boundary is held by the query's own condition, taken from its text and evaluated here,
since no reference graph runs these queries yet. By hand: t is Monday 1 July 2024 00:00 UTC, so
the lookback starts on 2 April; a payment of 12 April whose sender last paid on 1 March, 122 days
before t, reads no sender gap, where the stamp alone holds 42 days.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
import operator
import re

import pytest

from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    parameters,
    select_of,
    statements,
    version_rule,
    without_comments,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    MARK_PAIR_HISTORY,
    MARK_PAYMENT_GAPS,
    QUERY_FILES,
    READ_PAYMENTS,
    VERSIONS,
)
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS

DEFINITIONS = {
    name: definitions((GSQL_DIR / QUERY_FILES[name]).read_text())[name]
    for name in (READ_PAYMENTS, MARK_PAYMENT_GAPS, MARK_PAIR_HISTORY)
}
READ = DEFINITIONS[READ_PAYMENTS]
GAPS = DEFINITIONS[MARK_PAYMENT_GAPS]
PAIRS = DEFINITIONS[MARK_PAIR_HISTORY]

DAY_MS = 86_400_000
T_MS = 1_719_792_000_000  # Monday 2024-07-01 00:00 UTC
START_MS = T_MS - LOOKBACK_MS  # 2024-04-02 00:00 UTC, the lookback's start
PAYMENT_MS = T_MS - 80 * DAY_MS  # 2024-04-12 00:00 UTC

# A condition's keywords in Python, the operators it may hold, and its comparisons.
KEYWORDS = {"AND": "and", "OR": "or", "NOT": "not", "TRUE": "True", "FALSE": "False"}
OPERATORS = frozenset({"(", ")", "<", "<=", ">", ">=", "==", "!=", "+", "-", "%"})
COMPARISONS: dict[type[ast.cmpop], Callable[[int, int], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _python(token: str, values: Mapping[str, int]) -> str:
    """A token of a GSQL condition in Python: a keyword, a name bound in values, an integer or
    an operator. A name the test does not bind fails it, so each test binds all a condition
    reads."""
    if token in KEYWORDS:
        return KEYWORDS[token]
    if token in values:
        return str(values[token])
    if token.isdigit() or token in OPERATORS:
        return token
    raise KeyError(f"the condition reads {token}, which the test does not bind")


def _value(node: ast.expr) -> int:
    """The value of a parsed condition, a truth value as an int. Every integer is a UINT, so a
    subtraction below zero, which wraps in GSQL, fails the test."""
    match node:
        case ast.Constant(value=int() as value):
            return value
        case ast.BoolOp(op=ast.And(), values=operands):
            return all(_value(operand) for operand in operands)
        case ast.BoolOp(op=ast.Or(), values=operands):
            return any(_value(operand) for operand in operands)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _value(operand)
        case ast.Compare(left=left, ops=[op], comparators=[right]):
            return COMPARISONS[type(op)](_value(left), _value(right))
        case ast.BinOp(left=left, op=ast.Sub(), right=right):
            difference = _value(left) - _value(right)
            assert difference >= 0, "a UINT subtraction below zero wraps"
            return difference
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            return _value(left) + _value(right)
        case ast.BinOp(left=left, op=ast.Mod(), right=right):
            return _value(left) % _value(right)
        case _:
            raise ValueError(f"no condition of these queries reads {ast.dump(node)}")


def holds(condition: Sequence[str], values: Mapping[str, int]) -> bool:
    """Whether a GSQL condition, as its tokens, holds with the names it reads bound in values."""
    python = " ".join(_python(token, values) for token in condition)
    return bool(_value(ast.parse(python, mode="eval").body))


def _conditions(items: Sequence[str]) -> list[list[str]]:
    """The condition of each IF among items, in order: its tokens between IF and THEN."""
    return [
        list(items[index + 1 : items.index("THEN", index)])
        for index, item in enumerate(items)
        if item == "IF"
    ]


def _where(definition: str, target: str, values: Mapping[str, int]) -> bool:
    """Whether the WHERE clause of the SELECT that assigns target holds: all its conjuncts."""
    conjuncts = select_of(definition, target).conjuncts
    return all(holds(conjunct.split(" "), values) for conjunct in conjuncts)


def _source(definition: str, target: str) -> str:
    """The vertex set the SELECT that assigns target starts from."""
    statement = select_of(definition, target).statement
    return statement[statement.index("FROM") + 1]


@pytest.mark.parametrize("name", [READ_PAYMENTS, MARK_PAYMENT_GAPS, MARK_PAIR_HISTORY])
def test_its_version_is_its_hash_and_its_lookback_is_the_settings(name: str) -> None:
    assert version_rule(name, DEFINITIONS[name], VERSIONS[name]) == []
    assert f"UINT LOOKBACK_MS = {LOOKBACK_MS};" in DEFINITIONS[name]


# Each end of a payment: the SELECT of read_payments that reads its gap, and its account's alias.
ENDS = [("sender", "sent", "a"), ("receiver", "received", "b")]


def _gap_values(end: str, account: str, earlier_ms: int) -> dict[str, int]:
    """What read_payments' test of a stamped gap reads: the payment of 12 April, its stamp for
    an earlier payment at earlier_ms, no hub stamp on the account at that end, and t."""
    return {
        "p.event_ts_ms": PAYMENT_MS,
        f"p.{end}_gap_ms": PAYMENT_MS - earlier_ms,
        f"{account}.hub_at_ms": 0,
        "t_ms": T_MS,
        "LOOKBACK_MS": LOOKBACK_MS,
    }


@pytest.mark.parametrize(("end", "target", "account"), ENDS)
@pytest.mark.parametrize(
    ("earlier_ms", "read"),
    [
        (START_MS - 1, False),  # a millisecond before t - 90 days
        (START_MS, True),  # exactly at it
        (T_MS - 122 * DAY_MS, False),  # 1 March, 42 days before the payment
        (PAYMENT_MS - 1, True),  # a millisecond before the payment
    ],
)
def test_a_stamped_gap_is_read_only_when_its_earlier_payment_is_in_the_lookback(
    end: str, target: str, account: str, earlier_ms: int, read: bool
) -> None:
    """The stamp is the time since the latest strictly earlier payment at that end, so the
    payment's time minus the stamp is the earlier payment's time, and read_payments reads the
    stamp only when that is at or after t - 90 days. That is the first run's gap over
    [t - 90 days, ts): when the latest earlier payment is before t - 90 days, none lies inside.
    A stamp that is read passes unchanged."""
    accumulations = select_of(READ, target).accumulations
    (condition,) = _conditions(accumulations)
    assert holds(condition, _gap_values(end, account, earlier_ms)) is read
    then = accumulations[accumulations.index("THEN") + 1 : accumulations.index("END")]
    assert " ".join(then) == f"p.@{end}_gap_present += TRUE , p.@{end}_gap_ms += p.{end}_gap_ms"


@pytest.mark.parametrize(("end", "target", "account"), ENDS)
def test_no_gap_is_read_without_a_stamp_or_at_a_hub_end(
    end: str, target: str, account: str
) -> None:
    """A stamp of 0 is no earlier payment within 90 days of the payment, and an end stamped a
    hub at t reads no gap, as in the first run; a hub stamp of another time is harmless."""
    (condition,) = _conditions(select_of(READ, target).accumulations)
    inside = _gap_values(end, account, START_MS)
    assert holds(condition, inside)
    assert not holds(condition, inside | {f"p.{end}_gap_ms": 0})
    assert not holds(condition, inside | {f"{account}.hub_at_ms": T_MS})
    assert holds(condition, inside | {f"{account}.hub_at_ms": T_MS - 7 * DAY_MS})


@pytest.mark.parametrize("payment_ms", [START_MS + 1, PAYMENT_MS, T_MS - 1])
def test_the_gap_stamps_hold_every_earlier_payment_the_lookback_can_need(payment_ms: int) -> None:
    """mark_payment_gaps walks each account's payments each way oldest first and keeps, for
    each, the time of the one before it when that is strictly earlier and at most 90 days
    earlier. A payment a reader at t reads lies in [t - 90 days, t), so an earlier payment at or
    after t - 90 days is always within the 90 days kept, even for a payment a millisecond before
    t: the filter of read_payments loses nothing the first run read."""
    walked = _conditions(select_of(GAPS, "walked").accumulations)
    kept = [condition for condition in walked if condition[:1] == ["last"]]
    assert len(kept) == 2  # payments sent, then received
    for condition in kept:
        values = {"last": START_MS, "x.ts": payment_ms, "LOOKBACK_MS": LOOKBACK_MS}
        assert holds(condition, values)
        # A payment in the same millisecond is no earlier payment.
        assert not holds(condition, values | {"last": payment_ms})


@pytest.mark.parametrize(
    ("ts_ms", "inside"),
    [(START_MS - 1, False), (START_MS, True), (T_MS - 1, True), (T_MS, False)],
)
def test_the_pair_marks_window_is_the_lookback(ts_ms: int, inside: bool) -> None:
    """mark_pair_history reads and stamps the payments of [t - 90 days, t), the window of a
    reader at t: from exactly t - 90 days, and strictly before t."""
    window = {"p.event_ts_ms": ts_ms, "t_ms": T_MS, "LOOKBACK_MS": LOOKBACK_MS}
    assert _where(PAIRS, "recent", window) is inside


@pytest.mark.parametrize(
    ("earlier_ms", "counted"),
    [
        (START_MS - 1, False),  # a millisecond before t - 90 days
        (START_MS, True),  # exactly at it
        (PAYMENT_MS - 1, True),  # a millisecond before the payment
        (PAYMENT_MS, False),  # the payment's own millisecond
    ],
)
def test_an_earlier_pair_payment_counts_only_from_the_lookbacks_start(
    earlier_ms: int, counted: bool
) -> None:
    """mark_pair_history reads the payments of [t - 90 days, t) alone: each sender notes when
    it paid each receiver among them, and a payment counts those strictly before it. So an
    earlier payment of the pair a millisecond before t - 90 days is no part of the pair, one
    exactly at it is, and read_payments reads the pair as present only with one. By hand: a
    payment of 12 April whose pair also paid at the lookback's start, 2 April 00:00, has one
    earlier pair payment, and a pair gap and a pair age of 10 days."""
    window = {"t_ms": T_MS, "LOOKBACK_MS": LOOKBACK_MS}
    assert _where(PAIRS, "recent", window | {"p.event_ts_ms": PAYMENT_MS})
    seen = _where(PAIRS, "recent", window | {"p.event_ts_ms": earlier_ms})
    (earlier,) = _conditions(select_of(PAIRS, "counted").accumulations)
    count = int(seen and holds(earlier, {"ts": earlier_ms, "p.event_ts_ms": PAYMENT_MS}))
    assert count == counted
    present = select_of(READ, "rows").accumulations
    assert " ".join(present) == "p.@pair_present += p.pair_prior_count > 0"
    assert holds(present[2:], {"p.pair_prior_count": count}) is counted


# What mark_pair_history declares, exactly, in order: the latest of a pair's earlier times,
# which gives its gap, is a MaxAccum, and the earliest, which gives its age, a MinAccum.
PAIR_DECLARATIONS = [
    f"UINT LOOKBACK_MS = {LOOKBACK_MS}",
    "SetAccum<STRING> @@payment_vertex_type",
    "SetAccum<STRING> @@payment_to_sender_edge_type",
    "SetAccum<STRING> @@payment_to_receiver_edge_type",
    "MinAccum<VERTEX> @receiver",
    "MapAccum<VERTEX, ListAccum<UINT>> @times_paid",
    "SumAccum<INT> @prior_count",
    "MaxAccum<UINT> @prior_latest",
    "MinAccum<UINT> @prior_earliest",
]
DECLARES = re.compile(r"\w+Accum\b|TYPEDEF |(?:INT|UINT|FLOAT|DOUBLE|STRING|BOOL|DATETIME) ")


def test_the_pair_is_read_from_the_lookbacks_payments_and_stamped_on_them() -> None:
    """Every SELECT of mark_pair_history after its window starts from it: the receivers, the
    times each sender paid each receiver, self-payments aside, and the counts all come from the
    payments of [t - 90 days, t), and only those are stamped, with the count, the time since the
    latest and the time since the earliest. A payment outside the window keeps a stamp of
    another time, which no reader at t reads, since read_payments refuses a payment outside its
    lookback."""
    code = without_comments(PAIRS)
    found = [" ".join(part.split()) for part in code[code.index("{") + 1 :].split(";")]
    assert [text for text in found if DECLARES.match(text)] == PAIR_DECLARATIONS
    sources = {
        target: _source(PAIRS, target)
        for target in ("recent", "received", "senders", "counted", "marked")
    }
    assert sources == {
        "recent": "payments",
        "received": "recent",
        "senders": "received",
        "counted": "received",
        "marked": "recent",
    }
    assert ["payments", "=", "{", "@@payment_vertex_type", "}"] in statements(PAIRS)
    for target in ("senders", "counted"):
        assert select_of(PAIRS, target).conjuncts == {"a != p.@receiver"}
    senders = " ".join(select_of(PAIRS, "senders").accumulations)
    assert senders == "a.@times_paid += ( p.@receiver -> p.event_ts_ms )"
    assert " ".join(select_of(PAIRS, "marked").accumulations) == (
        "p.pair_prior_count = p.@prior_count , "
        "p.pair_gap_ms = CASE WHEN p.@prior_count > 0 THEN p.event_ts_ms - p.@prior_latest "
        "ELSE 0 END , "
        "p.pair_age_ms = CASE WHEN p.@prior_count > 0 THEN p.event_ts_ms - p.@prior_earliest "
        "ELSE 0 END , "
        "p.pair_marked = TRUE"
    )
    assert select_of(READ, "payments").conjuncts == {
        "p.event_ts_ms >= t_ms - LOOKBACK_MS",
        "p.event_ts_ms < t_ms",
    }


@pytest.mark.parametrize(
    ("t_ms", "refused"),
    [(LOOKBACK_MS, True), (LOOKBACK_MS + 1000, False), (T_MS, False), (T_MS + 1, True)],
)
def test_the_pair_mark_refuses_a_t_other_than_a_whole_second_after_the_lookback(
    t_ms: int, refused: bool
) -> None:
    """mark_pair_history takes t, and refuses with invalid_parameters, before it reads or
    writes anything, a t that is not a whole second after the lookback, below which
    t - 90 days would wrap."""
    assert parameters(PAIRS) == {"t_ms": ["UINT"]}
    found = statements(PAIRS)
    (test,) = [s for s in found if s[:1] == ["IF"] and '"invalid_parameters"' in s]
    assert holds(_conditions(test)[0], {"t_ms": t_ms, "LOOKBACK_MS": LOOKBACK_MS}) is refused
    at = found.index(test)
    assert found[at + 1 : at + 3] == [["RETURN"], ["END"]]
    assert at < found.index(["payments", "=", "{", "@@payment_vertex_type", "}"])


def test_the_pair_mark_records_its_t_and_read_payments_checks_it_beside_the_hubs() -> None:
    """mark_pair_history records its t on the "pairs" Mark once every payment of its window is
    stamped, and read_payments refuses with unmarked_time unless both the "hubs" Mark and the
    "pairs" Mark hold its own t, so it never reads the pair stamps of another time; and with
    unmarked_payment unless every payment it reads carries both marks' stamps."""
    marked = [" ".join(s) for s in statements(PAIRS)]
    insert = marked.index('INSERT INTO Mark VALUES ( "pairs" , t_ms )')
    assert marked[insert - 1].startswith("marked = SELECT p FROM recent : p")
    assert marked[insert + 1] == 'PRINT "ok" AS status , marked.size ( ) AS marked'
    assert select_of(READ, "current_marks").conjuncts == {
        '( m.id == "hubs" OR m.id == "pairs" )',
        "m.at_ms == t_ms",
    }
    read = [" ".join(s) for s in statements(READ)]
    check = read.index('IF current_marks.size ( ) != 2 THEN PRINT "unmarked_time" AS status')
    assert read[check + 1 : check + 3] == ["RETURN", "END"]
    assert select_of(READ, "marked").conjuncts == {"p.gaps_marked", "p.pair_marked"}
    check = read.index(
        'IF marked.size ( ) != payments.size ( ) THEN PRINT "unmarked_payment" AS status'
    )
    assert read[check + 1 : check + 3] == ["RETURN", "END"]
