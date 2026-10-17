"""The rules every query's text must pass (testing.gsql_rules), on small queries written here:
one that keeps every form passes them all, and each rule fails on a slip of its own, the seven
planted slips of the October 17 build among them, all of which once passed every text test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing import gsql_rules
from mule_pattern_learner.testing.gsql_rules import (
    accumulator_rule,
    attribute_rule,
    bound_rule,
    call_rule,
    clock_rule,
    common_problems,
    field_rule,
    heap_rule,
    label_rule,
    leaf_rule,
    name_rule,
    parameter_rule,
    pattern_rule,
    projection_fields,
    projection_rule,
    refusal_rule,
    status_fields,
    structure_rule,
    time_rule,
    tuple_rule,
    version_rule,
    window_rule,
    write_rule,
)
from mule_pattern_learner.tigergraph.installer import definitions, text_version
from mule_pattern_learner.tigergraph.queries import written_files

NAME = "read_example"
# A query that keeps every form the rules read: the refusal of scored accounts that do not
# exist at t, payments in a window before t, visible and current links, an account that
# exists, known starts, the crowd count and the leaf test, a capped heap, a tuple and a
# projected PRINT.
GOOD = """USE GRAPH Mule_Pattern_Learner

/* Every form the text rules read. */
CREATE OR REPLACE QUERY read_example(
    UINT t_ms,
    SET<VERTEX<Account>> scored
) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    TYPEDEF TUPLE<UINT ts, STRING id> Lot;
    HeapAccum<Lot>(256, ts DESC, id DESC) @@inflows;
    SumAccum<INT> @@received;
    MinAccum<UINT> @first_seen;
    MaxAccum<UINT> @since_start;
    SetAccum<STRING> @crowd_customers, @co_users;
    OrAccum @leaf;
    UINT lookback_ms = 7776000000;

    IF t_ms <= lookback_ms THEN
        PRINT "invalid_parameters" AS status;
        RETURN;
    END;
    Scored = {scored};
    Existing = SELECT s FROM Scored:s WHERE s.opened_ms < t_ms;
    IF Existing.size() != Scored.size() THEN
        PRINT "invisible_account" AS status;
        RETURN;
    END;
    Inflows = SELECT p FROM Scored:s -(Account_Received_Transaction>:r)- Payment_Transaction:p
      WHERE p.event_ts_ms >= t_ms - 2592000000 AND p.event_ts_ms < t_ms
      ACCUM @@inflows += Lot(p.event_ts_ms, p.id), @@received += 1;
    Owners = SELECT q FROM Scored:s -(Account_Owned_By_Party>:o)- Party:q
      WHERE o.valid_from_ms < t_ms AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms);
    Devices = SELECT d FROM Owners:q -(Party_Uses_Device>:l)- Device:d
      WHERE l.valid_from_ms < t_ms
        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)
      ACCUM IF l.valid_from_ms > q.joined_ms AND l.valid_from_ms > t_ms - lookback_ms THEN
          d.@since_start += t_ms - l.valid_from_ms
      END;
    Crowd = SELECT d FROM Devices:d -(Device_Used_By_Party>:l)- Party:q
      WHERE l.valid_from_ms < t_ms AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms)
      ACCUM d.@crowd_customers += q.id;
    Leaves = SELECT d FROM Devices:d POST-ACCUM d.@leaf += d.@crowd_customers.size() > 8;
    CoUsers = SELECT q FROM Devices:d -(Device_Used_By_Party>:l)- Party:q
      WHERE NOT d.@leaf AND l.valid_from_ms < t_ms
        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)
      ACCUM d.@co_users += q.id;
    Accounts = SELECT a FROM CoUsers:q -(Party_Owns_Account>:o)- Account:a
      WHERE a.opened_ms < t_ms AND o.valid_from_ms < t_ms
        AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms)
        AND NOT a.is_external AND a.account_type == "deposit";
    Seen = SELECT a FROM Accounts:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:p
      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms
      ACCUM a.@first_seen += p.event_ts_ms;
    PRINT "ok" AS status, "read_example-000000000000" AS version, @@received AS received;
    PRINT Accounts[Accounts.id AS account_id, Accounts.@first_seen AS first_seen] AS accounts;
}
"""


def _definition(text: str = GOOD, name: str = NAME) -> str:
    return definitions(text)[name]


def _changed(old: str, new: str, text: str = GOOD) -> str:
    """GOOD with one change, which must apply exactly once."""
    assert text.count(old) == 1, old
    return text.replace(old, new)


def test_a_query_that_keeps_every_form_passes_every_rule() -> None:
    assert common_problems(NAME, GOOD) == []


def test_every_written_query_passes_the_rules_that_read_its_file() -> None:
    # Each query step also holds its query to the rules that read the Python above it.
    for name, relative in written_files().items():
        assert common_problems(name, (GSQL_DIR / relative).read_text()) == [], relative


# The seven planted slips of the October 17 build, as one-line changes to GOOD, and a word
# the problem each causes names.
PLANTED = [
    (
        "p.event_ts_ms < t_ms\n      ACCUM @@inflows",
        "p.event_ts_ms <= t_ms\n      ACCUM @@inflows",
        "<=",
    ),
    (
        "WHERE o.valid_from_ms < t_ms AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms);",
        "WHERE o.valid_from_ms < t_ms;",
        "valid_from_ms",
    ),
    (
        "o.valid_to_ms > t_ms - lookback_ms);",
        "o.valid_to_ms > t_ms - lookback_ms) ORDER BY q.joined_ms ASC LIMIT 8;",
        "joined_ms",
    ),
    (
        "PRINT Accounts[Accounts.id AS account_id, Accounts.@first_seen AS first_seen] AS accounts;",
        "PRINT Accounts;",
        "prints all of Accounts",
    ),
    ("WHERE NOT d.@leaf AND l.valid_from_ms", "WHERE l.valid_from_ms", "leaf test"),
    ('a.account_type == "deposit"', 'a.account_type == "deposit" AND a.pu_label == 0', "pu_label"),
    ("HeapAccum<Lot>(256, ts DESC, id DESC)", "HeapAccum<Lot>(ts DESC, id DESC)", "capacity"),
]


@pytest.mark.parametrize(("old", "new", "word"), PLANTED)
def test_each_planted_slip_of_the_october_17_build_is_caught(old: str, new: str, word: str) -> None:
    problems = common_problems(NAME, _changed(old, new))
    assert problems and any(word in problem for problem in problems), problems


# Changes the time rule refuses: comparisons with t outside the allowed forms, t read as a
# value, and dates of record or link ends read as values.
TIME_SLIPS = [
    ("p.event_ts_ms >= t_ms - 2592000000", "p.event_ts_ms > t_ms - 2592000000"),
    ("p.event_ts_ms < t_ms\n      ACCUM @@inflows", "t_ms > p.event_ts_ms\n      ACCUM @@inflows"),
    ("a.opened_ms < t_ms", "a.opened_ms <= t_ms"),
    ("l.valid_from_ms > q.joined_ms", "l.valid_from_ms >= q.joined_ms"),
    ("l.valid_from_ms > q.joined_ms", "q.joined_ms < l.valid_from_ms"),
    ("l.valid_from_ms > t_ms - lookback_ms", "t_ms < l.valid_from_ms"),
    # Known after the lookback's start is strict; new over w is for windows inside it.
    ("l.valid_from_ms > t_ms - lookback_ms", "l.valid_from_ms >= t_ms - lookback_ms"),
    ("d.@since_start += t_ms - l.valid_from_ms", "d.@since_start += t_ms"),
    ("d.@since_start += t_ms - l.valid_from_ms", "d.@since_start += l.valid_to_ms"),
    ("d.@since_start += t_ms - l.valid_from_ms", "d.@since_start += q.joined_ms"),
    ("@@received += 1", "@@received += t_ms + 1"),
    ("ACCUM a.@first_seen += p.event_ts_ms", "ACCUM a.@first_seen += a.opened_ms"),
    (
        "(l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n      ACCUM d.@co_users",
        "(l.valid_to_ms == 0 OR l.valid_to_ms >= t_ms - 2592000000)\n      ACCUM d.@co_users",
    ),
    (
        "(l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n      ACCUM d.@co_users",
        "(l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 604800000)\n      ACCUM d.@co_users",
    ),
    (
        "AND NOT a.is_external",
        "AND (a.closed_ms == 0 OR a.closed_ms >= t_ms) AND NOT a.is_external",
    ),
]


@pytest.mark.parametrize(("old", "new"), TIME_SLIPS)
def test_the_time_rule_refuses_every_other_comparison_at_t(old: str, new: str) -> None:
    assert time_rule(NAME, _definition()) == []
    assert time_rule(NAME, _definition(_changed(old, new))), new


def test_the_time_rule_allows_the_designs_other_forms() -> None:
    allowed = [
        # An own payment before a link's start, for the known-start rule.
        "Seen = SELECT a FROM Accounts:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:p",
        "Seen = SELECT a FROM Accounts:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:p"
        " -(Transaction_Used_Device>:u)- Device:d -(<Party_Uses_Device:l)- Party:q",
    ]
    text = _changed(allowed[0], allowed[1])
    text = _changed(
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms",
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < l.valid_from_ms\n"
        "        AND l.valid_from_ms < t_ms\n"
        "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms)\n"
        "        AND l.valid_from_ms >= t_ms - 2592000000",
        text,
    )
    assert time_rule(NAME, _definition(text)) == []
    assert bound_rule(NAME, _definition(text)) == []
    # A known start held by an accumulator, and t tested against the parameters alone.
    text = _changed(
        "IF t_ms <= lookback_ms THEN",
        "IF t_ms <= lookback_ms OR t_ms % 86400000 != 0 OR @@received > t_ms - lookback_ms THEN",
    )
    assert time_rule(NAME, _definition(text)) == []


def test_closed_ms_and_opened_ms_are_read_only_where_the_design_says() -> None:
    open_test = "AND (a.closed_ms == 0 OR a.closed_ms >= t_ms) AND NOT a.is_external"
    text = _changed("AND NOT a.is_external", open_test)
    # The open test belongs to list_accounts alone.
    accounts = text.replace(NAME, "list_accounts")
    assert time_rule("list_accounts", _definition(accounts, "list_accounts")) == []
    assert time_rule(NAME, _definition(text))
    # read_answer_key prints opened_ms, for evaluation's groups; no other query does.
    printed = _changed(
        "Accounts.id AS account_id,", "Accounts.id AS account_id, Accounts.opened_ms AS opened_ms,"
    )
    assert (
        time_rule(
            "read_answer_key",
            _definition(printed.replace(NAME, "read_answer_key"), "read_answer_key"),
        )
        == []
    )
    assert any(
        "opened_ms as a value" in problem for problem in time_rule(NAME, _definition(printed))
    )
    # read_world has no t, and sums every time into the world's identity.
    world = _changed("@@received += 1", "@@received += floor(p.event_ts_ms / 1000) + q.joined_ms")
    assert (
        time_rule("read_world", _definition(world.replace(NAME, "read_world"), "read_world")) == []
    )


BOUND_SLIPS = [
    # A payment with no window's start, and one with no bound below t.
    (
        "WHERE p.event_ts_ms >= t_ms - 2592000000 AND p.event_ts_ms < t_ms",
        "WHERE p.event_ts_ms < t_ms",
    ),
    (
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms",
        "WHERE p.event_ts_ms >= t_ms - lookback_ms",
    ),
    # A bound inside an OR bounds nothing.
    (
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms",
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND (p.event_ts_ms < t_ms OR p.amount > 0)",
    ),
    # A link stepped along without its end test, and an owned account without existing.
    (
        "WHERE l.valid_from_ms < t_ms AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms)",
        "WHERE l.valid_from_ms < t_ms",
    ),
    ("WHERE a.opened_ms < t_ms AND o.valid_from_ms", "WHERE o.valid_from_ms"),
]


@pytest.mark.parametrize(("old", "new"), BOUND_SLIPS)
def test_the_bound_rule_requires_a_bound_at_t_on_every_step(old: str, new: str) -> None:
    assert bound_rule(NAME, _definition()) == []
    assert bound_rule(NAME, _definition(_changed(old, new))), new
    # read_world and read_answer_key have no t.
    other = _changed(old, new).replace(NAME, "read_world")
    assert bound_rule("read_world", _definition(other, "read_world")) == []


# Where GOOD's last statements begin, so a test can add statements before them.
BEFORE_PRINT = '    PRINT "ok" AS status'


def _added(statements: str, text: str = GOOD) -> str:
    """GOOD with statements added before its PRINTs."""
    return _changed(BEFORE_PRINT, statements + BEFORE_PRINT, text)


# Steps no bound covers, each with a word of the problem it causes: a set of a whole type or
# from to_vertex_set was bounded nowhere; a wildcard, a mixed type, an edge without a type or
# a set the rules cannot follow could reach anything; an account scanned whole may open
# after t.
UNBOUNDED_STEPS = [
    (
        "    All = {Payment_Transaction.*};\n    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "does not bound p",
    ),
    (
        '    All = to_vertex_set(@@ids, "Zelle_Transfer");\n'
        "    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "does not bound p",
    ),
    (
        "    Whole = {Payment_Transaction.*};\n"
        "    All = Inflows UNION Whole;\n"
        "    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "does not bound p",
    ),
    (
        "    X = SELECT x FROM Scored:s -(Account_Received_Transaction>:r)-"
        " (Payment_Transaction|Account):x ACCUM @@received += 1;\n",
        "does not bound x",
    ),
    (
        "    X = SELECT x FROM Scored:s -(Account_Received_Transaction>:r)- _:x"
        " ACCUM @@received += 1;\n",
        "steps to _",
    ),
    (
        "    X = SELECT x FROM Scored:s -(Account_Received_Transaction>:r)- ANY:x"
        " ACCUM @@received += 1;\n",
        "steps to ANY",
    ),
    ("    X = SELECT q FROM Scored:s -(_>:o)- Party:q ACCUM @@received += 1;\n", "steps along _"),
    ("    X = SELECT q FROM Scored:s -(:o)- Party:q ACCUM @@received += 1;\n", "edge of no type"),
    # A set declared with its type, or assigned by a set expression in parentheses, is
    # followed like any other.
    (
        "    All (Payment_Transaction) = {Payment_Transaction.*};\n"
        "    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "does not bound p",
    ),
    (
        "    Whole = {Payment_Transaction.*};\n"
        "    All = (Inflows UNION (Whole MINUS Inflows));\n"
        "    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "does not bound p",
    ),
    # A set assigned in a form the rules do not follow is no set they know.
    (
        "    All = Inflows UNION @@ids;\n    X = SELECT p FROM All:p ACCUM @@received += 1;\n",
        "steps to All",
    ),
    ("    X = SELECT a FROM Account:a ACCUM @@received += 1;\n", "may not exist at t"),
    (
        "    All = {Account.*};\n"
        "    X = SELECT a FROM All:a -(Account_Bound_From_Token>:b)- Token:k"
        " ACCUM @@received += 1;\n",
        "may not exist at t",
    ),
]


@pytest.mark.parametrize(("statements", "word"), UNBOUNDED_STEPS)
def test_a_step_no_bound_covers_is_refused(statements: str, word: str) -> None:
    text = _changed(
        "SumAccum<INT> @@received;",
        "SumAccum<INT> @@received;\n    SetAccum<STRING> @@ids;",
        _added(statements),
    )
    problems = common_problems(NAME, text)
    assert any(word in problem for problem in problems), problems


def test_sets_of_whole_types_and_of_vertex_accumulators_are_bounded_where_they_are_read() -> None:
    # A set of a whole type is bounded in the SELECT that reads it.
    whole = _added(
        "    All = {Payment_Transaction.*};\n"
        "    X = SELECT p FROM All:p\n"
        "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n"
        "      ACCUM @@received += 1;\n"
        "    Every = {Account.*};\n"
        "    Y = SELECT a FROM Every:a WHERE a.opened_ms < t_ms ACCUM @@received += 1;\n"
    )
    assert common_problems(NAME, whole) == []
    # Accounts reached as a payment's sender or receiver opened before it.
    paid = _added(
        "    Z = SELECT a FROM Inflows:p -(Transaction_From_Account>:f)- Account:a"
        " ACCUM @@received += 1;\n"
    )
    assert common_problems(NAME, paid) == []
    # A vertex accumulator holds vertices that SELECTs bounded, so its set is chosen and its
    # accounts need no second existence test; a set of a whole type does.
    kept = _changed(
        "SumAccum<INT> @@received;",
        "SumAccum<INT> @@received;\n    SetAccum<VERTEX<Account>> @@kept;",
        _added(
            "    Kept = {@@kept};\n"
            "    K = SELECT q FROM Kept:a -(Account_Owned_By_Party>:o)- Party:q\n"
            "      WHERE o.valid_from_ms < t_ms\n"
            "        AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms);\n"
        ),
    )
    kept = _changed(
        "ACCUM a.@first_seen += p.event_ts_ms",
        "ACCUM a.@first_seen += p.event_ts_ms, @@kept += a",
        kept,
    )
    assert common_problems(NAME, kept) == []
    scanned = _changed("Kept = {@@kept};", "Kept = {Account.*};", kept)
    (problem,) = common_problems(NAME, scanned)
    assert problem.endswith("reaches an account that may not exist at t")
    # A query with no t, read_labels, scans every account.
    labels = _definition(
        "USE GRAPH Mule_Pattern_Learner\n"
        "CREATE OR REPLACE QUERY read_labels() FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {\n"
        "    All = {Account.*};\n"
        "    Identified = SELECT a FROM All:a WHERE a.pu_label == 1;\n"
        "    PRINT Identified[Identified.id AS account_id] AS accounts;\n"
        "}\n",
        "read_labels",
    )
    assert bound_rule("read_labels", labels) == []


def test_a_payment_may_be_bounded_by_another_payment_of_its_pattern() -> None:
    text = _changed(
        "Seen = SELECT a FROM Accounts:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:p\n"
        "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms",
        "Seen = SELECT a FROM Payment_Transaction:p -(Transaction_From_Account>:e)- Account:a\n"
        "      -(Account_Initiated_Transaction>:f)- Payment_Transaction:x\n"
        "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n"
        "        AND x.event_ts_ms >= p.event_ts_ms AND x.event_ts_ms < t_ms\n"
        "        AND x.event_ts_ms < p.event_ts_ms + 86400000",
    )
    assert bound_rule(NAME, _definition(text)) == []
    assert time_rule(NAME, _definition(text)) == []
    (problem,) = bound_rule(
        NAME, _definition(text.replace(" AND x.event_ts_ms >= p.event_ts_ms", ""))
    )
    assert problem.endswith("does not bound x from a window's start")
    # Within a day of a payment before t can reach past t: x needs its own bound below t.
    (problem,) = bound_rule(NAME, _definition(text.replace(" AND x.event_ts_ms < t_ms", "")))
    assert problem.endswith("does not bound x below t")


def test_the_leaf_rule_lets_only_the_crowd_count_step_to_a_leafs_users() -> None:
    assert leaf_rule(NAME, _definition()) == []
    # A crowd count that accumulates anything else is no crowd count.
    text = _changed(
        "ACCUM d.@crowd_customers += q.id;",
        "ACCUM d.@crowd_customers += q.id, d.@co_users += q.id;",
    )
    assert leaf_rule(NAME, _definition(text))
    # A step to a leaf's users that accumulates nothing collects them.
    text = _changed("\n      ACCUM d.@crowd_customers += q.id;", ";")
    assert leaf_rule(NAME, _definition(text))
    # So does a step to the payments seen on it.
    text = _changed(
        "CoUsers = SELECT q FROM Devices:d -(Device_Used_By_Party>:l)- Party:q\n"
        "      WHERE NOT d.@leaf AND l.valid_from_ms < t_ms\n"
        "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n"
        "      ACCUM d.@co_users += q.id;",
        "Used = SELECT p FROM Devices:d -(Device_Used_By_Transaction>:u)- Payment_Transaction:p\n"
        "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n"
        "      ACCUM d.@co_users += p.id;",
    )
    text = text.replace("FROM CoUsers:q", "FROM Owners:q")
    assert leaf_rule(NAME, _definition(text))
    excluded = _changed(
        "WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n      ACCUM d.@co_users",
        "WHERE NOT d.@leaf AND p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n"
        "      ACCUM d.@co_users",
        text,
    )
    assert leaf_rule(NAME, _definition(excluded)) == []


def test_label_columns_are_named_only_where_the_labels_guard_allows() -> None:
    labelled = _changed('a.account_type == "deposit"', "a.mule_onset_ms > 0 AND a.pu_label == 1")
    assert label_rule(NAME, _definition(labelled)) == [
        f"{NAME} names the label column mule_onset_ms",
        f"{NAME} names the label column pu_label",
    ]
    reader = labelled.replace(NAME, "read_labels")
    assert label_rule("read_labels", _definition(reader, "read_labels")) == []
    key = labelled.replace(NAME, "read_answer_key")
    assert label_rule("read_answer_key", _definition(key, "read_answer_key")) == [
        "read_answer_key names the label column pu_label"
    ]
    # A label in a string literal is named too.
    named = _changed('"invalid_parameters"', '"is_mule"')
    assert label_rule(NAME, _definition(named)) == [f"{NAME} names the label column is_mule"]


def test_writes_are_refused_in_a_query() -> None:
    assert write_rule(NAME, _definition()) == []
    written = _changed(
        "ACCUM a.@first_seen += p.event_ts_ms",
        "ACCUM a.@first_seen += p.event_ts_ms, a.opened_ms = 0",
    )
    assert write_rule(NAME, _definition(written)) == [
        f"{NAME} writes to the graph: attribute assignment"
    ]


def test_accumulators_tuples_and_attributes_are_declared() -> None:
    definition = _definition()
    assert accumulator_rule(NAME, definition) == []
    assert tuple_rule(NAME, definition) == []
    assert attribute_rule(NAME, definition) == []
    undeclared = _changed("@@received += 1", "@@receivd += 1")
    assert accumulator_rule(NAME, _definition(undeclared)) == [
        f"{NAME} uses the undeclared @@receivd"
    ]
    short = _changed("Lot(p.event_ts_ms, p.id)", "Lot(p.event_ts_ms)")
    assert tuple_rule(NAME, _definition(short)) == [f"{NAME} builds Lot with 1 fields"]
    misread = _changed("a.account_type ==", "a.payment_rail ==")
    assert attribute_rule(NAME, _definition(misread)) == [
        f"{NAME} reads a.payment_rail, which Account does not have"
    ]
    edge = _changed(
        "o.valid_from_ms < t_ms AND (o.valid_to_ms == 0 OR o.valid_to_ms > t_ms - lookback_ms);",
        "o.event_ts_ms < t_ms;",
    )
    assert (
        f"{NAME} reads o.event_ts_ms, which Account_Owned_By_Party does not have"
        in attribute_rule(NAME, _definition(edge))
    )
    stray = _changed("@@received += 1", "@@received += x.amount")
    assert attribute_rule(NAME, _definition(stray)) == [
        f"{NAME} reads x.amount of no alias or tuple"
    ]


def test_a_name_starts_with_a_verb() -> None:
    assert [name_rule(name) for name in ("list_accounts", "read_hubs", "profile_hubs")] == [
        [],
        [],
        [],
    ]
    assert name_rule("fetch_subgraphs") == [
        "fetch_subgraphs does not start with list_ or read_ or profile_"
    ]
    assert name_rule("hubs_on") != []


def test_a_query_file_is_for_the_graph_in_syntax_v2() -> None:
    assert gsql_rules.header_rule(NAME, GOOD) == []
    assert gsql_rules.header_rule(NAME, GOOD.replace("SYNTAX V2 ", "")) == [
        f"{NAME} is not FOR GRAPH Mule_Pattern_Learner SYNTAX V2"
    ]
    assert gsql_rules.header_rule(NAME, "\n" + GOOD)


def test_a_file_defines_its_query_alone() -> None:
    assert common_problems("read_other", GOOD) == [
        "The file of read_other defines ['read_example']"
    ]


def test_the_version_is_the_hash_of_the_text() -> None:
    definition = _definition()
    version = text_version(f"{NAME}-", definition, "read_example-000000000000")
    stamped = GOOD.replace("read_example-000000000000", version)
    assert version_rule(NAME, _definition(stamped), version) == []
    assert version_rule(NAME, definition, version) == [
        f"{NAME} does not print {version} AS version"
    ]
    changed = _changed("HeapAccum<Lot>(256,", "HeapAccum<Lot>(255,", stamped)
    assert version_rule(NAME, _definition(changed), version) == [
        f"{NAME}'s text has changed since {version}: run testing.versions"
    ]
    assert version_rule(NAME, definition, None) == [f"{NAME} has no version in the registry"]


@dataclass(frozen=True)
class ExampleRow:
    account_id: str
    first_seen: int


@dataclass(frozen=True)
class ExampleHead:
    received: int


def test_parameters_and_printed_fields_are_the_pythons() -> None:
    definition = _definition()

    def params(t_ms: int, scored: list[str]) -> dict[str, object]:
        return {"t_ms": t_ms, "scored": scored}

    built: Callable[..., dict[str, object]] = params
    assert parameter_rule(NAME, definition, built(1, [])) == []
    assert parameter_rule(NAME, definition, ["t_ms"]) == [
        f"{NAME} takes ['t_ms', 'scored'], not ['t_ms']"
    ]
    assert projection_fields(definition, "Accounts") == ["account_id", "first_seen"]
    assert status_fields(definition) == ["received"]
    assert field_rule("Accounts", projection_fields(definition, "Accounts"), ExampleRow) == []
    assert field_rule("status", status_fields(definition), ExampleHead) == []
    assert field_rule("status", ["received", "extra"], ExampleHead) == [
        "status prints ['received', 'extra'], not ['received']"
    ]


def test_projections_and_heaps_alone() -> None:
    definition = _definition()
    assert projection_rule(NAME, definition) == []
    assert heap_rule(NAME, definition) == []
    # A vertex set printed among other fields is still printed whole.
    whole = _changed("@@received AS received;", "@@received AS received, Owners AS owners;")
    assert projection_rule(NAME, _definition(whole)) == [f"{NAME} prints all of Owners"]
    # Printing a set's size prints no attribute.
    size = _changed("@@received AS received;", "@@received AS received, Owners.size() AS owners;")
    assert projection_rule(NAME, _definition(size)) == []


def _with_ids(text: str) -> str:
    """text with a global SetAccum @@ids declared."""
    return _changed(
        "SumAccum<INT> @@received;", "SumAccum<INT> @@received;\n    SetAccum<STRING> @@ids;", text
    )


@pytest.mark.parametrize(
    ("statements", "printed"),
    [
        # A set declared with its type, a set expression in parentheses, a vertex parameter
        # and a set the rules cannot follow each print every attribute, labels among them.
        ("    All (Account) = {Account.*};\n    PRINT All;\n", "All"),
        ("    All = (Accounts UNION Scored);\n    PRINT All;\n", "All"),
        ("    All = Accounts UNION Scored;\n    PRINT All AS everyone;\n", "All"),
        ("    PRINT scored;\n", "scored"),
        ("    All = Accounts UNION @@ids;\n    PRINT @@received AS received, All;\n", "All"),
    ],
)
def test_a_vertex_set_is_never_printed_whole(statements: str, printed: str) -> None:
    problems = common_problems(NAME, _with_ids(_added(statements)))
    assert f"{NAME} prints all of {printed}" in problems, problems


def test_a_print_of_values_is_no_print_of_a_set() -> None:
    values = _added("    PRINT t_ms AS t, lookback_ms AS lookback, Accounts.size() AS accounts;\n")
    assert projection_rule(NAME, _definition(values)) == []


# A step between a device and its users in another spelling than GOOD's, without the leaf
# test: users first, payments seen on the device, two steps through a device, and a step
# between a device and users chosen before (a closure, which also reads leaves).
LEAF_SLIPS = [
    "    X = SELECT q FROM Party:q -(Party_Uses_Device>:l)- Devices:d\n"
    "      WHERE l.valid_from_ms < t_ms\n"
    "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n"
    "      ACCUM d.@co_users += q.id;\n",
    "    X = SELECT p FROM Payment_Transaction:p -(Transaction_Used_Device>:u)- Devices:d\n"
    "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms\n"
    "      ACCUM d.@co_users += p.id;\n",
    "    X = SELECT r FROM Owners:q -(Party_Uses_Device>:l)- Device:d"
    " -(Device_Used_By_Party>:m)- Party:r\n"
    "      WHERE l.valid_from_ms < t_ms\n"
    "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n"
    "        AND m.valid_from_ms < t_ms\n"
    "        AND (m.valid_to_ms == 0 OR m.valid_to_ms > t_ms - 2592000000)\n"
    "      ACCUM d.@co_users += r.id;\n",
    "    X = SELECT q FROM Devices:d -(Device_Used_By_Party>:l)- Owners:q\n"
    "      WHERE l.valid_from_ms < t_ms\n"
    "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - 2592000000)\n"
    "      ACCUM d.@co_users += q.id;\n",
]


@pytest.mark.parametrize("statements", LEAF_SLIPS)
def test_every_step_between_a_device_and_its_users_applies_the_leaf_test(statements: str) -> None:
    problems = common_problems(NAME, _added(statements))
    assert any("without the leaf test" in problem for problem in problems), problems
    # With the leaf test the same step passes.
    where = "WHERE l.valid_from_ms" if "WHERE l." in statements else "WHERE p.event_ts_ms"
    tested = statements.replace(where, where.replace("WHERE ", "WHERE NOT d.@leaf AND "), 1)
    assert common_problems(NAME, _added(tested)) == []


# Changes that defeat the leaf test without dropping it: a crowd count that selects the
# users, a leaf flag never set or set in another form, a crowd count that reads no user, and
# a crowd read other than through its size.
LEAF_FLAG_SLIPS = [
    ("Crowd = SELECT d FROM Devices:d", "Crowd = SELECT q FROM Devices:d", "leaf test"),
    (
        "    Leaves = SELECT d FROM Devices:d POST-ACCUM d.@leaf += d.@crowd_customers.size() > 8;\n",
        "",
        "before a crowd count sets it",
    ),
    ("d.@crowd_customers.size() > 8", "d.@crowd_customers.size() > 80", "sets the leaf flag"),
    (
        "POST-ACCUM d.@leaf += d.@crowd_customers.size() > 8",
        'POST-ACCUM IF d.type == "IP" THEN d.@leaf += d.@crowd_customers.size() > 8 END',
        "sets the leaf flag",
    ),
    (
        "Crowd = SELECT d FROM Devices:d -(Device_Used_By_Party>:l)- Party:q\n"
        "      WHERE l.valid_from_ms < t_ms AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms)\n"
        "      ACCUM d.@crowd_customers += q.id;",
        "Crowd = SELECT d FROM Devices:d ACCUM d.@crowd_customers += d.id;",
        "sets the leaf flag",
    ),
    (
        "ACCUM d.@co_users += q.id;",
        "ACCUM d.@co_users += q.id, @@ids += d.@crowd_customers;",
        "other than through its size",
    ),
]


@pytest.mark.parametrize(("old", "new", "word"), LEAF_FLAG_SLIPS)
def test_the_leaf_flag_comes_from_a_crowd_count(old: str, new: str, word: str) -> None:
    problems = leaf_rule(NAME, _definition(_changed(old, new)))
    assert any(word in problem for problem in problems), problems


def test_the_crowd_threshold_is_the_settings() -> None:
    named = _changed("d.@crowd_customers.size() > 8", "d.@crowd_customers.size() > crowd_threshold")
    declared = _changed(
        "UINT lookback_ms = 7776000000;",
        "UINT lookback_ms = 7776000000;\n    UINT crowd_threshold = 8;",
        named,
    )
    assert common_problems(NAME, declared) == []
    raised = declared.replace("UINT crowd_threshold = 8;", "UINT crowd_threshold = 100;")
    assert window_rule(NAME, _definition(raised)) == [
        f"{NAME} reads crowd_threshold without declaring it once as UINT crowd_threshold = 8 "
        "or taking it as a parameter"
    ]


# Edges of several hops hide the payments and users between their ends.
@pytest.mark.parametrize(
    "statements",
    [
        "    X = SELECT b FROM Scored:s"
        " -(Account_Initiated_Transaction>.Transaction_To_Account>)- Account:b"
        " ACCUM @@received += 1;\n",
        "    X = SELECT b FROM Scored:s"
        " -((Account_Initiated_Transaction>|Transaction_To_Account>)*2)- Account:b"
        " ACCUM @@received += 1;\n",
        "    X = SELECT a FROM Devices:d"
        " -((Device_Used_By_Transaction>|Transaction_From_Account>)*2)- Account:a"
        " WHERE NOT d.@leaf ACCUM @@received += 1;\n",
    ],
)
def test_an_edge_of_several_hops_is_refused(statements: str) -> None:
    problems = common_problems(NAME, _added(statements))
    assert any("several edges at once" in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("old", "new", "call"),
    [
        (
            "@@received += 1",
            '@@received += s.outdegree("Account_Received_Transaction")',
            "outdegree",
        ),
        (
            "@@received += 1",
            '@@received += s.neighbors("Account_Received_Transaction").size()',
            "neighbors",
        ),
        (
            "ACCUM a.@first_seen += p.event_ts_ms",
            'ACCUM a.@first_seen += a.getAttr("opened_ms", "UINT")',
            "getattr",
        ),
        ("@@received += 1", '@@received += evaluate("p.amount", "int")', "evaluate"),
    ],
)
def test_a_built_in_that_reads_around_the_rules_is_refused(old: str, new: str, call: str) -> None:
    assert structure_rule(NAME, _definition()) == []
    assert structure_rule(NAME, _definition(_changed(old, new))) == [
        f"{NAME} calls {call}, which reads the graph around the rules"
    ]


HUBS = """USE GRAPH Mule_Pattern_Learner
CREATE OR REPLACE QUERY read_hubs(UINT t_ms) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    SumAccum<INT> @@busy;
    All = {Account.*};
    Busy = SELECT a FROM All:a
      WHERE a.opened_ms < t_ms
        AND a.outdegree("Account_Initiated_Transaction")
          + a.outdegree("Account_Received_Transaction") > 200
      ACCUM @@busy += 1;
    PRINT "ok" AS status, @@busy AS busy;
}
"""


def test_read_hubs_alone_prefilters_by_all_time_payments() -> None:
    hubs = definitions(HUBS)["read_hubs"]
    assert structure_rule("read_hubs", hubs) == []
    assert common_problems("read_hubs", HUBS) == []
    # Anywhere else, over another edge, or against another bound, it reads after t.
    other = HUBS.replace("read_hubs", NAME)
    assert structure_rule(NAME, _definition(other)) == [
        f"{NAME} calls outdegree, which reads the graph around the rules"
    ]
    for old, new in (
        ('"Account_Received_Transaction"', '"Account_Owned_By_Party"'),
        ("> 200", "> 100"),
        ("ACCUM @@busy += 1", 'ACCUM @@busy += a.outdegree("Account_Initiated_Transaction")'),
    ):
        changed = definitions(HUBS.replace(old, new))["read_hubs"]
        assert structure_rule("read_hubs", changed), new


# Windows and durations that reach before the lookback's start.
WINDOW_SLIPS = [
    ("p.event_ts_ms >= t_ms - 2592000000", "p.event_ts_ms >= t_ms - 31536000000"),
    ("UINT lookback_ms = 7776000000;", "UINT lookback_ms = 31536000000;"),
    ("UINT lookback_ms = 7776000000;", "UINT lookback_ms = 7776000001;"),
    (
        "UINT lookback_ms = 7776000000;",
        "UINT lookback_ms = 7776000000;\n    lookback_ms = 31536000000;",
    ),
    # Days since a date before the lookback, which is the Monday's index.
    ("@@received += 1", "@@received += t_ms - 1704067200000"),
    # A start new over the whole lookback is the known start's slip in another spelling.
    ("l.valid_from_ms > t_ms - lookback_ms", "l.valid_from_ms >= t_ms - 7776000000"),
]


@pytest.mark.parametrize(("old", "new"), WINDOW_SLIPS)
def test_every_window_lies_inside_the_lookback(old: str, new: str) -> None:
    assert window_rule(NAME, _definition()) == []
    problems = common_problems(NAME, _changed(old, new))
    assert problems and all(" reads " in problem for problem in problems), problems


def test_the_lookback_may_be_a_parameter_never_set() -> None:
    text = _changed("    UINT t_ms,\n", "    UINT t_ms,\n    UINT lookback_ms,\n")
    text = _changed("    UINT lookback_ms = 7776000000;\n", "", text)
    assert common_problems(NAME, text) == []
    changed = _changed(
        "    Scored = {scored};", "    lookback_ms = 31536000000;\n    Scored = {scored};", text
    )
    assert window_rule(NAME, _definition(changed)) == [f"{NAME} sets its parameter lookback_ms"]


@pytest.mark.parametrize(
    ("new", "word"),
    [
        ("@@received += datetime_to_epoch(now())", "now"),
        ("@@received += hour(epoch_to_datetime(p.event_ts_ms / 1000))", "hour"),
        ("@@received += p.event_ts_ms % 86400000", "remainder"),
        ("@@received += (t_ms - p.event_ts_ms) % 3600000", "remainder"),
    ],
)
def test_no_query_reads_the_clock(new: str, word: str) -> None:
    assert clock_rule(NAME, _definition()) == []
    problems = clock_rule(NAME, _definition(_changed("@@received += 1", new)))
    assert any(word in problem for problem in problems), problems
    # A test of the parameter t reads no event's clock.
    test = _changed(
        "IF t_ms <= lookback_ms THEN", "IF t_ms <= lookback_ms OR t_ms % 86400000 != 0 THEN"
    )
    assert clock_rule(NAME, _definition(test)) == []


def test_bounds_by_other_payments_must_reach_a_window() -> None:
    # Each payment bounded below only by the other bounds neither.
    circular = _changed(
        "Seen = SELECT a FROM Accounts:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:p\n"
        "      WHERE p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < t_ms",
        "Seen = SELECT a FROM Payment_Transaction:p -(Transaction_From_Account>:e)- Account:a\n"
        "      -(Account_Initiated_Transaction>:f)- Payment_Transaction:x\n"
        "      WHERE p.event_ts_ms >= x.event_ts_ms AND p.event_ts_ms < t_ms\n"
        "        AND x.event_ts_ms >= p.event_ts_ms AND x.event_ts_ms < t_ms",
    )
    problems = bound_rule(NAME, _definition(circular))
    assert len(problems) == 2, problems
    assert problems[0].endswith("does not bound p from a window's start"), problems
    assert problems[1].endswith("does not bound x from a window's start"), problems
    # Above, a chain through each other never reaches t either.
    above = circular.replace(
        "p.event_ts_ms >= x.event_ts_ms AND p.event_ts_ms < t_ms",
        "p.event_ts_ms >= t_ms - lookback_ms AND p.event_ts_ms < x.event_ts_ms",
    ).replace(
        "x.event_ts_ms >= p.event_ts_ms AND x.event_ts_ms < t_ms",
        "x.event_ts_ms >= p.event_ts_ms AND x.event_ts_ms < p.event_ts_ms",
    )
    problems = bound_rule(NAME, _definition(above))
    assert len(problems) == 2 and all("below t" in p for p in problems), problems


TOKEN_STEP = (
    "    X = SELECT k FROM Owners:q -(Party_Uses_Token>:l)- Token:k\n"
    "      WHERE l.valid_from_ms < t_ms\n"
    "        AND (l.valid_to_ms == 0 OR l.valid_to_ms > t_ms - lookback_ms)\n"
    "      ACCUM @@received += 1;\n"
)


def test_only_read_world_reads_a_token() -> None:
    problems = common_problems(NAME, _added(TOKEN_STEP))
    assert any("steps to a Token" in problem for problem in problems), problems
    assert any("a Token's link" in problem for problem in problems), problems
    world = _added(TOKEN_STEP).replace(NAME, "read_world")
    assert pattern_rule("read_world", _definition(world, "read_world")) == []


@pytest.mark.parametrize(
    ("statement", "problem"),
    [
        (
            '    tg_wcc(["Party"], ["Party_Uses_Device"], 100, FALSE, "");\n',
            "calls the library algorithm tg_wcc",
        ),
        ("    read_hubs(t_ms);\n", "calls the query read_hubs"),
    ],
)
def test_no_query_calls_a_library_algorithm_or_another_query(statement: str, problem: str) -> None:
    assert call_rule(NAME, _definition()) == []
    assert call_rule(NAME, _definition(_added(statement))) == [f"{NAME} {problem}"]


REFUSAL = (
    "    Existing = SELECT s FROM Scored:s WHERE s.opened_ms < t_ms;\n"
    "    IF Existing.size() != Scored.size() THEN\n"
    '        PRINT "invisible_account" AS status;\n'
    "        RETURN;\n"
    "    END;\n"
)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (REFUSAL, ""),
        ("WHERE s.opened_ms < t_ms;", "WHERE NOT s.is_external;"),
        ("IF Existing.size() != Scored.size()", "IF Existing.size() > Scored.size()"),
        ('PRINT "invisible_account" AS status;', 'PRINT "ok" AS status;'),
        ('"invisible_account" AS status;\n        RETURN;', '"invisible_account" AS status;'),
    ],
)
def test_a_query_with_t_refuses_accounts_that_do_not_exist_at_t(old: str, new: str) -> None:
    assert refusal_rule(NAME, _definition()) == []
    assert refusal_rule(NAME, _definition(_changed(old, new))) == [
        f"{NAME} does not refuse accounts of scored that do not exist at t"
    ]


def test_a_refusal_after_the_ok_status_comes_too_late() -> None:
    # The parser reads the first row, so the "ok" status would already have been taken.
    late = _changed("AS received;\n", "AS received;\n" + REFUSAL, _changed(REFUSAL, ""))
    assert refusal_rule(NAME, _definition(late)) == [
        f"{NAME} does not refuse accounts of scored that do not exist at t"
    ]


def test_only_account_parameters_of_a_query_with_t_need_the_refusal() -> None:
    no_t = _changed("    UINT t_ms,\n", "").replace(REFUSAL, "")
    assert refusal_rule(NAME, _definition(no_t)) == []
    parties = _changed("SET<VERTEX<Account>> scored", "SET<VERTEX<Party>> scored").replace(
        REFUSAL, ""
    )
    assert refusal_rule(NAME, _definition(parties)) == []
