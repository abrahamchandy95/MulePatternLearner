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
    common_problems,
    field_rule,
    heap_rule,
    label_rule,
    leaf_rule,
    name_rule,
    parameter_rule,
    projection_fields,
    projection_rule,
    status_fields,
    time_rule,
    tuple_rule,
    version_rule,
    write_rule,
)
from mule_pattern_learner.tigergraph.installer import definitions, text_version
from mule_pattern_learner.tigergraph.queries import written_files

NAME = "read_example"
# A query that keeps every form the rules read: payments in a window before t, visible and
# current links, an account that exists, known starts, the crowd count and the leaf test, a
# capped heap, a tuple and a projected PRINT.
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
