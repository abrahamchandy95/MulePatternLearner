"""The examples and their weights on the calendar world, worked out by hand (docs/design.md,
"Worlds, examples and weights").

At every Monday the calendar world has 17 scorable accounts (18 before a_gone closes on
2024-06-01): the twelve ordinary ones, a_hidden, and the identified a_long, a_short, a_edge and
a_early. The unlabelled pool is the 13 that are no identified mule: the twelve and a_hidden.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, DataConfig
from mule_pattern_learner.data.draw import lowest, sampled
from mule_pattern_learner.data.examples import (
    POSITIVE,
    SELECTION_FILE,
    SELECTION_SCHEMA,
    TEST_FILE,
    TEST_SCHEMA,
    TRAINING_FILE,
    TRAINING_SCHEMA,
    UNLABELLED,
    ExampleKeys,
    Monday,
    TableWriter,
    examples_folder,
    examples_key,
    monday,
    read_first,
    read_table,
    scored_frame,
    selection_frame,
    training_frame,
    unlabelled,
)
from mule_pattern_learner.data.labels import Labels, read_identified
from mule_pattern_learner.data.population import population
from mule_pattern_learner.testing.calendar import ORDINARY, Spy, calendar_world
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries import (
    LIST_ACCOUNTS,
    READ_LABELS,
    READ_MONEY_FLOW,
    VERSIONS,
)
from mule_pattern_learner.tigergraph.queries.read_labels import IdentifiedMule

W = 7 * DAY_MS
MONDAYS = DEFAULT_CONFIG.calendar.mondays()
DATA = DataConfig(per_positive=2, min_unlabelled=3)
POOL = sorted([f"a_{name}" for name in ORDINARY] + ["a_hidden"])
IDENTITY = "0123456789abcdef0123456789abcdef"


@pytest.fixture(scope="module")
def days(tmp_path_factory: pytest.TempPathFactory) -> tuple[Labels, list[Monday]]:
    """The identified mules and every scored Monday of the calendar world."""
    spy = Spy(calendar_world().graph())
    root = tmp_path_factory.mktemp("cache")
    labels = read_identified(spy)
    found = [
        monday(t_ms, [row.account_id for row in population(spy, root, t_ms)], labels.mules)
        for t_ms in MONDAYS
    ]
    return labels, found


def _on(days: list[Monday], t_ms: int) -> Monday:
    return next(day for day in days if day.t_ms == t_ms)


def test_the_positives_are_identified_mules_active_and_scorable(
    days: tuple[Labels, list[Monday]],
) -> None:
    _, found = days
    assert _on(found, T).positives == ("a_long",)
    assert _on(found, T + 9 * W).positives == ("a_long", "a_short")
    # a_gone is active on T + 1 week but closed, so not scorable; a_hidden is not identified.
    assert _on(found, T + W).positives == ("a_long",)
    # a_edge's onset is exactly at T + 30 weeks, so it is not active then; a_long's last
    # payment plus 7 days was T + 25 weeks.
    assert _on(found, T + 30 * W).positives == ()
    assert _on(found, T + 31 * W).positives == ("a_edge",)
    assert _on(found, T - W).positives == ()
    assert len(_on(found, T).accounts) == 17 and len(found[0].accounts) == 18
    positive = sum(len(day.positives) for day in found)
    assert positive == 26 + 1 + 1


def test_the_dropped_mules_are_the_identified_scorable_and_not_active(
    days: tuple[Labels, list[Monday]],
) -> None:
    _, found = days
    # On the first Monday, 2024-04-01, all five identified mules are scorable (a_gone has not
    # closed) and none is active yet: 5 dropped.
    assert (found[0].dropped, found[0].positives) == (5, ())
    # A week before T, a_long is not active yet (its onset is T - 1 day) and is dropped with
    # a_short, a_edge and a_early; a_gone closed on 2024-06-01, so it is no example.
    assert (_on(found, T - W).dropped, _on(found, T - W).positives) == (4, ())
    # At T, a_long is a positive, and the other three scorable ones are dropped, not a_gone.
    assert (_on(found, T).dropped, _on(found, T).positives) == (3, ("a_long",))
    # Two weeks later a_gone's activity is over and it is still closed: not counted. Counting
    # every identified mule that is not active would give 4.
    assert (_on(found, T + 2 * W).dropped, _on(found, T + 2 * W).positives) == (3, ("a_long",))
    # a_short is active at T + 9 weeks, so a_edge and a_early are dropped; at T + 30 weeks
    # a_edge's onset is exactly then, so all four scorable ones are.
    assert _on(found, T + 9 * W).dropped == 2
    assert (_on(found, T + 30 * W).dropped, _on(found, T + 30 * W).positives) == (4, ())
    assert _on(found, T + 30 * W).tally().dropped == 4


def test_a_mule_not_scorable_that_monday_is_neither_positive_nor_dropped() -> None:
    mules = (
        IdentifiedMule("a_past", T - 3 * W, T - 2 * W),
        IdentifiedMule("a_later", T + W, T + 2 * W),
        IdentifiedMule("a_now", T - W, T + W),
        IdentifiedMule("a_closed", T - W, T + W),
    )
    # a_later is not scorable yet and a_closed is closed: neither is an example at T.
    day = monday(T, ["a_n1", "a_past", "a_now"], mules)
    assert (day.positives, day.dropped) == (("a_now",), 1)
    assert monday(T, [], mules).dropped == 0 and monday(T, ["a_past"], mules).dropped == 1


def test_each_mule_counts_once_however_long_it_ran(days: tuple[Labels, list[Monday]]) -> None:
    labels, found = days
    draws = [unlabelled(day, labels.accounts, DATA) for day in found]
    frame = training_frame([day.tally() for day in found], draws)
    positives = frame[frame["label"] == POSITIVE]
    # Before division: a_long 1/26 on each of its 26 Mondays, a_short and a_edge 1 each; their
    # mean is 3/28, so a_long's rows weigh 28/78 and the others 28/3, every mule 28/3 in all.
    totals = positives.groupby("account_id")["weight"].sum()
    for account in ("a_long", "a_short", "a_edge"):
        assert totals[account] == pytest.approx(28 / 3)
    long_rows = positives[positives["account_id"] == "a_long"]
    assert len(long_rows) == 26 and long_rows["weight"].iloc[0] == pytest.approx(28 / 78)
    assert positives["weight"].mean() == pytest.approx(1.0)


def test_the_unlabelled_draw_stands_for_its_mondays_pool(
    days: tuple[Labels, list[Monday]],
) -> None:
    labels, found = days
    draws = {day.t_ms: unlabelled(day, labels.accounts, DATA) for day in found}
    # n = min(13, max(2 x positives, 3)): 3 on a Monday of 0 or 1 positive, 4 on T + 9 weeks.
    assert draws[T].pool == 13 and draws[T].drawn == tuple(lowest(DATA.draw_key, POOL, T, 3))
    assert len(draws[T + 9 * W].drawn) == 4 and draws[T + 9 * W].weight == 13 / 4
    assert draws[MONDAYS[0]].pool == 13 and draws[MONDAYS[0]].weight == 13 / 3
    # Identified mules are never drawn, on any Monday; the hidden mule can be.
    drawn = {account for draw in draws.values() for account in draw.drawn}
    assert not drawn & labels.accounts
    assert "a_hidden" in drawn
    frame = training_frame([day.tally() for day in found], list(draws.values()))
    rows = frame[frame["label"] == UNLABELLED]
    # 51 Mondays of 3 at 13/3 and one of 4 at 13/4: they sum to 676 over 157 rows, so each
    # divides by 676/157.
    assert len(rows) == 157
    assert 51 * 3 * Fraction(13, 3) + 4 * Fraction(13, 4) == 676
    assert Fraction(13, 3) / Fraction(676, 157) == Fraction(2041, 2028)
    at_t = rows[rows["t_ms"] == T]["weight"]
    assert at_t.tolist() == pytest.approx([float(Fraction(2041, 2028))] * 3)
    at_9 = rows[rows["t_ms"] == T + 9 * W]["weight"]
    assert at_9.tolist() == pytest.approx([float(Fraction(2041, 2704))] * 4)
    assert rows["weight"].mean() == pytest.approx(1.0)
    # The table is by Monday, positives first, then by account.
    assert frame[frame["t_ms"] == T]["label"].tolist() == [POSITIVE] + [UNLABELLED] * 3
    assert list(frame.columns) == ["t_ms", "account_id", "label", "weight"]


def test_the_draw_is_at_least_the_minimum_and_at_most_the_pool() -> None:
    day = Monday(T, tuple(POOL), (), 0)
    assert len(unlabelled(day, (), DATA).drawn) == 3
    assert len(unlabelled(day, (), DEFAULT_CONFIG.data).drawn) == 13
    busy = Monday(T, tuple(POOL), tuple(POOL[:5]), 0)
    assert len(unlabelled(busy, POOL[:5], DATA).drawn) == 8
    assert unlabelled(Monday(T, (), (), 0), (), DATA).weight == 0.0


def test_the_selection_keys_rank_by_pu_label_alone(days: tuple[Labels, list[Monday]]) -> None:
    labels, found = days
    frame = selection_frame(_on(found, T + W), labels.accounts, DEFAULT_CONFIG.data)
    assert len(frame) == 17 and frame["weight"].eq(1.0).all()
    by_id = frame.set_index("account_id")
    assert by_id.loc["a_long", "positive"] and by_id.loc["a_long", "ranked"]
    # Identified mules on their other Mondays are out of the ranking; a hidden mule is ranked
    # like anyone else.
    for account in ("a_short", "a_edge", "a_early"):
        assert not by_id.loc[account, "positive"] and not by_id.loc[account, "ranked"]
    assert by_id.loc["a_hidden", "ranked"] and not by_id.loc["a_hidden", "positive"]
    assert by_id["ranked"].sum() == 14


def test_the_fallback_selection_keys_sample_the_rest(days: tuple[Labels, list[Monday]]) -> None:
    labels, found = days
    data = replace(DEFAULT_CONFIG.data, scoring="fallback")
    frame = selection_frame(_on(found, T), labels.accounts, data)
    sample = sampled(data.draw_key, POOL, T, 10)
    assert sorted(frame["account_id"]) == sorted(["a_long", *sample])
    by_id = frame.set_index("account_id")
    assert by_id.loc["a_long", "weight"] == 1.0 and by_id.loc["a_long", "positive"]
    assert all(by_id.loc[account, "weight"] == 10.0 for account in sample)
    assert frame["ranked"].all()


def test_the_test_keys_are_every_scorable_account(days: tuple[Labels, list[Monday]]) -> None:
    _, found = days
    frame = scored_frame(_on(found, T))
    assert frame["account_id"].tolist() == sorted(_on(found, T).accounts)
    assert frame["t_ms"].eq(T).all() and list(frame.columns) == ["t_ms", "account_id"]


def test_the_examples_are_keyed_by_the_settings_they_read(monkeypatch: pytest.MonkeyPatch) -> None:
    calendar = DEFAULT_CONFIG.calendar
    key = examples_key(calendar, DEFAULT_CONFIG.data)
    assert len(key) == 32 and key == examples_key(calendar, DataConfig())
    assert examples_key(calendar, DATA) != key
    assert examples_key(calendar, replace(DEFAULT_CONFIG.data, draw_key=b"other")) != key
    assert examples_key(replace(calendar, scored_mondays=51), DEFAULT_CONFIG.data) != key
    # The versions of the two queries the examples are made of move the key; those of the
    # queries only the cache holds do not.
    for query in (LIST_ACCOUNTS, READ_LABELS):
        monkeypatch.setitem(VERSIONS, query, f"{query}-000000000000")
        assert examples_key(calendar, DEFAULT_CONFIG.data) != key
        monkeypatch.undo()
        assert examples_key(calendar, DEFAULT_CONFIG.data) == key
    monkeypatch.setitem(VERSIONS, READ_MONEY_FLOW, "read_money_flow-000000000000")
    assert examples_key(calendar, DEFAULT_CONFIG.data) == key
    monkeypatch.undo()
    found = examples_folder(IDENTITY, key, IDENTITY)
    assert found.parts[-4:] == ("examples", IDENTITY, key, IDENTITY)
    for bad in ("", "../x", key.upper()):
        with pytest.raises(ValueError, match="32 hex digits"):
            examples_folder(IDENTITY, bad, IDENTITY)


def _write(folder: Path, name: str, schema: object, frame: pd.DataFrame) -> None:
    writer = TableWriter(folder / name, schema)  # pyright: ignore[reportArgumentType]
    writer.write(frame)
    writer.commit()


def test_the_keys_of_the_examples_tables_come_a_monday_at_a_time(tmp_path: Path) -> None:
    # The training table holds a_1 and a_2 on T and a_2 on T + 1 week; the selection table holds
    # a_2 (ranked) and a_3 (not ranked) on T and a_4 on T + 2 weeks; the test table a_5 on T.
    _write(
        tmp_path,
        TRAINING_FILE,
        TRAINING_SCHEMA,
        pd.DataFrame(
            {
                "t_ms": [T, T, T + W],
                "account_id": ["a_2", "a_1", "a_2"],
                "label": [1, 0, 0],
                "weight": [1.0, 1.0, 1.0],
            }
        ),
    )
    _write(
        tmp_path,
        SELECTION_FILE,
        SELECTION_SCHEMA,
        pd.DataFrame(
            {
                "t_ms": [T, T, T + 2 * W],
                "account_id": ["a_2", "a_3", "a_4"],
                "positive": [False, False, False],
                "ranked": [True, False, True],
                "weight": [1.0, 1.0, 1.0],
            }
        ),
    )
    _write(tmp_path, TEST_FILE, TEST_SCHEMA, pd.DataFrame({"t_ms": [T], "account_id": ["a_5"]}))
    keys = ExampleKeys(tmp_path, [TRAINING_FILE, SELECTION_FILE, TEST_FILE])
    # Each Monday once, the accounts of every table together by increasing id, and an account
    # the selection table does not rank left out.
    assert list(keys) == [T, T + W, T + 2 * W] and len(keys) == 3
    assert keys[T] == ["a_1", "a_2", "a_5"]
    assert keys[T + W] == ["a_2"] and keys[T + 2 * W] == ["a_4"]
    assert ExampleKeys(tmp_path, [SELECTION_FILE])[T] == ["a_2"]
    with pytest.raises(KeyError):
        keys[T + 3 * W]
    with pytest.raises(FileNotFoundError, match="run `mule prepare` first"):
        ExampleKeys(tmp_path / "other", [TRAINING_FILE])


def test_the_first_rows_of_a_table_are_its_first_row_group_alone(tmp_path: Path) -> None:
    # A table written a Monday at a time holds a row group each: the first look reads the first.
    writer = TableWriter(tmp_path / TEST_FILE, TEST_SCHEMA)
    writer.write(pd.DataFrame({"t_ms": [T, T], "account_id": ["a_1", "a_2"]}))
    writer.write(pd.DataFrame({"t_ms": [T + W], "account_id": ["a_3"]}))
    writer.commit()
    first = read_first(tmp_path / TEST_FILE, TEST_SCHEMA)
    assert first["account_id"].tolist() == ["a_1", "a_2"] and first["t_ms"].dtype == "int64"
    assert read_table(tmp_path / TEST_FILE, TEST_SCHEMA)["account_id"].tolist() == [
        "a_1",
        "a_2",
        "a_3",
    ]
    # A table with no row has no row group, and an empty first look with the table's columns.
    empty = TableWriter(tmp_path / "empty.parquet", TEST_SCHEMA)
    empty.commit()
    nothing = read_first(tmp_path / "empty.parquet", TEST_SCHEMA)
    assert nothing.empty and list(nothing.columns) == ["t_ms", "account_id"]
