"""The examples and their weights on the calendar world, worked out by hand (docs/design.md,
"Worlds, examples and weights").

At every Monday the calendar world has 17 scorable accounts (18 before a_gone closes on
2024-06-01): the twelve ordinary ones, a_hidden, and the identified a_long, a_short, a_edge and
a_early. The unlabelled pool is the 13 that are no identified mule: the twelve and a_hidden.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, DataConfig
from mule_pattern_learner.data.draw import lowest, sampled
from mule_pattern_learner.data.examples import (
    POSITIVE,
    UNLABELLED,
    Monday,
    examples_folder,
    examples_key,
    monday,
    scored_frame,
    selection_frame,
    training_frame,
    unlabelled,
)
from mule_pattern_learner.data.labels import Labels, read_identified
from mule_pattern_learner.data.population import population
from mule_pattern_learner.testing.calendar import ORDINARY, Spy, calendar_world
from mule_pattern_learner.testing.world import DAY_MS, T

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
    day = Monday(T, tuple(POOL), ())
    assert len(unlabelled(day, (), DATA).drawn) == 3
    assert len(unlabelled(day, (), DEFAULT_CONFIG.data).drawn) == 13
    busy = Monday(T, tuple(POOL), tuple(POOL[:5]))
    assert len(unlabelled(busy, POOL[:5], DATA).drawn) == 8
    assert unlabelled(Monday(T, (), ()), (), DATA).weight == 0.0


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


def test_the_examples_are_keyed_by_the_settings_they_read() -> None:
    calendar = DEFAULT_CONFIG.calendar
    key = examples_key(calendar, DEFAULT_CONFIG.data)
    assert len(key) == 32 and key == examples_key(calendar, DataConfig())
    assert examples_key(calendar, DATA) != key
    assert examples_key(calendar, replace(DEFAULT_CONFIG.data, draw_key=b"other")) != key
    assert examples_key(replace(calendar, scored_mondays=51), DEFAULT_CONFIG.data) != key
    found = examples_folder(IDENTITY, key, IDENTITY)
    assert found.parts[-4:] == ("examples", IDENTITY, key, IDENTITY)
    for bad in ("", "../x", key.upper()):
        with pytest.raises(ValueError, match="32 hex digits"):
            examples_folder(IDENTITY, bad, IDENTITY)
