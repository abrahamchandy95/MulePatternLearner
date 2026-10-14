"""The windows and their weekly as-of dates."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.grid import DAY_MS, Window, day_ms, windows
from mule_pattern_learner.testing.world import T
from mule_pattern_learner.tigergraph.queries import Phase

CONFIG = DEFAULT_CONFIG.data


def _days(dates: list[int]) -> list[date]:
    return [datetime.fromtimestamp(ms / 1000, UTC).date() for ms in dates]


def test_the_windows_are_the_designs_with_their_labels_and_phases() -> None:
    training, validation, test = windows(CONFIG)
    # 2024-07-01 is the test world's T, a Monday.
    assert day_ms(date(2024, 7, 1)) == T
    assert (training.phase, training.labels_before_ms) == (Phase.TRAINING, T)
    assert (validation.phase, validation.labels_before_ms) == (Phase.TRAINING, T + 92 * DAY_MS)
    assert (test.phase, test.labels_before_ms) == (Phase.TEST, None)
    # The 60-day gap stops training at 2024-05-02, so its last Monday is 2024-04-29.
    assert (training.first, training.last) == (date(2024, 1, 1), date(2024, 5, 2))
    days = _days(training.as_of_dates())
    assert (days[0], days[-1], len(days)) == (date(2024, 1, 1), date(2024, 4, 29), 18)
    days = _days(validation.as_of_dates())
    assert (days[0], days[-1], len(days)) == (date(2024, 7, 1), date(2024, 9, 30), 14)
    # The test window starts on a Tuesday.
    days = _days(test.as_of_dates())
    assert (days[0], days[-1], len(days)) == (date(2024, 10, 7), date(2024, 12, 30), 13)


def test_every_as_of_date_is_a_monday_at_midnight_utc_a_week_apart() -> None:
    for window in windows(CONFIG):
        dates = window.as_of_dates()
        assert all(ms % DAY_MS == 0 for ms in dates)
        assert all(day.weekday() == 0 for day in _days(dates))
        assert {later - earlier for earlier, later in zip(dates, dates[1:])} == {7 * DAY_MS}


@pytest.mark.parametrize(
    ("gap", "last"),
    [(0, date(2024, 6, 24)), (7, date(2024, 6, 24)), (8, date(2024, 6, 17))],
)
def test_no_training_date_comes_nearer_the_labels_cutoff_than_the_gap(gap: int, last: date) -> None:
    training = windows(replace(CONFIG, maturity_gap_days=gap)).training
    assert _days(training.as_of_dates())[-1] == last


def test_a_window_without_a_monday_is_refused() -> None:
    with pytest.raises(ValueError, match="no Monday"):
        windows(replace(CONFIG, maturity_gap_days=365))
    with pytest.raises(ValueError, match="no Monday"):
        Window(date(2024, 10, 1), date(2024, 10, 6), Phase.TEST, None)
    assert len(Window(date(2024, 10, 7), date(2024, 10, 7), Phase.TEST, None).as_of_dates()) == 1
