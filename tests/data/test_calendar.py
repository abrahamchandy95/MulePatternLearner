"""The scored weeks and scoring times, and the check of the export window against the loaded
world (docs/design.md, "Worlds and calendar" and the Lookback guard)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, CalendarConfig
from mule_pattern_learner.data.calendar import check_window, epoch_ms, scoring_times, week_starts
from mule_pattern_learner.testing.calendar import calendar_world, world_of
from mule_pattern_learner.tigergraph.queries.rows import DAY_MS

CALENDAR = DEFAULT_CONFIG.calendar
# 2024-01-01 00:00 UTC is 19,723 days after 1970-01-01.
DAY_0 = 1_704_067_200_000
LAST_DAY = DAY_0 + 454 * DAY_MS


def test_the_export_starts_on_day_0_at_midnight_utc() -> None:
    assert date(2024, 1, 1) - date(1970, 1, 1) == date.resolution * 19_723
    assert epoch_ms(CALENDAR.export_start) == DAY_0 == 19_723 * DAY_MS


def test_the_week_starts_are_the_calendars() -> None:
    found = week_starts(CALENDAR)
    # 2024-04-01 is day 91, the first Monday at or after day 90; 2025-03-24 is day 448.
    assert found[0] == DAY_0 + 91 * DAY_MS and found[-1] == DAY_0 + 448 * DAY_MS
    assert len(found) == 52 and found == CALENDAR.week_starts()


def test_the_scoring_times_are_the_calendars_for_the_worlds_seed() -> None:
    found = scoring_times(CALENDAR, 1001)
    assert found == CALENDAR.scoring_times(1001) and found != scoring_times(CALENDAR, 1002)
    # One time inside each week, a whole second after the week's Monday midnight.
    for start, t_ms in zip(week_starts(CALENDAR), found, strict=True):
        assert start <= t_ms < start + 7 * DAY_MS and t_ms % 1_000 == 0


def test_a_world_holding_the_export_window_passes() -> None:
    world = world_of(calendar_world().graph())
    # The calendar world's first payment is at 01:00 on day 0 and its last at 23:00 on day 454.
    assert world.first_event_ms == DAY_0 + 3_600_000
    assert world.last_event_ms == LAST_DAY + 23 * 3_600_000
    check_window(world, CALENDAR)
    # The window's edges: day 0 at 00:00 and day 454 at its last millisecond.
    check_window(
        replace(world, first_event_ms=DAY_0, last_event_ms=LAST_DAY + DAY_MS - 1), CALENDAR
    )
    check_window(
        replace(world, first_event_ms=DAY_0 + DAY_MS - 1, last_event_ms=LAST_DAY), CALENDAR
    )


@pytest.mark.parametrize(
    ("first", "last", "refusal"),
    [
        (DAY_0 - 1, LAST_DAY, "first payment"),
        (DAY_0 + DAY_MS, LAST_DAY, "first payment"),
        (0, LAST_DAY, "first payment"),
        (DAY_0, LAST_DAY - 1, "last payment"),
        (DAY_0, LAST_DAY + DAY_MS, "last payment"),
    ],
)
def test_a_world_outside_the_export_window_is_refused(first: int, last: int, refusal: str) -> None:
    world = replace(world_of(calendar_world().graph()), first_event_ms=first, last_event_ms=last)
    with pytest.raises(ValueError, match=refusal):
        check_window(world, CALENDAR)


def test_another_calendar_refuses_the_world() -> None:
    # A world exported from 2024-01-08 would not hold this calendar's day 0.
    world = world_of(calendar_world().graph())
    later = CalendarConfig(export_start=date(2024, 1, 8))
    with pytest.raises(ValueError, match="first payment"):
        check_window(world, later)
