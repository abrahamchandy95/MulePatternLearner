"""The scored weeks and scoring times, and the check that the loaded world holds the export
window they need (docs/design.md, "Worlds and calendar" and the Lookback guard).

Every world shares one calendar, which config holds: the export starts on export_start (day
0) and lasts 455 days, its first 90 history only, and the 52 scored weeks follow, each from a
Monday at 00:00 UTC. A world scores each week at one time inside it, drawn from the world's own
seed (CalendarConfig.scoring_times). mule prepare derives the weeks and times from config and
checks the calendar against the loaded world through read_world: its first payment must fall on
day 0 and its last on day 454, the same 455 days the loader verifies after loading. The check
reads the first payment, never the first join, since backdated joins put the first join years
before the data. That every scoring time's lookback lies inside the export needs no check here:
CalendarConfig holds history_days at least lookback_days, which puts the first scored week at or
after day 90.
"""

from __future__ import annotations

from datetime import date

from ..config import CalendarConfig
from ..tigergraph.queries.read_world import World
from ..tigergraph.queries.rows import DAY_MS


def epoch_ms(day: date) -> int:
    """Milliseconds from 1970-01-01 00:00 UTC to day at 00:00 UTC."""
    return (day - date(1970, 1, 1)).days * DAY_MS


def week_starts(calendar: CalendarConfig) -> tuple[int, ...]:
    """The start of each scored week, its Monday at 00:00 UTC, as UTC epoch milliseconds in
    increasing order."""
    return calendar.week_starts()


def scoring_times(calendar: CalendarConfig, seed: int) -> tuple[int, ...]:
    """The time each scored week is scored at in the world of that seed, as UTC epoch
    milliseconds in increasing order."""
    return calendar.scoring_times(seed)


def check_window(world: World, calendar: CalendarConfig) -> None:
    """Refuse a loaded world that does not hold the calendar's export window: its first payment
    on day 0 and its last on the window's last day."""
    start = epoch_ms(calendar.export_start)
    last_day = start + (calendar.export_days - 1) * DAY_MS
    if not start <= world.first_event_ms < start + DAY_MS:
        raise ValueError(
            f"The loaded world's first payment is at {world.first_event_ms} ms, not on day 0 of "
            f"the export ({calendar.export_start}, {start} ms): it is not the configured "
            "calendar's world"
        )
    if not last_day <= world.last_event_ms < last_day + DAY_MS:
        raise ValueError(
            f"The loaded world's last payment is at {world.last_event_ms} ms, not on day "
            f"{calendar.export_days - 1} of the export ({last_day} ms): it is not the "
            "configured calendar's world"
        )
