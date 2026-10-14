"""The training, validation and test windows (docs/design.md, "Splits and hidden labels")
and their as-of dates: every Monday at 00:00 UTC from a window's first day to its last."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import NamedTuple

from ..config import DataConfig
from ..tigergraph.queries import Phase

DAY_MS = 86_400_000
_EPOCH = date(1970, 1, 1)
_DAY = timedelta(days=1)


def day_ms(day: date) -> int:
    """00:00 UTC of day, in UTC epoch milliseconds."""
    return (day - _EPOCH).days * DAY_MS


@dataclass(frozen=True)
class Window:
    """A window: its as-of days from first to last, the customers its phase sees, and the
    labels it may use, the mules discovered before labels_before_ms (all of the answer key
    when None). It refuses to hold no Monday."""

    first: date
    last: date
    phase: Phase
    labels_before_ms: int | None

    def __post_init__(self) -> None:
        if not self.as_of_dates():
            raise ValueError(f"The window from {self.first} to {self.last} holds no Monday")

    def as_of_dates(self) -> list[int]:
        """Every Monday at 00:00 UTC from first to last, in UTC epoch milliseconds."""
        monday = self.first + timedelta(days=(7 - self.first.weekday()) % 7)
        weeks = (self.last - monday).days // 7 + 1
        return [day_ms(monday + timedelta(weeks=week)) for week in range(max(weeks, 0))]


class Windows(NamedTuple):
    training: Window
    validation: Window
    test: Window


def windows(config: DataConfig) -> Windows:
    """The windows of config. Training and validation see the training customers and use the
    mules discovered before the day after their last day; training's as-of days also stop
    the maturity gap before that cutoff. Test sees every customer and the full answer key."""
    first, last = config.training
    cutoff = last + _DAY
    nearest = cutoff - timedelta(days=config.maturity_gap_days)
    training = Window(first, min(last, nearest), Phase.TRAINING, day_ms(cutoff))
    first, last = config.validation
    validation = Window(first, last, Phase.TRAINING, day_ms(last + _DAY))
    test = Window(*config.test, Phase.TEST, None)
    return Windows(training, validation, test)
