"""The loaded world through read_world (docs/design.md, "Extraction and caching"): its
identity, which keys every cache, and the data start that bounds the lookback."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json

from .executor import QueryExecutor
from .queries import READ_WORLD, World, parse_world

DAY_MS = 86_400_000


def read_world(executor: QueryExecutor) -> World:
    return parse_world(executor.run(READ_WORLD, {}))


def identity(world: World) -> str:
    """The sha256 of the world's canonical JSON. Every value is an integer, so a world has
    the same identity on every run, and a world changed in any count, time, amount, category
    or wiring the query reads has another."""
    text = json.dumps(asdict(world), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def data_start(world: World) -> int:
    """The data's first millisecond: the earliest customer joining or payment."""
    return min(world.first_joined_ms, world.first_event_ms)


def check_lookback(world: World, first_date: date, lookback_days: int) -> None:
    """Refuse unless the lookback before first_date (00:00 UTC) lies inside the data, so no
    feature of any date reaches before the data starts (docs/design.md, "Time")."""
    first_ms = (first_date - date(1970, 1, 1)).days * DAY_MS
    if first_ms - lookback_days * DAY_MS < data_start(world):
        raise ValueError(
            f"The {lookback_days}-day lookback before {first_date} starts before the data "
            f"(at {data_start(world)} ms): the warm-up is shorter than the lookback"
        )
