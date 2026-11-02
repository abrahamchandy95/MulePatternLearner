"""A reference world that spans the configured calendar, with identified mules on chosen weeks;
the identity file the loader writes for a world from PhantomLedger's world record; and a spy
executor that answers through a reference graph and records every query it runs.

The world of calendar_world holds the export window of the default calendar (its first payment
on 2024-01-01, day 0, and its last on 2025-03-30, day 454) and these identified mules, around
T, the start of the 14th of the 52 scored weeks, Monday 2024-07-01. A mule is positive at a
scoring time t when its onset < t and t <= its last payment + W(t), W(t) being the longer of 7
days and the time since the previous scoring time (7 days when there is none); positive_times
works that out for any scoring times, and early_warning_times the one time each mule is at its
early warning:

- a_long, active from the Sunday before T (onset T - 1 day) until T + 24 weeks (last), so
  positive at every scoring time after its onset up to about T + 25 weeks, where its last
  payment leaves the window;
- a_short, active from Wednesday T + 8 weeks + 2 days to the next day, positive at the scoring
  times in (T + 8 weeks + 2 days, T + 9 weeks + 3 days] and, when the gap between scoring times
  that holds its last payment is longer than a week, at the scoring time that ends the gap;
- a_edge, whose onset and last payment are both exactly at the Monday T + 30 weeks: not positive
  at a scoring time exactly then, since an onset at t is not in view, but positive at the times
  after it up to T + 31 weeks, and at the scoring time that ends the gap holding its last
  payment when that is longer than a week;
- a_gone, active from T + 2 days to T + 3 days, whose account closed at T - 30 days: positive in
  the same way as a_short, but never scorable;
- a_early, active in February 2024, before the first scored week (2024-04-01): positive at no
  scoring time, since the first one is more than 7 days after its last payment, so in
  Unscored.no_active_week.

a_hidden is a mule the bank never identified. Twelve ordinary customers own a_n00 to a_n11;
c_card is a credit card a_n00's customer owns; x_far is another bank's account.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DEFAULT_CONFIG
from ..data.calendar import epoch_ms
from ..tigergraph.executor import ResponseTooLarge
from ..tigergraph.queries import READ_WORLD, read_world
from ..tigergraph.queries.read_world import World
from ..tigergraph.world import shared_fields
from .reference import ReferenceGraph
from .world import DAY_MS, HOUR_MS, Mule, T, WorldBuilder

D, H, W = DAY_MS, HOUR_MS, 7 * DAY_MS
DAY_0 = epoch_ms(DEFAULT_CONFIG.calendar.export_start)
LAST_DAY = DAY_0 + 454 * D
JOINED = T - 900 * D
# The activity periods of the identified mules, and of the hidden one.
LONG = Mule(T - D, T + 24 * W, ring=1, identified=True)
SHORT = Mule(T + 8 * W + 2 * D, T + 8 * W + 3 * D, ring=2, identified=True)
EDGE = Mule(T + 30 * W, T + 30 * W, ring=3, identified=True)
GONE = Mule(T + 2 * D, T + 3 * D, ring=4, identified=True)
EARLY = Mule(DAY_0 + 31 * D, DAY_0 + 40 * D, ring=5, identified=True)
HIDDEN = Mule(T - 2 * D, T + 2 * W, ring=1)
ORDINARY = tuple(f"n{index:02d}" for index in range(12))


def _window(t_ms: int, scoring_times: Sequence[int]) -> int:
    """W(t) of the label rule, from its words: the longer of 7 days and the time since the
    previous scoring time, the largest below t, and 7 days when there is none."""
    earlier = [s for s in scoring_times if s < t_ms]
    return max(W, t_ms - max(earlier)) if earlier else W


def positive_times(mule: Mule, scoring_times: Sequence[int]) -> tuple[int, ...]:
    """The scoring times, in increasing order, at which a mule is positive, by the label rule of
    docs/design.md read literally: the onset before t, and t no later than the last payment plus
    W(t). Written apart from data.labels, so that a test of that module compares two."""
    return tuple(
        t_ms
        for t_ms in scoring_times
        if mule.onset_ms < t_ms <= mule.last_ms + _window(t_ms, scoring_times)
    )


def early_warning_times(mule: Mule, scoring_times: Sequence[int]) -> tuple[int, ...]:
    """The scoring times, in increasing order, at which a mule is at its early warning, by the
    rule of docs/design.md read literally: t at or before the onset, and the onset before the
    next scoring time, the smallest above t, which is t plus 7 days when there is none. Written
    apart from evaluation.answer_key, so that a test of that module compares two."""
    found: list[int] = []
    for t_ms in scoring_times:
        later = [s for s in scoring_times if s > t_ms]
        if t_ms <= mule.onset_ms < (min(later) if later else t_ms + W):
            found.append(t_ms)
    return tuple(found)


def calendar_world() -> WorldBuilder:
    """The world of this module's docstring."""
    w = WorldBuilder()
    for name in ORDINARY:
        w.customer(name, JOINED)
    w.account("c_card", "p_n00", JOINED, account_type="credit")
    w.external("x_far", JOINED)
    for name, mule in (
        ("long", LONG),
        ("short", SHORT),
        ("edge", EDGE),
        ("early", EARLY),
        ("hidden", HIDDEN),
    ):
        w.customer(name, JOINED, mule=mule)
    w.customer("gone", JOINED, closed_ms=T - 30 * D, mule=GONE)
    w.pay("x_first", "a_n00", "x_far", DAY_0 + H, 25.0, "ach")
    w.pay("x_last", "x_far", "a_n01", LAST_DAY + 23 * H, 40.0, "ach")
    w.pay("x_mid", "a_long", "a_n02", T - 3 * D, 600.0, "zelle")
    return w


@dataclass
class Spy:
    """A query executor that answers through a reference graph and records every query it
    runs, with its parameters. With too_large, (query, most), it refuses every request of that
    query for more than most accounts, as TigerGraph refuses an answer for its size."""

    graph: ReferenceGraph
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=lambda: [])
    too_large: tuple[str, int] | None = None

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        self.calls.append((name, params))
        if self.too_large is not None:
            query, most = self.too_large
            if name == query and len(params["account_ids"]) > most:
                raise ResponseTooLarge(f"{name} answered more than the response limit")
        return self.graph.run(
            name, params, timeout_s=timeout_s, attempts=attempts, timeout_retries=timeout_retries
        )

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def sizes(self, query: str) -> list[int]:
        """The accounts of each request of query, in order."""
        return [len(params["account_ids"]) for name, params in self.calls if name == query]


def world_of(graph: ReferenceGraph) -> World:
    """read_world's fields of a reference graph."""
    return read_world.parse(graph.run(READ_WORLD, read_world.params()))


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'


def identity_text(
    world: World,
    seed: int,
    *,
    nested: bool = False,
    changed: Mapping[str, object] | None = None,
    dropped: tuple[str, ...] = (),
) -> str:
    """An identity file of the world as the loader writes it from PhantomLedger's world record:
    the record's own keys, then the fields it shares with read_world, as quoted dotted keys,
    or as tables of keys when nested. changed replaces or adds keys; dropped leaves keys out."""
    keys: dict[str, object] = {
        "phantomledger_commit": "5754a26",
        "clean_tree": True,
        "code_gates_certified": True,
        "seed": seed,
        "population": 30_000,
        "start": DAY_0,
        "days": 455,
        "burn_in_days": 275,
        "label_tail_days": 90,
        "first_scored_monday": DEFAULT_CONFIG.calendar.week_starts()[0],
        **shared_fields(world),
        **(changed or {}),
    }
    for name in dropped:
        del keys[name]
    if not nested:
        return "".join(f'"{name}" = {_toml(value)}\n' for name, value in keys.items())
    flat = [f"{name} = {_toml(value)}\n" for name, value in keys.items() if "." not in name]
    tables: dict[str, list[str]] = {}
    for name, value in keys.items():
        if "." in name:
            table, rest = name.split(".", 1)
            tables.setdefault(table, []).append(f'"{rest}" = {_toml(value)}\n')
    return "".join(flat) + "".join(f"[{table}]\n{''.join(rows)}" for table, rows in tables.items())
