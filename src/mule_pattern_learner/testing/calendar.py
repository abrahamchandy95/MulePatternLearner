"""A reference world that spans the configured calendar, with identified mules on chosen Mondays;
the identity file the loader writes for a world from PhantomLedger's world record; and a spy
executor that answers through a reference graph and records every query it runs.

The world of calendar_world holds the export window of the default calendar (its first payment
on 2024-01-01, day 0, and its last on 2025-03-30, day 454) and these identified mules, around
T, the scored Monday 2024-07-01 (the 14th of 52):

- a_long, active from the Sunday before T (onset T - 1 day) until T + 24 weeks (last), so
  positive on the 26 Mondays from T to T + 25 weeks;
- a_short, active from Wednesday T + 8 weeks + 2 days to the next day, so positive on the one
  Monday T + 9 weeks;
- a_edge, whose onset and last payment are both exactly at the Monday T + 30 weeks: not active
  at T + 30 weeks, since an onset at t is not in view, and active at T + 31 weeks, since its
  last payment plus 7 days is exactly then; one positive Monday;
- a_gone, active from T + 2 days to T + 3 days, whose account closed at T - 30 days: active on
  the Monday T + 1 week but never scorable;
- a_early, active in February 2024, before the first scored Monday (2024-04-01): active on no
  scored Monday.

a_hidden is a mule the bank never identified. Twelve ordinary customers own a_n00 to a_n11;
c_card is a credit card a_n00's customer owns; x_far is another bank's account.
"""

from __future__ import annotations

from collections.abc import Mapping
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
        "first_scored_monday": DEFAULT_CONFIG.calendar.mondays()[0],
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
