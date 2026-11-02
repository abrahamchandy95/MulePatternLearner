"""What the evaluation tests share: a reference world with mules of every kind the groups and
checks read, a calendar of three scored weeks around it, the identity file and gate record
PhantomLedger would write for it, a parity record and a data folder of the test's own.

The world is testing.fixtures.catalogue (the look-alikes, and ring 7: a_mule, which the bank
identified, its collector a_collector and a_mule2 and a_mule3), with:

- ring 11, which the bank never surfaced: a_hr1, a_hr2 and a_hr3, enrolled on the device
  d_hidden from 20 days before T, active from 10 days before T to 9 days after it, each paying
  the next 200.00 by Zelle from the device 5 days before T and 4 days after it;
- a_drop, a ringless mule (an account-takeover drop) the bank identified, active from 3 days
  before T to T, receiving 900.00 from x_drop 3 days before T;
- a_late, a ring 11 mule active from 10 to 12 days after T, so that T + 1 week is its
  early-warning time and T + 2 weeks its one active time, enrolled on d_hidden from 3
  days after T, so that its ring link is visible at its early-warning time;
- twelve ordinary customers, a_o00 to a_o11, each paying x_shop 20.00 by card 2 days before T.

The calendar's three scored weeks start on T, T + 1 week and T + 2 weeks (2024-07-01 to
2024-07-15), and the tests score each at its start.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from mule_pattern_learner import paths
from mule_pattern_learner.config import FULL_DESIGN_CONFIG, CalendarConfig, RunConfig
from mule_pattern_learner.evaluation.gate_records import CODE, GATES, WORLD
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import identity_text, world_of
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, Mule, T, WorldBuilder

D, H, W = DAY_MS, HOUR_MS, 7 * DAY_MS
JOINED = T - 400 * D
HIDDEN = Mule(T - 10 * D, T + 9 * D, ring=11)
DROP = Mule(T - 3 * D, T, identified=True)
LATE = Mule(T + 10 * D, T + 12 * D, ring=11)
ORDINARY = tuple(f"o{index:02d}" for index in range(12))
# The calendar of the tests: three scored weeks from T.
CALENDAR = CalendarConfig(export_start=date(2024, 1, 1), history_days=182, scored_weeks=3)
TIMES = (T, T + W, T + 2 * W)
# The full design's settings (its four rotations, whole scoring), with small samples and few
# replicates, so that an audit runs in seconds.
CONFIG = replace(
    FULL_DESIGN_CONFIG,
    calendar=CALENDAR,
    audit=replace(FULL_DESIGN_CONFIG.audit, replicates=40, nonmule_sample=60),
)


def evaluation_world() -> WorldBuilder:
    """The world of this module's docstring."""
    w = fixtures.catalogue()
    hidden = w.ring(
        11, ["hr1", "hr2", "hr3"], JOINED, "d_hidden", T - 20 * D, HIDDEN.onset_ms, HIDDEN.last_ms
    )
    for at in (T - 5 * D + 12 * H, T + 4 * D + 12 * H):
        for sender, receiver in zip(hidden, hidden[1:], strict=False):
            w.pay(f"z_{sender}_{at}", sender, receiver, at, 200.0, "zelle", device="d_hidden")
    w.customer("drop", JOINED, mule=DROP)
    w.external("x_drop", fixtures.OPENED)
    w.pay("x_drop_in", "x_drop", "a_drop", T - 3 * D + 9 * H, 900.0, "ach")
    w.customer("late", JOINED, mule=LATE)
    w.device("p_late", "d_hidden", T + 3 * D)
    w.external("x_shop", fixtures.OPENED)
    for name in ORDINARY:
        w.customer(name, JOINED)
        w.pay(f"x_{name}_card", f"a_{name}", "x_shop", T - 2 * D + 15 * H, 20.0, "card")
    return w


@pytest.fixture
def config() -> RunConfig:
    """The settings of the tests: the calendar of three weeks, small samples."""
    return CONFIG


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    """The world's contract frames."""
    return evaluation_world().tables()


@pytest.fixture(scope="session")
def graph(tables: dict[str, pd.DataFrame]) -> ReferenceGraph:
    return ReferenceGraph(tables)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The folder the cache, the examples and the evaluation store go under, the test's own."""
    folder = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_DIR", folder)
    return folder


@pytest.fixture
def parity(tmp_path: Path) -> Path:
    record = tmp_path / "parity.toml"
    fixtures.write_parity_record(record)
    return record


def gates_text(passed: Mapping[str, bool] | None = None) -> str:
    """A gate record holding a passing world check and a passing code gate of every gate of
    GATES; passed sets the pass flag of both rows of the entries it names."""
    rows: list[str] = []
    for entry in GATES:
        flag = "true" if (passed or {}).get(entry, True) else "false"
        for kind in (CODE, WORLD):
            rows.append(f'[[gate]]\nentry = "{entry}"\nkind = "{kind}"\npass = {flag}\n')
    return "\n".join(rows)


Records = Callable[..., tuple[Path, Path]]


@pytest.fixture
def records(tmp_path: Path, graph: ReferenceGraph) -> Records:
    """A writer of the world's identity file and gate record, as the loader writes them for
    the seed given: records(seed, identity changes, gate pass flags)."""
    folder = tmp_path / "records"
    folder.mkdir()

    def write(
        seed: int,
        changed: Mapping[str, object] | None = None,
        passed: Mapping[str, bool] | None = None,
    ) -> tuple[Path, Path]:
        identity = folder / f"world_{seed}.identity.toml"
        identity.write_text(identity_text(world_of(graph), seed, changed=changed))
        gates = folder / f"world_{seed}.gates.toml"
        gates.write_text(gates_text(passed))
        return identity, gates

    return write
