"""What the cache and fetch tests share: a reference world and a parity record of the test's
own.

The world is the catalogue of look-alikes (testing.fixtures) around T, 2024-07-01, with:

- a household, p_h1 and p_h2, on the device d_home and the IP i_home since joining, a_h1
  paying a_h2 from them 3 days before t;
- a ring, p_r1 (identified) and p_r2, on the device d_ring since 20 days before t, a_r1 paying
  a_r2 from it 5 days before t;
- a shop, a_shop, paid 10.00 by card once in the lookback by each of 201 customers, a_c000 to
  a_c200, so that it is an account hub at T: more than 200 customers own the other accounts of
  its payments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import VERSIONS

JOINED = fixtures.JOINED
SHOPPERS = tuple(f"c{index:03d}" for index in range(201))


def subgraph_world() -> WorldBuilder:
    """The world of this module's docstring."""
    w = fixtures.catalogue()
    w.household(["h1", "h2"], JOINED, "d_home", "i_home")
    w.pay("x_home", "a_h1", "a_h2", T - 3 * DAY_MS, 75.0, "ach", device="d_home", ip="i_home")
    w.ring(
        9,
        ["r1", "r2"],
        JOINED,
        "d_ring",
        T - 20 * DAY_MS,
        T - 10 * DAY_MS,
        T - 2 * DAY_MS,
        identified=["r1"],
    )
    w.pay("z_ring", "a_r1", "a_r2", T - 5 * DAY_MS, 480.0, "zelle", device="d_ring")
    w.customer("shop", JOINED)
    for index, name in enumerate(SHOPPERS):
        w.customer(name, JOINED)
        at = T - (index % 80 + 1) * DAY_MS + index * fixtures.MINUTE_MS
        w.pay(f"x_shop_{name}", f"a_{name}", "a_shop", at, 10.0, "card")
    return w


@pytest.fixture(scope="session")
def graph() -> ReferenceGraph:
    """The reference graph of the world of this module's docstring."""
    return subgraph_world().graph()


@pytest.fixture
def parity(tmp_path: Path) -> Path:
    """A parity record of the test's own holding every query's current version."""
    record = tmp_path / "parity.toml"
    record.write_text(
        "".join(
            f'[[passed]]\nquery = "{name}"\nversion = "{version}"\ntigergraph = "4.2.5"\n'
            f'world = "{"0" * 32}"\ndate = 2026-10-18\n'
            for name, version in VERSIONS.items()
        )
    )
    return record
