"""What the encoding and model tests share: a reference world around T, its rows fetched into a
cache for 32 scored accounts, and the batch built from them.

The world is the catalogue of look-alikes (testing.fixtures) with a household on one device
and IP, a ring on its device, the owner's Account Z enrolled on the ring's device without a
payment, and a shop that 201 customers paid by card, an account hub at T.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.subgraph.build import Batch, ExampleRows, build, read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.world import DAY_MS, T, WorldBuilder
from mule_pattern_learner.tigergraph.queries import VERSIONS

SHOPPERS = tuple(f"c{index:03d}" for index in range(201))
# The 32 scored accounts: the catalogue's 17, the household's, the ring's, Account Z, the shop
# and nine of its customers.
SCORED = (
    "a_collector",
    "a_landlord",
    "a_mule",
    "a_mule2",
    "a_mule3",
    "a_newjob",
    "a_parent",
    "a_plumber",
    "a_saver",
    "a_saver_save",
    "a_sleeper",
    "a_split",
    "a_split_save",
    "a_student",
    "a_tenant1",
    "a_tenant2",
    "a_vic1",
    "a_h1",
    "a_h2",
    "a_r1",
    "a_r2",
    "a_z",
    "a_shop",
    *(f"a_{name}" for name in SHOPPERS[:9]),
)


def model_world() -> WorldBuilder:
    """The world of this module's docstring."""
    w = fixtures.catalogue()
    w.household(["h1", "h2"], fixtures.JOINED, "d_home", "i_home")
    w.pay("x_home", "a_h1", "a_h2", T - 3 * DAY_MS, 75.0, "ach", device="d_home", ip="i_home")
    w.ring(
        9,
        ["r1", "r2"],
        fixtures.JOINED,
        "d_ring",
        T - 20 * DAY_MS,
        T - 10 * DAY_MS,
        T - 2 * DAY_MS,
        identified=["r1"],
    )
    w.pay("z_ring", "a_r1", "a_r2", T - 5 * DAY_MS, 480.0, "zelle", device="d_ring")
    w.customer("z", fixtures.JOINED)
    w.device("p_z", "d_ring", T - 18 * DAY_MS)
    w.customer("shop", fixtures.JOINED)
    for index, name in enumerate(SHOPPERS):
        w.customer(name, fixtures.JOINED)
        at = T - (index % 80 + 1) * DAY_MS + index * fixtures.MINUTE_MS
        w.pay(f"x_shop_{name}", f"a_{name}", "a_shop", at, 10.0, "card")
    return w


@pytest.fixture(scope="session")
def model_rows(tmp_path_factory: pytest.TempPathFactory) -> list[ExampleRows]:
    """The rows of the 32 examples at T, fetched into a cache from the reference graph."""
    folder: Path = tmp_path_factory.mktemp("model")
    parity = folder / "parity.toml"
    parity.write_text(
        "".join(
            f'[[passed]]\nquery = "{name}"\nversion = "{version}"\ntigergraph = "4.2.5"\n'
            f'world = "{"0" * 32}"\ndate = 2026-10-18\n'
            for name, version in VERSIONS.items()
        )
    )
    cache = Cache(folder / "cache")
    fetch(Spy(model_world().graph()), cache, {T: SCORED}, parity=parity)
    return read_rows(cache, [(T, account) for account in SCORED])


@pytest.fixture
def raw_batch(model_rows: list[ExampleRows]) -> Batch:
    """The training batch of the 32 examples, raw: the ring's a_r1 a positive, the rest
    unlabelled, each weighing 1."""
    labels = [int(account == "a_r1") for account in SCORED]
    return build(model_rows, "w", labels=labels, weights=[1.0] * len(SCORED))
