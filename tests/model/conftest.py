"""What the encoding and model tests share: a reference world around T, its rows fetched into a
cache for 32 scored accounts, and the batch built from them.

The world is testing.fixtures.batch_world: the catalogue of look-alikes with a household, a
ring, the owner's Account Z and a shop that is an account hub at T.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.subgraph.build import Batch, ExampleRows, build, read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.world import T

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
    *(f"a_{name}" for name in fixtures.SHOPPERS[:9]),
)


@pytest.fixture(scope="session")
def model_rows(tmp_path_factory: pytest.TempPathFactory) -> list[ExampleRows]:
    """The rows of the 32 examples at T, fetched into a cache from the reference graph."""
    folder: Path = tmp_path_factory.mktemp("model")
    parity = folder / "parity.toml"
    fixtures.write_parity_record(parity)
    cache = Cache(folder / "cache")
    fetch(Spy(fixtures.batch_world().graph()), cache, {T: SCORED}, parity=parity)
    return read_rows(cache, [(T, account) for account in SCORED])


@pytest.fixture
def raw_batch(model_rows: list[ExampleRows]) -> Batch:
    """The training batch of the 32 examples, raw: the ring's a_r1 a positive, the rest
    unlabelled, each weighing 1."""
    labels = [int(account == "a_r1") for account in SCORED]
    return build(model_rows, "w", labels=labels, weights=[1.0] * len(SCORED))
