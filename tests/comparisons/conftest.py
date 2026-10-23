"""What the comparison tests share: the batch world's rows in a cache, its training batches with
its six mules planted as identified, and its selection world of T.

The world is testing.fixtures.batch_world, as in the training tests: its 224 scorable accounts
of T fetched for T and T + 1 week; a_mule, a_mule2 and a_mule3 forward victims' money to the
collector a_collector, and a_r1 and a_r2 pay each other from the ring's device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing import fixtures
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.training.batches import training_batches
from mule_pattern_learner.training.trainer import SelectionWorld

MULES = ("a_collector", "a_mule", "a_mule2", "a_mule3", "a_r1", "a_r2")
WORLD = "0123456789abcdef0123456789abcdef"


@dataclass(frozen=True)
class ComparisonWorld:
    """The training batches of both Mondays with the mules positive, each example weighing
    its own number (weights), and the selection world of T with the mules positive."""

    training: list[Batch]
    selection: SelectionWorld
    mules: tuple[str, ...]
    weights: dict[tuple[int, str], float]


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> ComparisonWorld:
    folder: Path = tmp_path_factory.mktemp("comparisons")
    parity = folder / "parity.toml"
    fixtures.write_parity_record(parity)
    graph = fixtures.batch_world().graph()
    accounts = [account.account_id for account in list_population(graph, T)]
    mondays = (T, T + 7 * DAY_MS)
    cache = Cache(folder / "cache")
    fetch(Spy(graph), cache, {t_ms: accounts for t_ms in mondays}, parity=parity)
    # Every example weighs its own number, so that a weight read in another order is caught.
    weights = {
        (t_ms, account): 1.0 + 0.5 * day + index / 1_000
        for day, t_ms in enumerate(mondays)
        for index, account in enumerate(accounts)
    }
    examples = pd.DataFrame(
        [
            (t_ms, account, int(account in MULES), weights[t_ms, account])
            for t_ms in mondays
            for account in accounts
        ],
        columns=["t_ms", "account_id", "label", "weight"],
    )
    keys = pd.DataFrame(
        {
            "t_ms": np.full(len(accounts), T, dtype=np.int64),
            "account_id": accounts,
            "positive": [account in MULES for account in accounts],
            "ranked": [True] * len(accounts),
            "weight": np.ones(len(accounts)),
        }
    )
    return ComparisonWorld(
        training_batches(cache, WORLD, examples, 64, seed=DEFAULT_CONFIG.training.shuffle_seed),
        SelectionWorld(cache, WORLD, keys, 64),
        MULES,
        weights,
    )
