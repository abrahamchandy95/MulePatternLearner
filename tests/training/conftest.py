"""What the training tests share: the batch world's rows in a cache, with the examples and
selection keys made from them, and the training batches with its mules planted as positives.

The world is testing.fixtures.batch_world: the catalogue of look-alikes with a household, a
ring, the owner's Account Z and a shop that is an account hub at T. Its 224 scorable accounts
of T (internal deposit accounts with an owner) are fetched for T and for T + 1 week, whose
lookback still holds every payment of the world. Six of them are the world's mules, which the
tests plant as identified: a_mule, a_mule2 and a_mule3 forward victims' money to the collector
a_collector, and a_r1 and a_r2 pay each other from the ring's device. Their behaviour sets them
apart from the rest, so a model trained on them learns to rank them high.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

T1 = T + 7 * DAY_MS
TIMES = (T, T1)
MULES = ("a_collector", "a_mule", "a_mule2", "a_mule3", "a_r1", "a_r2")
WORLD = "0123456789abcdef0123456789abcdef"
# Short training runs: a few epochs of small batches at a higher learning rate, one prior.
SHORT = replace(
    DEFAULT_CONFIG.training, learning_rate=0.01, batch_size=32, max_epochs=3, patience=3
)


@dataclass(frozen=True)
class TrainingWorld:
    """The cache of the batch world's rows, its world identity, its times, its scorable
    accounts of T and its mules."""

    cache: Cache
    identity: str
    times: tuple[int, ...]
    accounts: tuple[str, ...]
    mules: tuple[str, ...]

    def examples(self, positives: tuple[str, ...] | None = None) -> pd.DataFrame:
        """The training examples of both weeks: every scorable account, positive when it is
        one of positives (the mules by default), each weighing 1."""
        chosen = self.mules if positives is None else positives
        rows = [
            (t_ms, account, int(account in chosen), 1.0)
            for t_ms in self.times
            for account in self.accounts
        ]
        return pd.DataFrame(rows, columns=["t_ms", "account_id", "label", "weight"])

    def selection_keys(self, t_ms: int = T) -> pd.DataFrame:
        """The selection keys of one week: every scorable account ranked, the mules
        positive, each weighing 1."""
        return pd.DataFrame(
            {
                "t_ms": np.full(len(self.accounts), t_ms, dtype=np.int64),
                "account_id": list(self.accounts),
                "positive": [account in self.mules for account in self.accounts],
                "ranked": [True] * len(self.accounts),
                "weight": np.ones(len(self.accounts)),
            }
        )

    def shuffled(self) -> tuple[str, ...]:
        """Six ordinary accounts drawn with a fixed seed: the positives of shuffled labels."""
        ordinary = [account for account in self.accounts if account not in self.mules]
        drawn = np.random.default_rng(7).choice(len(ordinary), len(self.mules), replace=False)
        return tuple(ordinary[index] for index in sorted(drawn))


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> TrainingWorld:
    folder: Path = tmp_path_factory.mktemp("training")
    parity = folder / "parity.toml"
    fixtures.write_parity_record(parity)
    graph = fixtures.batch_world().graph()
    accounts = tuple(account.account_id for account in list_population(graph, T))
    assert len(accounts) == 224 and set(MULES) <= set(accounts)
    cache = Cache(folder / "cache")
    fetch(Spy(graph), cache, {t_ms: accounts for t_ms in TIMES}, parity=parity)
    return TrainingWorld(cache, WORLD, TIMES, accounts, MULES)


@pytest.fixture(scope="session")
def planted(world: TrainingWorld) -> list[Batch]:
    """The training batches with the mules planted as positives."""
    return training_batches(
        world.cache, world.identity, world.examples(), SHORT.batch_size, seed=SHORT.shuffle_seed
    )
