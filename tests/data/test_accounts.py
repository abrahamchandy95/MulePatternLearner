"""Seed reservoirs read the scope population with the labels revealed in the graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from mule_pattern_learner.config import DatasetConfig, SeedLimits, SplitDates
from mule_pattern_learner.contract.server import POPULATION_QUERY
from mule_pattern_learner.data.accounts import select_accounts
from mule_pattern_learner.testing.builders import unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader
from mule_pattern_learner.training.schedule import pu_batches


def population_row(account: str, positive: bool, known: int) -> dict[str, Any]:
    return {
        "account_id": account,
        "partition": 1,
        "group_id": "g",
        "first_seen_ts_ms": 1,
        "observed_positive": positive,
        "known_from_ms": known,
    }


def test_the_population_is_read_with_the_labels_revealed_in_the_graph() -> None:
    from mule_pattern_learner.data.accounts import select_accounts

    graph = FakeTigerGraph(population=[population_row("A1", True, 5)])
    config = unit_config()
    frame, _ = select_accounts(TigerGraphScopeReader(graph), config.scope.id, config.dataset)
    seen = [params["include_observed"] for _, params in graph.calls]
    assert graph.names() == [POPULATION_QUERY] and seen == [True]
    assert frame.in_marginal.tolist() == [True]
    assert frame.observed_positive.tolist() == [True] and frame.known_from_ms.tolist() == [5]


def test_bounded_seed_reservoir_does_not_enrich_the_nnpu_marginal() -> None:
    # Server has already assigned ownership groups; the client never holds all owners.
    # The graph revealed the first three accounts, discovered at 1 ms.
    known = ["A00000", "A00001", "A00002"]
    rows = [
        {
            "account_id": f"A{i:05}",
            "partition": i % 3 + 1,
            "group_id": str(i),
            "first_seen_ts_ms": 1,
            "observed_positive": i < len(known),
            "known_from_ms": int(i < len(known)),
        }
        for i in range(10050)
    ]
    graph = FakeTigerGraph(population=rows)
    dataset = DatasetConfig(
        dates=SplitDates(("2024-01-01",), ("2024-01-02",), ("2024-01-03",)),
        seed_limits=SeedLimits(10, 10, 10),
        seed=42,
    )
    selected, counts = select_accounts(TigerGraphScopeReader(graph), "strict", dataset)
    # Two pages of 10,000 accounts, with the labels revealed in the graph.
    assert [(name, p["after_id"], p["include_observed"]) for name, p in graph.calls] == [
        (POPULATION_QUERY, "", True),
        (POPULATION_QUERY, "A09999", True),
    ]
    assert sum(counts.values()) == len(rows)
    assert len(selected) <= 33 and selected.in_marginal.sum() == 30
    assert set(known) <= set(selected.account_id)
    observed = selected.account_id.isin(known).to_numpy()
    train = selected.split.eq("train").to_numpy()
    marginal = np.flatnonzero(train & selected.in_marginal.to_numpy())
    positive = np.flatnonzero(train & observed)
    draws = list(
        pu_batches(marginal, observed, np.random.default_rng(42), 8, positive_indices=positive)
    )
    assert sorted(np.concatenate([u for _, u in draws])) == sorted(marginal)
    assert all(observed[p].all() for p, _ in draws)
