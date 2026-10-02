"""Seed reservoirs read the scope population with the labels revealed in the graph."""

from __future__ import annotations

from typing import Any

from mule_pattern_learner.contract.server import POPULATION_QUERY
from mule_pattern_learner.testing.builders import unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph
from mule_pattern_learner.tigergraph.scope import TigerGraphScopeReader


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
