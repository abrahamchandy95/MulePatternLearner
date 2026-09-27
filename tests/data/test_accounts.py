"""Seed reservoirs read graph labels only for the graph's label source."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.testing.builders import FrameObservedLabels, unit_config
from mule_pattern_learner.testing.fake_graph import Runner
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabels
from mule_pattern_learner.tigergraph.scope import TigerGraphScope

# A table source with no labels: population queries then run without include_observed.
NO_LABELS = pd.DataFrame(columns=["account_id", "known_positive", "known_from_ms"])


def population_row(account: str, positive: bool, known: int) -> dict[str, Any]:
    return {
        "account_id": account,
        "partition": 1,
        "group_id": "g",
        "first_seen_ts_ms": 1,
        "observed_positive": positive,
        "known_from_ms": known,
    }


def test_only_graph_labels_read_labels_from_the_graph() -> None:
    from mule_pattern_learner.data.accounts import select_accounts

    seen = []

    def run(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        seen.append(params["include_observed"])
        graph = params["include_observed"]
        row = population_row("A1", graph, 5 if graph else 0)
        return [{"status": "ok", "accounts": [row]}]

    fake = Runner(run)
    config = unit_config()
    select_accounts(
        TigerGraphScope(fake), config.scope.id, config.dataset, FrameObservedLabels(NO_LABELS)
    )
    frame, _ = select_accounts(
        TigerGraphScope(fake), config.scope.id, config.dataset, TigerGraphObservedLabels()
    )
    assert seen == [False, True] and frame.in_marginal.tolist() == [True]
    with pytest.raises(ValueError, match="explicit"):
        select_accounts(TigerGraphScope(fake), config.scope.id, config.dataset, None)


def test_stale_population_queries_fail_fast() -> None:
    from mule_pattern_learner.data.accounts import select_accounts

    config = unit_config()
    # An old query emits the discovery time of hidden or negative labels.
    stale = Runner(lambda n, p: [{"status": "ok", "accounts": [population_row("A1", False, 5)]}])
    with pytest.raises(ValueError, match="predates the masked-label predicate"):
        select_accounts(
            TigerGraphScope(stale), config.scope.id, config.dataset, TigerGraphObservedLabels()
        )
    # Without include_observed the query must return no label information.
    leaky = Runner(lambda n, p: [{"status": "ok", "accounts": [population_row("A1", True, 5)]}])
    with pytest.raises(ValueError, match="include_observed is false"):
        select_accounts(
            TigerGraphScope(leaky), config.scope.id, config.dataset, FrameObservedLabels(NO_LABELS)
        )
    metadata = pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3"],
            "split": ["train", "train", "test"],
            "observed_positive": [True, False, False],
            "known_from_ms": [5, 0, 7],
        }
    )
    with pytest.raises(ValueError, match="1 account.*'A3'.*mule install"):
        TigerGraphObservedLabels().read(metadata)
    metadata.loc[2, "known_from_ms"] = 0
    labels = TigerGraphObservedLabels().read(metadata)
    assert labels.known_positive.tolist() == [True, False, False]
    assert labels.pu_label.tolist() == [1, 0, 0]
