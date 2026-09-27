"""Seed reservoirs read the scope population with the labels revealed in the graph."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.testing.builders import unit_config
from mule_pattern_learner.testing.fake_graph import Runner
from mule_pattern_learner.tigergraph.labels import TigerGraphObservedLabels
from mule_pattern_learner.tigergraph.scope import TigerGraphScope


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

    seen = []

    def run(name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        seen.append(params["include_observed"])
        return [{"status": "ok", "accounts": [population_row("A1", True, 5)]}]

    config = unit_config()
    frame, _ = select_accounts(TigerGraphScope(Runner(run)), config.scope.id, config.dataset)
    assert seen == [True] and frame.in_marginal.tolist() == [True]
    assert frame.observed_positive.tolist() == [True] and frame.known_from_ms.tolist() == [5]


def test_stale_population_queries_fail_fast() -> None:
    from mule_pattern_learner.data.accounts import select_accounts

    config = unit_config()
    # An old query emits the discovery time of hidden or negative labels.
    stale = Runner(lambda n, p: [{"status": "ok", "accounts": [population_row("A1", False, 5)]}])
    with pytest.raises(ValueError, match="predates the masked-label predicate"):
        select_accounts(TigerGraphScope(stale), config.scope.id, config.dataset)
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
