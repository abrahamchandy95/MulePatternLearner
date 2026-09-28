"""The graph's label contract: revealed positives for training, and the contract audit."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mule_pattern_learner.contract.server import LABEL_CONTRACT_QUERY
from mule_pattern_learner.tigergraph.labels import (
    TigerGraphObservedLabels,
    validate_supervision,
)

POPULATION = pd.DataFrame(
    {
        "account_id": ["a", "b", "c"],
        "split": ["train", "validation", "train"],
        "observed_positive": [True, True, False],
        "known_from_ms": [1_000, 2_000, 0],
    }
)


class Audit:
    """An executor answering the label-contract query with the given counts."""

    def __init__(self, **violations: int) -> None:
        self.violations = violations

    def run(self, name: str, params: dict[str, Any], **_: Any) -> list[dict[str, Any]]:
        assert name == LABEL_CONTRACT_QUERY and params == {}
        return [{"status": "ok", "known_labels": 2, "invalid_mule": 0, **self.violations}]


def test_the_population_rows_give_the_revealed_positives_and_their_clocks() -> None:
    labels = TigerGraphObservedLabels().read(POPULATION)
    assert labels.known_positive.tolist() == [True, True, False]
    assert labels.known_from_ms.tolist() == [1_000, 2_000, 0]
    # Only revealed train positives are training labels.
    assert labels.pu_label.tolist() == [1, 0, 0]


def test_the_contract_audit_refuses_any_violation() -> None:
    assert validate_supervision(Audit())["known_labels"] == 2
    with pytest.raises(
        ValueError, match=r"contract violated after the reveal: \{'invalid_pu': 3\}"
    ):
        validate_supervision(Audit(invalid_pu=3))
