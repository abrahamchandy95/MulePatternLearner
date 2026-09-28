"""Proxy validity: the oracle metrics of a run's proxy predictions, in three subsets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import write_json, write_predictions
from mule_pattern_learner.diagnostics.proxy_validity import COLUMNS, proxy_validity
from mule_pattern_learner.paths import RunPaths

# One split's predictions: a revealed mule (R), two hidden mules (H1, H2), three non-mules
# (N1 to N3), and an account whose truth is unknown (U).
ROWS = [
    ("R", 1, 0.9),
    ("N1", 0, 0.8),
    ("H1", 0, 0.7),
    ("H2", 0, 0.3),
    ("N2", 0, 0.2),
    ("N3", 0, 0.1),
    ("U", 0, 0.5),
]
TRUTH = pd.DataFrame(
    {
        "account_id": ["R", "H1", "H2", "N1", "N2", "N3", "U"],
        "is_mule": [1, 1, 1, 0, 0, 0, -1],
        "ring_id": [0, 0, -1, -1, -1, -1, -1],
        "label_source": "phantomledger_role",
    }
)


def predictions(date: str) -> pd.DataFrame:
    frame = pd.DataFrame(ROWS, columns=["account_id", "observed_label", "score"])
    frame.insert(1, "group_id", "G" + frame.account_id)
    frame.insert(2, "date", date)
    return frame


def test_the_proxy_predictions_are_scored_against_the_truth_in_each_subset(
    tmp_path: Path,
) -> None:
    run = RunPaths(tmp_path / "run")
    run.root.mkdir()
    write_json(run.metrics, {"validation_proxy": {"threshold": 0.75}})
    for split, date in (("validation", "2024-10-01"), ("test", "2025-01-01")):
        write_predictions(run.predictions(split), predictions(date))
    table = proxy_validity(run, TRUTH)
    assert tuple(table.columns) == COLUMNS
    assert set(table.split) == {"validation", "test"}
    values = table[table.split == "test"].set_index(["subset", "metric"]).value
    # All six known accounts rank R, N1, H1, H2, N2, N3: the mules come 1st, 3rd and 4th,
    # so AP = (1/1 + 2/3 + 3/4) / 3 = 29/36, and 7 of the 9 mule and non-mule pairs rank
    # the mule higher. The threshold 0.75 flags R and N1.
    assert values["all", "n"] == 6 and values["all", "positives"] == 3
    assert values["all", "unknown_truth"] == 1
    assert values["all", "average_precision"] == pytest.approx(29 / 36)
    assert values["all", "roc_auc"] == pytest.approx(7 / 9)
    assert values["all", "precision"] == pytest.approx(1 / 2)
    assert values["all", "recall"] == pytest.approx(1 / 3)
    # Hidden: without R the ranking is N1, H1, H2, N2, N3, so AP = (1/2 + 2/3) / 2 = 7/12
    # and 4 of 6 pairs rank the mule higher.
    assert values["hidden", "n"] == 5 and values["hidden", "positives"] == 2
    assert values["hidden", "average_precision"] == pytest.approx(7 / 12)
    assert values["hidden", "roc_auc"] == pytest.approx(4 / 6)
    assert values["hidden", "recall"] == 0
    # Revealed: R above every non-mule.
    assert values["revealed", "n"] == 4 and values["revealed", "positives"] == 1
    assert values["revealed", "average_precision"] == 1 and values["revealed", "roc_auc"] == 1
    # The splits are scored alike.
    validation = table[table.split == "validation"].value.to_numpy()
    assert np.array_equal(validation, table[table.split == "test"].value.to_numpy())


def test_an_observed_positive_that_is_no_mule_breaks_the_label_contract(tmp_path: Path) -> None:
    run = RunPaths(tmp_path / "run")
    run.root.mkdir()
    write_json(run.metrics, {"validation_proxy": {"threshold": 0.5}})
    for split in ("validation", "test"):
        write_predictions(run.predictions(split), predictions("2025-01-01"))
    wrong = TRUTH.assign(is_mule=TRUTH.is_mule.where(TRUTH.account_id != "R", 0))
    with pytest.raises(ValueError, match="label contract"):
        proxy_validity(run, wrong)
