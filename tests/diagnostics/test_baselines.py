"""The baselines of the no_graph question, on a synthetic feature table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.baselines import (
    BASELINE_FAMILIES,
    CUTOFF_COLUMNS,
    KINDS,
    SINGLE_FEATURES,
    attribute_columns,
    baselines,
    model_columns,
    pu_labels,
)
from mule_pattern_learner.metrics import ranking_metrics
from mule_pattern_learner.testing.builders import feature_frame

AUDITS: dict[str, dict[str, Any]] = {
    split: {
        "metrics": {"average_precision": ap, "roc_auc": 0.93, "threshold": 0.99},
        "intervals": {"average_precision": [ap / 2, ap * 1.5], "roc_auc": [0.9, 0.95]},
    }
    for split, ap in (("validation", 0.2), ("test", 0.13))
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return baselines(feature_frame(), run="baseline/seed-42", audits=AUDITS, replicates=40)


def test_every_baseline_is_scored_on_validation_and_test_with_intervals(
    table: pd.DataFrame,
) -> None:
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["baselines"]
    pu = table[table.baseline == "pu"]
    assert set(pu.features) == {*BASELINE_FAMILIES, "attributes"} and set(pu.model) == set(KINDS)
    assert set(table.split) == {"validation", "test"}
    singles = table[table.baseline == "single_feature"]
    assert singles.features.nunique() == SINGLE_FEATURES and set(singles.model) == {"raw"}
    ranking = table[table.baseline != "model"]
    assert {"average_precision", "roc_auc", "recall_at_1pct", "precision_at_10pct"} <= set(
        ranking.metric
    )
    assert (ranking.low <= ranking.high).all()
    # The account's own history ranks the synthetic mules well above chance, though its
    # visible event count grows with the cutoff, which costs the trees the most.
    account = pu[(pu.features == "account") & (pu.metric == "roc_auc")]
    assert (account[account.model == "lr"].value > 0.8).all()


def test_the_attribute_floor_ranks_at_chance_and_no_model_reads_the_cutoff(
    table: pd.DataFrame,
) -> None:
    frame = feature_frame()
    assert attribute_columns(frame) == []
    floor = table[(table.features == "attributes")].pivot_table(
        index=["model", "split"], columns="metric", values="value"
    )
    assert np.allclose(floor.roc_auc, 0.5)
    for split in ("validation", "test"):
        part = frame[(frame.split == split) & ~frame.rejected]
        prevalence = part.weight[part.is_mule == 1].sum() / part.weight.sum()
        assert np.allclose(floor.xs(split, level="split").average_precision, prevalence)
    for families in BASELINE_FAMILIES.values():
        assert not CUTOFF_COLUMNS & set(model_columns(frame, families))


def test_the_run_is_added_from_its_audit_reports(table: pd.DataFrame) -> None:
    model = table[table.baseline == "model"]
    assert set(model.features) == {"baseline/seed-42"} and set(model.model) == {"audit"}
    # Only the metrics with intervals: the ranking metrics, not the threshold.
    assert set(model.metric) == {"average_precision", "roc_auc"}
    test = model[(model.split == "test") & (model.metric == "average_precision")]
    assert test[["value", "low", "high"]].to_numpy().tolist() == [[0.13, 0.065, 0.13 * 1.5]]


def test_single_features_are_ranked_in_their_train_direction(table: pd.DataFrame) -> None:
    frame = feature_frame()
    singles = table[(table.baseline == "single_feature") & (table.split == "test")]
    test = frame[(frame.split == "test") & ~frame.rejected]
    for name in singles.features.unique():
        train = frame[(frame.split == "train") & ~frame.rejected]
        higher = train[train.is_mule == 1][name].mean() >= train[train.is_mule == 0][name].mean()
        score = (1.0 if higher else -1.0) * test[name].to_numpy()
        expected = ranking_metrics(test.is_mule.to_numpy(), score, test.weight.to_numpy())
        found = singles[singles.features == name].set_index("metric").value
        assert found["average_precision"] == pytest.approx(expected["average_precision"])


def test_pu_labels_weigh_the_unlabelled_to_their_population_share() -> None:
    train = pd.DataFrame({"revealed": [True, False, False, False], "weight": [1.0, 1.0, 3.0, 4.0]})
    labels, weights = pu_labels(train)
    assert labels.tolist() == [1, 0, 0, 0]
    assert weights.tolist() == [1.0, 0.375, 1.125, 1.5]
