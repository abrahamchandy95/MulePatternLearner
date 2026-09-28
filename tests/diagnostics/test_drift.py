"""The shift of the features between the splits' cutoffs, on a synthetic feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.drift import (
    SETUPS,
    drift,
    split_rank_transform,
    strongest_shifts,
)
from mule_pattern_learner.testing.builders import feature_frame


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return drift(feature_frame())


def test_the_features_that_grow_with_history_shift_the_most(table: pd.DataFrame) -> None:
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["drift"]
    shifted = strongest_shifts(table, 2)
    assert shifted == ["account__visible_event_count", "messages__max_pair_prior_count"]
    smd = table[table.metric == "smd"].pivot_table(index="feature", columns="split", values="value")
    # A year of history shifts further than nine months, and noise barely shifts.
    assert (smd.loc[shifted, "test"] > smd.loc[shifted, "validation"]).all()
    noise = smd.loc["message_context__mean_pair_count_7d"].to_numpy(np.float64)
    assert np.abs(noise).max() < 0.3
    auc = table[(table.metric == "shift_auc") & (table.feature == shifted[0])]
    assert (auc.value > 0.9).all()
    above = table[(table.metric == "above_train_q90") & (table.feature == shifted[0])]
    assert (above.value > 0.9).all()
    quantiles = table[table.metric.str.startswith("q")]
    assert set(quantiles.split) == {"train", "validation", "test"}
    assert set(quantiles.metric) == {"q10", "q50", "q90"}


def test_the_standardised_mean_difference_is_computed_on_non_mules(table: pd.DataFrame) -> None:
    frame = feature_frame()
    name = "messages__max_pair_prior_count"
    negatives = frame[(frame.is_mule == 0) & ~frame.rejected]
    train = negatives[negatives.split == "train"][name]
    test = negatives[negatives.split == "test"][name]
    expected = (test.mean() - train.mean()) / np.sqrt((test.var(ddof=0) + train.var(ddof=0)) / 2)
    found = table[(table.feature == name) & (table.split == "test") & (table.metric == "smd")]
    assert found.value.tolist() == [pytest.approx(expected)]


def test_the_shift_cost_compares_each_learner_across_setups(table: pd.DataFrame) -> None:
    cost = table[table.feature == ""]
    assert set(cost.setup) == set(SETUPS) and set(cost.model) == {"lr", "hgb"}
    assert set(cost.split) == {"validation", "test"} and set(cost.family) == {"all"}
    assert {"average_precision", "roc_auc"} <= set(cost.metric)
    auc = cost[cost.metric == "roc_auc"].value
    assert ((auc > 0.5) & (auc <= 1)).all()


def test_split_percentiles_remove_the_shift_of_every_feature() -> None:
    frame = feature_frame()
    frame = frame[~frame.rejected]
    name = "account__visible_event_count"
    ranked = split_rank_transform(frame, [name])
    for _, part in ranked.groupby("split"):
        values = part[name]
        assert ((values > 0) & (values < 1)).all()
        # The percentiles keep the order of the values inside a split.
        original = frame.loc[part.index, name]
        assert values.rank().tolist() == original.rank().tolist()
        # Weighted, the median account of a split sits at about one half.
        middle = np.average(values, weights=part.weight)
        assert middle == pytest.approx(0.5, abs=0.02)
