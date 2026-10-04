"""Each feature alone, on a synthetic feature table."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.univariate import univariate, varying
from mule_pattern_learner.testing.builders import feature_frame


def test_each_feature_that_varies_on_train_is_ranked_in_every_split() -> None:
    frame = feature_frame()
    table = univariate(frame)
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["univariate"]
    # The entity flags are the same for every account and the age for every account of a
    # split, so none of them is ranked.
    ranked = set(table.feature)
    assert ranked == set(varying(frame))
    assert not {"model__type_Account", "model__is_deposit", "account__age_days"} & ranked
    assert set(table.split) == {"train", "validation", "test"}
    assert set(table.metric) == {
        "hidden_roc_auc",
        "hidden_average_precision",
        "roc_auc",
        "average_precision",
    }
    # Weighted by 1 / inclusion probability, without the rejected account.
    test = frame[(frame.split == "test") & ~frame.rejected]
    name = "model__pool_first_in_internal"
    row = table[(table.feature == name) & (table.split == "test")].set_index("metric").value
    weight = test.weight.to_numpy()
    assert row["roc_auc"] == roc_auc_score(test.is_mule, test[name], sample_weight=weight)
    assert row["average_precision"] == average_precision_score(
        test.is_mule, test[name], sample_weight=weight
    )
    # The hidden mules' rank without the revealed ones.
    hidden = test[~test.revealed]
    assert row["hidden_roc_auc"] == roc_auc_score(
        hidden.is_mule, hidden[name], sample_weight=hidden.weight.to_numpy()
    )
    assert table.family[table.feature == name].unique().tolist() == ["model"]


def test_average_precision_ranks_in_the_train_direction() -> None:
    frame = feature_frame()
    table = univariate(frame)
    auc = table[table.metric == "roc_auc"].pivot_table(
        index="feature", columns="split", values="value"
    )
    # A feature lower for mules on train is ranked from its lowest value on every split.
    lower = auc.index[auc.train < 0.5]
    assert len(lower)
    name = lower[0]
    validation = frame[(frame.split == "validation") & ~frame.rejected]
    expected = average_precision_score(
        validation.is_mule, -validation[name], sample_weight=validation.weight
    )
    found = table[
        (table.feature == name)
        & (table.split == "validation")
        & (table.metric == "average_precision")
    ].value
    assert found.tolist() == [expected]
    assert np.isfinite(table.value).all()
