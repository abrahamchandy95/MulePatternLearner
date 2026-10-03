"""Observed labels carry no oracle field and keep splits apart."""

from __future__ import annotations

import pytest

from mule_pattern_learner.contract.graph_schema import TRUTH_COLUMNS
from mule_pattern_learner.data.observed_labels import (
    ORACLE_COLUMNS,
    align_observed_labels,
    label_summary,
)
from mule_pattern_learner.testing.builders import assigned_accounts, supplied_labels


def test_observed_labels_have_no_oracle_and_preserve_split_isolation() -> None:
    a = assigned_accounts()
    labels = align_observed_labels(a, supplied_labels())
    assert label_summary(labels) == {"train": 20, "validation": 20, "test": 20}
    assert labels.pu_label.sum() == 20
    assert not labels.loc[labels.split != "train", "pu_label"].any()
    # Every field of the truth table and every label attribute only the oracle reads.
    for field in ("is_mule", "ring_id", "label_source", "mule_ring_id", "mule_label_known"):
        bad = supplied_labels().assign(**{field: 1})
        with pytest.raises(ValueError, match="Oracle"):
            align_observed_labels(a, bad)
    assert {"mule_label_source", "is_mule_masked"} <= ORACLE_COLUMNS
    assert set(TRUTH_COLUMNS) - ORACLE_COLUMNS == {"account_id"}
    with pytest.raises(ValueError, match="discovery"):
        align_observed_labels(a, supplied_labels().assign(known_from_ms=0))
