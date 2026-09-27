"""Observed labels carry no oracle field and keep splits apart."""

from __future__ import annotations

import pytest

from mule_pattern_learner.data.observed_labels import align_observed_labels, label_summary
from mule_pattern_learner.testing.builders import assigned_accounts, supplied_labels


def test_observed_labels_have_no_oracle_and_preserve_split_isolation() -> None:
    a = assigned_accounts()
    labels = align_observed_labels(a, supplied_labels())
    assert label_summary(labels) == {"train": 20, "validation": 20, "test": 20}
    assert labels.pu_label.sum() == 20
    assert not labels.loc[labels.split != "train", "pu_label"].any()
    bad = supplied_labels().assign(is_mule=1)
    with pytest.raises(ValueError, match="Oracle"):
        align_observed_labels(a, bad)
    with pytest.raises(ValueError, match="discovery"):
        align_observed_labels(a, supplied_labels().assign(known_from_ms=0))
