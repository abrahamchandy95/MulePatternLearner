"""The selection rules put every criterion on one scale and stop early as they say."""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import SELECTION_RULES
from mule_pattern_learner.training.selection import selection_value, stops_early

ROW = {"epoch": 3, "validation_ap": 0.4, "validation_roc_auc": 0.9, "validation_pu_risk": 0.25}


def test_every_rule_values_an_epoch_higher_when_it_is_better() -> None:
    values = {rule: selection_value(rule, ROW) for rule in SELECTION_RULES}
    # The risk is negated, since a lower risk is better; "none" prefers the later epoch.
    assert values == {
        "validation_ap": 0.4,
        "validation_roc_auc": 0.9,
        "validation_pu_risk": -0.25,
        "none": 3.0,
    }
    assert selection_value("validation_ap", {**ROW, "validation_ap": None}) is None
    with pytest.raises(ValueError, match="Unknown selection rule"):
        selection_value("test_ap", ROW)


def test_early_stopping_follows_the_rule() -> None:
    assert stops_early("validation_roc_auc", 2, epoch=5, best_epoch=3)
    assert not stops_early("validation_roc_auc", 2, epoch=4, best_epoch=3)
    # Patience 0 never stops, and "none" trains every epoch.
    assert not stops_early("validation_ap", 0, epoch=9, best_epoch=1)
    assert not stops_early("none", 2, epoch=9, best_epoch=1)
