"""Which epoch a run keeps: the rules of training.selection.

After every epoch the trainer scores the validation proxy (validation's revealed mules
and a sample of its unlabeled accounts) and writes the epoch's row of epochs.csv, with
every rule's criterion: the proxy AP, the proxy ROC AUC and the nnPU risk
(objective.pu_risk). The rule picks the criterion the kept epoch is best at, and early
stopping counts the epochs since that one. selection_value puts every rule on one scale,
higher being better, so a run keeps one best value whatever its rule, and the resume
state stores it in the slot it had when AP was the only rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import NO_SELECTION


def selection_value(rule: str, row: Mapping[str, Any]) -> float | None:
    """An epoch's value under a selection rule, higher being better; None if undefined.

    ``row`` is the epoch's epochs.csv row. "validation_ap" and "validation_roc_auc" are
    its columns; "validation_pu_risk" is its risk negated, since a lower risk is better;
    "none" values each epoch by its number, so each epoch beats the one before and the
    last is kept. An AP without positives, or a ROC AUC without both classes, is None.
    """
    if rule == NO_SELECTION:
        return float(row["epoch"])
    if rule not in ("validation_ap", "validation_roc_auc", "validation_pu_risk"):
        raise ValueError(f"Unknown selection rule {rule!r}")
    value = row[rule]
    if value is None:
        return None
    return -float(value) if rule == "validation_pu_risk" else float(value)


def stops_early(rule: str, patience: int, epoch: int, best_epoch: int) -> bool:
    """Whether training ends after ``epoch`` (from 1): ``patience`` epochs without a gain.

    A patience of 0 never stops early, and neither does the rule "none", which trains
    every epoch.
    """
    return rule != NO_SELECTION and patience > 0 and epoch - best_epoch >= patience
