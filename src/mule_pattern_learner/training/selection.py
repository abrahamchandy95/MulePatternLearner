"""How validation chooses the epoch and the class prior (docs/design.md, "Training"): the
recall of the revealed mules within the top 1% and 5% of the validation accounts, on each
weekly date, averaged over the dates.

The validation examples (data.examples.validation_examples) hold, on each date, every
revealed mule active that date and a weighted sample of the other accounts, so the
budgets are shares of the population the weights estimate (metrics.capture_at_budgets).
A date with no revealed mule active has no recall and is left out of the average.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..metrics import capture_at_budgets


@dataclass(frozen=True)
class Recall:
    """The recall of the revealed mules in the top 1% and 5%, averaged over dates."""

    top_1pct: float
    top_5pct: float

    @property
    def score(self) -> float:
        """What selection maximises: the mean of the two recalls."""
        return (self.top_1pct + self.top_5pct) / 2


def validation_recall(
    as_of_ms: NDArray[Any], label: NDArray[Any], weight: NDArray[Any], score: NDArray[Any]
) -> Recall:
    """The Recall of validation examples, each with its as-of time, its label (1 for a
    revealed mule), its weight and its score."""
    recalls: list[tuple[float, float]] = []
    for day in np.unique(as_of_ms):
        on = as_of_ms == day
        if not (label[on] == 1).any():
            continue
        found = capture_at_budgets(label[on], score[on], weight[on])
        recalls.append((found["recall_at_1pct"], found["recall_at_5pct"]))
    if not recalls:
        raise ValueError("No validation date has a revealed mule to recall")
    top_1pct, top_5pct = np.mean(recalls, axis=0)
    return Recall(float(top_1pct), float(top_5pct))
