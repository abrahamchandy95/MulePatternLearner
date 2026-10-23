"""What the XGBoost comparisons need that XGBoost itself does not (docs/design.md,
"Comparisons"): the rounds trained and the step of the round counts the choice tries, the
weights at equal totals, and the choice of a round count by the selection score.

Both sides of the process boundary read it: the process that runs XGBoost (comparisons.boosting)
and the one that starts it (comparisons.process). So it imports neither xgboost nor torch; the
selection score it reads, training.selection, imports neither either."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..training.selection import Found, found

# The rounds trained, and the step of the round counts the choice tries.
MAX_ROUNDS = 1_000
ROUND_STEP = 10


def equal_totals(labels: NDArray[np.integer], weights: NDArray[np.floating]) -> NDArray[np.float64]:
    """The weights rescaled so that each label's total is half the number of examples."""
    rescaled = np.asarray(weights, dtype=np.float64).copy()
    for label in (0, 1):
        chosen = labels == label
        if not chosen.any():
            raise ValueError(f"The training examples hold no label {label}")
        rescaled[chosen] *= (len(rescaled) / 2) / rescaled[chosen].sum()
    return rescaled


def check_rounds(rounds: int, trained: int = MAX_ROUNDS) -> None:
    """Refuse a round count that is not a multiple of ROUND_STEP up to the trees trained."""
    if rounds % ROUND_STEP or not 0 < rounds <= trained:
        raise ValueError(f"A round count is a multiple of {ROUND_STEP} up to the trees trained")


def choose_rounds(
    totals: Iterable[tuple[int, NDArray[np.float64]]], keys: pd.DataFrame
) -> tuple[int, Found, tuple[tuple[int, float], ...]]:
    """The round count with the best selection score on the ranked selection keys, given for
    each round count tried its totals, a score for every key in their order (a larger count
    only when strictly better); what it found; and the score of every count tried."""
    best: tuple[int, Found] | None = None
    history: list[tuple[int, float]] = []
    for rounds, total in totals:
        result = found(keys, total)
        history.append((rounds, result.score))
        if best is None or result.score > best[1].score:
            best = (rounds, result)
    if best is None:
        raise ValueError("No round count was tried")
    return best[0], best[1], tuple(history)
