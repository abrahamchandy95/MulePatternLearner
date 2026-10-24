"""A ranking of the rows of a test frame (docs/design.md, "Rankings" and "Metrics"): the rows, which
are positive, and their Monday ranking and pooled ranking, with the headline metrics and the
per-Monday table of the ranking.

The class lives apart from evaluation.rankings, which reads the truth through the answer key, so
that the benchmark can rank a synthetic frame without importing the answer key's module: this
module imports only the metrics, the statuses and the uncertainty, none of which reads the truth
(see the import contract of the answer key's parser in pyproject.toml).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..metrics import TEST_BUDGETS, MondayRanking, budget_name
from .statuses import NON_MULE
from .uncertainty import Ranked


@dataclass(frozen=True)
class Ranking:
    """A ranking of rows of a test frame: the rows (non-mules, and the positives chosen), which
    are positive, and their Monday ranking and pooled ranking."""

    frame: pd.DataFrame
    positive: NDArray[np.bool_]
    order: MondayRanking
    pooled: Ranked

    @classmethod
    def of(cls, frame: pd.DataFrame, positive: NDArray[np.bool_]) -> Ranking:
        """The ranking of the non-mule rows of frame and the rows positive marks."""
        kept = positive | (frame["status"] == NON_MULE).to_numpy()
        rows = frame.loc[kept].reset_index(drop=True)
        chosen = positive[kept]
        logits = rows["logit"].to_numpy(dtype=np.float64)
        return cls(
            rows,
            chosen,
            MondayRanking(rows["t_ms"].to_numpy(dtype=np.int64), logits),
            Ranked(logits, chosen),
        )

    @property
    def weight(self) -> NDArray[np.float64]:
        return self.frame["weight"].to_numpy(dtype=np.float64)

    def headline(self, weight: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """The headline metrics (HEADLINE) under weight, each row's weight by default: pooled
        recall and precision at each budget, ROC AUC and PR AUC; NaN where undefined."""
        used = self.weight if weight is None else weight
        shares = self.order.shares(used)
        found, positives, sizes = self.order.budget_found(shares, self.positive, used)
        total = positives.sum()
        recall = found.sum(axis=0) / total if total > 0 else np.full(len(TEST_BUDGETS), np.nan)
        precision = found.sum(axis=0) / sizes.sum(axis=0)
        return np.r_[
            recall, precision, self.pooled.roc_auc(used), self.pooled.average_precision(used)
        ]

    def per_monday(self) -> pd.DataFrame:
        """Recall and precision at each budget on each Monday, with its positives' weight."""
        weight = self.weight
        found, positives, sizes = self.order.budget_found(
            self.order.shares(weight), self.positive, weight
        )
        table = pd.DataFrame({"t_ms": self.order.mondays, "positives": positives})
        for column, budget in enumerate(TEST_BUDGETS):
            name = budget_name(budget)
            table[f"recall_at_{name}"] = np.divide(
                found[:, column],
                positives,
                out=np.full(len(positives), np.nan),
                where=positives > 0,
            )
            table[f"precision_at_{name}"] = found[:, column] / sizes[:, column]
        return table
