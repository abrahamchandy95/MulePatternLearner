"""Each feature alone: how well it ranks the mules of each split, and in which direction.

For every feature of the table that varies on train, and each split: the weighted ROC
AUC of its raw value (below 0.5 when mules have lower values), and the weighted average
precision of the value in the direction the train split gives it, a ranking fixed before
the held-out splits are read; of the hidden mules first, the split's revealed mules
removed (hidden_roc_auc and hidden_average_precision, metrics.hidden_name), then of every
mule. The direction is every train mule's. Each account is weighted by 1 / its inclusion
probability, so a split's sample stands for its population. Rejected accounts are left
out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import DIAGNOSTIC_TABLES
from ..metrics import average_precision, hidden_name, roc_auc
from .feature_table import FEATURE_SPLITS, family_of, feature_columns, usable

COLUMNS = DIAGNOSTIC_TABLES["univariate"]


def varying(frame: pd.DataFrame, split: str = "train") -> list[str]:
    """The feature columns with more than one value among a split's accepted accounts."""
    rows = usable(frame)
    rows = rows[rows.split == split]
    return [name for name in feature_columns(rows) if rows[name].nunique(dropna=True) > 1]


def univariate(frame: pd.DataFrame) -> pd.DataFrame:
    """The univariate table: roc_auc and average_precision per feature and split.

    Each of the hidden mules first (hidden_roc_auc, hidden_average_precision), then of
    every mule. ``average_precision`` ranks by the value times the train split's
    direction (+1 when every train mule's ROC AUC is at least 0.5). A metric that is
    undefined (a split without both classes) is left out.
    """
    rows = usable(frame)
    by_split = {split: rows[rows.split == split] for split in FEATURE_SPLITS}
    records: list[tuple[str, str, str, str, float]] = []
    for name in varying(frame):
        family = family_of(name)
        train = by_split["train"]
        auc = roc_auc(train.is_mule.to_numpy(), train[name].to_numpy(), train.weight.to_numpy())
        direction = 1.0 if auc is None or auc >= 0.5 else -1.0
        for split, part in by_split.items():
            y, weight = part.is_mule.to_numpy(np.int64), part.weight.to_numpy(np.float64)
            value = part[name].to_numpy(np.float64)
            hidden = ~part.revealed.to_numpy(bool)
            found = {
                hidden_name("roc_auc"): roc_auc(y[hidden], value[hidden], weight[hidden]),
                hidden_name("average_precision"): average_precision(
                    y[hidden], direction * value[hidden], weight[hidden]
                ),
                "roc_auc": roc_auc(y, value, weight),
                "average_precision": average_precision(y, direction * value, weight),
            }
            records += [
                (name, family, split, metric, number)
                for metric, number in found.items()
                if number is not None
            ]
    return pd.DataFrame(records, columns=list(COLUMNS))
