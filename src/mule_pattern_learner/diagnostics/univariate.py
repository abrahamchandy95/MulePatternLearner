"""Each feature alone: how well it ranks the mules of each split, and in which direction.

For every feature of the table that varies on train, and each split: the weighted ROC
AUC of its raw value (below 0.5 when mules have lower values), and the weighted average
precision of the value in the direction the train split gives it, a ranking fixed before
the held-out splits are read. Each account is weighted by 1 / its inclusion probability,
so a split's sample stands for its population. Rejected accounts are left out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..artifacts import DIAGNOSTIC_TABLES
from ..metrics import average_precision
from .feature_table import FEATURE_SPLITS, family_of, feature_columns, usable

COLUMNS = DIAGNOSTIC_TABLES["univariate"]


def varying(frame: pd.DataFrame, split: str = "train") -> list[str]:
    """The feature columns with more than one value among a split's accepted accounts."""
    rows = usable(frame)
    rows = rows[rows.split == split]
    return [name for name in feature_columns(rows) if rows[name].nunique(dropna=True) > 1]


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float | None:
    """The weighted ROC AUC, None unless both classes are present."""
    if len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y, score, sample_weight=weight))


def univariate(frame: pd.DataFrame) -> pd.DataFrame:
    """The univariate table: roc_auc and average_precision per feature and split.

    ``average_precision`` ranks by the value times the train split's direction (+1 when
    its train ROC AUC is at least 0.5). A metric that is undefined (a split without both
    classes) is left out.
    """
    rows = usable(frame)
    by_split = {split: rows[rows.split == split] for split in FEATURE_SPLITS}
    records: list[tuple[str, str, str, str, float]] = []
    for name in varying(frame):
        family = family_of(name)
        train = by_split["train"]
        auc = weighted_auc(
            train.is_mule.to_numpy(), train[name].to_numpy(), train.weight.to_numpy()
        )
        direction = 1.0 if auc is None or auc >= 0.5 else -1.0
        for split, part in by_split.items():
            y, weight = part.is_mule.to_numpy(np.int64), part.weight.to_numpy(np.float64)
            value = part[name].to_numpy(np.float64)
            found = {
                "roc_auc": weighted_auc(y, value, weight),
                "average_precision": average_precision(y, direction * value, weight),
            }
            records += [
                (name, family, split, metric, number)
                for metric, number in found.items()
                if number is not None
            ]
    return pd.DataFrame(records, columns=list(COLUMNS))


def strongest(table: pd.DataFrame, split: str = "validation", count: int = 30) -> list[str]:
    """The features whose ROC AUC on a split is farthest from 0.5, strongest first.

    Decisions use the validation split, so the figure ranks the features there.
    """
    auc = table[(table.split == split) & (table.metric == "roc_auc")]
    distance = (auc.value - 0.5).abs().to_numpy()
    order = np.argsort(-distance, kind="stable")
    return auc.feature.iloc[order].head(count).tolist()
