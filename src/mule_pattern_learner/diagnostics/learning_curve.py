"""How ranking quality grows with the number of labelled training mules.

The oracle curve: a learner trained at the train cutoff on k train mules drawn at random
(revealed or hidden, labelled from the ground truth) against every sampled train
non-mule, for k from 10 up to every train mule, REPEATS draws each (one when k takes
them all), scored on the validation and test audit samples, of the hidden mules first
(the split's revealed mules removed, metrics named hidden_...) and then of every mule.
Beside it, the same learner
on the revealed train mules alone against the same non-mules (the study's A3 setup: the
mules training knows, but clean negatives, where training's unlabelled accounts hold
hidden mules), and a run's audits at the revealed count, when the run is audited. A
curve still rising at every train mule says that more labels would help; revealed mules
that train better than random draws of the same count are the easy ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import DIAGNOSTIC_TABLES
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import hidden_first, hidden_name
from .baselines import KINDS, fit_scores, model_columns
from .feature_table import FAMILIES, usable

COLUMNS = DIAGNOSTIC_TABLES["learning_curve"]
# The mule counts of the curve, and the random draws of each.
LABEL_COUNTS = (10, 20, 40, 80, 160)
REPEATS = 5
# The metrics the curve records, of the hidden mules and then of every mule.
MEASURED = ("average_precision", "roc_auc", "recall_at_1pct", "recall_at_5pct")
CURVE_METRICS = (*(hidden_name(metric) for metric in MEASURED), *MEASURED)


def learning_curve(
    frame: pd.DataFrame,
    *,
    counts: Sequence[int] = LABEL_COUNTS,
    repeats: int = REPEATS,
    audits: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """The learning curve table (see the module docstring).

    A count above the train mules is replaced by all of them, once. ``audits`` are a run's
    audit reports by split, recorded as model "model" at the revealed count.
    """
    rows = usable(frame)
    train = rows[rows.split == "train"]
    mules, negatives = train[train.is_mule == 1], train[train.is_mule == 0]
    revealed = train[(train.is_mule == 1) & train.revealed.astype(bool)]
    held = [rows[rows.split == split] for split in HELD_OUT_SPLITS]
    columns = model_columns(frame, FAMILIES)
    records: list[tuple[Any, ...]] = []

    def record(kind: str, labels: str, k: int, repeat: int, scores: list[np.ndarray]) -> None:
        for split, part, score in zip(HELD_OUT_SPLITS, held, scores, strict=True):
            y, weight = part.is_mule.to_numpy(np.int64), part.weight.to_numpy(np.float64)
            found = hidden_first(y, score, weight, part.revealed.to_numpy(bool))
            for metric in CURVE_METRICS:
                value = found[metric]
                if value is not None:
                    records.append((kind, labels, k, repeat, split, metric, value))

    drawn = sorted({min(k, len(mules)) for k in counts if k > 0})
    for kind in KINDS:
        for k in drawn:
            for repeat in range(1 if k == len(mules) else repeats):
                rng = np.random.default_rng([seed, k, repeat])
                chosen = mules.iloc[rng.choice(len(mules), size=k, replace=False)]
                fitted = pd.concat([chosen, negatives])
                labels = fitted.is_mule.to_numpy(np.int64)
                record(kind, "random", k, repeat, fit_scores(kind, fitted, labels, columns, held))
        if len(revealed):
            fitted = pd.concat([revealed, negatives])
            labels = fitted.is_mule.to_numpy(np.int64)
            scores = fit_scores(kind, fitted, labels, columns, held)
            record(kind, "revealed", len(revealed), 0, scores)
    for split, report in (audits or {}).items():
        recorded = {
            **{hidden_name(name): value for name, value in report["hidden_metrics"].items()},
            **report["metrics"],
        }
        for metric in CURVE_METRICS:
            value = recorded.get(metric)
            if value is not None:
                records.append(("model", "revealed", len(revealed), 0, split, metric, float(value)))
    return pd.DataFrame(records, columns=list(COLUMNS))
