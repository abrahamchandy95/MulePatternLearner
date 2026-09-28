"""How the features shift between the splits' cutoffs, and what the shift costs a model.

Each split is read at its own cutoff and with its own visibility, so its accounts have
seen more history than the train split's (about 6, 9 and 12 months of it). The non-mules
of a split are a uniform sample, so their shift against train's non-mules is feature
drift, not a change of labels. Per feature and held-out split: the standardised mean
difference from train (the difference of the weighted means over the root mean of the
two weighted variances), the ROC AUC of telling the split's non-mules from train's by
the value (0.5 means no shift), and the share of the split's non-mules above train's
90th percentile (10% without shift); the weighted 10th, 50th and 90th percentiles of
every split's non-mules sit beside them.

The shift cost: a learner trained at the train cutoff with every train mule labelled,
scored on validation and test, against the same learner on each split's percentiles
(split_rank_transform, which removes the shift of each feature's distribution), and a
cross-validation inside each held-out split, which no shift touches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..artifacts import DIAGNOSTIC_TABLES
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import ranking_metrics, weighted_quantiles
from .baselines import KINDS, fit_scores, model_columns
from .feature_table import FAMILIES, FEATURE_SPLITS, family_of, usable
from .univariate import varying, weighted_auc

COLUMNS = DIAGNOSTIC_TABLES["drift"]
QUANTILES = (0.1, 0.5, 0.9)
# The shift costs' setups, and the folds of the cross-validation inside a split.
SETUPS = ("train_cutoff", "split_percentiles", "within_split")
FOLDS = 5


def weighted_moments(values: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weight))
    return mean, float(np.average((values - mean) ** 2, weights=weight))


def feature_shift(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    """The shift rows of every feature that varies on train (see the module docstring)."""
    rows = usable(frame)
    negatives = {
        split: rows[(rows.split == split) & (rows.is_mule == 0)] for split in FEATURE_SPLITS
    }
    base = negatives["train"]
    records: list[tuple[Any, ...]] = []
    for name in varying(frame):
        family = family_of(name)
        train_x, train_w = base[name].to_numpy(np.float64), base.weight.to_numpy(np.float64)
        train_mean, train_var = weighted_moments(train_x, train_w)
        high = float(weighted_quantiles(train_x, train_w, [0.9])[0])
        for split, part in negatives.items():
            x, w = part[name].to_numpy(np.float64), part.weight.to_numpy(np.float64)
            for q, value in zip(QUANTILES, weighted_quantiles(x, w, QUANTILES), strict=True):
                records.append((name, family, "", "", split, f"q{round(q * 100)}", float(value)))
            if split == "train":
                continue
            mean, var = weighted_moments(x, w)
            spread = np.sqrt((var + train_var) / 2)
            smd = (mean - train_mean) / spread if spread > 0 else 0.0
            labels = np.concatenate([np.zeros(len(train_x)), np.ones(len(x))])
            auc = weighted_auc(labels, np.concatenate([train_x, x]), np.concatenate([train_w, w]))
            above = float(np.average(x > high, weights=w))
            for metric, value in (("smd", smd), ("shift_auc", auc), ("above_train_q90", above)):
                if value is not None:
                    records.append((name, family, "", "", split, metric, float(value)))
    return records


def split_rank_transform(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Each feature as its weighted population percentile within its own split.

    Every accepted account of the split counts with its inclusion weight, so the
    percentiles estimate the split population's own distribution; tied values share the
    middle of their block. Mules weigh 1 and barely move it.
    """
    out = frame.copy()
    out[columns] = out[columns].astype(np.float64)
    for _, part in frame.groupby("split"):
        weight = part.weight.to_numpy(np.float64)
        for name in columns:
            x = part[name].to_numpy(np.float64)
            order = np.argsort(x, kind="mergesort")
            ordered, weights = x[order], weight[order]
            unique, start = np.unique(ordered, return_index=True)
            cumulative = np.concatenate([[0.0], np.cumsum(weights)])
            end = np.append(start[1:], len(ordered))
            middle = (cumulative[start] + cumulative[end]) / 2 / cumulative[-1]
            out.loc[part.index, name] = middle[np.searchsorted(unique, x)]
    return out


def shift_cost(frame: pd.DataFrame, *, seed: int = 0) -> list[tuple[Any, ...]]:
    """The ranking metrics of each learner and setup on validation and test."""
    rows = usable(frame)
    columns = model_columns(frame, FAMILIES)
    ranked = split_rank_transform(rows, columns)
    records: list[tuple[Any, ...]] = []

    def record(kind: str, setup: str, split: str, part: pd.DataFrame, score: np.ndarray) -> None:
        y, weight = part.is_mule.to_numpy(np.int64), part.weight.to_numpy(np.float64)
        for metric, value in ranking_metrics(y, score, weight).items():
            if value is not None:
                records.append(("", "all", kind, setup, split, metric, float(value)))

    for kind in KINDS:
        for setup, source in (("train_cutoff", rows), ("split_percentiles", ranked)):
            train = source[source.split == "train"]
            parts = [source[source.split == split] for split in HELD_OUT_SPLITS]
            labels = train.is_mule.to_numpy(np.int64)
            scores = fit_scores(kind, train, labels, columns, parts, seed=seed)
            for split, part, score in zip(HELD_OUT_SPLITS, parts, scores, strict=True):
                record(kind, setup, split, part, score)
        for split in HELD_OUT_SPLITS:
            part = rows[rows.split == split].reset_index(drop=True)
            y = part.is_mule.to_numpy(np.int64)
            if min(int(y.sum()), int((y == 0).sum())) < FOLDS:
                continue
            folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
            score = np.zeros(len(part))
            for fitted, held in folds.split(np.zeros(len(part)), y):
                inside, outside = part.iloc[fitted], part.iloc[held]
                (found,) = fit_scores(kind, inside, y[fitted], columns, [outside], seed=seed)
                score[held] = found
            record(kind, "within_split", split, part, score)
    return records


def drift(frame: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """The drift table: the feature shift rows, then the shift cost rows."""
    return pd.DataFrame(feature_shift(frame) + shift_cost(frame, seed=seed), columns=list(COLUMNS))


def strongest_shifts(table: pd.DataFrame, count: int = 25) -> list[str]:
    """The features whose non-mules shift most from train (largest absolute SMD), first."""
    smd = table[table.metric == "smd"]
    largest = smd.assign(size=smd.value.abs()).groupby("feature")["size"].max()
    return largest.sort_values(ascending=False, kind="stable").head(count).index.tolist()
