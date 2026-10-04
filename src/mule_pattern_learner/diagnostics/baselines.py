"""Baselines that rank mules from the feature table alone, beside the model's audits.

They answer the question of the retired no_graph control: how well does a table of the
account's own activity rank mules, with no neighbour, association or pool input? The
`account` baseline reads only the account's own history, which the analytics query
computes and no model reads. The others read the root's own model inputs (`model`: the
entity flags, the hub flag and the pool counts, what the no_attention control sees),
the summaries of its candidate pool (`messages`), and everything (`all`). The
`attributes` floor reads the root's entity attributes alone, which are the same for
every scored account, so it ranks at chance.

Each baseline is a logistic regression and a gradient-boosted tree with fixed settings,
trained at the train cutoff as the model is, PU-style: the revealed train mules are
the positives, and every other sampled train account is unlabelled, weighted to the
population, so the hidden mules among them count at their population share. Beside
them, single-feature rankings with no training: the features whose train ROC AUC is
farthest from 0.5, each in its train direction. All are scored on the validation audit
sample (for decisions) and the test one (for reporting), with the metrics and the
ring-clustered intervals of the audits (metrics.bootstrap_intervals), so they compare
with a run's audits of the same accounts, which the table adds as `model` rows. Like the
audits, every ranking is measured on the hidden mules first, the split's revealed mules
removed from it (metrics named hidden_..., metrics.hidden_name), then on every mule.
The `chance` rows are what a random ranking scores there in expectation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..artifacts import DIAGNOSTIC_TABLES
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import (
    BOOTSTRAP_REPLICATES,
    REVIEW_BUDGETS,
    bootstrap_intervals,
    budget_name,
    hidden_name,
    ranking_metrics,
    roc_auc,
)
from .feature_table import FAMILIES, family_of, usable
from .univariate import varying

COLUMNS = DIAGNOSTIC_TABLES["baselines"]
# The learners: an L2 logistic regression on signed log1p, standardised inputs, and a
# small gradient-boosted tree; both balance the classes. The settings are the diagnostic
# study's, fixed, never tuned on a held-out split.
KINDS = ("lr", "hgb")
# The feature families each baseline reads.
BASELINE_FAMILIES = {
    "account": ("account",),
    "model": ("model",),
    "messages": ("messages",),
    "all": FAMILIES,
}
# The entity attributes of the floor.
ATTRIBUTES = ("type_", "is_external", "is_deposit", "history_withheld")
# The account's age at the cutoff is nearly the same for every account of a split (almost
# all were first seen in the first days of the data), so it encodes the cutoff, not the
# account: no model reads it.
CUTOFF_COLUMNS = frozenset({"account__age_days"})
# The single features ranked without training.
SINGLE_FEATURES = 5


def signed_log(x: NDArray[Any]) -> NDArray[Any]:
    return np.sign(x) * np.log1p(np.abs(x))


def make_model(kind: str, seed: int = 0) -> Pipeline | HistGradientBoostingClassifier:
    """A learner of KINDS with the study's fixed settings."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=8,
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced",
            # sklearn takes a bool here, which its type stubs do not allow for.
            early_stopping=False,  # pyright: ignore[reportArgumentType]
            random_state=seed,
        )
    if kind == "lr":
        return make_pipeline(
            FunctionTransformer(signed_log),
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000),
        )
    raise ValueError(f"Unknown learner {kind!r}; the learners are {list(KINDS)}")


def fit_scores(
    kind: str,
    train: pd.DataFrame,
    labels: NDArray[Any],
    columns: Sequence[str],
    scored: Sequence[pd.DataFrame],
    *,
    weights: NDArray[Any] | None = None,
    seed: int = 0,
) -> list[NDArray[np.float64]]:
    """Fit a learner on train's columns and labels, and score each frame of ``scored``.

    Without columns every account scores the same, a ranking at chance.
    """
    if not columns:
        return [np.zeros(len(frame)) for frame in scored]
    model = make_model(kind, seed)
    x = train[list(columns)].to_numpy(np.float64)
    fitted: dict[str, Any] = {}
    if weights is not None:
        key = "logisticregression__sample_weight" if kind == "lr" else "sample_weight"
        fitted[key] = weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, labels, **fitted)
    return [
        model.predict_proba(frame[list(columns)].to_numpy(np.float64))[:, 1].astype(np.float64)
        for frame in scored
    ]


def model_columns(frame: pd.DataFrame, families: Sequence[str]) -> list[str]:
    """The inputs a baseline of these families reads: varying on train, not the cutoff."""
    chosen = set(families)
    return [
        name for name in varying(frame) if family_of(name) in chosen and name not in CUTOFF_COLUMNS
    ]


def attribute_columns(frame: pd.DataFrame) -> list[str]:
    """The floor's inputs: the root's entity attributes that vary on train (none, usually)."""
    return [
        name
        for name in model_columns(frame, ("model",))
        if name.removeprefix("model__").startswith(ATTRIBUTES)
    ]


def pu_labels(train: pd.DataFrame) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """PU training labels and weights: revealed mules against the population-weighted rest.

    Each unlabelled account is weighted by its inclusion weight over their mean, so the
    hidden mules among them count at their population share, not their sample share.
    """
    labels = train.revealed.to_numpy(bool).astype(np.int64)
    weight = train.weight.to_numpy(np.float64)
    unlabelled = labels == 0
    weights = np.where(unlabelled, weight / weight[unlabelled].mean(), 1.0)
    return labels, weights


def scored_rows(
    baseline: str,
    features: str,
    model: str,
    split: str,
    part: pd.DataFrame,
    score: NDArray[np.float64],
    replicates: int,
) -> list[tuple[Any, ...]]:
    """One ranking's rows: its ranking metrics with their ring-clustered intervals.

    The hidden mules' first, the split's revealed mules left out of the ranking (named
    hidden_...), then every mule's.
    """
    y = part.is_mule.to_numpy(np.int64)
    weight = part.weight.to_numpy(np.float64)
    rings = part.ring_id.to_numpy(np.int64)
    rows = []
    for hidden, kept in mule_views(part):
        found = ranking_metrics(y[kept], score[kept], weight[kept])
        intervals = bootstrap_intervals(
            y[kept], score[kept], weight[kept], rings[kept], replicates=replicates
        )
        for metric, value in found.items():
            if value is None:
                continue
            interval = intervals.get(metric) or [np.nan, np.nan]
            name = hidden_name(metric) if hidden else metric
            rows.append((baseline, features, model, split, name, value, *interval))
    return rows


def mule_views(part: pd.DataFrame) -> list[tuple[bool, NDArray[np.bool_]]]:
    """The rows each ranking is measured on: without the revealed mules (hidden), then all."""
    revealed = part.revealed.astype(bool).to_numpy()
    return [(True, ~revealed), (False, np.ones(len(part), dtype=bool))]


def chance_rows(split: str, part: pd.DataFrame) -> list[tuple[Any, ...]]:
    """What a random ranking scores on a split in expectation: the `chance` rows.

    Its AP is the split's weighted prevalence and its ROC AUC 0.5, and each review
    budget finds that share of the mules at the prevalence's precision; of the hidden
    mules first, the revealed ones removed, then of every mule. They have no interval.
    """
    y = part.is_mule.to_numpy(np.int64)
    weight = part.weight.to_numpy(np.float64)
    rows = []
    for hidden, kept in mule_views(part):
        prevalence = float(weight[kept][y[kept] == 1].sum() / weight[kept].sum())
        values = {"average_precision": prevalence, "roc_auc": 0.5}
        for fraction in REVIEW_BUDGETS:
            values[f"precision_at_{budget_name(fraction)}"] = prevalence
            values[f"recall_at_{budget_name(fraction)}"] = fraction
        for metric, value in values.items():
            name = hidden_name(metric) if hidden else metric
            rows.append(("chance", "", "", split, name, value, np.nan, np.nan))
    return rows


def audit_rows(run: str, audits: Mapping[str, Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    """A run's recorded audit metrics and intervals as `model` rows, per audited split.

    The hidden mules' first (named hidden_...), then every mule's; only the metrics with
    an interval, the ranking metrics.
    """
    rows = []
    for split, report in audits.items():
        for prefix in ("hidden_", ""):
            intervals = report.get(f"{prefix}intervals", {})
            for metric, value in report[f"{prefix}metrics"].items():
                if metric not in intervals or value is None:
                    continue
                interval = intervals.get(metric) or [np.nan, np.nan]
                name = hidden_name(metric) if prefix else metric
                rows.append(("model", run, "audit", split, name, float(value), *interval))
    return rows


def baselines(
    frame: pd.DataFrame,
    *,
    run: str = "",
    audits: Mapping[str, Mapping[str, Any]] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> pd.DataFrame:
    """The baselines table: every baseline's metrics on validation and test, with intervals.

    ``audits`` are a run's audit reports by split (its audit/<split>.json), added as the
    `model` rows under the run's name ``run``.
    """
    rows = usable(frame)
    train = rows[rows.split == "train"]
    held = {split: rows[rows.split == split] for split in HELD_OUT_SPLITS}
    labels, weights = pu_labels(train)
    records: list[tuple[Any, ...]] = []
    chosen = {**{name: model_columns(frame, f) for name, f in BASELINE_FAMILIES.items()}}
    chosen["attributes"] = attribute_columns(frame)
    for kind in KINDS:
        for name, columns in chosen.items():
            scores = fit_scores(
                kind, train, labels, columns, list(held.values()), weights=weights, seed=seed
            )
            for (split, part), score in zip(held.items(), scores, strict=True):
                records += scored_rows("pu", name, kind, split, part, score, replicates)
    for name, direction in single_features(frame, SINGLE_FEATURES):
        for split, part in held.items():
            score = direction * part[name].to_numpy(np.float64)
            records += scored_rows("single_feature", name, "raw", split, part, score, replicates)
    for split, part in held.items():
        records += chance_rows(split, part)
    records += audit_rows(run, audits or {})
    return pd.DataFrame(records, columns=list(COLUMNS))


def single_features(frame: pd.DataFrame, count: int) -> list[tuple[str, float]]:
    """The features whose train ROC AUC is farthest from 0.5, with their train direction."""
    train = usable(frame)
    train = train[train.split == "train"]
    y, weight = train.is_mule.to_numpy(), train.weight.to_numpy()
    found: list[tuple[float, str, float]] = []
    for name in model_columns(frame, FAMILIES):
        auc = roc_auc(y, train[name].to_numpy(np.float64), weight)
        if auc is not None:
            found.append((abs(auc - 0.5), name, 1.0 if auc >= 0.5 else -1.0))
    found.sort(key=lambda item: -item[0])
    return [(name, direction) for _, name, direction in found[:count]]
