"""Proxy validity: how well a run's ranking, which training chose by the proxy, finds the truth.

Training selects its epoch and threshold, and reports, on observed labels only: the
proxy. Its predictions (predictions/<split>.parquet) score the observed positives and a
sample of the unlabelled accounts of validation and test, which holds the split's hidden
mules only at their population rate: a handful at most, often none. So proxy_validity
takes the hidden and revealed subsets from the run's audit sample of the split
(audit/<split>.parquet), which holds every mule of the split and is scored by the same
model, so its hidden rows are never empty. In each audited split it scores:
- hidden: the audit sample's hidden mules against its non-mules, the revealed mules
  left out, as the audit's hidden-mule metrics are;
- revealed: its revealed mules against its non-mules, the hidden mules left out;
- all: every predicted account against its ground truth.
A ranking that finds the revealed mules and the hidden ones no better than chance
measures the reveal, not mule detection.

The audit subsets are weighted, each sampled account standing for 1 / its inclusion
probability accounts, so they estimate the split's population (n and positives count
the sample, and prevalence is weighted). The predicted accounts are unweighted
(metrics.proxy_metrics with the ground truth as labels): they are the proxy's sample,
not a probability sample of the population, so they say how the proxy ranks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import hidden_rows, read_audit_scores, read_json, read_predictions
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..evaluation.truth import checked_truth
from ..metrics import proxy_metrics, weighted_metrics
from ..paths import RunPaths

# The subsets of a split, each scored on its own, the hidden mules first.
SUBSETS = ("hidden", "revealed", "all")
# The long-format columns of the result.
COLUMNS = ("split", "subset", "metric", "value")


def audit_subset(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """The weighted metrics of an audit sample's rows, named as proxy_metrics names them.

    n and positives count the sampled accounts and mules; prevalence and every other
    metric are weighted by 1 / inclusion probability (metrics.weighted_metrics).
    """
    y = frame.is_mule.to_numpy(np.int64)
    weight = 1 / frame.inclusion_probability.to_numpy(np.float64)
    found = weighted_metrics(y, frame.score.to_numpy(np.float64), weight, threshold)
    return {
        "n": found.pop("sample_accounts"),
        "positives": found.pop("sample_positives"),
        "prevalence": found.pop("weighted_prevalence"),
        **found,
    }


def audit_subsets(audit: pd.DataFrame, threshold: float) -> dict[str, dict[str, Any]]:
    """The hidden and revealed subsets of a split's scored audit sample.

    The hidden mules against the non-mules, the revealed mules removed
    (artifacts.hidden_rows), and the revealed mules against the non-mules, the hidden
    ones removed.
    """
    revealed = audit[audit.is_mule.eq(0) | audit.revealed.astype(bool)]
    return {
        "hidden": audit_subset(hidden_rows(audit), threshold),
        "revealed": audit_subset(revealed.reset_index(drop=True), threshold),
    }


def predicted_subset(
    predictions: pd.DataFrame, truth: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """The unweighted metrics of every predicted account against its ground truth.

    Accounts whose truth is unknown (-1) or missing are left out and counted
    (``unknown_truth``).
    """
    frame = predictions.merge(truth[["account_id", "is_mule"]], on="account_id", how="left")
    known = frame.is_mule.isin([0, 1])
    frame = frame[known].astype({"is_mule": "int64"})
    if (frame.observed_label.eq(1) & frame.is_mule.ne(1)).any():
        raise ValueError("An observed positive is not a mule: the label contract is broken")
    y = frame.is_mule.to_numpy(np.int64)
    found = proxy_metrics(y, frame.score.to_numpy(np.float64), threshold)
    return {**found, "unknown_truth": int((~known).sum())}


def validity_table(
    predictions: Mapping[str, pd.DataFrame],
    audits: Mapping[str, pd.DataFrame],
    truth: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """The validity of a run's ranking in each audited split, in long format.

    ``predictions`` and ``audits`` are each split's proxy predictions and scored audit
    sample; a split without an audit has no rows. One row per split, subset (SUBSETS)
    and metric (COLUMNS); a metric that is undefined for a subset (AP without mules, ROC
    AUC without both classes) is NaN. ``threshold`` is the run's validation threshold.
    """
    answer = checked_truth(truth)
    rows: list[tuple[str, str, str, float]] = []
    for split, audit in audits.items():
        found = {
            **audit_subsets(audit, threshold),
            "all": predicted_subset(predictions[split], answer, threshold),
        }
        for subset in SUBSETS:
            for metric, value in found[subset].items():
                rows.append((split, subset, metric, np.nan if value is None else float(value)))
    return pd.DataFrame(rows, columns=list(COLUMNS))


def audited_splits(run: RunPaths) -> list[str]:
    """The held-out splits the run has audited: a split's report is written last."""
    return [split for split in HELD_OUT_SPLITS if run.audit_report(split).exists()]


def proxy_validity(run: RunPaths, truth: pd.DataFrame) -> pd.DataFrame:
    """The validity table of a complete run's audited splits (validity_table).

    The threshold is the one training chose on validation (metrics.json).
    """
    threshold = float(read_json(run.metrics)["validation_proxy"]["threshold"])
    splits = audited_splits(run)
    predictions = {split: read_predictions(run.predictions(split)) for split in splits}
    audits = {split: read_audit_scores(run.audit_scores(split)) for split in splits}
    return validity_table(predictions, audits, truth, threshold)
