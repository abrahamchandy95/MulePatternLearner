"""Proxy validity: how well a run's proxy predictions rank the ground truth.

Training selects its epoch and threshold, and reports, on observed labels only: the
proxy. These predictions (predictions/<split>.parquet) score the observed positives and a
sample of the unlabeled accounts of validation and test. proxy_validity scores them
against the ground truth in three subsets: all the predicted accounts; the hidden mules
against the non-mules, leaving the revealed mules out; and the revealed mules against
the non-mules. A proxy that ranks the revealed mules well and the hidden ones no better
than chance measures the reveal, not mule detection.

The metrics are unweighted (metrics.proxy_metrics with the ground truth as labels): the
predicted accounts are the proxy's sample, not a probability sample of the population,
so they say how the proxy ranks, not what the population holds; the audits estimate
that (evaluation.audit).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import read_json, read_predictions
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..evaluation.truth import checked_truth
from ..metrics import proxy_metrics
from ..paths import RunPaths

# The subsets of a predicted split's accounts, each scored on its own.
SUBSETS = ("all", "hidden", "revealed")
# The long-format columns of the result.
COLUMNS = ("split", "subset", "metric", "value")


def subset_rows(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    """The predicted accounts of a subset: all, or the hidden or revealed mules with the non-mules.

    A revealed mule has observed_label 1; a hidden one is a mule the proxy saw unlabeled.
    """
    if subset == "all":
        return frame
    mule, revealed = frame.is_mule.eq(1), frame.observed_label.eq(1)
    if subset == "hidden":
        return frame[~revealed]
    if subset == "revealed":
        return frame[~mule | revealed]
    raise ValueError(f"Unknown subset {subset!r}; the subsets are {list(SUBSETS)}")


def split_validity(
    predictions: pd.DataFrame, truth: pd.DataFrame, threshold: float
) -> dict[str, dict[str, Any]]:
    """The oracle metrics of one split's proxy predictions in each subset.

    Accounts whose truth is unknown (-1) or missing are left out and counted
    (``unknown_truth``). ``threshold`` is the run's validation threshold.
    """
    frame = predictions.merge(truth[["account_id", "is_mule"]], on="account_id", how="left")
    known = frame.is_mule.isin([0, 1])
    frame = frame[known].astype({"is_mule": "int64"})
    if (frame.observed_label.eq(1) & frame.is_mule.ne(1)).any():
        raise ValueError("An observed positive is not a mule: the label contract is broken")
    result: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        rows = subset_rows(frame, subset)
        y = rows.is_mule.to_numpy(np.int64)
        result[subset] = proxy_metrics(y, rows.score.to_numpy(np.float64), threshold)
    result["all"]["unknown_truth"] = int((~known).sum())
    return result


def proxy_validity(run: RunPaths, truth: pd.DataFrame) -> pd.DataFrame:
    """The oracle metrics of a complete run's proxy predictions, in long format.

    One row per split, subset and metric (COLUMNS); a metric that is undefined for a
    subset (AP without mules, ROC AUC without both classes) is NaN. The threshold is the
    one training chose on validation (metrics.json).
    """
    answer = checked_truth(truth)
    threshold = float(read_json(run.metrics)["validation_proxy"]["threshold"])
    rows: list[tuple[str, str, str, float]] = []
    for split in HELD_OUT_SPLITS:
        found = split_validity(read_predictions(run.predictions(split)), answer, threshold)
        for subset, metrics in found.items():
            for metric, value in metrics.items():
                rows.append((split, subset, metric, np.nan if value is None else float(value)))
    return pd.DataFrame(rows, columns=list(COLUMNS))
