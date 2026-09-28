"""The audit sample: every positive plus uniform negatives with inclusion probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.graph_schema import TRUTH_COLUMNS

# The negatives an audit sample draws from its split; every positive is kept.
AUDIT_NEGATIVES = 2000


def audit_sample(
    population: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    seed: int,
    negatives: int = AUDIT_NEGATIVES,
) -> pd.DataFrame:
    """Case/control sample of one split's population with known inclusion probabilities.

    The caller supplies the COMPLETE frozen population of the split (audit_population),
    not the preparation reservoir; the sample must never feed training or threshold
    selection. It keeps every positive and draws ``negatives`` negatives uniformly with
    ``seed`` (dataset.split_seed), so it depends only on the population, the truth and
    the seed: every run of a dataset is audited on the same accounts. The rows, in
    account order, carry the population's columns, the truth's (TRUTH_COLUMNS) and
    inclusion_probability. Unknown truth is an error: otherwise neither prevalence nor
    weights is known.
    """
    if negatives < 1 or "split" not in population or population.split.nunique() != 1:
        raise ValueError("Provide the complete population of one split and a positive count")
    if population.account_id.duplicated().any() or truth.account_id.duplicated().any():
        raise ValueError("The population and the truth must have unique account IDs")
    if "is_mule" in population:
        raise ValueError("Evaluation population must be label-blind")
    frame = population.merge(
        truth[list(TRUTH_COLUMNS)], on="account_id", how="left", validate="one_to_one"
    )
    if not frame.is_mule.isin([0, 1]).all():
        raise ValueError("Complete binary truth is required for population-weighted evaluation")
    positive = frame[frame.is_mule == 1].copy()
    negative = frame[frame.is_mule == 0].sort_values("account_id")
    n = min(len(negative), negatives)
    chosen = np.random.default_rng(seed).choice(len(negative), size=n, replace=False)
    negative_sample = negative.iloc[chosen].copy()
    positive["inclusion_probability"] = 1.0
    negative_sample["inclusion_probability"] = n / len(negative) if len(negative) else 1.0
    return (
        pd.concat([positive, negative_sample], ignore_index=True)
        .sort_values("account_id")
        .reset_index(drop=True)
    )
