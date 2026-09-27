"""The audit sample: every positive plus uniform negatives with inclusion probabilities."""

from __future__ import annotations

import pandas as pd


def final_evaluation_sample(
    universe: pd.DataFrame, truth: pd.DataFrame, *, negative_limit: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Final-only case/control sample with known inclusion probabilities.

    The caller supplies the COMPLETE frozen test population, not the preparation
    reservoir. This function must never feed training or threshold selection.
    Unknown truth is an error: otherwise neither prevalence nor weights is known.
    """
    import numpy as np

    if negative_limit < 1 or "split" not in universe or not universe.split.eq("test").all():
        raise ValueError("Provide a complete test-only population and positive sample limit")
    if universe.account_id.duplicated().any() or truth.account_id.duplicated().any():
        raise ValueError("Final population and truth must have unique account IDs")
    if "is_mule" in universe:
        raise ValueError("Evaluation population must be label-blind")
    frame = universe.merge(
        truth[["account_id", "is_mule"]], on="account_id", how="left", validate="one_to_one"
    )
    if not frame.is_mule.isin([0, 1]).all():
        raise ValueError("Complete binary truth is required for population-weighted evaluation")
    positive = frame[frame.is_mule == 1].copy()
    negative = frame[frame.is_mule == 0].sort_values("account_id")
    n = min(len(negative), negative_limit)
    chosen = np.random.default_rng(seed).choice(len(negative), size=n, replace=False)
    negative_sample = negative.iloc[chosen].copy()
    positive["inclusion_probability"] = 1.0
    negative_sample["inclusion_probability"] = n / len(negative) if len(negative) else 1.0
    return (
        pd.concat([positive, negative_sample], ignore_index=True)
        .sort_values("account_id")
        .reset_index(drop=True)
    )
