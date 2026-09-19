"""Optional post-training oracle evaluation, isolated from the trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch

from ..metrics import evaluate


class EvaluationTruthSource(Protocol):
    def read(self) -> pd.DataFrame: ...


@dataclass
class ParquetEvaluationTruth:
    path: Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


def evaluate_predictions(
    predictions: Path, checkpoint: Path, truth: EvaluationTruthSource
) -> dict[str, Any]:
    """Apply the frozen checkpoint threshold; never choose an epoch or threshold."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    frame = pd.read_parquet(predictions)
    answer = truth.read()
    if "is_mule" not in answer or not answer.is_mule.isin([-1, 0, 1]).all():
        raise ValueError("Evaluation truth requires integer is_mule (-1 unknown, 0 or 1)")
    keys = ["account_id", "date"] if "date" in answer else ["account_id"]
    if answer.duplicated(keys).any():
        raise ValueError("Duplicate evaluation truth keys")
    frame = frame.merge(answer[keys + ["is_mule"]], on=keys, how="left", validate="many_to_one")
    observed = frame[frame.is_mule.isin([0, 1])]
    threshold = float(payload["threshold"])
    result: dict[str, Any] = {
        "evaluated": len(observed),
        "evaluation_cohort": "supplied_prediction_rows_unweighted",
        "population_performance_claim": False,
        "unknown_or_missing_truth": len(frame) - len(observed),
        "selection": payload.get("selected_on"),
        "all": evaluate(
            observed.is_mule.to_numpy(dtype="int64"), observed.score.to_numpy(), threshold
        ),
    }
    if "observed_label" in observed:
        hidden = observed[observed.observed_label == 0]
        result["unlabeled_accounts"] = evaluate(
            hidden.is_mule.to_numpy(dtype="int64"), hidden.score.to_numpy(), threshold
        )
    return result
