"""What a finished run records: the model.pt payload, the split predictions and metrics.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig
from ..contract.feature_groups import FeaturePlan, contract_fingerprint
from ..contract.graph_schema import EVALUATION_PROTOCOL
from ..contract.sampler_plan import SamplerPlan
from ..contract.time_basis import BASIS_ID
from ..data.manifest import manifest_digest
from ..paths import DatasetPaths
from .history import Progress
from .objective import objective_name
from .schedule import EvaluationSample

TRAINING_PROTOCOL = "scoped_observed_label_nnpu_v5"


def model_payload(
    *,
    state: dict[str, torch.Tensor],
    config: RunConfig,
    dataset: DatasetPaths,
    threshold: float,
    plan: FeaturePlan,
    sampler: SamplerPlan,
    known_mules: dict[str, int],
    device: torch.device,
    backend: str,
) -> dict[str, Any]:
    """The model.pt payload of the selected state (read by saved_model.ModelCheckpoint)."""
    return {
        "state_dict": state,
        "config": config.to_dict(),
        "basis_id": BASIS_ID,
        "contract": contract_fingerprint(),
        "dataset": str(dataset.root.resolve()),
        "dataset_manifest_sha256": manifest_digest(dataset),
        "threshold": threshold,
        "feature_dim": len(plan.node_names),
        "input_fingerprint": plan.fingerprint(),
        "sampler": sampler.query_params(),
        "sampler_fingerprint": sampler.fingerprint(),
        "selected_on": "validation_observed_label_proxy_ap",
        "training_protocol": TRAINING_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "known_mules": known_mules,
        "training_device": str(device),
        "sampler_backend": backend,
    }


def prediction_frame(
    accounts: pd.DataFrame,
    samples: list[EvaluationSample],
    scores: np.ndarray,
    accepted: np.ndarray,
) -> pd.DataFrame:
    """The accepted rows of a split's samples, with their observed labels and scores."""
    frames = []
    offset = 0
    for sample in samples:
        frame = accounts.iloc[sample.indices][["account_id", "group_id"]].copy()
        frame["date"] = sample.date
        frame["observed_label"] = sample.labels.astype(np.int64)
        frame["score"] = scores[offset : offset + len(sample.indices)]
        offset += len(sample.indices)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result[accepted].reset_index(drop=True)


def run_summary(
    *,
    config: RunConfig,
    manifest: dict[str, Any],
    seed: int,
    known_mules: dict[str, int],
    device: torch.device,
    prior: float,
    positive_weight: float,
    plan: FeaturePlan,
    parameter_count: int,
    best_epoch: int,
    history: list[dict[str, Any]],
    results: dict[str, Any],
    selection: dict[str, Any],
    checkpoint: Path,
    progress: Progress,
    rejected_rows: dict[str, int],
    rejected_roots: dict[str, dict[str, int]],
    limit: float,
) -> dict[str, Any]:
    """The metrics.json record of a complete run."""
    return {
        "status": "complete",
        "cohort": manifest["cohort"],
        "label_policy": "graph_observed",
        "seed": seed,
        "known_mules": known_mules,
        "device": str(device),
        "loss": "nnPU",
        "class_prior": prior,
        "positive_weight": positive_weight,
        "objective": objective_name(prior, positive_weight),
        "input_fingerprint": plan.fingerprint(),
        "parameter_count": parameter_count,
        "revealed_training_accounts": known_mules["train"],
        "best_epoch": best_epoch,
        "history": history,
        "observed_label_proxy": results,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "validation_proxy": selection,
        "checkpoint": str(checkpoint),
        "database_calls_during_training": progress.calls(),
        "contexts": progress.contexts(),
        "rejections": progress.rejections(),
        "sampler_backend": progress.backend,
        "sampler_totals": dict(progress.totals),
        "rejected_evaluation_rows": rejected_rows,
        "rejected_roots": rejected_roots,
        "max_rejected_root_fraction": limit,
        "performance_claim": EVALUATION_PROTOCOL + "_observed_label_proxy_only",
        "evaluation_unlabeled_limit": config.training.proxy_unlabeled_limit,
        "training_protocol": TRAINING_PROTOCOL,
    }
