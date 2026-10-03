"""What a run records: its provenance, predictions and metrics (model.pt is SavedModel's)."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig, RuntimeConfig
from ..contract.feature_groups import FeaturePlan
from ..contract.graph_schema import EVALUATION_PROTOCOL
from ..paths import REPOSITORY_ROOT
from .history import RunTotals, disk_hit_rate
from .objective import objective_name
from .schedule import EvaluationSample

# The distributions whose versions config.json records.
PACKAGES = ("mule-pattern-learner", "torch", "numpy", "scikit-learn", "pyTigerGraph")


def _git(*args: str) -> str | None:
    """The output of a git command in the repository, or None without git or a repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def host_settings(device: torch.device, runtime: RuntimeConfig) -> dict[str, Any]:
    """The device, CPU threads and determinism a segment of a run trains with.

    The configuration's fingerprint leaves them out, so a resumed run may change them,
    but each can change floating-point results.
    """
    return {
        "device": str(device),
        "threads": runtime.threads,
        "deterministic": runtime.deterministic,
    }


def provenance(host: dict[str, Any], backend: str, dataset_id: str) -> dict[str, Any]:
    """Where and how a run ran, for config.json: the code, the versions, the host and data.

    The commit is the repository's HEAD and dirty says whether its working tree had
    changes (both None without git); host is the run's host_settings and started is
    when the run started, in UTC.
    """
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "versions": {package: _version(package) for package in PACKAGES},
        **host,
        "sampler_backend": backend,
        "dataset_id": dataset_id,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    dataset_id: str,
    seed: int,
    known_mules: dict[str, int],
    device: torch.device,
    prior: float,
    positive_weight: float,
    plan: FeaturePlan,
    parameter_count: int,
    best_epoch: int,
    results: dict[str, Any],
    selection: dict[str, Any],
    progress: RunTotals,
    rejected_rows: dict[str, int],
    rejected_roots: dict[str, dict[str, int]],
    limit: float,
) -> dict[str, Any]:
    """The metrics.json record of a complete run (its epochs are in epochs.csv)."""
    contexts = progress.context_counts()
    return {
        "status": "complete",
        "dataset_id": dataset_id,
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
        "observed_label_proxy": results,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "validation_proxy": selection,
        "database_calls_during_training": progress.calls(),
        # Wall-clock seconds of every segment of the run, test scoring included.
        "elapsed_seconds": round(time.perf_counter() - progress.started, 3),
        "contexts": {**contexts, "disk_hit_rate": disk_hit_rate(contexts)},
        "rejections": progress.rejections(),
        "sampler_backend": progress.backend,
        "sampler_totals": dict(progress.totals),
        "rejected_evaluation_rows": rejected_rows,
        "rejected_roots": rejected_roots,
        "max_rejected_root_fraction": limit,
        "performance_claim": EVALUATION_PROTOCOL + "_observed_label_proxy_only",
        "proxy_unlabeled_limit": config.training.proxy_unlabeled_limit,
    }
