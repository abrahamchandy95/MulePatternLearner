"""Optional post-training oracle evaluation, isolated from the trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from ..common import cutoff_ms
from ..metrics import evaluate
from .checkpoint import ModelCheckpoint

if TYPE_CHECKING:
    from .executor import QueryExecutor


class EvaluationTruthSource(Protocol):
    def read(self) -> pd.DataFrame: ...


@dataclass
class ParquetEvaluationTruth:
    path: Path

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


@dataclass
class GraphEvaluationTruth:
    """Oracle truth paged from the graph's label contract, for evaluation only.

    temporal_get_account_supervision is the oracle endpoint; training never calls
    it. An account whose label is not known (mule_label_known false) reports
    is_mule = -1, which the evaluators treat as unknown, never as a negative.
    """

    executor: QueryExecutor | None = None

    def read(self) -> pd.DataFrame:
        from .executor import TigerGraphExecutor, account_pages

        executor = self.executor if self.executor is not None else TigerGraphExecutor()
        rows: list[dict[str, Any]] = []
        pages = account_pages(executor, "temporal_get_account_supervision", {}, timeout_s=900.0)
        for page in pages:
            for row in page:
                known = bool(row["mule_label_known"])
                rows.append(
                    {
                        "account_id": str(row["account_id"]),
                        "is_mule": int(row["is_mule"]) if known else -1,
                    }
                )
        return pd.DataFrame(rows, columns=["account_id", "is_mule"])


def evaluate_predictions(
    predictions: Path, checkpoint: Path | ModelCheckpoint, truth: EvaluationTruthSource
) -> dict[str, Any]:
    """Apply the frozen checkpoint threshold; never choose an epoch or threshold."""
    saved = ModelCheckpoint.of(checkpoint)
    frame = pd.read_parquet(predictions)
    answer = truth.read()
    if "is_mule" not in answer or not answer.is_mule.isin([-1, 0, 1]).all():
        raise ValueError("Evaluation truth requires integer is_mule (-1 unknown, 0 or 1)")
    keys = ["account_id", "date"] if "date" in answer else ["account_id"]
    if answer.duplicated(keys).any():
        raise ValueError("Duplicate evaluation truth keys")
    frame = frame.merge(answer[keys + ["is_mule"]], on=keys, how="left", validate="many_to_one")
    observed = frame[frame.is_mule.isin([0, 1])]
    threshold = saved.threshold
    result: dict[str, Any] = {
        "evaluated": len(observed),
        "evaluation_cohort": "supplied_prediction_rows_unweighted",
        "population_performance_claim": False,
        "unknown_or_missing_truth": len(frame) - len(observed),
        "selection": saved.selected_on,
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


def evaluate_weighted(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Weighted AP/ROC/threshold metrics; estimates, not census measurements."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not len(frame) or not frame.is_mule.isin([0, 1]).all():
        raise ValueError("Weighted evaluation needs binary truth and nonempty predictions")
    p = frame.inclusion_probability.to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Invalid inclusion probabilities")
    y, score = frame.is_mule.to_numpy(int), frame.score.to_numpy(float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    w = 1 / p
    predicted = score >= threshold
    positives, tp = float(w[y == 1].sum()), float(w[(y == 1) & predicted].sum())
    precision, recall = tp / max(float(w[predicted].sum()), 1), tp / max(positives, 1)
    return {
        "sample_accounts": len(frame),
        "sample_positives": int(y.sum()),
        "estimated_population": float(w.sum()),
        "weighted_prevalence": positives / float(w.sum()),
        "average_precision": float(average_precision_score(y, score, sample_weight=w))
        if y.any()
        else None,
        "roc_auc": float(roc_auc_score(y, score, sample_weight=w))
        if len(np.unique(y)) == 2
        else None,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "evaluation_cohort": "all_test_positives_plus_uniform_negatives_inverse_probability_weighted",
    }


def evaluate_final_population(
    checkpoint: Path | ModelCheckpoint,
    truth: EvaluationTruthSource,
    output: Path,
    *,
    negative_limit: int = 2000,
    executor: Any = None,
    dataset: Path | None = None,
    contexts: Any = None,
    hubs: Any = None,
) -> dict[str, Any]:
    """Score a fresh final-only sample from the entire frozen test partition.

    This POC audit bounds host metadata to one million test accounts. It never
    changes a checkpoint and refuses to overwrite an existing final report. The
    prepared dataset (``dataset`` or the path recorded in the checkpoint) supplies
    the test cutoff clock and the hub registry, so scoring matches training.

    Accounts TigerGraph rejects are not scored. A rejected test positive, or a
    rejected fraction of the sample above the checkpoint's
    ``max_rejected_root_fraction`` (default 0), fails the audit before anything is
    written: the weighted metrics would silently describe a censored population.
    Rejected negatives within the limit are listed in ``<output>.rejected.txt`` and
    the metrics' ``evaluation_cohort`` says that they were dropped.
    """
    import json

    from .config_schema import setting, split_seed
    from .contract import SPLIT_PHASE
    from .dataset import MANIFEST, load_prepared, sample_keys
    from .executor import account_pages, live_executor
    from .hubs import load_hub_registry
    from .policy import exceeds_rejection_limit
    from .predictor import TemporalPredictor, write_rejected
    from .source import close_source, rejection_summary

    if output.suffix != ".json":
        raise ValueError("Final audit output must be a .json report path")
    rejected_output = output.with_suffix(".rejected.txt")
    if output.exists() or output.with_suffix(".parquet").exists() or rejected_output.exists():
        raise FileExistsError(output)
    saved = ModelCheckpoint.of(checkpoint)
    config = saved.validated_config()
    if config.get("evaluation_protocol") != "strict_inductive" or len(config["dates"]["test"]) != 1:
        raise ValueError("Final population audit requires a frozen scope and one test cutoff")
    date = config["dates"]["test"][0]
    last_ms = cutoff_ms(date)
    if dataset is None:
        dataset = saved.dataset
    if dataset is None or not (dataset / MANIFEST).exists():
        raise ValueError(
            "Final audit needs the prepared dataset of this checkpoint for its cutoff clock "
            "and hub registry; pass --dataset"
        )
    manifest, _ = load_prepared(dataset)
    saved.check_dataset(dataset)
    if executor is None:
        from .installation import verify_frozen_source

        # The checkpoint's retry budgets (max_query_attempts, max_outage_s).
        executor = live_executor(config)
        verify_frozen_source(executor, manifest)
    population: list[dict[str, Any]] = []
    for page in account_pages(
        executor,
        "temporal_scope_population",
        {"scope_id": config["scope_id"], "include_observed": False},
    ):
        for row in page:
            if row["partition"] == SPLIT_PHASE["test"] and row["first_seen_ts_ms"] <= last_ms:
                population.append({"account_id": row["account_id"], "split": "test"})
                if len(population) > 1_000_000:
                    raise ValueError(
                        "Final audit metadata budget exceeded; use a streamed truth provider"
                    )
    if not population:
        raise ValueError("No eligible accounts in final test population")
    answer = truth.read()
    if "date" in answer and not answer.date.eq(date).all():
        raise ValueError("Evaluation truth date differs from the frozen test cutoff")
    selected = final_evaluation_sample(
        pd.DataFrame(population),
        answer,
        negative_limit=negative_limit,
        seed=split_seed(config),
    )
    if len(selected) > 100_000:
        raise ValueError("Final scoring sample exceeds audit budget")
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    predictor = TemporalPredictor(saved, contexts, executor=executor, hubs=registry)
    failed = True
    try:
        size = predictor.batch_size
        # The prepared test keys: the dataset's cutoff clock, scope and phase 3.
        frames, rejected = predictor.score_keys(
            sample_keys(selected.iloc[start : start + size], date, manifest)
            for start in range(0, len(selected), size)
        )
        failed = False
    finally:
        close_source(predictor.contexts, failed=failed)
    scores: dict[str, float] = {}
    for frame in frames:
        scores.update(zip(frame.account_id, frame.score.astype(float), strict=True))
    selected["score"] = selected.account_id.astype(str).map(scores)
    unscored = selected[selected.score.isna()]
    scored = selected[selected.score.notna()].reset_index(drop=True)
    rejected_positives = int(unscored.is_mule.sum())
    limit = float(setting(config, "max_rejected_root_fraction"))
    if exceeds_rejection_limit(len(unscored), rejected_positives, len(selected), limit):
        examples = unscored.account_id.astype(str).head(20).tolist()
        raise ValueError(
            f"TigerGraph rejected {len(unscored)} of {len(selected)} final audit accounts "
            f"({rejected_positives} test positives; max_rejected_root_fraction={limit}); "
            f"statuses {dict(getattr(predictor.contexts, 'rejections', {}) or {})}; "
            f"first {examples}. Weighted metrics over the remaining accounts would describe "
            "a censored population, so no report was written"
        )
    metrics = evaluate_weighted(scored, saved.threshold)
    if len(unscored):
        metrics["evaluation_cohort"] += "_minus_rejected_negatives"
    result = {
        "selection": saved.selected_on,
        "test_date": date,
        "test_population_accounts": len(population),
        "metrics": metrics,
        "rejected_accounts": len(unscored),
        "rejected_positives": rejected_positives,
        "rejected_negatives": len(unscored) - rejected_positives,
        **rejection_summary(predictor.contexts, len(rejected), predictor.totals),
        "scope": config["scope_id"],
        "model_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scored.to_parquet(output.with_suffix(".parquet"), index=False)
    if rejected:
        write_rejected(rejected_output, rejected)
    return result
