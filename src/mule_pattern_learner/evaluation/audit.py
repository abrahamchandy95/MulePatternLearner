"""Optional post-training oracle evaluation, isolated from the trainer.

The final audit writes a run's audit/test.json (the report), audit/test.parquet (the
scored sample) and audit/test_rejected.txt (the accounts TigerGraph rejected, if any).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..artifacts import write_audit_scores, write_json, write_rejected
from ..contract.bounds import AUDIT_POPULATION, AUDIT_SAMPLE
from ..contract.clock import cutoff_ms
from ..inference.saved_model import SavedModel
from ..metrics import weighted_metrics
from ..paths import DATA_DIR, DatasetPaths, RunPaths
from ..runtime.progress import emit
from .sample import audit_sample
from .truth import TruthReader

if TYPE_CHECKING:
    from ..data.contexts import ContextReader
    from ..data.hub_registry import HubRegistry
    from ..data.ports import ScopeReader


def audit_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """The weighted metrics (``metrics.weighted_metrics``) of an audit sample.

    Each account is weighted by 1 / its inclusion probability. Account IDs are not
    read, so they cannot break ties.
    """
    import numpy as np

    if not len(frame) or not frame.is_mule.isin([0, 1]).all():
        raise ValueError("Weighted evaluation needs binary truth and nonempty predictions")
    p = frame.inclusion_probability.to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Invalid inclusion probabilities")
    y, score = frame.is_mule.to_numpy(int), frame.score.to_numpy(float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    return {
        **weighted_metrics(y, score, 1 / p, threshold),
        "evaluation_sample": "all_test_positives_plus_uniform_negatives_inverse_probability_weighted",
    }


# The split the final audit samples: its frozen population at the test cutoff.
AUDITED_SPLIT = "test"


def audit_inputs(
    run: RunPaths, dataset: DatasetPaths | None = None, data: Path = DATA_DIR
) -> tuple[SavedModel, DatasetPaths, dict[str, Any]]:
    """The run's frozen model, its prepared dataset and the dataset's manifest, all checked.

    The final audit reads nothing from the graph before these checks pass: an audit
    the run already has, a model with more than one test cutoff and a missing or
    changed dataset are refused. ``dataset`` defaults to the model's own in data.
    """
    from ..data.manifest import load_prepared

    split = AUDITED_SPLIT
    for path in (run.audit_metrics(split), run.audit_scores(split), run.audit_rejected(split)):
        if path.exists():
            raise FileExistsError(path)
    saved = SavedModel.load(run.model)
    if len(saved.config.dataset.dates.test) != 1:
        raise ValueError("Final population audit requires one test cutoff")
    if dataset is None:
        dataset = saved.dataset(data)
    if dataset is None or not dataset.manifest.exists():
        raise ValueError(
            f"The audit needs the prepared dataset of this model in {data} for its cutoff "
            "clock and hub registry; `mule train` prepares it"
        )
    manifest, _ = load_prepared(dataset)
    saved.check_dataset(dataset)
    return saved, dataset, manifest


def audit(
    run: RunPaths,
    truth: TruthReader,
    *,
    scope: ScopeReader,
    contexts: ContextReader,
    negative_limit: int = 2000,
    dataset: DatasetPaths | None = None,
    hubs: HubRegistry | None = None,
) -> dict[str, Any]:
    """Score a fresh final-only sample from the entire frozen test partition.

    The model is the run's model.pt, and the audit goes into the run's audit/ files.
    This POC audit bounds host metadata to one million test accounts. It never
    changes a model and refuses to overwrite an existing audit. The
    prepared dataset (``dataset`` or the one recorded in the model) supplies
    the test cutoff clock and the hub registry, so scoring matches training. The test
    population comes from ``scope`` and the contexts from ``contexts``, which the audit
    closes; the pipeline opens both on a frozen source it has verified
    (pipeline.evaluate.evaluate_run).

    Accounts TigerGraph rejects are not scored. A rejected test positive, or a
    rejected fraction of the sample above the model's
    ``runtime.max_rejected_root_fraction`` (default 0), fails the audit before anything is
    written: the weighted metrics would silently describe a censored population.
    Rejected negatives within the limit are listed in audit/test_rejected.txt and
    the metrics' ``evaluation_sample`` says that they were dropped.
    """
    from ..contract.graph_schema import SPLIT_PHASE
    from ..data.accounts import scope_accounts
    from ..data.contexts import close_source
    from ..data.hub_registry import load_hub_registry
    from ..data.splits import sample_keys
    from ..inference.predictor import Predictor
    from ..inference.rejections import exceeds_rejection_limit, rejection_summary

    saved, dataset, manifest = audit_inputs(run, dataset)
    split = AUDITED_SPLIT
    config = saved.config
    (date,) = config.dataset.dates.test
    last_ms = cutoff_ms(date)
    population: list[dict[str, Any]] = []
    for row in scope_accounts(scope, config.scope.id, include_observed=False):
        if row["partition"] == SPLIT_PHASE[split] and row["first_seen_ts_ms"] <= last_ms:
            population.append({"account_id": row["account_id"], "split": split})
            if len(population) > AUDIT_POPULATION:
                raise ValueError(
                    "Final audit metadata budget exceeded; use a streamed truth provider"
                )
    if not population:
        raise ValueError("No eligible accounts in final test population")
    answer = truth.read()
    if "date" in answer and not answer.date.eq(date).all():
        raise ValueError("Evaluation truth date differs from the frozen test cutoff")
    selected = audit_sample(
        pd.DataFrame(population),
        answer,
        negative_limit=negative_limit,
        seed=config.dataset.split_seed,
    )
    if len(selected) > AUDIT_SAMPLE:
        raise ValueError("Final scoring sample exceeds audit budget")
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    failed = True
    try:
        predictor = Predictor(saved, contexts, hubs=registry)
        size = predictor.batch_size
        # The prepared test keys: the dataset's cutoff clock, scope and phase 3.
        with predictor.runtime():
            frames, rejected = predictor.score_keys(
                sample_keys(selected.iloc[start : start + size], date, manifest)
                for start in range(0, len(selected), size)
            )
        failed = False
    finally:
        close_source(contexts, failed=failed)
    scores: dict[str, float] = {}
    for frame in frames:
        scores.update(zip(frame.account_id, frame.score.astype(float), strict=True))
    selected["score"] = selected.account_id.astype(str).map(scores)
    unscored = selected[selected.score.isna()]
    scored = selected[selected.score.notna()].reset_index(drop=True)
    rejected_positives = int(unscored.is_mule.sum())
    limit = config.runtime.max_rejected_root_fraction
    if exceeds_rejection_limit(len(unscored), rejected_positives, len(selected), limit):
        examples = unscored.account_id.astype(str).head(20).tolist()
        raise ValueError(
            f"TigerGraph rejected {len(unscored)} of {len(selected)} final audit accounts "
            f"({rejected_positives} test positives; max_rejected_root_fraction={limit}); "
            f"statuses {dict(predictor.contexts.rejections)}; "
            f"first {examples}. Weighted metrics over the remaining accounts would describe "
            "a censored population, so no report was written"
        )
    metrics = audit_metrics(scored, saved.threshold)
    if len(unscored):
        metrics["evaluation_sample"] += "_minus_rejected_negatives"
    result = {
        "selection": saved.selected_on,
        "test_date": date,
        "test_population_accounts": len(population),
        "metrics": metrics,
        "rejected_accounts": len(unscored),
        "rejected_positives": rejected_positives,
        "rejected_negatives": len(unscored) - rejected_positives,
        **rejection_summary(predictor.contexts, len(rejected), predictor.totals),
        "scope": config.scope.id,
        "model_changed": False,
    }
    run.audit_metrics(split).parent.mkdir(parents=True, exist_ok=True)
    write_json(run.audit_metrics(split), result)
    write_audit_scores(run.audit_scores(split), scored)
    if rejected:
        write_rejected(run.audit_rejected(split), rejected)
    emit(
        {
            "event": "audit",
            "split": split,
            "date": date,
            "accounts": len(scored),
            "rejected_accounts": len(unscored),
            "output": str(run.audit_metrics(split)),
        }
    )
    return result
