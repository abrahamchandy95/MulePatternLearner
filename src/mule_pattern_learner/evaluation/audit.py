"""The ground-truth audit of a run's model on one split, isolated from the trainer.

An audit of a split writes the run's audit/<split>.json (the report),
audit/<split>.parquet (the scored sample) and audit/<split>_rejected.txt (the accounts
TigerGraph rejected, if any). `mule evaluate` audits validation and test: decisions use
the validation audit, and the test audit is for reporting. audit runs the steps:
audit_population reads the split's frozen population from the scope, audit_sample
(evaluation.sample) draws the sample, score_sample scores it with the run's model, and
write_audit writes the files. audit_inputs loads and checks the model and its dataset
once for every split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import AUDIT_COLUMNS, write_audit_scores, write_json, write_rejected
from ..contract.bounds import AUDIT_POPULATION, AUDIT_SAMPLE
from ..contract.clock import timestamp
from ..contract.graph_schema import HELD_OUT_SPLITS, SPLIT_PHASE
from ..data.accounts import scope_accounts
from ..data.contexts import ContextReader
from ..data.hub_registry import HubRegistry, load_hub_registry
from ..data.manifest import load_prepared
from ..data.ports import ScopeReader
from ..data.splits import eligible_mask, sample_keys
from ..inference.predictor import Predictor
from ..inference.rejections import (
    SourceRejections,
    exceeds_rejection_limit,
    rejection_summary,
)
from ..inference.saved_model import SavedModel
from ..metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    INTERVAL,
    REVIEW_BUDGETS,
    bootstrap_intervals,
    weighted_metrics,
)
from ..paths import DATA_DIR, DatasetPaths, RunPaths
from ..runtime.console import plural
from ..runtime.progress import emit
from .sample import AUDIT_NEGATIVES, audit_sample
from .truth import checked_truth

# The splits `mule evaluate` audits, and what each audit is for: decisions use the
# validation audit, and the test audit is for reporting.
AUDIT_SPLITS = dict(zip(HELD_OUT_SPLITS, ("decisions", "reporting"), strict=True))


@dataclass(frozen=True)
class AuditedRun:
    """A run's frozen model and the prepared dataset it was trained on, loaded and checked.

    ``hubs`` is the dataset's hub registry, which scoring uses as training did.
    """

    paths: RunPaths
    model: SavedModel
    dataset: DatasetPaths
    manifest: dict[str, Any]
    hubs: HubRegistry

    def audited(self, split: str) -> bool:
        """Whether the run has the split's audit: its report is written last."""
        return self.paths.audit_report(split).exists()


def audit_inputs(
    run: RunPaths,
    *,
    data: Path = DATA_DIR,
    dataset: DatasetPaths | None = None,
    hubs: HubRegistry | None = None,
) -> AuditedRun:
    """The run's frozen model, its prepared dataset and the dataset's hub registry, checked.

    An audit reads nothing from the graph before these checks pass: a model with other
    than one cutoff in an audited split and a missing or changed dataset are refused.
    ``dataset`` defaults to the model's own in data, and ``hubs`` to its registry.
    """
    saved = SavedModel.load(run.model)
    dates = saved.config.dataset.dates
    for split in AUDIT_SPLITS:
        if len(dates[split]) != 1:
            raise ValueError(f"An audit needs one {split} cutoff, not {len(dates[split])}")
    if dataset is None:
        dataset = saved.dataset(data)
    if dataset is None or not dataset.manifest.exists():
        raise ValueError(
            f"The audit needs the prepared dataset of this model in {data} for its cutoff "
            "clock and hub registry; `mule train` prepares it"
        )
    manifest, _ = load_prepared(dataset)
    saved.check_dataset(dataset)
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    return AuditedRun(run, saved, dataset, manifest, registry)


def audit_population(scope: ScopeReader, scope_id: str, split: str, date: str) -> pd.DataFrame:
    """The frozen population of a split at its cutoff: account_id, split and revealed.

    These are the accounts of the split's partition of the scope that existed before
    the cutoff (data.splits.eligible_mask), in account order. The population is read with
    its observed labels: revealed says that the graph revealed the account's label
    before the cutoff, as the proxy metrics see it. At most
    contract.bounds.AUDIT_POPULATION accounts of the partition are held.
    """
    phase = SPLIT_PHASE[split]
    rows: list[dict[str, Any]] = []
    for row in scope_accounts(scope, scope_id, include_observed=True):
        if row["partition"] == phase:
            rows.append(row)
            if len(rows) > AUDIT_POPULATION:
                raise ValueError(
                    f"The {split} population exceeds the audit's {AUDIT_POPULATION} accounts"
                )
    columns = ["account_id", "first_seen_ts_ms", "observed_positive", "known_from_ms"]
    frame = pd.DataFrame(rows, columns=columns).assign(split=split)
    frame = frame[eligible_mask(frame, split, date)]
    revealed = frame.observed_positive.eq(True) & (frame.known_from_ms < timestamp(date))
    return pd.DataFrame(
        {
            "account_id": frame.account_id.astype(str).to_numpy(),
            "split": split,
            "revealed": revealed.to_numpy(bool),
        }
    )


def audit_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """The weighted metrics (``metrics.weighted_metrics``) of an audit sample.

    Each account is weighted by 1 / its inclusion probability. Account IDs are not
    read, so they cannot break ties.
    """
    if not len(frame) or not frame.is_mule.isin([0, 1]).all():
        raise ValueError("Weighted evaluation needs binary truth and nonempty predictions")
    p = frame.inclusion_probability.to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Invalid inclusion probabilities")
    y, score = frame.is_mule.to_numpy(int), frame.score.to_numpy(float)
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid prediction probabilities")
    return weighted_metrics(y, score, 1 / p, threshold)


def audit_intervals(frame: pd.DataFrame) -> dict[str, list[float] | None]:
    """The ring-clustered bootstrap intervals of an audit sample's ranking metrics.

    Mules are resampled by ring (ring_id; a mule without a ring alone) and non-mules
    within their class, each keeping its inclusion weight (metrics.bootstrap_intervals).
    """
    return bootstrap_intervals(
        frame.is_mule.to_numpy(int),
        frame.score.to_numpy(float),
        1 / frame.inclusion_probability.to_numpy(float),
        frame.ring_id.to_numpy(int),
    )


def audit_constants(seed: int) -> dict[str, Any]:
    """What an audit's numbers depend on besides the model, recorded in its report."""
    return {
        "audit_negatives": AUDIT_NEGATIVES,
        "sample_seed": seed,
        "review_budgets": list(REVIEW_BUDGETS),
        "interval": INTERVAL,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": "positives_by_ring_negatives_within_class",
    }


def score_sample(
    run: AuditedRun, sample: pd.DataFrame, date: str, contexts: ContextReader
) -> tuple[dict[str, float], list[str], dict[str, Any]]:
    """Score an audit sample of one split with the run's model at its split's cutoff.

    The keys are the prepared dataset's: its cutoff clock, scope and the split's phase,
    and the hub registry is the dataset's, so scoring matches training. Returns the
    scores of the accepted accounts by account id, the ids TigerGraph rejected and the
    rejection summary (inference.rejections.rejection_summary) of this sample alone,
    whatever ``contexts`` served before. A terminal shows how far it has come. The
    caller closes ``contexts``.
    """
    predictor = Predictor(run.model, contexts, hubs=run.hubs)
    before = SourceRejections.of(contexts)
    size = predictor.batch_size
    splits = ", ".join(sorted({str(split) for split in sample.split}))
    with predictor.runtime():
        frames, rejected = predictor.score_keys(
            (
                sample_keys(sample.iloc[start : start + size], date, run.manifest)
                for start in range(0, len(sample), size)
            ),
            shown=f"the {splits} audit sample",
            total=len(sample),
        )
    scores: dict[str, float] = {}
    for frame in frames:
        scores.update(zip(frame.account_id, frame.score.astype(float), strict=True))
    summary = rejection_summary(contexts, len(rejected), predictor.totals, since=before)
    return scores, rejected, summary


def write_audit(
    run: RunPaths, split: str, report: dict[str, Any], scored: pd.DataFrame, rejected: list[str]
) -> None:
    """Write a split's audit: the scored sample, the rejected accounts, then the report.

    The report comes last, so a split is audited exactly when its report exists; the
    files an interrupted audit left are replaced.
    """
    run.audit_report(split).parent.mkdir(parents=True, exist_ok=True)
    write_audit_scores(run.audit_scores(split), scored[list(AUDIT_COLUMNS)])
    if rejected:
        write_rejected(run.audit_rejected(split), rejected)
    else:
        run.audit_rejected(split).unlink(missing_ok=True)
    write_json(run.audit_report(split), report)


def audit(
    run: AuditedRun,
    split: str,
    *,
    truth: pd.DataFrame,
    scope: ScopeReader,
    contexts: ContextReader,
) -> dict[str, Any]:
    """Audit the run's model on a fresh sample of one split's entire frozen population.

    ``truth`` is the ground truth (contract.graph_schema.TRUTH_COLUMNS), the split's
    population comes from ``scope`` and the contexts from ``contexts``; the pipeline
    opens both on a frozen source it has verified and closes the contexts
    (pipeline.evaluate.evaluate_run). The report's rejection counts are this audit's
    alone, whatever ``contexts`` served before, so splits can share a source. The audit
    never changes a model and refuses to overwrite an audit the run already has.

    Accounts TigerGraph rejects are not scored. A rejected positive, or a rejected
    fraction of the sample above the model's ``runtime.max_rejected_root_fraction``
    (default 0), fails the audit before anything is written: the weighted metrics
    would silently describe a censored population. Rejected negatives within the limit
    are listed in audit/<split>_rejected.txt, and the metrics' ``evaluation_sample``
    says that they were dropped.
    """
    if split not in AUDIT_SPLITS:
        raise ValueError(f"Audits cover {list(AUDIT_SPLITS)}, not {split!r}")
    if run.audited(split):
        raise FileExistsError(run.paths.audit_report(split))
    config = run.model.config
    (date,) = config.dataset.dates[split]
    population = audit_population(scope, config.scope.id, split, date)
    if not len(population):
        raise ValueError(f"No eligible accounts in the {split} population")
    selected = audit_sample(population, checked_truth(truth), seed=config.dataset.split_seed)
    if len(selected) > AUDIT_SAMPLE:
        raise ValueError(f"The {split} audit sample exceeds the audit's {AUDIT_SAMPLE} accounts")
    scores, rejected, summary = score_sample(run, selected, date, contexts)
    selected["score"] = selected.account_id.map(scores)
    unscored = selected[selected.score.isna()]
    scored = selected[selected.score.notna()].reset_index(drop=True)
    rejected_positives = int(unscored.is_mule.sum())
    limit = config.runtime.max_rejected_root_fraction
    if exceeds_rejection_limit(len(unscored), rejected_positives, len(selected), limit):
        examples = unscored.account_id.astype(str).head(20).tolist()
        raise ValueError(
            f"TigerGraph rejected {len(unscored)} of "
            f"{plural(len(selected), f'{split} audit account')} "
            f"({plural(rejected_positives, f'{split} positive')}; "
            f"max_rejected_root_fraction={limit}); "
            f"statuses {summary['rejection_events_by_status']}; "
            f"first {examples}. Weighted metrics over the remaining accounts would describe "
            "a censored population, so no report was written"
        )
    metrics = {
        **audit_metrics(scored, run.model.threshold),
        "evaluation_sample": f"all_{split}_positives_plus_uniform_negatives_"
        "inverse_probability_weighted",
    }
    if len(unscored):
        metrics["evaluation_sample"] += "_minus_rejected_negatives"
    mules = scored[scored.is_mule == 1]
    report = {
        "split": split,
        "purpose": AUDIT_SPLITS[split],
        "date": date,
        "selection": run.model.selected_on,
        "population_accounts": len(population),
        "metrics": metrics,
        # The ring-clustered 90% bootstrap interval of each ranking metric.
        "intervals": audit_intervals(scored),
        "constants": audit_constants(config.dataset.split_seed),
        "revealed_positives": int(mules.revealed.sum()),
        "hidden_positives": int((~mules.revealed).sum()),
        "rejected_accounts": len(unscored),
        "rejected_positives": rejected_positives,
        "rejected_negatives": len(unscored) - rejected_positives,
        **summary,
        "scope": config.scope.id,
        "model_changed": False,
    }
    write_audit(run.paths, split, report, scored, rejected)
    emit(
        {
            "event": "audit",
            "split": split,
            "date": date,
            "accounts": len(scored),
            "rejected_accounts": len(unscored),
            "output": str(run.paths.audit_report(split)),
        }
    )
    return report
