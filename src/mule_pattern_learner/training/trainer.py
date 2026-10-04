"""nnPU with observed-label selection over an injected context source.

A run writes its files into its own directory (paths.RunPaths). It is resumable:
resume.pt holds the model, optimizer, weight average, RNG and schedule position plus the
selection state, written every epoch and every ``runtime.checkpoint_every_steps`` steps.
``train(..., resume=True)`` continues from it and, on the same device with the same
threads and determinism, reproduces the uninterrupted run exactly: every epoch
schedule is drawn up front from the saved generator state, and every step reseeds
torch from a stable hash of (seed, epoch, step), so dropout and sampler draws never
depend on history. A resumed segment may change those three host settings; events.jsonl
then records the change, since floating-point results may differ from there on.
The sampler backend is resolved once per run and stored in resume.pt; a resume that
would sample with another backend is refused unless the sampler section names that
backend explicitly. Reported database calls, rejections and batch totals are saved
there too, so they cover every segment of a resumed run.

Roots that TigerGraph rejects are dropped from a batch, but only within
``runtime.max_rejected_root_fraction`` (default 0: any rejection fails). A rejected
observed positive always fails the epoch or evaluation, and validation must keep
both observed classes after its rejections.

The run drives the other training modules: the nnPU step (objective), the weight
average (averaging), the progress records (history) and what a finished run writes
(summary). Validation and test go through the scoring loop of inference.predictor.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..artifacts import (
    EPOCH_COLUMNS,
    HISTORY_COLUMNS,
    append_history,
    keep_history,
    read_run_provenance,
    write_epochs,
    write_json,
    write_predictions,
    write_run_config,
)
from ..batching.assemble import RootBatch, batch_device, build_root_batch, to_device
from ..batching.limits import BatchLimits
from ..config import RunConfig, SplitDates, TrainingConfig
from ..contract.feature_groups import FeaturePlan, extraction_plan
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..data.contexts import ContextOpener, ContextReader, check_coverage, close_source
from ..data.hub_registry import HubRegistry, load_hub_registry, warn_hub_stubs
from ..data.manifest import dataset_id, dataset_mismatches, load_prepared, manifest_digest
from ..data.observed_labels import label_summary, load_observed_labels, visible_labels
from ..data.splits import eligible_mask, marginal_mask, sample_keys
from ..inference.predictor import accepted_scores, score_batches
from ..inference.rejections import (
    TrainingRejections,
    check_split_rejections,
    rejection_counts,
)
from ..inference.saved_model import SavedModel
from ..metrics import proxy_metrics, select_threshold
from ..model.build import build_model
from ..model.loss import NonNegativePULoss
from ..paths import DatasetPaths, RunPaths
from ..runtime.console import show_scoring
from ..runtime.device import choose_device, torch_runtime
from ..runtime.progress import emit, recording
from ..runtime.workers import BatchPrefetcher
from ..sampling.backend import resolve_backend
from .averaging import WeightAverage, evaluated_weights
from .checkpoint import (
    ResumeState,
    check_run_dataset,
    load_resume_state,
    restore_cuda_rng,
    run_started,
)
from .history import LogInterval, RunTotals, epoch_record, plain
from .objective import StepLoss, nnpu_objective, nnpu_step, pu_risk
from .schedule import (
    EvaluationSample,
    PUSample,
    TrainingStep,
    epoch_schedule,
    evaluation_indices,
    schedule_steps,
)
from .selection import selection_value, stops_early
from .summary import host_settings, prediction_frame, provenance, run_summary

Batch = dict[str, torch.Tensor]
BatchRequest = tuple[list[ContextKey], str, int]
# The epochs.csv column an epoch row of the resume state never holds: record_epochs adds it.
UNSAVED = ("selected",)


def check_limits(config: RunConfig, plan: FeaturePlan) -> None:
    """Raise when a training batch of these settings would exceed the memory limits."""
    BatchLimits().validate_model(
        config.training.batch_size,
        config.sampler.fanouts,
        config.model.hidden,
        plan,
        config.sampler,
    )


def check_source(
    contexts: ContextReader, prepared: FeaturePlan, model: FeaturePlan, sampler: SamplerPlan
) -> None:
    """The source extracts the prepared plan, covers the model inputs and uses its sampler."""
    if contexts.plan.fingerprint() != prepared.fingerprint():
        raise ValueError("Context source extraction plan differs from the prepared extraction plan")
    if contexts.sampler != sampler:
        raise ValueError("Context source sampler differs from the training sampler")
    check_coverage(contexts, model, sampler)


def build_optimizer(model: nn.Module, training: TrainingConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )


def train(
    config: RunConfig,
    dataset: DatasetPaths,
    run: RunPaths,
    *,
    contexts: ContextReader | None = None,
    open_contexts: ContextOpener | None = None,
    hubs: HubRegistry | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train, select by training.selection on observed validation labels, save, score test.

    Every file of the run goes into the directory ``run`` names. Without ``contexts``,
    ``open_contexts`` opens the dataset's context source once the settings and the
    prepared dataset passed their checks (the pipeline passes
    pipeline.connect.open_context_source). ``contexts`` and ``hubs`` replace the
    dataset's source and hub registry (tests, offline replays). With ``resume`` a
    started run continues from its resume.pt, but only on the dataset it trained on;
    without it a started run is an error.
    """
    plan = config.feature_plan()
    # Fails fast on per-hop candidate pools too, before any database work.
    check_limits(config, plan)
    started = run_started(run)
    if started and not resume:
        raise FileExistsError(f"Run already exists: {run.root}; resume it or pick a new one")
    state = load_resume_state(config, run) if started else None
    manifest, accounts = load_prepared(dataset)
    differences = dataset_mismatches(config, manifest)
    if differences:
        raise ValueError(f"Training settings differ from the prepared dataset: {differences}")
    identity = dataset_id(manifest["source"]["source_id"], config)
    if started:
        check_run_dataset(run, state, identity, manifest_digest(dataset))
    # The source requests this model's groups; its hop-2 flags follow the architecture.
    source_plan = extraction_plan(plan)
    mask = load_observed_labels(accounts, dataset)
    training, evaluation = _samples(config, accounts, mask)
    prior, positive_weight = nnpu_objective(config.loss)
    runtime = config.runtime
    device = choose_device(runtime.device)
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    if contexts is None:
        if open_contexts is None:
            raise ValueError(
                "Training needs contexts, or open_contexts to open the dataset's source"
            )
        contexts = open_contexts(dataset, manifest, config)
    failed = True
    try:
        check_source(contexts, source_plan, plan, config.sampler)
        with torch_runtime(device, deterministic=runtime.deterministic, threads=runtime.threads):
            training_run = _TrainingRun(
                config=config,
                dataset=dataset,
                manifest=manifest,
                accounts=accounts,
                mask=mask,
                plan=plan,
                contexts=contexts,
                hubs=registry,
                device=device,
                prior=prior,
                positive_weight=positive_weight,
                training=training,
                evaluation=evaluation,
                run=run,
                dataset_id=identity,
            )
            result = training_run.execute(state)
        failed = False
        return result
    finally:
        # After an error or Ctrl-C, do not wait for in-flight REST calls.
        close_source(contexts, failed=failed)


def training_samples(
    dates: SplitDates, accounts: pd.DataFrame, mask: pd.DataFrame
) -> list[PUSample]:
    """The PUSample of every train cutoff with visible positives, as train() schedules them."""
    marginal = marginal_mask(accounts)
    samples: list[PUSample] = []
    for date in dates.train:
        eligible = np.flatnonzero(eligible_mask(accounts, "train", date))
        observed = visible_labels(mask, date)
        if observed[eligible].any():
            samples.append(
                PUSample(date, eligible[marginal[eligible]], observed, eligible[observed[eligible]])
            )
    return samples


def _samples(
    config: RunConfig, accounts: pd.DataFrame, mask: pd.DataFrame
) -> tuple[list[PUSample], dict[str, list[EvaluationSample]]]:
    training = training_samples(config.dataset.dates, accounts, mask)
    marginal = marginal_mask(accounts)
    evaluation: dict[str, list[EvaluationSample]] = {"validation": [], "test": []}
    for name, samples in evaluation.items():
        for date in config.dataset.dates[name]:
            eligible = np.flatnonzero(eligible_mask(accounts, name, date))
            observed = visible_labels(mask, date)
            chosen = evaluation_indices(
                eligible,
                observed,
                limit=config.training.proxy_unlabeled_limit,
                seed=config.dataset.split_seed,
                marginal=marginal,
            )
            samples.append(EvaluationSample(date, chosen, observed[chosen]))
    validation = [s.labels for s in evaluation["validation"]]
    if not training or len(np.unique(np.concatenate(validation or [np.zeros(0)]))) != 2:
        raise ValueError(
            "Training needs revealed positives and validation needs both observed classes; "
            "masks/splits were not changed"
        )
    return training, evaluation


class _TrainingRun:
    def __init__(
        self,
        *,
        config: RunConfig,
        dataset: DatasetPaths,
        manifest: dict[str, Any],
        accounts: pd.DataFrame,
        mask: pd.DataFrame,
        plan: FeaturePlan,
        contexts: ContextReader,
        hubs: HubRegistry,
        device: torch.device,
        prior: float,
        positive_weight: float,
        training: list[PUSample],
        evaluation: dict[str, list[EvaluationSample]],
        run: RunPaths,
        dataset_id: str,
    ) -> None:
        self.config, self.dataset = config, dataset
        self.dataset_id, self.dataset_sha256 = dataset_id, manifest_digest(dataset)
        self.training_config, self.runtime = config.training, config.runtime
        self.manifest, self.accounts, self.mask = manifest, accounts, mask
        self.plan, self.sampler, self.contexts, self.hubs = plan, config.sampler, contexts, hubs
        self.device, self.prior, self.positive_weight = device, prior, positive_weight
        self.loss = NonNegativePULoss(prior=prior, positive_weight=positive_weight)
        self.training, self.evaluation = training, evaluation
        self.run = run
        self.batch_device = batch_device(device)
        self.prefetch = config.runtime.prefetch_batches
        # Resolved once here, on the main thread, before any prefetch worker starts
        # (the cuGraph probe runs at most once), then passed to every batch.
        self.backend = resolve_backend(self.sampler, self.batch_device)
        self.limit = config.runtime.max_rejected_root_fraction
        self.observed = {sample.date: sample.observed for sample in training}
        self.progress = RunTotals(time.perf_counter(), contexts, self.backend)
        # Seeded right before the model is built: initial weights depend only on the seed.
        torch.manual_seed(config.training.seed)
        self.model = build_model(config.model, plan, self.sampler.fanouts[0]).to(device)
        self.optimizer = build_optimizer(self.model, config.training)
        decay = config.training.weight_average_decay
        self.average = WeightAverage(self.model, decay) if decay > 0 else None
        self.rng = np.random.default_rng(config.training.seed)
        self.epoch, self.step, self.stopped = 0, 0, False
        self.epoch_started = time.perf_counter()
        # The selection rule, and the kept epoch's value under it (selection_value).
        self.selection = config.training.selection
        self.best_value, self.best_epoch = -math.inf, 0
        self.best_state = self._state_copy()
        self.best_scores: np.ndarray | None = None
        self.best_accepted: np.ndarray | None = None
        # One epochs.csv row per finished epoch, without its selected flag.
        self.epoch_rows: list[dict[str, Any]] = []
        self.loss_sum = torch.zeros((), device=device)
        self.loss_steps = 0
        self.rejected_rows: dict[str, int] = {}
        self.rejections = TrainingRejections(self.limit)
        self.epoch_rng_state: Mapping[str, Any] = self.rng.bit_generator.state

    def _state_copy(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # Batches -----------------------------------------------------------------

    def build_eval(self, keys: list[ContextKey]) -> RootBatch:
        return self.build((keys, "eval", 0))

    def build(self, request: BatchRequest) -> RootBatch:
        keys, mode, seed = request
        return build_root_batch(
            self.contexts,
            keys,
            device=self.batch_device,
            plan=self.plan,
            sampler=self.sampler,
            hubs=self.hubs,
            mode=mode,
            step_seed=seed,
            sampler_backend=self.backend,
        )

    def keys(self, indices: np.ndarray, date: str) -> list[ContextKey]:
        return sample_keys(self.accounts.iloc[indices], date, self.manifest)

    def batches(self, requests: Iterator[BatchRequest]) -> BatchPrefetcher[BatchRequest, Any]:
        # Keys are built lazily on the consumer thread; workers only fetch and assemble.
        return BatchPrefetcher(self.build, requests, depth=self.prefetch)

    # Checkpoints -------------------------------------------------------------

    def save_last(self) -> None:
        """Write resume.pt: everything a resumed run needs to continue exactly."""
        best_scores, best_accepted = self.best_scores, self.best_accepted
        ResumeState(
            model=self._state_copy(),
            optimizer=self.optimizer.state_dict(),
            weight_average=None if self.average is None else self.average.saved(),
            numpy_rng=self.epoch_rng_state,
            torch_rng=torch.get_rng_state(),
            # The training device only: a resume may see a different number of GPUs.
            cuda_rng=(
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
            epoch=self.epoch,
            step=self.step,
            stopped=self.stopped,
            loss_sum=self.loss_sum.detach().cpu(),
            loss_steps=self.loss_steps,
            best_state=self.best_state,
            best_ap=self.best_value,
            best_epoch=self.best_epoch,
            best_scores=None if best_scores is None else torch.from_numpy(best_scores),
            best_accepted=None if best_accepted is None else torch.from_numpy(best_accepted),
            epoch_rows=self.epoch_rows,
            sampler_backend=self.backend,
            dataset_id=self.dataset_id,
            dataset_manifest_sha256=self.dataset_sha256,
            progress=self.progress.saved(),
            rejections=self.rejections.saved(),
        ).save(self.run.resume, self.config)

    def restore(self, state: ResumeState) -> None:
        saved = state.sampler_backend
        if saved != self.backend:
            if self.sampler.backend != self.backend:
                raise ValueError(
                    f"The run was sampled with the {saved} backend but this host resolves "
                    f"{self.backend}; resume on a matching host, or set sampler.backend = "
                    f'"{self.backend}" to accept a different sampling stream from here on'
                )
            emit(
                {
                    "event": "sampler_backend",
                    "saved": saved,
                    "resumed": self.backend,
                    "note": "the remaining steps sample a different stream",
                }
            )
        self.model.load_state_dict(state.model)
        self.optimizer.load_state_dict(state.optimizer)
        if self.average is not None:
            if state.weight_average is None:
                raise ValueError("The resume state holds no weight average")
            self.average.load(state.weight_average)
        self.epoch_rng_state = state.numpy_rng
        self.rng.bit_generator.state = state.numpy_rng
        torch.set_rng_state(state.torch_rng)
        restore_cuda_rng(state.cuda_rng, self.device)
        self.epoch, self.step, self.stopped = state.epoch, state.step, state.stopped
        self.loss_sum = state.loss_sum.to(self.device)
        self.loss_steps = state.loss_steps
        self.best_state, self.best_value = state.best_state, state.best_ap
        # A state saved before validation_pu_risk joined epochs.csv lacks it in its rows.
        unset = dict.fromkeys(name for name in EPOCH_COLUMNS if name not in UNSAVED)
        self.best_epoch = state.best_epoch
        self.epoch_rows = [{**unset, **row} for row in state.epoch_rows]
        scores, accepted = state.best_scores, state.best_accepted
        self.best_scores = None if scores is None else scores.numpy()
        self.best_accepted = None if accepted is None else accepted.numpy()
        self.progress.restore(state.progress)
        self.rejections.restore(state.rejections)

    # Phases ------------------------------------------------------------------

    def execute(self, state: ResumeState | None) -> dict[str, Any]:
        self.run.root.mkdir(parents=True, exist_ok=True)
        with recording(self.run.events):
            warn_hub_stubs(self.hubs, self.plan)
            if state is not None:
                self.restore(state)
            host = host_settings(self.device, self.runtime)
            if state is None or not self.run.config.exists():
                # Without a resume state the run trains from its first step on this host.
                run_provenance = provenance(host, self.backend, self.dataset_id)
                write_run_config(self.run.config, self.config, run_provenance)
            else:
                self.note_host_changes(host)
            # The intervals after the resume position are logged again.
            keep_history(self.run.history, self.epoch, self.step)
            if self.epoch_rows:
                self.record_epochs()
            self.emit_event(
                {
                    "event": "resume" if state is not None else "start",
                    **host,
                    "known_mules": label_summary(self.mask),
                    "loss": "nnPU",
                    "run": str(self.run.root),
                    "epoch": self.epoch,
                    "step": self.step,
                    "stopped": self.stopped,
                    "epochs": self.training_config.epochs,
                    "steps_per_epoch": self.training_config.steps_per_epoch,
                    # The steps each epoch's schedule takes, which the plan's line gives.
                    "steps": schedule_steps(
                        self.training,
                        self.training_config.batch_size,
                        self.training_config.steps_per_epoch,
                    ),
                    "patience": self.training_config.patience,
                    "selection": self.selection,
                    "prefetch_batches": self.prefetch,
                    "max_rejected_root_fraction": self.limit,
                }
            )
            while self.epoch < self.training_config.epochs and not self.stopped:
                self.run_epoch()
            return self.finish()

    def note_host_changes(self, host: dict[str, Any]) -> None:
        """Record in events.jsonl the host settings a resumed segment changes.

        config.json's provenance holds those the run started with. The fingerprint
        leaves them out, so a resume may change them, but then the remaining steps may
        give other floating-point results than an uninterrupted run.
        """
        recorded = read_run_provenance(self.run.config)
        changed = sorted(k for k, v in host.items() if k in recorded and recorded[k] != v)
        if changed:
            emit(
                {
                    "event": "host_settings",
                    "saved": {k: recorded[k] for k in changed},
                    "resumed": {k: host[k] for k in changed},
                    "note": "the remaining steps may not reproduce an uninterrupted run",
                }
            )

    def emit_event(self, record: dict[str, Any]) -> dict[str, Any]:
        """Emit record with the run's totals, into the run's events.jsonl; return it."""
        record = self.progress.record(record)
        emit(record)
        return record

    def run_epoch(self) -> None:
        """Train the current epoch's remaining steps, then select on validation."""
        epoch = self.epoch
        self.epoch_started = time.perf_counter()
        self.epoch_rng_state = self.rng.bit_generator.state
        schedule = epoch_schedule(
            self.training,
            self.rng,
            self.training_config.batch_size,
            epoch=epoch,
            seed=self.training_config.seed,
            max_steps=self.training_config.steps_per_epoch,
        )
        if self.step > len(schedule):
            raise ValueError("The resume state's step lies beyond the epoch schedule")
        if self.step == 0:
            self.loss_sum = torch.zeros((), device=self.device)
            self.loss_steps = 0
            self.rejections.start_epoch()
        self._train_steps(epoch, schedule)
        if not self.loss_steps:
            raise ValueError(f"Every training batch of epoch {epoch + 1} was rejected")
        self._select_epoch(epoch)

    def _train_steps(self, epoch: int, schedule: list[TrainingStep]) -> None:
        """Run the schedule from self.step on, with interval logging and step checkpoints."""
        requested = sum(len(s.indices) for s in schedule)
        remaining = schedule[self.step :]
        self.model.train()
        requests = ((self.keys(s.indices, s.date), "train", s.seed) for s in remaining)
        interval = LogInterval(self.device)
        every = self.runtime.checkpoint_every_steps
        mark = interval.started
        with self.batches(requests) as batches:
            for step, prepared in zip(remaining, batches, strict=True):
                interval.waited += time.perf_counter() - mark
                stats = prepared.stats
                used = stats.get("sampler_backend")
                if used is not None and used != self.backend:
                    raise RuntimeError(
                        f"A training batch used the {used} sampler backend; the run "
                        f"resolved {self.backend}"
                    )
                self.progress.add(stats)
                self.rejections.count(
                    epoch,
                    self.observed[step.date][step.indices],
                    prepared.accepted,
                    requested,
                    self.progress.rejections,
                )
                if prepared.batch is not None:
                    # Rejected roots were dropped; the leading accepted rows are positives.
                    positives = int(prepared.accepted[: len(step.positives)].sum())
                    value, risk = self.train_step(
                        step, positives, to_device(prepared.batch, self.device)
                    )
                    if self.average is not None:
                        self.average.update(self.model)
                    # Losses stay on the device; the host reads them once per log interval.
                    self.loss_sum += value
                    interval.add(value, risk)
                    self.loss_steps += 1
                self.step = step.step + 1
                logged = self.step == len(schedule) or self.step % self.runtime.log_every_steps == 0
                saved = bool(every) and self.step % every == 0
                if logged or saved:
                    loss, risk, corrections, finite = interval.totals()
                    if not finite:
                        # Raised before any checkpoint can persist non-finite weights.
                        raise ValueError(
                            f"Non-finite training loss at or before epoch {epoch + 1} "
                            f"step {self.step}"
                        )
                    now = time.perf_counter()
                    record = self.progress.record(
                        {
                            "event": "train",
                            "epoch": epoch + 1,
                            "step": self.step,
                            "steps": len(schedule),
                            "date": step.date,
                            **interval.record(loss, risk, corrections, now),
                            "rejected_roots": int(self.rejections.run["rejected"]),
                            "batch": {
                                k: v for k, v in plain(stats).items() if k != "sampler_backend"
                            },
                        }
                    )
                    # Every interval is a row of history.csv; a logged one is an event too.
                    append_history(self.run.history, {k: record[k] for k in HISTORY_COLUMNS})
                    if logged:
                        emit(record)
                    interval.start(now)
                    if saved:
                        self.save_last()
                mark = time.perf_counter()

    def _select_epoch(self, epoch: int) -> None:
        """Score validation, keep the state the rule prefers, apply patience, checkpoint."""
        with evaluated_weights(self.model, self.average):
            scores, accepted = self.score("validation")
            selected = self._state_copy()
        labels = self.labels("validation")
        check_split_rejections(
            "validation", labels, accepted, self.limit, self.progress.rejections()
        )
        observed, scored = labels[accepted].astype(np.int64), scores[accepted]
        metrics = proxy_metrics(observed, scored, 0.5)
        risk = pu_risk(observed, scored, self.prior, self.positive_weight)
        row = epoch_record(
            epoch + 1,
            self.loss_sum,
            self.loss_steps,
            metrics,
            risk,
            averaged=self.average is not None,
        )
        value = selection_value(self.selection, row)
        if value is not None and value > self.best_value:
            self.best_value, self.best_epoch = value, epoch + 1
            self.best_state, self.best_scores = selected, scores
            self.best_accepted = accepted
        patience = self.training_config.patience
        self.stopped = stops_early(self.selection, patience, epoch + 1, self.best_epoch)
        self.epoch_rows.append({**row, "stopped": self.stopped})
        self.epoch, self.step = epoch + 1, 0
        self.epoch_rng_state = self.rng.bit_generator.state
        self.save_last()
        # After the resume state, so epochs.csv never holds an epoch that resume.pt lacks.
        rows = self.record_epochs()
        # Beside the epochs.csv row: the best epoch so far, and the seconds this segment
        # spent on the epoch, its validation included.
        seconds = round(time.perf_counter() - self.epoch_started, 3)
        record = {**rows[-1], "best_epoch": self.best_epoch, "epoch_seconds": seconds}
        self.emit_event({"event": "epoch", **record})

    def record_epochs(self) -> list[dict[str, Any]]:
        """Replace epochs.csv with the epochs so far, the selected one marked; return them."""
        rows = [{**row, "selected": row["epoch"] == self.best_epoch} for row in self.epoch_rows]
        write_epochs(self.run.epochs, rows)
        return rows

    def labels(self, split: str) -> np.ndarray:
        return np.concatenate([s.labels for s in self.evaluation[split]])

    def train_step(self, step: TrainingStep, positives: int, batch: Batch) -> StepLoss:
        return nnpu_step(self.model, self.optimizer, self.loss, batch, positives, step.seed)

    def score(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities and the accepted mask for the split's rows (sample, then row order).

        Rejected roots (NaN scores) are left out of metrics and outputs; see
        ``predictor.accepted_scores``.
        """
        self.model.eval()
        size = self.training_config.batch_size
        samples = self.evaluation[split]
        chunks = [(s, start) for s in samples for start in range(0, len(s.indices), size)]
        requests = (self.keys(s.indices[start : start + size], s.date) for s, start in chunks)
        logits: list[torch.Tensor] = []
        accepted: list[np.ndarray] = []
        total = sum(len(s.indices) for s in samples)
        done = 0
        scored = score_batches(
            self.model, self.build_eval, requests, device=self.device, prefetch=self.prefetch
        )
        with contextlib.closing(scored):
            for number, ((sample, start), item) in enumerate(zip(chunks, scored, strict=True), 1):
                self.progress.add(item.prepared.stats)
                accepted.append(item.prepared.accepted)
                if item.logits is not None:
                    logits.append(item.logits)
                done += min(size, len(sample.indices) - start)
                show_scoring(split, done, total)
                if number % self.runtime.log_every_steps == 0 or number == len(chunks):
                    self.emit_event(
                        {"event": "score", "split": split, "accounts": done, "total": total}
                    )
        return accepted_scores(logits, accepted, split)

    def frame(self, split: str, scores: np.ndarray, accepted: np.ndarray) -> pd.DataFrame:
        frame = prediction_frame(self.accounts, self.evaluation[split], scores, accepted)
        self.rejected_rows[split] = int((~accepted).sum())
        return frame

    def finish(self) -> dict[str, Any]:
        """Save the selected model, then score test and write metrics.json."""
        if self.best_epoch == 0 or self.best_scores is None or self.best_accepted is None:
            raise ValueError(
                f"No epoch produced a value of the selection rule {self.selection}; "
                "refusing to save untrained weights"
            )
        self.model.load_state_dict(self.best_state)
        validation = self.frame("validation", self.best_scores, self.best_accepted)
        rejected_roots = {
            "train": self.rejections.totals(),
            "validation": rejection_counts(self.labels("validation"), self.best_accepted),
        }
        labels = validation["observed_label"].to_numpy()
        threshold = select_threshold(labels, validation["score"].to_numpy())
        selection = proxy_metrics(labels, validation["score"].to_numpy(), threshold)
        known_mules = label_summary(self.mask)
        # Save the selected model before any test context is requested.
        SavedModel.selected(
            self.run.model,
            state=self.best_state,
            config=self.config,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            threshold=threshold,
            known_mules=known_mules,
            device=self.device,
            backend=self.backend,
        ).save()
        test, rejected_roots["test"] = self._score_test()
        results = {}
        for split, frame in (("validation", validation), ("test", test)):
            write_predictions(self.run.predictions(split), frame)
            results[split] = proxy_metrics(
                frame["observed_label"].to_numpy(), frame["score"].to_numpy(), threshold
            )
        result = run_summary(
            config=self.config,
            dataset_id=self.dataset_id,
            seed=self.training_config.seed,
            known_mules=known_mules,
            device=self.device,
            prior=self.prior,
            positive_weight=self.positive_weight,
            plan=self.plan,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            best_epoch=self.best_epoch,
            results=results,
            selection=selection,
            progress=self.progress,
            rejected_rows=self.rejected_rows,
            rejected_roots=rejected_roots,
            limit=self.limit,
        )
        self.emit_event({"event": "complete", "best_epoch": self.best_epoch})
        write_json(self.run.metrics, result)
        return result

    def _score_test(self) -> tuple[pd.DataFrame, dict[str, int]]:
        """Test predictions of the accepted roots, and the test split's rejection counts."""
        scores, accepted = self.score("test")
        labels = self.labels("test")
        check_split_rejections("test", labels, accepted, self.limit, self.progress.rejections())
        counts = rejection_counts(labels, accepted)
        return self.frame("test", scores, accepted), counts
