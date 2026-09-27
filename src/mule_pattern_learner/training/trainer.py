"""nnPU with observed-label selection over an injected temporal context source.

A run writes its files into its own directory (paths.RunPaths). It is resumable:
resume.pt holds the model, optimizer, weight average, RNG and schedule position plus the
selection state, written every epoch and every ``runtime.checkpoint_every_steps`` steps.
``train(..., resume=True)`` continues from it and reproduces the uninterrupted run
exactly: every epoch schedule is drawn up front from the saved generator state, and
every step reseeds torch from a stable hash of (seed, epoch, step), so dropout and
sampler draws never depend on history.
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

from collections import Counter
from collections.abc import Iterator, Mapping
import contextlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..artifacts import (
    HISTORY_COLUMNS,
    append_history,
    keep_history,
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
from ..data.contexts import ContextOpener, ContextSource, check_coverage, close_source
from ..data.hub_registry import HubRegistry, load_hub_registry, warn_hub_stubs
from ..data.manifest import dataset_id, dataset_mismatches, load_prepared
from ..data.observed_labels import label_summary, load_observed_labels, visible_labels
from ..data.splits import eligible_mask, marginal_mask, sample_keys
from ..inference.predictor import accepted_scores, score_batches
from ..inference.rejections import exceeds_rejection_limit
from ..inference.saved_model import SavedModel
from ..metrics import evaluate, select_threshold
from ..model.build import build_model
from ..model.loss import NonNegativePULoss
from ..paths import DatasetPaths, RunPaths
from ..runtime.device import choose_device, torch_runtime
from ..runtime.progress import emit, recording
from ..runtime.workers import BatchPrefetcher
from ..sampling.backend import resolve_backend
from .averaging import WeightAverage, evaluated_weights
from .checkpoint import ResumeState, load_resume_state, restore_cuda_rng, run_started
from .history import LogInterval, Progress, epoch_record, plain
from .objective import StepLoss, nnpu_objective, nnpu_step
from .schedule import (
    EvaluationSample,
    PUSample,
    TrainingStep,
    epoch_schedule,
    evaluation_indices,
)
from .summary import model_payload, prediction_frame, provenance, run_summary

Batch = dict[str, torch.Tensor]
BatchRequest = tuple[list[ContextKey], str, int]


def check_limits(config: RunConfig, plan: FeaturePlan) -> None:
    """Raise when a training batch of these settings would exceed the memory limits."""
    BatchLimits().validate_model(
        config.training.batch_size,
        config.sampler.fanouts,
        config.model.hidden,
        plan,
        config.sampler,
    )


def rejection_counts(labels: np.ndarray, accepted: np.ndarray) -> dict[str, int]:
    """Requested and rejected roots, split into observed positives and unlabeled."""
    lost = ~accepted
    positives = int(labels[lost].astype(bool).sum())
    rejected = int(lost.sum())
    return {
        "requested": len(labels),
        "rejected": rejected,
        "positive": positives,
        "unlabeled": rejected - positives,
    }


def check_source(
    store: ContextSource, prepared: FeaturePlan, model: FeaturePlan, sampler: SamplerPlan
) -> None:
    """The source extracts the prepared plan, covers the model inputs and uses its sampler."""
    if store.plan.fingerprint() != prepared.fingerprint():
        raise ValueError("Context source extraction plan differs from the prepared extraction plan")
    if store.sampler != sampler:
        raise ValueError("Context source sampler differs from the training sampler")
    check_coverage(store, model, sampler)


def build_optimizer(model: nn.Module, training: TrainingConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )


def train(
    config: RunConfig,
    dataset: DatasetPaths,
    run: RunPaths,
    *,
    contexts: ContextSource | None = None,
    open_contexts: ContextOpener | None = None,
    hubs: HubRegistry | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train, select on observed validation labels, save the model, then score test.

    The run's files go into its directory, run. Without ``contexts``,
    ``open_contexts`` opens the dataset's live source once the settings and the
    prepared dataset passed their checks (the pipeline passes
    pipeline.connect.open_context_source). ``contexts`` and ``hubs`` replace the
    dataset's source and hub registry (tests, offline replays). With ``resume`` a
    started run continues from its resume.pt; without it a started run is an error.
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
    # The source requests this model's groups; its hop-2 flags follow the architecture.
    source_plan = extraction_plan(plan)
    mask = load_observed_labels(accounts, dataset)
    training, evaluation = _samples(config, accounts, mask)
    prior, positive_weight = nnpu_objective(config.loss)
    runtime = config.runtime
    device = choose_device(runtime.device)
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    warn_hub_stubs(registry, plan)
    if contexts is not None:
        store = contexts
    elif open_contexts is not None:
        store = open_contexts(dataset, manifest, config)
    else:
        raise ValueError("Training needs contexts, or open_contexts to open the dataset's source")
    failed = True
    try:
        check_source(store, source_plan, plan, config.sampler)
        with torch_runtime(device, deterministic=runtime.deterministic, threads=runtime.threads):
            training_run = _TrainingRun(
                config=config,
                dataset=dataset,
                manifest=manifest,
                accounts=accounts,
                mask=mask,
                plan=plan,
                store=store,
                hubs=registry,
                device=device,
                prior=prior,
                positive_weight=positive_weight,
                training=training,
                evaluation=evaluation,
                run=run,
            )
            result = training_run.execute(state)
        failed = False
        return result
    finally:
        # After an error or Ctrl-C, do not wait for in-flight REST calls.
        close_source(store, failed=failed)


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
        store: ContextSource,
        hubs: HubRegistry,
        device: torch.device,
        prior: float,
        positive_weight: float,
        training: list[PUSample],
        evaluation: dict[str, list[EvaluationSample]],
        run: RunPaths,
    ) -> None:
        self.config, self.dataset = config, dataset
        self.dataset_id = dataset_id(manifest["source"]["source_id"], config)
        self.training_config, self.runtime = config.training, config.runtime
        self.manifest, self.accounts, self.mask = manifest, accounts, mask
        self.plan, self.sampler, self.store, self.hubs = plan, config.sampler, store, hubs
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
        self.progress = Progress(time.perf_counter(), store, self.backend)
        # Seeded right before the model is built: initial weights depend only on the seed.
        torch.manual_seed(config.training.seed)
        self.model = build_model(config.model, plan, self.sampler.fanouts[0]).to(device)
        self.optimizer = build_optimizer(self.model, config.training)
        decay = config.training.weight_average_decay
        self.average = WeightAverage(self.model, decay) if decay > 0 else None
        self.rng = np.random.default_rng(config.training.seed)
        self.epoch, self.step, self.stopped = 0, 0, False
        self.best_ap, self.best_epoch = -1.0, 0
        self.best_state = self._state_copy()
        self.best_scores: np.ndarray | None = None
        self.best_accepted: np.ndarray | None = None
        # One epochs.csv row per finished epoch, without its selected flag.
        self.epoch_rows: list[dict[str, Any]] = []
        self.loss_sum = torch.zeros((), device=device)
        self.loss_steps = 0
        self.rejected_rows: dict[str, int] = {}
        # Rejected training roots of the whole run and of the current epoch.
        self.train_rejections: Counter[str] = Counter()
        self.epoch_rejections: Counter[str] = Counter()
        self.epoch_rng_state: Mapping[str, Any] = self.rng.bit_generator.state

    def _state_copy(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # Batches -----------------------------------------------------------------

    def build_eval(self, keys: list[ContextKey]) -> RootBatch:
        return self.build((keys, "eval", 0))

    def build(self, request: BatchRequest) -> RootBatch:
        keys, mode, seed = request
        return build_root_batch(
            self.store,
            keys,
            fanouts=self.sampler.fanouts,
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
        state: dict[str, Any] = {
            "model": self._state_copy(),
            "optimizer": self.optimizer.state_dict(),
            "numpy_rng": self.epoch_rng_state,
            "torch_rng": torch.get_rng_state(),
            "epoch": self.epoch,
            "step": self.step,
            "stopped": self.stopped,
            "loss_sum": self.loss_sum.detach().cpu(),
            "loss_steps": self.loss_steps,
            "best_state": self.best_state,
            "weight_average": None if self.average is None else self.average.saved(),
            "best_ap": self.best_ap,
            "best_epoch": self.best_epoch,
            "best_scores": None if self.best_scores is None else torch.from_numpy(self.best_scores),
            "best_accepted": (
                None if self.best_accepted is None else torch.from_numpy(self.best_accepted)
            ),
            "epoch_rows": self.epoch_rows,
            "elapsed_seconds": time.perf_counter() - self.progress.started,
            # Plain dicts and ints: torch.load(weights_only=True) refuses a Counter.
            "sampler_backend": self.backend,
            "progress_totals": dict(self.progress.totals),
            "database_calls": self.progress.calls(),
            "rejections": self.progress.rejections(),
            "context_counts": {
                k: v for k, v in self.progress.contexts().items() if k != "distinct"
            },
            # The distinct contexts asked for, as their context_hash values.
            "context_keys": torch.tensor(sorted(self.store.counts.seen), dtype=torch.int64),
            "train_rejections": dict(self.train_rejections),
            "epoch_rejections": dict(self.epoch_rejections),
        }
        if self.device.type == "cuda":
            # The training device only: a resume may see a different number of GPUs.
            state["cuda_rng"] = torch.cuda.get_rng_state(self.device)
        ResumeState(state).save(self.run.resume, self.config)

    def restore(self, resumed: ResumeState) -> None:
        state = resumed.values
        saved = state["sampler_backend"]
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
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.average is not None:
            self.average.load(state["weight_average"])
        self.epoch_rng_state = state["numpy_rng"]
        self.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        restore_cuda_rng(state.get("cuda_rng"), self.device)
        self.epoch, self.step, self.stopped = state["epoch"], state["step"], state["stopped"]
        self.loss_sum = state["loss_sum"].to(self.device)
        self.loss_steps = state["loss_steps"]
        self.best_state, self.best_ap = state["best_state"], state["best_ap"]
        self.best_epoch, self.epoch_rows = state["best_epoch"], state["epoch_rows"]
        scores, accepted = state["best_scores"], state["best_accepted"]
        self.best_scores = None if scores is None else scores.numpy()
        self.best_accepted = None if accepted is None else accepted.numpy()
        self.progress.started -= float(state["elapsed_seconds"])
        self.progress.totals = Counter(state["progress_totals"])
        self.progress.base_calls = int(state["database_calls"])
        self.progress.base_rejections = Counter(state["rejections"])
        self.progress.base_contexts = Counter(state["context_counts"])
        self.store.counts.seen.update(state["context_keys"].tolist())
        self.train_rejections = Counter(state["train_rejections"])
        self.epoch_rejections = Counter(state["epoch_rejections"])

    # Phases ------------------------------------------------------------------

    def execute(self, state: ResumeState | None) -> dict[str, Any]:
        self.run.root.mkdir(parents=True, exist_ok=True)
        with recording(self.run.events):
            if state is not None:
                self.restore(state)
            if not self.run.config.exists():
                run_provenance = provenance(self.device, self.backend, self.dataset_id)
                write_run_config(self.run.config, self.config, run_provenance)
            # The intervals after the resume position are logged again.
            keep_history(self.run.history, self.epoch, self.step)
            if self.epoch_rows:
                self.record_epochs()
            self.emit_event(
                {
                    "event": "resume" if state is not None else "start",
                    "device": str(self.device),
                    "known_mules": label_summary(self.mask),
                    "loss": "nnPU",
                    "run": str(self.run.root),
                    "epoch": self.epoch,
                    "step": self.step,
                    "prefetch_batches": self.prefetch,
                    "max_rejected_root_fraction": self.limit,
                }
            )
            while self.epoch < self.training_config.epochs and not self.stopped:
                self.run_epoch()
            return self.finish()

    def emit_event(self, record: dict[str, Any]) -> dict[str, Any]:
        """Print record with the run's totals, recording it in events.jsonl; return it."""
        record = self.progress.record(record)
        emit(record)
        return record

    def run_epoch(self) -> None:
        """Train the current epoch's remaining steps, then select on validation."""
        epoch = self.epoch
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
            self.epoch_rejections = Counter()
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
                self.count_training_rejections(step, prepared.accepted, requested)
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
                            "rejected_roots": int(self.train_rejections["rejected"]),
                            "batch": {
                                k: v for k, v in plain(stats).items() if k != "sampler_backend"
                            },
                        }
                    )
                    # Every interval is a row of history.csv; logged ones are printed too.
                    append_history(self.run.history, {k: record[k] for k in HISTORY_COLUMNS})
                    if logged:
                        emit(record)
                    interval.start(now)
                    if saved:
                        self.save_last()
                mark = time.perf_counter()

    def _select_epoch(self, epoch: int) -> None:
        """Score validation, keep the best state, apply patience and checkpoint the epoch."""
        with evaluated_weights(self.model, self.average):
            scores, accepted = self.score("validation")
            selected = self._state_copy()
        labels = self.labels("validation")
        self.check_rejections("validation", labels, accepted)
        metrics = evaluate(labels[accepted].astype(np.int64), scores[accepted], 0.5)
        ap = metrics["average_precision"]
        if ap is not None and ap > self.best_ap:
            self.best_ap, self.best_epoch = ap, epoch + 1
            self.best_state, self.best_scores = selected, scores
            self.best_accepted = accepted
        # patience = 0 disables early stopping.
        patience = self.training_config.patience
        self.stopped = patience > 0 and epoch + 1 - self.best_epoch >= patience
        record = epoch_record(
            epoch + 1,
            self.loss_sum,
            self.loss_steps,
            metrics,
            averaged=self.average is not None,
            stopped=self.stopped,
        )
        self.epoch_rows.append(record)
        self.epoch, self.step = epoch + 1, 0
        self.epoch_rng_state = self.rng.bit_generator.state
        self.save_last()
        # After the resume state, so epochs.csv never holds an epoch that resume.pt lacks.
        rows = self.record_epochs()
        self.emit_event({"event": "epoch", **rows[-1]})

    def record_epochs(self) -> list[dict[str, Any]]:
        """Replace epochs.csv with the epochs so far, the selected one marked; return them."""
        rows = [{**row, "selected": row["epoch"] == self.best_epoch} for row in self.epoch_rows]
        write_epochs(self.run.epochs, rows)
        return rows

    def count_training_rejections(
        self, step: TrainingStep, accepted: np.ndarray, requested: int
    ) -> None:
        """Count a step's rejected roots; fail past the limit or on an observed positive.

        The limit applies to the whole epoch (``requested`` roots). The count only
        grows, so failing as soon as it is crossed equals failing at the epoch end.
        """
        self.train_rejections["requested"] += len(accepted)
        if accepted.all():
            return
        counts = rejection_counts(self.observed[step.date][step.indices], accepted)
        del counts["requested"]
        self.epoch_rejections.update(counts)
        self.train_rejections.update(counts)
        rejected, positives = self.epoch_rejections["rejected"], self.epoch_rejections["positive"]
        if exceeds_rejection_limit(rejected, positives, requested, self.limit):
            raise ValueError(
                f"Epoch {step.epoch + 1}: TigerGraph rejected {rejected} of {requested} training "
                f"roots so far ({positives} observed positives; max_rejected_root_fraction="
                f"{self.limit}); statuses {self.progress.rejections()}"
            )

    def labels(self, split: str) -> np.ndarray:
        return np.concatenate([s.labels for s in self.evaluation[split]])

    def check_rejections(self, split: str, labels: np.ndarray, accepted: np.ndarray) -> None:
        """Fail an evaluation whose rejected roots would censor its metrics."""
        counts = rejection_counts(labels, accepted)
        rejected, positives = counts["rejected"], counts["positive"]
        problem = None
        if exceeds_rejection_limit(rejected, positives, len(labels), self.limit):
            problem = (
                f"{positives} observed positives among them"
                if positives
                else f"above max_rejected_root_fraction={self.limit}"
            )
        elif split == "validation" and len(np.unique(labels[accepted])) != 2:
            problem = "validation no longer has both observed classes"
        if problem is not None:
            raise ValueError(
                f"{split}: TigerGraph rejected {rejected} of {len(labels)} roots ({problem}); "
                f"statuses {self.progress.rejections()}"
            )

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
                if number % self.runtime.log_every_steps == 0 or number == len(chunks):
                    self.emit_event(
                        {"event": "evaluate", "split": split, "accounts": done, "total": total}
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
                "No epoch produced a finite validation AP; refusing to save untrained weights"
            )
        self.model.load_state_dict(self.best_state)
        validation = self.frame("validation", self.best_scores, self.best_accepted)
        rejected_roots = {
            "train": {
                key: int(self.train_rejections[key])
                for key in ("requested", "rejected", "positive", "unlabeled")
            },
            "validation": rejection_counts(self.labels("validation"), self.best_accepted),
        }
        labels = validation["observed_label"].to_numpy()
        threshold = select_threshold(labels, validation["score"].to_numpy())
        selection = evaluate(labels, validation["score"].to_numpy(), threshold)
        known_mules = label_summary(self.mask)
        payload = model_payload(
            state=self.best_state,
            config=self.config,
            dataset=self.dataset,
            dataset_id=self.dataset_id,
            threshold=threshold,
            plan=self.plan,
            sampler=self.sampler,
            known_mules=known_mules,
            device=self.device,
            backend=self.backend,
        )
        # Save the selected model before any test context is requested.
        SavedModel(self.run.model, payload).save()
        test, rejected_roots["test"] = self._score_test()
        results = {}
        for split, frame in (("validation", validation), ("test", test)):
            write_predictions(self.run.predictions(split), frame)
            results[split] = evaluate(
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
        self.check_rejections("test", labels, accepted)
        counts = rejection_counts(labels, accepted)
        return self.frame("test", scores, accepted), counts
