"""nnPU with observed-label selection over an injected temporal context source.

A run is resumable. ``run_dir/checkpoint_last.pt`` holds the model, optimizer,
weight average, RNG and schedule position plus the selection state, written every epoch and every
``checkpoint_every_steps`` steps. ``train(..., resume=True)`` continues from it and
reproduces the uninterrupted run exactly: every epoch schedule is drawn up front
from the saved generator state, and every step reseeds torch from a stable hash
of (seed, epoch, step), so dropout and sampler draws never depend on history.
The sampler backend is resolved once per run and stored in the checkpoint; a
resume that would sample with another backend is refused unless the config names
that backend explicitly. Reported REST calls, rejections and batch totals are
checkpointed too, so they cover every segment of a resumed run.

Roots that TigerGraph rejects are dropped from a batch, but only within
``max_rejected_root_fraction`` (default 0: any rejection fails). A rejected
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
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..artifacts import atomic_write
from ..batching.assemble import RootBatch, batch_device, build_root_batch, to_device
from ..batching.limits import BatchLimits
from ..config import fanouts, validate_config
from ..contract.bounds import PREFETCH_BATCHES
from ..contract.feature_groups import FeaturePlan, extraction_plan
from ..contract.graph_schema import ContextKey, context_scope
from ..contract.sampler_plan import SamplerPlan
from ..data.contexts import ContextOpener, ContextSource, check_coverage, close_source
from ..data.hub_registry import HubRegistry, load_hub_registry, warn_hub_stubs
from ..data.manifest import load_prepared, preparation_mismatches
from ..data.observed_labels import label_summary, load_observed_labels, visible_labels
from ..data.splits import eligible_mask, marginal_mask, sample_keys
from ..inference.predictor import accepted_scores, score_batches
from ..inference.rejections import exceeds_rejection_limit
from ..metrics import evaluate, select_threshold
from ..model.build import build_model
from ..model.loss import NonNegativePULoss
from ..paths import output_paths
from ..runtime.device import choose_device, torch_runtime
from ..runtime.workers import BatchPrefetcher
from ..sampling.backend import resolve_backend
from .averaging import WeightAverage, evaluated_weights
from .checkpoint import (
    CHECKPOINT_FORMAT,
    RUN_STATE_FILES,
    explicit_backend,
    load_resume_state,
    restore_cuda_rng,
    resume_fingerprint,
)
from .history import LogInterval, Progress, epoch_record, plain
from .objective import StepLoss, nnpu_objective, nnpu_step
from .schedule import (
    EvaluationSample,
    PUSample,
    TrainingStep,
    epoch_schedule,
    evaluation_indices,
)
from .summary import model_payload, prediction_frame, run_summary

Batch = dict[str, torch.Tensor]
BatchRequest = tuple[list[ContextKey], str, int]


@dataclass(frozen=True)
class RunSettings:
    batch_size: int
    fanouts: tuple[int, int]
    hidden: int
    epochs: int
    steps_per_epoch: int | None
    patience: int
    seed: int
    split_seed: int
    device: str
    threads: int
    deterministic: bool | str
    prefetch_batches: int
    checkpoint_every_steps: int
    log_every_steps: int
    # Largest fraction of an epoch's (or evaluation split's) roots TigerGraph may reject.
    max_rejected_root_fraction: float = 0.0
    # 0 validates and saves the raw weights; d in (0, 1) their moving average (WeightAverage).
    weight_average_decay: float = 0.0

    def __post_init__(self) -> None:
        # patience = 0 disables early stopping; n > 0 stops after n epochs without a
        # better validation AP.
        if len(self.fanouts) != 2 or self.epochs < 1 or self.patience < 0:
            raise ValueError("Training needs two fanouts, epochs >= 1 and patience >= 0")
        if not 0.0 <= self.max_rejected_root_fraction <= 1.0:
            raise ValueError("max_rejected_root_fraction must be in [0,1]")
        if not PREFETCH_BATCHES.holds(self.prefetch_batches):
            raise ValueError(
                f"prefetch_batches must be in [{PREFETCH_BATCHES.low},{PREFETCH_BATCHES.high}]"
            )
        if self.checkpoint_every_steps < 0 or self.log_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be >= 0 and log_every_steps >= 1")
        if not 0.0 <= self.weight_average_decay < 1.0:
            raise ValueError("weight_average_decay must be in [0,1)")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RunSettings:
        """The settings of a validated configuration."""
        steps = config["steps_per_epoch"]
        return cls(
            batch_size=int(config["batch_size"]),
            fanouts=fanouts(config),
            hidden=int(config["hidden"]),
            epochs=int(config["epochs"]),
            steps_per_epoch=None if steps is None else int(steps),
            patience=int(config["patience"]),
            seed=int(config["seed"]),
            split_seed=int(config["split_seed"]),
            device=str(config["device"]),
            threads=int(config["threads"]),
            deterministic=config["deterministic"],
            prefetch_batches=int(config["prefetch_batches"]),
            checkpoint_every_steps=int(config["checkpoint_every_steps"]),
            log_every_steps=int(config["log_every_steps"]),
            max_rejected_root_fraction=float(config["max_rejected_root_fraction"]),
            weight_average_decay=float(config["weight_average_decay"]),
        )

    def check_limits(self, plan: FeaturePlan, sampler: SamplerPlan) -> None:
        """Raise when a training batch of these settings would exceed the memory limits."""
        BatchLimits().validate_model(self.batch_size, self.fanouts, self.hidden, plan, sampler)


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
    source_plan = getattr(store, "plan", None)
    if (
        not isinstance(source_plan, FeaturePlan)
        or source_plan.fingerprint() != prepared.fingerprint()
    ):
        raise ValueError("Context source extraction plan differs from the prepared extraction plan")
    if getattr(store, "sampler", None) != sampler:
        raise ValueError("Context source sampler differs from the training sampler")
    check_coverage(store, model, sampler)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def train(
    config: dict[str, Any],
    dataset: Path,
    output: Path,
    *,
    contexts: ContextSource | None = None,
    open_contexts: ContextOpener | None = None,
    hubs: HubRegistry | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train, select on observed validation labels, save the model, then score test.

    Without ``contexts``, ``open_contexts`` opens the dataset's live source once the
    settings and the prepared dataset passed their checks (the pipeline passes
    pipeline.connect.open_context_source). ``contexts`` and ``hubs`` replace the
    dataset's source and hub registry (tests, offline replays). With ``resume`` an
    existing run directory continues from its last checkpoint; without it an existing
    run is an error.
    """
    config = validate_config(config)
    context_scope(config)
    plan = FeaturePlan.from_config(config)
    sampler = SamplerPlan.from_config(config)
    settings = RunSettings.from_config(config)
    # Fails fast on per-hop candidate pools too, before any database work.
    settings.check_limits(plan, sampler)
    checkpoint_path, run_dir = output_paths(output)
    # The run directory may already hold the prepared cache (run_dir/prepared); a run
    # has started once training wrote its own state there.
    started = any((run_dir / name).exists() for name in RUN_STATE_FILES)
    resuming = resume and started
    if not resuming and (checkpoint_path.exists() or started):
        raise FileExistsError(f"Experiment already exists: {output}; resume it or pick a new one")
    state = load_resume_state(config, run_dir) if resuming else None
    manifest, accounts = load_prepared(dataset)
    differences = preparation_mismatches(config, manifest)
    if differences:
        raise ValueError(f"Training settings differ from preparation: {differences}")
    # The source requests this model's groups; its hop-2 flags follow the architecture.
    source_plan = extraction_plan(config)
    mask = load_observed_labels(accounts, dataset)
    training, evaluation = _samples(config, settings, accounts, mask)
    prior, positive_weight = nnpu_objective(config)
    device = choose_device(settings.device)
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
        check_source(store, source_plan, plan, sampler)
        with torch_runtime(device, deterministic=settings.deterministic, threads=settings.threads):
            run = _TrainingRun(
                config=config,
                settings=settings,
                dataset=dataset,
                manifest=manifest,
                accounts=accounts,
                mask=mask,
                plan=plan,
                sampler=sampler,
                store=store,
                hubs=registry,
                device=device,
                prior=prior,
                positive_weight=positive_weight,
                training=training,
                evaluation=evaluation,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
            )
            result = run.execute(state)
        failed = False
        return result
    finally:
        # After an error or Ctrl-C, do not wait for in-flight REST calls.
        close_source(store, failed=failed)


def training_samples(
    config: dict[str, Any], accounts: pd.DataFrame, mask: pd.DataFrame
) -> list[PUSample]:
    """The PUSample of every train cutoff with visible positives, as train() schedules them."""
    marginal = marginal_mask(accounts)
    samples: list[PUSample] = []
    for date in config["dates"]["train"]:
        eligible = np.flatnonzero(eligible_mask(accounts, "train", date))
        observed = visible_labels(mask, date)
        if observed[eligible].any():
            samples.append(
                PUSample(date, eligible[marginal[eligible]], observed, eligible[observed[eligible]])
            )
    return samples


def _samples(
    config: dict[str, Any], settings: RunSettings, accounts: pd.DataFrame, mask: pd.DataFrame
) -> tuple[list[PUSample], dict[str, list[EvaluationSample]]]:
    training = training_samples(config, accounts, mask)
    marginal = marginal_mask(accounts)
    evaluation: dict[str, list[EvaluationSample]] = {"validation": [], "test": []}
    for name, samples in evaluation.items():
        for date in config["dates"][name]:
            eligible = np.flatnonzero(eligible_mask(accounts, name, date))
            observed = visible_labels(mask, date)
            chosen = evaluation_indices(
                eligible,
                observed,
                limit=config.get("evaluation_unlabeled_limit"),
                seed=settings.split_seed,
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
        config: dict[str, Any],
        settings: RunSettings,
        dataset: Path,
        manifest: dict[str, Any],
        accounts: pd.DataFrame,
        mask: pd.DataFrame,
        plan: FeaturePlan,
        sampler: SamplerPlan,
        store: ContextSource,
        hubs: HubRegistry,
        device: torch.device,
        prior: float,
        positive_weight: float,
        training: list[PUSample],
        evaluation: dict[str, list[EvaluationSample]],
        checkpoint_path: Path,
        run_dir: Path,
    ) -> None:
        self.config, self.settings, self.dataset = config, settings, dataset
        self.manifest, self.accounts, self.mask = manifest, accounts, mask
        self.plan, self.sampler, self.store, self.hubs = plan, sampler, store, hubs
        self.device, self.prior, self.positive_weight = device, prior, positive_weight
        self.loss = NonNegativePULoss(prior=prior, positive_weight=positive_weight)
        self.training, self.evaluation = training, evaluation
        self.checkpoint_path, self.run_dir = checkpoint_path, run_dir
        self.last_path = run_dir / "checkpoint_last.pt"
        self.batch_device = batch_device(device)
        self.prefetch = settings.prefetch_batches
        # Resolved once here, on the main thread, before any prefetch worker starts
        # (the cuGraph probe runs at most once), then passed to every batch.
        self.backend = resolve_backend(sampler, self.batch_device)
        self.limit = settings.max_rejected_root_fraction
        self.observed = {sample.date: sample.observed for sample in training}
        self.progress = Progress(time.perf_counter(), store, self.backend)
        # Seeded right before the model is built: initial weights depend only on the seed.
        torch.manual_seed(settings.seed)
        self.model = build_model(config, plan).to(device)
        self.optimizer = build_optimizer(self.model, config)
        decay = settings.weight_average_decay
        self.average = WeightAverage(self.model, decay) if decay > 0 else None
        self.rng = np.random.default_rng(settings.seed)
        self.epoch, self.step, self.stopped = 0, 0, False
        self.best_ap, self.best_epoch = -1.0, 0
        self.best_state = self._state_copy()
        self.best_scores: np.ndarray | None = None
        self.best_accepted: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []
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
            fanouts=self.settings.fanouts,
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
        state: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "config_fingerprint": resume_fingerprint(self.config),
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
            "history": self.history,
            "elapsed_seconds": time.perf_counter() - self.progress.started,
            # Plain dicts and ints: torch.load(weights_only=True) refuses a Counter.
            "sampler_backend": self.backend,
            "progress_totals": dict(self.progress.totals),
            "query_calls": self.progress.calls(),
            "rejections": self.progress.rejections(),
            "train_rejections": dict(self.train_rejections),
            "epoch_rejections": dict(self.epoch_rejections),
        }
        if self.device.type == "cuda":
            # The training device only: a resume may see a different number of GPUs.
            state["cuda_rng"] = torch.cuda.get_rng_state(self.device)
        with atomic_write(self.last_path) as pending:
            torch.save(state, pending)

    def restore(self, state: dict[str, Any]) -> None:
        saved = state["sampler_backend"]
        if saved != self.backend:
            if explicit_backend(self.config) != self.backend:
                raise ValueError(
                    f"Checkpoint was sampled with the {saved} backend but this host resolves "
                    f"{self.backend}; resume on a matching host, or set [sampler] backend = "
                    f'"{self.backend}" to accept a different sampling stream from here on'
                )
            print(
                f"Resuming a {saved} checkpoint with the explicitly configured {self.backend} "
                "sampler backend: the remaining steps sample a different stream",
                flush=True,
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
        self.best_epoch, self.history = state["best_epoch"], state["history"]
        scores, accepted = state["best_scores"], state["best_accepted"]
        self.best_scores = None if scores is None else scores.numpy()
        self.best_accepted = None if accepted is None else accepted.numpy()
        self.progress.started -= float(state["elapsed_seconds"])
        self.progress.totals = Counter(state["progress_totals"])
        self.progress.base_calls = int(state["query_calls"])
        self.progress.base_rejections = Counter(state["rejections"])
        self.train_rejections = Counter(state["train_rejections"])
        self.epoch_rejections = Counter(state["epoch_rejections"])

    # Phases ------------------------------------------------------------------

    def execute(self, state: dict[str, Any] | None) -> dict[str, Any]:
        if state is not None:
            self.restore(state)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not (self.run_dir / "config.json").exists():
            (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
            self.mask.to_parquet(self.run_dir / "observed_labels.parquet", index=False)
        self.progress.path = self.run_dir / "progress.jsonl"
        self.progress.emit(
            {
                "event": "resume" if state is not None else "start",
                "device": str(self.device),
                "known_mules": label_summary(self.mask),
                "loss": "nnPU",
                "model": str(self.checkpoint_path),
                "epoch": self.epoch,
                "step": self.step,
                "prefetch_batches": self.prefetch,
                "max_rejected_root_fraction": self.limit,
            }
        )
        while self.epoch < self.settings.epochs and not self.stopped:
            self.run_epoch()
        return self.finish()

    def run_epoch(self) -> None:
        """Train the current epoch's remaining steps, then select on validation."""
        epoch = self.epoch
        self.epoch_rng_state = self.rng.bit_generator.state
        schedule = epoch_schedule(
            self.training,
            self.rng,
            self.settings.batch_size,
            epoch=epoch,
            seed=self.settings.seed,
            max_steps=self.settings.steps_per_epoch,
        )
        if self.step > len(schedule):
            raise ValueError("Checkpoint step lies beyond the epoch schedule")
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
        every = self.settings.checkpoint_every_steps
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
                logged = (
                    self.step == len(schedule) or self.step % self.settings.log_every_steps == 0
                )
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
                    self.progress.emit(
                        {
                            "event": "train",
                            "epoch": epoch + 1,
                            "step": self.step,
                            "steps": len(schedule),
                            "date": step.date,
                            **interval.record(loss, risk, corrections, now),
                            "batch": {
                                k: v for k, v in plain(stats).items() if k != "sampler_backend"
                            },
                        },
                        echo=logged,
                    )
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
        self.history.append(epoch_record(epoch + 1, self.loss_sum, self.loss_steps, metrics))
        if ap is not None and ap > self.best_ap:
            self.best_ap, self.best_epoch = ap, epoch + 1
            self.best_state, self.best_scores = selected, scores
            self.best_accepted = accepted
        # patience = 0 disables early stopping.
        patience = self.settings.patience
        self.stopped = patience > 0 and epoch + 1 - self.best_epoch >= patience
        self.epoch, self.step = epoch + 1, 0
        self.epoch_rng_state = self.rng.bit_generator.state
        self.save_last()
        self.progress.emit({"event": "epoch", **self.history[-1], "stopped": self.stopped})

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
        size = self.settings.batch_size
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
                if number % self.settings.log_every_steps == 0 or number == len(chunks):
                    self.progress.emit(
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
            threshold=threshold,
            plan=self.plan,
            sampler=self.sampler,
            known_mules=known_mules,
            device=self.device,
            backend=self.backend,
        )
        # Save the selected model before any test context is requested.
        with atomic_write(self.checkpoint_path) as pending:
            torch.save(payload, pending)
        test, rejected_roots["test"] = self._score_test()
        results = {}
        for split, frame in (("validation", validation), ("test", test)):
            frame.to_parquet(self.run_dir / (split + "_predictions.parquet"), index=False)
            results[split] = evaluate(
                frame["observed_label"].to_numpy(), frame["score"].to_numpy(), threshold
            )
        result = run_summary(
            config=self.config,
            manifest=self.manifest,
            seed=self.settings.seed,
            known_mules=known_mules,
            device=self.device,
            prior=self.prior,
            positive_weight=self.positive_weight,
            plan=self.plan,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            best_epoch=self.best_epoch,
            history=self.history,
            results=results,
            selection=selection,
            checkpoint=self.checkpoint_path,
            progress=self.progress,
            rejected_rows=self.rejected_rows,
            rejected_roots=rejected_roots,
            limit=self.limit,
        )
        self.progress.emit({"event": "complete", "best_epoch": self.best_epoch}, echo=False)
        (self.run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result

    def _score_test(self) -> tuple[pd.DataFrame, dict[str, int]]:
        """Test predictions of the accepted roots, and the test split's rejection counts."""
        scores, accepted = self.score("test")
        labels = self.labels("test")
        self.check_rejections("test", labels, accepted)
        counts = rejection_counts(labels, accepted)
        return self.frame("test", scores, accepted), counts
