"""nnPU with observed-label selection over an injected temporal context source.

A run is resumable. ``run_dir/checkpoint_last.pt`` holds the model, optimizer,
RNG and schedule position plus the selection state, written every epoch and every
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
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from mule_pattern_learner.device import choose_device, torch_runtime

from ..encoding import BASIS_ID
from ..loss import NonNegativePULoss
from ..metrics import evaluate, select_threshold
from .batching import RootBatch, build_root_batch
from .checkpoint import (
    CHECKPOINT_FORMAT,
    RUN_STATE_FILES,
    atomic_save,
    explicit_backend,
    load_resume_state,
    restore_cuda_rng,
    resume_fingerprint,
)
from .config_schema import fanouts, model_seed, setting, split_seed, validate_config
from .contract import (
    CLIENT_GROUPS,
    ContextKey,
    FeaturePlan,
    SamplerPlan,
    contract_fingerprint,
    extraction_plan,
)
from .dataset import (
    eligible_mask,
    load_prepared,
    manifest_digest,
    marginal_mask,
    preparation_mismatches,
    sample_keys,
)
from .hubs import HubRegistry, load_hub_registry, warn_hub_stubs
from .memory import BatchLimits
from .model import build_model
from .policy import exceeds_rejection_limit, validate_protocol
from .sampler import resolve_backend
from .sampling import (
    MAX_PREFETCH,
    BatchPrefetcher,
    PUSample,
    TrainingStep,
    epoch_schedule,
    evaluation_indices,
)
from .source import ContextSource, check_coverage, close_source, open_context_source
from .supervision import label_summary, load_observed_labels, visible_labels

TRAINING_PROTOCOL = "scoped_observed_label_nnpu_v5"
Batch = dict[str, torch.Tensor]
BatchRequest = tuple[list[ContextKey], str, int]


def output_paths(output: Path) -> tuple[Path, Path]:
    """A .pt output names the model; directory outputs retain the experiment API."""
    if output.suffix == ".pt":
        return output, output.with_name(output.stem + "_run")
    return output / "model.pt", output


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

    def __post_init__(self) -> None:
        # patience = 0 disables early stopping; n > 0 stops after n epochs without a
        # better validation AP.
        if len(self.fanouts) != 2 or self.epochs < 1 or self.patience < 0:
            raise ValueError("Training needs two fanouts, epochs >= 1 and patience >= 0")
        if not 0.0 <= self.max_rejected_root_fraction <= 1.0:
            raise ValueError("max_rejected_root_fraction must be in [0,1]")
        if not 0 <= self.prefetch_batches <= MAX_PREFETCH:
            raise ValueError(f"prefetch_batches must be in [0,{MAX_PREFETCH}]")
        if self.checkpoint_every_steps < 0 or self.log_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be >= 0 and log_every_steps >= 1")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RunSettings:
        steps = config.get("steps_per_epoch")
        return cls(
            batch_size=int(setting(config, "batch_size")),
            fanouts=fanouts(config),
            hidden=int(setting(config, "hidden")),
            epochs=int(setting(config, "epochs")),
            steps_per_epoch=None if steps is None else int(steps),
            patience=int(setting(config, "patience")),
            seed=model_seed(config),
            split_seed=split_seed(config),
            device=str(setting(config, "device")),
            threads=int(setting(config, "threads")),
            deterministic=setting(config, "deterministic"),
            prefetch_batches=int(setting(config, "prefetch_batches")),
            checkpoint_every_steps=int(setting(config, "checkpoint_every_steps")),
            log_every_steps=int(setting(config, "log_every_steps")),
            max_rejected_root_fraction=float(setting(config, "max_rejected_root_fraction")),
        )

    def check_limits(self, plan: FeaturePlan, sampler: SamplerPlan) -> None:
        """Raise when a training batch of these settings would exceed the memory limits."""
        BatchLimits().validate_model(self.batch_size, self.fanouts, self.hidden, plan, sampler)


@dataclass(frozen=True)
class EvaluationSample:
    date: str
    indices: np.ndarray
    labels: np.ndarray


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


def nnpu_objective(config: dict[str, Any]) -> tuple[float, float]:
    """The class prior and the positive-risk weight.

    "prior" is textbook nnPU (the weight is the prior). "balanced" is imbalanced nnPU
    (Su, Chen and Xu, IJCAI 2021) with a balanced target prior of 0.5: its risk
    0.5 * R_p^+ + 0.5 / (1 - prior) * (R_u^- - prior * R_p^-) is this loss with weight
    1 - prior, scaled by a constant (exactly so for the loss's beta = 0, gamma = 1).
    """
    prior = float(config["class_prior"])
    weight = setting(config, "positive_weight")
    if weight == "prior":
        return prior, prior
    if weight == "balanced":
        return prior, 1.0 - prior
    return prior, float(weight)


def objective_name(prior: float, positive_weight: float) -> str:
    if positive_weight == prior:
        return "nnPU"
    if math.isclose(positive_weight, 1.0 - prior):
        return "imbalanced_nnPU"
    return "positive_reweighted_nnPU"


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(setting(config, "learning_rate")),
        weight_decay=float(setting(config, "weight_decay")),
    )


class StepLoss(NamedTuple):
    """One step's backpropagated nnPU loss and its unclamped risk estimate (detached).

    They differ exactly when the non-negative correction fired, which a model that
    memorises its few revealed positives makes frequent.
    """

    value: torch.Tensor
    objective: torch.Tensor


def nnpu_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: NonNegativePULoss,
    batch: Batch,
    positives: int,
    seed: int,
) -> StepLoss:
    """One optimizer step; the batch's leading ``positives`` rows are observed positives.

    Dropout masks depend only on ``seed`` (the step seed), never on earlier history.
    """
    torch.manual_seed(seed)
    logits = model(batch)
    targets = torch.zeros_like(logits)
    targets[:positives] = 1
    value, objective = loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    value.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return StepLoss(value.detach(), objective.detach())


def train(
    config: dict[str, Any],
    dataset: Path,
    output: Path,
    *,
    contexts: ContextSource | None = None,
    hubs: HubRegistry | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train, select on observed validation labels, save the model, then score test.

    ``contexts`` and ``hubs`` replace the dataset's live source and hub registry
    (tests, offline replays). With ``resume`` an existing run directory continues
    from its last checkpoint; without it an existing run is an error.
    """
    config = validate_config(config)
    validate_protocol(config)
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
    # Same extraction groups as preparation (checked above); hop-2 flags follow this model.
    source_plan = extraction_plan(config)
    if set(plan.groups) - set(source_plan.groups) - CLIENT_GROUPS:
        raise ValueError("Prepared extraction does not contain requested feature groups")
    mask = load_observed_labels(accounts, dataset, manifest)
    training, evaluation = _samples(config, settings, accounts, mask)
    prior, positive_weight = nnpu_objective(config)
    device = choose_device(settings.device)
    registry = hubs if hubs is not None else load_hub_registry(dataset, manifest)
    warn_hub_stubs(registry, plan)
    store = contexts if contexts is not None else open_context_source(dataset, manifest, config)
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


def _plain(stats: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of batch statistics (numpy scalars become Python numbers)."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in stats.items()}


class _Progress:
    """Append JSON lines to run_dir/progress.jsonl; echo selected records to stdout."""

    def __init__(self, started: float, store: ContextSource, backend: str) -> None:
        self.started, self.store = started, store
        self.path: Path | None = None
        self.totals: Counter[str] = Counter()
        # The backend resolved once for the run (batch statistics name the same one).
        self.backend = backend
        # Counts of earlier segments of a resumed run; the source counts this one.
        self.base_calls = 0
        self.base_rejections: Counter[str] = Counter()

    def add(self, stats: dict[str, Any]) -> None:
        for key, value in _plain(stats).items():
            # Batch statistics are counts; the sampler backend is the only string.
            if isinstance(value, int) and not isinstance(value, bool):
                self.totals[key] += value

    def calls(self) -> int:
        """REST calls of every segment of the run."""
        return self.base_calls + int(getattr(self.store, "query_calls", 0))

    def rejections(self) -> dict[str, int]:
        """Rejected rows served by the source (both hops) in every segment, by status."""
        current: Counter[str] = Counter(getattr(self.store, "rejections", {}) or {})
        return dict(self.base_rejections + current)

    def emit(self, record: dict[str, Any], *, echo: bool = True) -> dict[str, Any]:
        record = {
            **record,
            "query_calls": self.calls(),
            "rejections": self.rejections(),
            "stub_children": int(self.totals["stub_children"]),
            "rejected_children": int(self.totals["rejected_children"]),
            "sampler_backend": self.backend,
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
        }
        line = json.dumps(record, allow_nan=False)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(line + "\n")
        if echo:
            print(line, flush=True)
        return record


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
        # CUDA batches are built (and resampled) on the device by the prefetch workers.
        # Other devices build on the CPU and copy on the training thread.
        self.batch_device = device if device.type == "cuda" else torch.device("cpu")
        self.prefetch = settings.prefetch_batches
        # Resolved once here, on the main thread, before any prefetch worker starts
        # (the cuGraph probe runs at most once), then passed to every batch.
        # "deterministic" for policies other than resample.
        self.backend = resolve_backend(sampler, self.batch_device)
        self.limit = settings.max_rejected_root_fraction
        self.observed = {sample.date: sample.observed for sample in training}
        self.progress = _Progress(time.perf_counter(), store, self.backend)
        # Seeded right before the model is built: initial weights depend only on the seed.
        torch.manual_seed(settings.seed)
        self.model = build_model(config, plan).to(device)
        self.optimizer = build_optimizer(self.model, config)
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

    def to_device(self, batch: Batch) -> Batch:
        # CUDA batches already live on the device; others copy synchronously.
        if self.batch_device == self.device:
            return batch
        return {k: v.to(self.device) for k, v in batch.items()}

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
        atomic_save(state, self.last_path)

    def restore(self, state: dict[str, Any]) -> None:
        saved = state.get("sampler_backend")
        if saved is not None and saved != self.backend:
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
        self.epoch_rng_state = state["numpy_rng"]
        self.rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        restore_cuda_rng(state.get("cuda_rng"), self.device)
        self.epoch, self.step, self.stopped = state["epoch"], state["step"], state["stopped"]
        self.loss_sum = state["loss_sum"].to(self.device)
        self.loss_steps = state["loss_steps"]
        self.best_state, self.best_ap = state["best_state"], state["best_ap"]
        self.best_epoch, self.history = state["best_epoch"], state["history"]
        scores = state["best_scores"]
        self.best_scores = None if scores is None else scores.numpy()
        accepted = state.get("best_accepted")
        if accepted is not None:
            self.best_accepted = accepted.numpy()
        elif self.best_scores is not None:
            self.best_accepted = ~np.isnan(self.best_scores)
        self.progress.started -= float(state.get("elapsed_seconds", 0.0))
        self.progress.totals = Counter(state.get("progress_totals", {}))
        self.progress.base_calls = int(state.get("query_calls", 0))
        self.progress.base_rejections = Counter(state.get("rejections", {}))
        self.train_rejections = Counter(state.get("train_rejections", {}))
        self.epoch_rejections = Counter(state.get("epoch_rejections", {}))

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
        interval = torch.zeros((), device=self.device)
        objective = torch.zeros((), device=self.device)
        corrected = torch.zeros((), device=self.device)
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        every = self.settings.checkpoint_every_steps
        clock = mark = time.perf_counter()
        waited, count = 0.0, 0
        with self.batches(requests) as batches:
            for step, prepared in zip(remaining, batches, strict=True):
                waited += time.perf_counter() - mark
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
                    value, risk = self.train_step(step, positives, self.to_device(prepared.batch))
                    # Losses stay on the device; the host reads them once per log interval.
                    self.loss_sum += value
                    interval += value
                    objective += risk
                    corrected += (value != risk).to(corrected.dtype)
                    finite &= torch.isfinite(value)
                    self.loss_steps += 1
                    count += 1
                self.step = step.step + 1
                logged = (
                    self.step == len(schedule) or self.step % self.settings.log_every_steps == 0
                )
                saved = bool(every) and self.step % every == 0
                if logged or saved:
                    loss, risk, corrections, ok = torch.stack(
                        (interval, objective, corrected, finite.to(interval.dtype))
                    ).tolist()
                    if not ok:
                        # Raised before any checkpoint can persist non-finite weights.
                        raise ValueError(
                            f"Non-finite training loss at or before epoch {epoch + 1} "
                            f"step {self.step}"
                        )
                    now = time.perf_counter()
                    trained = max(count, 1)
                    self.progress.emit(
                        {
                            "event": "train",
                            "epoch": epoch + 1,
                            "step": self.step,
                            "steps": len(schedule),
                            "date": step.date,
                            "loss": loss / trained,
                            # The unclamped risk and the steps whose nnPU correction fired.
                            "objective": risk / trained,
                            "corrected_steps": int(corrections),
                            "seconds_per_step": (now - clock) / trained,
                            "batch_wait_seconds": waited / trained,
                            "batch": {
                                k: v for k, v in _plain(stats).items() if k != "sampler_backend"
                            },
                        },
                        echo=logged,
                    )
                    interval = torch.zeros((), device=self.device)
                    objective = torch.zeros((), device=self.device)
                    corrected = torch.zeros((), device=self.device)
                    clock, waited, count = now, 0.0, 0
                    if saved:
                        self.save_last()
                mark = time.perf_counter()

    def _select_epoch(self, epoch: int) -> None:
        """Score validation, keep the best state, apply patience and checkpoint the epoch."""
        scores, accepted = self.score("validation")
        labels = self.labels("validation")
        self.check_rejections("validation", labels, accepted)
        ap = evaluate(labels[accepted].astype(np.int64), scores[accepted], 0.5)["average_precision"]
        self.history.append(
            {
                "epoch": epoch + 1,
                "loss": float(self.loss_sum.item() / self.loss_steps),
                "steps": self.loss_steps,
                "validation_proxy_ap": ap,
            }
        )
        if ap is not None and ap > self.best_ap:
            self.best_ap, self.best_epoch = ap, epoch + 1
            self.best_state, self.best_scores = self._state_copy(), scores
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

        Roots that TigerGraph rejects are False in the mask (their score is NaN) and are
        left out of metrics and outputs. A non-finite probability for an accepted root
        is an error, never a rejection.
        """
        self.model.eval()
        size = self.settings.batch_size
        samples = self.evaluation[split]
        chunks = [(s, start) for s in samples for start in range(0, len(s.indices), size)]
        requests = (
            (self.keys(s.indices[start : start + size], s.date), "eval", 0) for s, start in chunks
        )
        values: list[torch.Tensor] = []
        accepted: list[np.ndarray] = []
        total = sum(len(s.indices) for s in samples)
        done = 0
        with self.batches(requests) as batches, torch.inference_mode():
            for number, ((sample, start), prepared) in enumerate(
                zip(chunks, batches, strict=True), 1
            ):
                self.progress.add(prepared.stats)
                accepted.append(prepared.accepted)
                if prepared.batch is not None:
                    values.append(torch.sigmoid(self.model(self.to_device(prepared.batch))))
                done += min(size, len(sample.indices) - start)
                if number % self.settings.log_every_steps == 0 or number == len(chunks):
                    self.progress.emit(
                        {"event": "evaluate", "split": split, "accounts": done, "total": total}
                    )
        mask = np.concatenate(accepted) if accepted else np.zeros(0, dtype=bool)
        scores = np.full(total, np.nan)
        if values:
            # One device-to-host copy per split.
            probabilities = torch.cat(values).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Non-finite model probability for {int((~np.isfinite(probabilities)).sum())}"
                    f" accepted {split} roots"
                )
            scores[mask] = probabilities
        return scores, mask

    def frame(self, split: str, scores: np.ndarray, accepted: np.ndarray) -> pd.DataFrame:
        frames = []
        offset = 0
        for sample in self.evaluation[split]:
            frame = self.accounts.iloc[sample.indices][["account_id", "group_id"]].copy()
            frame["date"] = sample.date
            frame["observed_label"] = sample.labels.astype(np.int64)
            frame["score"] = scores[offset : offset + len(sample.indices)]
            offset += len(sample.indices)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        self.rejected_rows[split] = int((~accepted).sum())
        return result[accepted].reset_index(drop=True)

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
        # Save the selected model before any test context is requested.
        atomic_save(self._model_payload(threshold), self.checkpoint_path)
        test, rejected_roots["test"] = self._score_test()
        results = {}
        for split, frame in (("validation", validation), ("test", test)):
            frame.to_parquet(self.run_dir / (split + "_predictions.parquet"), index=False)
            results[split] = evaluate(
                frame["observed_label"].to_numpy(), frame["score"].to_numpy(), threshold
            )
        result = self._metrics(results, selection, rejected_roots)
        self.progress.emit({"event": "complete", "best_epoch": self.best_epoch}, echo=False)
        (self.run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result

    def _model_payload(self, threshold: float) -> dict[str, Any]:
        """The model.pt payload of the selected state (read by checkpoint.ModelCheckpoint)."""
        return {
            "state_dict": self.best_state,
            "config": self.config,
            "basis_id": BASIS_ID,
            "contract": contract_fingerprint(),
            "dataset": str(self.dataset.resolve()),
            "dataset_manifest_sha256": manifest_digest(self.dataset),
            "threshold": threshold,
            "feature_dim": len(self.plan.node_names),
            "input_fingerprint": self.plan.fingerprint(),
            "sampler": self.sampler.query_params(),
            "sampler_fingerprint": self.sampler.fingerprint(),
            "selected_on": "validation_observed_label_proxy_ap",
            "training_protocol": TRAINING_PROTOCOL,
            "evaluation_protocol": self.config["evaluation_protocol"],
            "known_mules": label_summary(self.mask),
            "training_device": str(self.device),
            "sampler_backend": self.backend,
        }

    def _score_test(self) -> tuple[pd.DataFrame, dict[str, int]]:
        """Test predictions of the accepted roots, and the test split's rejection counts."""
        scores, accepted = self.score("test")
        labels = self.labels("test")
        self.check_rejections("test", labels, accepted)
        counts = rejection_counts(labels, accepted)
        return self.frame("test", scores, accepted), counts

    def _metrics(
        self,
        results: dict[str, Any],
        selection: dict[str, Any],
        rejected_roots: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        """The metrics.json record of a complete run."""
        prior, positive_weight = self.prior, self.positive_weight
        return {
            "status": "complete",
            "cohort": self.manifest["cohort"],
            "label_policy": self.config.get("label_policy"),
            "variant": self.model.variant,
            "seed": self.settings.seed,
            "known_mules": label_summary(self.mask),
            "device": str(self.device),
            "loss": "nnPU",
            "class_prior": prior,
            "positive_weight": positive_weight,
            "objective": objective_name(prior, positive_weight),
            "input_fingerprint": self.plan.fingerprint(),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "revealed_training_accounts": label_summary(self.mask)["train"],
            "best_epoch": self.best_epoch,
            "history": self.history,
            "observed_label_proxy": results,
            "evaluation_protocol": self.config["evaluation_protocol"],
            "validation_proxy": selection,
            "checkpoint": str(self.checkpoint_path),
            "database_calls_during_training": self.progress.calls(),
            "rejections": self.progress.rejections(),
            "sampler_backend": self.backend,
            "sampler_totals": dict(self.progress.totals),
            "rejected_evaluation_rows": self.rejected_rows,
            "rejected_roots": rejected_roots,
            "max_rejected_root_fraction": self.limit,
            "performance_claim": self.config["evaluation_protocol"] + "_observed_label_proxy_only",
            "evaluation_unlabeled_limit": self.config.get("evaluation_unlabeled_limit"),
            "training_protocol": TRAINING_PROTOCOL,
        }
