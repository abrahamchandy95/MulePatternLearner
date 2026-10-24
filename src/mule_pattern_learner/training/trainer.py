"""The trainer of the GNN and its ablations, with selection after every epoch (docs/design.md,
"Training and selection" and "Comparisons").

For one model seed and one ranker (the full model, or an ablation's switch on the same
batches), train tries each prior it is given in turn:

- The model starts from weights drawn with the seed, the same for every prior. Adam (learning
  rate 0.001, weight decay 0.0001) minimises the balanced nnPU loss with the prior
  (model.loss), whose positive risk keeps each mule's weight and whose correction reads the
  positives at unit weight; data.examples divided each label's weights by their mean over the
  training examples once, so the trainer uses them as they are.
- Each epoch takes one step per training batch, in an order drawn from the seed and the epoch,
  and each step reseeds the generator of dropout from the seed, the epoch and the step. So a
  run resumed from a checkpoint takes exactly the steps it would have taken, and two runs of
  one seed are identical, under torch's deterministic algorithms.
- After each epoch the selection world is scored whole (or as its fallback keys say) and the
  selection score computed from pu_label alone (training.selection). An epoch is kept only when
  strictly better than the best so far, and a prior stops after patience epochs without one,
  or at max_epochs.
- After each epoch a checkpoint is written through a temporary file and a rename: the model,
  the optimiser, the best epoch so far and the history, under the contract lock and what the
  run is besides it: its seed, the optimiser's and the model's settings, the epochs and the
  patience, the training and selection worlds and a digest of each one's examples
  (checkpoint_fields). A run that stops resumes from it, and one of other settings, worlds or
  examples is refused with the differing field named; a prior whose checkpoint says it
  finished is not trained again. When epochs names a folder, every epoch's model is kept there
  too, under the contract lock alone, for evaluation's correlation of the selection score with
  recall (the development rotation's seed 0 only).

Of the priors, the one whose best epoch scores highest is kept; a later prior must be strictly
better. Each ablation trains at the prior the full model chose for that seed. A model file
holds the kept model with its prior, epoch, score, what it found and its history, and each
prior's best epoch with what it found, for the prior grid's decision (training.priors), under
the contract lock (training.contract). Before training, check_determinism runs PyG's attention
under the deterministic algorithms on the training device and stops on one where it has no
deterministic form.

The trainer reads the cache's batches and nothing else: no query, no answer key and no audit
record.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.nn import TransformerConv

from ..config import ModelConfig, SubgraphConfig, TrainingConfig
from ..model.encode import encode_batch
from ..model.hetero import HeteroModel
from ..model.loss import NonNegativePULoss
from ..subgraph.build import Batch
from ..subgraph.cache import Cache
from ..subgraph.layouts import LAYOUT
from ..subgraph.switches import Switch, apply
from . import contract
from .batches import ScoringBatches
from .contract import Lock
from .priors import Prior
from .selection import Found, found, ranked_keys

# The name of the full model; an ablation's is its switch's after it.
GNN = "gnn"
# The folders of a run's results that hold the checkpoints and every kept epoch's model.
CHECKPOINTS = "checkpoints"
EPOCHS = "epochs"
# The workspace cuBLAS needs to run deterministically on CUDA.
CUBLAS_WORKSPACE = ":4096:8"


def ranker_name(switch: Switch | None) -> str:
    """The name of the full model, or of the ablation of switch."""
    return GNN if switch is None else f"{GNN}_{switch.value}"


def model_config(config: ModelConfig, switch: Switch | None) -> ModelConfig:
    """The model of the full model, or of the ablation of switch: an ablation reads the
    community block only when its switch keeps it."""
    return config if switch is None else replace(config, community=switch.community)


def gnn_lock(
    switch: Switch | None,
    model: ModelConfig,
    subgraph: SubgraphConfig,
    training: TrainingConfig,
    extra: Mapping[str, object] | None = None,
) -> Lock:
    """The contract lock of the full model or of the ablation of switch; extra holds the
    ranker's own fields, which a checkpoint adds (checkpoint_fields) and a model file does
    not."""
    return contract.lock(
        ranker_name(switch),
        community=model_config(model, switch).community,
        subgraph=subgraph,
        seeds=len(training.seeds),
        extra=extra,
    )


def _digest(*parts: bytes) -> str:
    """A blake2b digest of parts, each preceded by its length so no two splits collide."""
    found = hashlib.blake2b(digest_size=16)
    for part in parts:
        found.update(len(part).to_bytes(8, "big"))
        found.update(part)
    return found.hexdigest()


def training_digest(training: Sequence[Batch]) -> str:
    """A digest of the training examples as the batches hold them, in their order: each one's
    Monday, account, label and weight."""
    keys = [key for batch in training for key in batch.keys]
    labels = torch.cat([batch.data.label.cpu() for batch in training]).to(torch.int64)
    weights = torch.cat([batch.data.weight.cpu() for batch in training]).to(torch.float64)
    return _digest(
        np.array([key.t_ms for key in keys], dtype=np.int64).tobytes(),
        "\x00".join(key.account_id for key in keys).encode(),
        labels.numpy().tobytes(),
        weights.numpy().tobytes(),
    )


def checkpoint_fields(
    training: Sequence[Batch],
    selection: SelectionWorld,
    config: ModelConfig,
    settings: TrainingConfig,
    seed: int,
) -> dict[str, object]:
    """What a checkpoint is a checkpoint of, beyond the contract lock: the seed, the
    optimiser's settings, the epochs and the patience, the model's settings, the worlds of the
    training and selection examples, and a digest of each one's examples. A run resumes only
    from a checkpoint with every one of them equal, so a finished run of other settings, or on
    other worlds or examples, is never taken for this one."""
    return {
        "seed": seed,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "batch_size": settings.batch_size,
        "max_epochs": settings.max_epochs,
        "patience": settings.patience,
        **{f"model.{name}": value for name, value in asdict(config).items()},
        "training_worlds": sorted({key.world for batch in training for key in batch.keys}),
        "training_examples": training_digest(training),
        "selection_world": selection.batches.world,
        "selection_examples": selection.digest,
    }


def model_file(folder: Path, ranker: str, seed: int) -> Path:
    """The model file of a ranker and a seed in a run's results folder."""
    return folder / f"{ranker}_seed{seed}.pt"


def selection_passes(settings: TrainingConfig, priors: int) -> int:
    """The most passes over the selection world one rotation makes. Every seed trains the full
    model over priors priors and then each ablation at one prior, and each of those runs at
    most max_epochs epochs, each ending in one pass over the selection world, which builds
    every selection example anew (training.batches.ScoringBatches) unless
    settings.reuse_built_examples keeps them. mule bench multiplies its build time per example
    by this, and by the selection world's examples, before whole-world selection is accepted."""
    if priors < 1:
        raise ValueError(f"A rotation tries at least one prior, not {priors}")
    return len(settings.seeds) * (priors + len(Switch)) * settings.max_epochs


class SelectionWorld:
    """The selection world a run scores after every epoch: its ranked keys (training.selection)
    and their scoring batches, built anew on every pass, or built once and kept when keep says
    so (training.batches.ScoringBatches)."""

    def __init__(
        self, cache: Cache, world: str, keys: pd.DataFrame, size: int, *, keep: bool = False
    ) -> None:
        self.keys = ranked_keys(keys)
        self.batches = ScoringBatches(cache, world, self.keys, size, keep=keep)
        ordered = self.batches.keys
        if not (
            ordered["t_ms"].tolist() == self.keys["t_ms"].tolist()
            and ordered["account_id"].tolist() == self.keys["account_id"].tolist()
        ):
            raise ValueError("The selection keys are not in the order of their batches")

    @cached_property
    def digest(self) -> str:
        """A digest of the ranked selection keys: each one's Monday, account, whether it is a
        positive and its weight."""
        keys = self.keys
        return _digest(
            keys["t_ms"].to_numpy(dtype=np.int64).tobytes(),
            "\x00".join(keys["account_id"]).encode(),
            keys["positive"].to_numpy(dtype=np.bool_).tobytes(),
            keys["weight"].to_numpy(dtype=np.float64).tobytes(),
        )


@dataclass(frozen=True)
class Epoch:
    """One epoch of one prior: the mean training loss of its steps and of the unclamped nnPU
    objective, the steps whose negative risk was corrected, and the selection score after
    it."""

    multiple: float
    epoch: int
    loss: float
    objective: float
    corrected: int
    score: float


@dataclass(frozen=True)
class PriorRun:
    """A prior's run: its best epoch, that epoch's score, what it found and its weights, and the
    history of its epochs."""

    prior: Prior
    epoch: int
    score: float
    found: Found
    state: dict[str, Tensor]
    history: tuple[Epoch, ...]


@dataclass(frozen=True)
class Trained:
    """A ranker trained with one seed: its model in evaluation mode with the kept weights, the
    prior and epoch selection kept, their score and what it found, and every prior's run."""

    ranker: str
    seed: int
    config: ModelConfig
    model: HeteroModel
    prior: Prior
    epoch: int
    score: float
    found: Found
    runs: tuple[PriorRun, ...]

    @property
    def history(self) -> tuple[Epoch, ...]:
        return tuple(epoch for run in self.runs for epoch in run.history)


def _step_seed(*parts: int) -> int:
    """A seed for torch's generator drawn from parts."""
    return int(np.random.SeedSequence(list(parts)).generate_state(1, dtype=np.uint64)[0] >> 1)


@contextmanager
def deterministic(device: torch.device) -> Generator[None]:
    """Run under torch's deterministic algorithms, as cuBLAS needs them on CUDA, and restore
    the settings found."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark


def check_determinism(device: torch.device) -> None:
    """Refuse a device on which PyG's scatter softmax, the attention of every TransformerConv,
    or its scatter sum, which gathers each node's messages, has no deterministic form under
    torch's deterministic algorithms: a TransformerConv with edge attributes must run, forward
    and backward, and give the same values and gradients twice. Its weights are drawn from a
    generator of its own, so the check leaves torch's generator as it found it."""
    generator = torch.Generator().manual_seed(0)
    nodes = torch.randn(512, 16, generator=generator)
    edges = torch.randint(0, 512, (2, 4_096), generator=generator)
    attributes = torch.randn(4_096, 8, generator=generator)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        conv = TransformerConv(16, 4, heads=4, edge_dim=8)
    for parameter in conv.parameters():
        parameter.data = torch.randn(parameter.shape, generator=generator) / 4
    conv = conv.to(device)
    found: list[tuple[Tensor, ...]] = []
    with deterministic(device):
        for _ in range(2):
            conv.zero_grad()
            x = nodes.to(device).requires_grad_(True)
            out = conv(x, edges.to(device), attributes.to(device))
            out.square().sum().backward()
            assert x.grad is not None
            grads = [parameter.grad for parameter in conv.parameters()]
            found.append(
                (
                    out.detach().cpu(),
                    x.grad.cpu(),
                    *(grad.cpu() for grad in grads if grad is not None),
                )
            )
    if not all(torch.equal(a, b) for a, b in zip(found[0], found[1], strict=True)):
        raise RuntimeError(f"PyG's attention is not deterministic on {device}")


def logits(
    model: HeteroModel,
    batches: ScoringBatches | Sequence[Batch],
    switch: Switch | None,
    device: torch.device,
) -> tuple[list[tuple[int, str]], NDArray[np.float64]]:
    """The model's logit of every example of batches, under switch, in evaluation mode, as
    float64, with each example's (Monday, account) key."""
    model.eval()
    keys: list[tuple[int, str]] = []
    found: list[Tensor] = []
    with torch.no_grad():
        for batch in batches:
            switched = batch if switch is None else apply(batch, switch)
            found.append(model(encode_batch(switched.data, LAYOUT, device)).double().cpu())
            keys += [(key.t_ms, key.account_id) for key in batch.keys]
    return keys, torch.cat(found).numpy() if found else np.zeros(0)


def selection_found(
    model: HeteroModel, selection: SelectionWorld, switch: Switch | None, device: torch.device
) -> Found:
    """What the model found on the selection world."""
    keys, scores = logits(model, selection.batches, switch, device)
    expected = list(zip(selection.keys["t_ms"], selection.keys["account_id"], strict=True))
    if keys != [(int(t_ms), str(account)) for t_ms, account in expected]:
        raise ValueError("The selection batches scored other keys than the selection world's")
    return found(selection.keys, scores)


def _state(model: HeteroModel) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _found_content(found: Found) -> dict[str, Any]:
    return {
        "accounts": list(found.accounts),
        "weights": torch.from_numpy(found.weights.copy()),
        "shares": torch.from_numpy(found.shares.copy()),
    }


def _found_of(content: Mapping[str, Any]) -> Found:
    return Found(
        tuple(content["accounts"]),
        content["weights"].cpu().numpy().astype(np.float64),
        content["shares"].cpu().numpy().astype(np.float64),
    )


def _history_content(history: Sequence[Epoch]) -> list[list[float]]:
    return [
        [epoch.multiple, epoch.epoch, epoch.loss, epoch.objective, epoch.corrected, epoch.score]
        for epoch in history
    ]


def _history_of(rows: Sequence[Sequence[float]]) -> tuple[Epoch, ...]:
    return tuple(
        Epoch(float(m), int(e), float(loss), float(o), int(c), float(s))
        for m, e, loss, o, c, s in rows
    )


@dataclass
class _Best:
    epoch: int
    score: float
    found: Found
    state: dict[str, Tensor]


class _PriorTraining:
    """The training of one prior for one seed and ranker, resumable from its checkpoint."""

    def __init__(
        self,
        training: Sequence[Batch],
        selection: SelectionWorld,
        prior: Prior,
        seed: int,
        *,
        config: ModelConfig,
        settings: TrainingConfig,
        lock: Lock,
        resume_lock: Lock,
        switch: Switch | None,
        device: torch.device,
        checkpoint: Path | None,
        epochs: Path | None,
    ) -> None:
        self.training = training
        self.selection = selection
        self.prior = prior
        self.seed = seed
        self.settings = settings
        self.lock = lock
        self.resume_lock = resume_lock
        self.switch = switch
        self.device = device
        self.checkpoint = checkpoint
        self.epochs = epochs
        torch.manual_seed(seed)
        self.model = HeteroModel(LAYOUT, config).to(device)
        self.config = config
        self.optimiser = Adam(
            self.model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        self.loss = NonNegativePULoss(prior.value)
        self.history: list[Epoch] = []
        self.best: _Best | None = None
        self.done = 0
        self.finished = False
        if checkpoint is not None and checkpoint.exists():
            self._resume(checkpoint)

    def _resume(self, path: Path) -> None:
        stored = contract.load(path, self.resume_lock, self.device)
        if stored["prior"] != [self.prior.multiple, self.prior.value]:
            raise ValueError(f"{path} is the checkpoint of another prior: {stored['prior']}")
        self.model.load_state_dict(stored["model"])
        self.optimiser.load_state_dict(stored["optimiser"])
        self.history = list(_history_of(stored["history"]))
        best = stored["best"]
        self.best = _Best(best["epoch"], best["score"], _found_of(best), best["state"])
        self.done = stored["epoch"]
        self.finished = stored["finished"]

    def _save(self) -> None:
        if self.checkpoint is None:
            return
        assert self.best is not None
        best = self.best
        contract.save(
            self.checkpoint,
            self.resume_lock,
            {
                "prior": [self.prior.multiple, self.prior.value],
                "epoch": self.done,
                "finished": self.finished,
                "model": _state(self.model),
                "optimiser": self.optimiser.state_dict(),
                "best": {
                    "epoch": best.epoch,
                    "score": best.score,
                    "state": best.state,
                    **_found_content(best.found),
                },
                "history": _history_content(self.history),
            },
        )

    def _epoch(self, epoch: int) -> tuple[float, float, int]:
        """One step per training batch in the epoch's order: the mean loss, the mean
        objective and the steps whose negative risk was corrected."""
        self.model.train()
        order = np.random.default_rng([self.seed, epoch]).permutation(len(self.training))
        losses: list[Tensor] = []
        objectives: list[Tensor] = []
        for step, index in enumerate(order.tolist()):
            torch.manual_seed(_step_seed(self.seed, epoch, step))
            data = encode_batch(self.training[index].data, LAYOUT, self.device)
            self.optimiser.zero_grad()
            loss, objective = self.loss(self.model(data), data.label, data.weight)
            loss.backward()
            self.optimiser.step()
            losses.append(loss.detach())
            objectives.append(objective.detach())
        loss_values = torch.stack(losses).double().cpu()
        objective_values = torch.stack(objectives).double().cpu()
        corrected = int((loss_values != objective_values).sum())
        return float(loss_values.mean()), float(objective_values.mean()), corrected

    def run(self) -> PriorRun:
        """Train until patience runs out or max_epochs, from the checkpoint if there is one."""
        settings = self.settings
        while not self.finished:
            epoch = self.done + 1
            loss, objective, corrected = self._epoch(epoch)
            found = selection_found(self.model, self.selection, self.switch, self.device)
            self.history.append(
                Epoch(self.prior.multiple, epoch, loss, objective, corrected, found.score)
            )
            if self.best is None or found.score > self.best.score:
                self.best = _Best(epoch, found.score, found, _state(self.model))
            self.done = epoch
            self.finished = (
                epoch >= settings.max_epochs or epoch - self.best.epoch >= settings.patience
            )
            if self.epochs is not None:
                self._keep_epoch(epoch, found)
            self._save()
        assert self.best is not None
        best = self.best
        return PriorRun(
            self.prior, best.epoch, best.score, best.found, best.state, tuple(self.history)
        )

    def _keep_epoch(self, epoch: int, found: Found) -> None:
        assert self.epochs is not None
        name = f"{ranker_name(self.switch)}_seed{self.seed}_prior{self.prior.multiple:g}"
        contract.save(
            self.epochs / f"{name}_epoch{epoch}.pt",
            self.lock,
            {
                "ranker": ranker_name(self.switch),
                "seed": self.seed,
                "prior": [self.prior.multiple, self.prior.value],
                "epoch": epoch,
                "score": found.score,
                "model_config": asdict(self.config),
                "state": _state(self.model),
                "found": _found_content(found),
            },
        )


def train(
    training: Sequence[Batch],
    selection: SelectionWorld,
    priors: Sequence[Prior],
    seed: int,
    *,
    model: ModelConfig,
    settings: TrainingConfig,
    subgraph: SubgraphConfig,
    switch: Switch | None = None,
    device: torch.device | str = "cpu",
    checkpoints: Path | None = None,
    epochs: Path | None = None,
) -> Trained:
    """The full model, or the ablation of switch, trained with seed over priors and selected
    on the selection world, as the module says. training holds the built training batches,
    which an ablation switches once; checkpoints names the folder of the checkpoints, and
    epochs the folder that keeps every epoch's model."""
    if not training:
        raise ValueError("Training needs at least one batch")
    if not priors:
        raise ValueError("Training needs at least one prior")
    if any("label" not in batch.data for batch in training):
        raise ValueError("A training batch has no labels and weights")
    target = torch.device(device)
    check_determinism(target)
    config = model_config(model, switch)
    lock = gnn_lock(switch, model, subgraph, settings)
    resume_lock = gnn_lock(
        switch,
        model,
        subgraph,
        settings,
        checkpoint_fields(training, selection, config, settings, seed),
    )
    batches = list(training) if switch is None else [apply(batch, switch) for batch in training]
    ranker = ranker_name(switch)
    runs: list[PriorRun] = []
    with deterministic(target):
        for prior in priors:
            checkpoint = None
            if checkpoints is not None:
                checkpoint = checkpoints / f"{ranker}_seed{seed}_prior{prior.multiple:g}.pt"
            runs.append(
                _PriorTraining(
                    batches,
                    selection,
                    prior,
                    seed,
                    config=config,
                    settings=settings,
                    lock=lock,
                    resume_lock=resume_lock,
                    switch=switch,
                    device=target,
                    checkpoint=checkpoint,
                    epochs=epochs,
                ).run()
            )
    kept = runs[0]
    for run in runs[1:]:
        if run.score > kept.score:
            kept = run
    final = HeteroModel(LAYOUT, config).to(target)
    final.load_state_dict(kept.state)
    final.eval()
    return Trained(
        ranker, seed, config, final, kept.prior, kept.epoch, kept.score, kept.found, tuple(runs)
    )


def save_trained(trained: Trained, path: Path, lock: Lock) -> None:
    """Write a trained ranker's model file under lock."""
    contract.save(
        path,
        lock,
        {
            "ranker": trained.ranker,
            "seed": trained.seed,
            "prior": [trained.prior.multiple, trained.prior.value],
            "epoch": trained.epoch,
            "score": trained.score,
            "model_config": asdict(trained.config),
            "state": _state(trained.model),
            "found": _found_content(trained.found),
            "history": _history_content(trained.history),
            "priors": [
                {
                    "prior": [run.prior.multiple, run.prior.value],
                    "epoch": run.epoch,
                    "score": run.score,
                    **_found_content(run.found),
                }
                for run in trained.runs
            ],
        },
    )


@dataclass(frozen=True)
class Selected:
    """A prior's best epoch, as a model file keeps it: the prior, the epoch, its selection score
    and what it found, which the prior grid's decision reads (training.priors)."""

    prior: Prior
    epoch: int
    score: float
    found: Found


@dataclass(frozen=True)
class Saved:
    """A model file read back: its ranker, seed, prior, epoch and selection score, its model in
    evaluation mode, what it found, its history, and every prior's best epoch."""

    ranker: str
    seed: int
    prior: Prior
    epoch: int
    score: float
    model: HeteroModel
    found: Found
    history: tuple[Epoch, ...]
    priors: tuple[Selected, ...]


def load_trained(path: Path, lock: Lock, device: torch.device | str = "cpu") -> Saved:
    """The model file at path, refused when its lock is not lock (training.contract)."""
    stored = contract.load(path, lock, device)
    config = ModelConfig(**stored["model_config"])
    model = HeteroModel(LAYOUT, config).to(torch.device(device))
    model.load_state_dict(stored["state"])
    model.eval()
    multiple, value = stored["prior"]
    return Saved(
        stored["ranker"],
        stored["seed"],
        Prior(multiple, value),
        stored["epoch"],
        stored["score"],
        model,
        _found_of(stored["found"]),
        _history_of(stored["history"]),
        tuple(
            Selected(Prior(*run["prior"]), run["epoch"], run["score"], _found_of(run))
            for run in stored["priors"]
        ),
    )


def train_rotation(
    training: Sequence[Batch],
    selection: SelectionWorld,
    priors: Sequence[Prior],
    folder: Path,
    *,
    model: ModelConfig,
    settings: TrainingConfig,
    subgraph: SubgraphConfig,
    device: torch.device | str = "cpu",
    keep_epochs_of_seed: int | None = None,
) -> dict[str, list[Trained]]:
    """Train the full model over priors and every ablation at the prior the full model chose,
    for every seed of settings, writing each model file into folder and the checkpoints
    under it; the full model of keep_epochs_of_seed keeps every epoch's model. The trained
    rankers by name, each a list over the seeds."""
    found: dict[str, list[Trained]] = {}
    for seed in settings.seeds:
        full = train(
            training,
            selection,
            priors,
            seed,
            model=model,
            settings=settings,
            subgraph=subgraph,
            device=device,
            checkpoints=folder / CHECKPOINTS,
            epochs=folder / EPOCHS if seed == keep_epochs_of_seed else None,
        )
        save_trained(full, model_file(folder, GNN, seed), gnn_lock(None, model, subgraph, settings))
        found.setdefault(GNN, []).append(full)
        for switch in Switch:
            ablation = train(
                training,
                selection,
                [full.prior],
                seed,
                model=model,
                settings=settings,
                subgraph=subgraph,
                switch=switch,
                device=device,
                checkpoints=folder / CHECKPOINTS,
            )
            save_trained(
                ablation,
                model_file(folder, ablation.ranker, seed),
                gnn_lock(switch, model, subgraph, settings),
            )
            found.setdefault(ablation.ranker, []).append(ablation)
    return found
