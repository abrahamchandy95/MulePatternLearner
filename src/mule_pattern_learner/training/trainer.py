"""Training the model on the examples, and its file model.pt (docs/design.md, "Training").

batches turns examples (the frames of data.examples) and their fetched subgraphs into the
model's batches. They are built once, on the training device, and serve every epoch: the
label-input variant hides confirmations on the device each step (model.hetero).

train fits one model per class prior of the config, each from the same seeded start: Adam
on the nnPU loss with each example's weight, one step per batch in a seeded random order,
for at most max_epochs epochs, stopping once patience epochs bring no better validation
score (training.selection). Each prior keeps its best epoch, and the model of the prior
whose best epoch scores highest is the one trained; an earlier epoch or prior wins a tie.

Before training, the weights of each label are scaled to a mean of 1 over the training
batches (scaled), because the loss divides each label's weighted sum by its count in the
batch (model.loss). An example then counts by its weight whatever shares its batch, so each
mule keeps the equal share data.examples gives it however many dates it spans.

model.pt holds the weights under the contract lock (docs/design.md, "The subgraph query"):
the version of fetch_subgraphs the model was trained on, a fingerprint of the feature
layout it reads, and the subgraph settings its batches were fetched with. load refuses a
model of another version, layout or settings, whose features would not mean what it
learned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import HeteroData

from ..config import ModelConfig, SubgraphConfig, TrainingConfig
from ..model.hetero import LABEL_WIDTH, EdgeType, HeteroModel
from ..model.loss import NonNegativePULoss
from ..model.time_encoding import FREQUENCIES, SCALE
from ..subgraph.build import COUNT_TRANSFORM, LAYOUTS, NODE_TYPES, build
from ..tigergraph.queries import NODES, RELATIONS, SUBGRAPH_VERSION, Subgraph, SubgraphBatch
from .selection import Recall, validation_recall

MODEL_FILE = "model.pt"


def dimensions() -> tuple[dict[str, int], dict[EdgeType, int]]:
    """The feature width of each node type and of each relation and its reverse, as
    subgraph.build lays them out."""
    nodes = {NODE_TYPES[name]: len(LAYOUTS[kind].columns) for name, kind in NODES.items()}
    edges: dict[EdgeType, int] = {}
    for name, (kind, start, end) in RELATIONS.items():
        source, target = NODE_TYPES[start], NODE_TYPES[end]
        edges[source, name, target] = edges[target, f"rev_{name}", source] = len(
            LAYOUTS[kind].columns
        )
    return nodes, edges


def fingerprint(label_inputs: bool) -> str:
    """A hash of the feature layout a model reads: the Layout of each node type and each
    relation (its categories, flags, counts, and durations with the fields that say they
    are present), how counts are transformed, the Fourier basis, and the width of the
    label inputs if it has them."""
    layout = {
        "nodes": {NODE_TYPES[name]: asdict(LAYOUTS[kind]) for name, kind in NODES.items()},
        "relations": {
            name: [start, end, asdict(LAYOUTS[kind])]
            for name, (kind, start, end) in RELATIONS.items()
        },
        "counts": COUNT_TRANSFORM,
        "fourier": [FREQUENCIES.tolist(), SCALE],
        "labels": LABEL_WIDTH if label_inputs else 0,
    }
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Batch:
    """Examples as the model reads them, in the order of data.seed_index: their accounts,
    their subgraphs, their labels and weights, and for the label-input variant the time
    from each account node's confirmation to its seed's as-of time (since_confirmed)."""

    account_ids: tuple[str, ...]
    data: HeteroData
    label: Tensor
    weight: Tensor
    since_confirmed_ms: Tensor | None


def since_confirmed(batches: Sequence[SubgraphBatch], discovered: Mapping[str, int]) -> Tensor:
    """The time from each account node's confirmation to its seed's as-of time, in the
    order build lays out the account nodes of batches, or -1 when the account was not
    confirmed by then. discovered holds the discovered_ms of each confirmed mule
    (data.supervision.read_supervision)."""

    def since(account_id: str, as_of_ms: int) -> int:
        found = discovered.get(account_id, 0)
        return as_of_ms - found if 0 < found <= as_of_ms else -1

    return torch.tensor(
        [
            since(node.id, batch.as_of_ms)
            for batch in batches
            for subgraph in batch.subgraphs
            for node in subgraph.accounts
        ],
        dtype=torch.long,
    )


def batches(
    examples: pd.DataFrame,
    subgraphs: Mapping[int, SubgraphBatch],
    *,
    size: int,
    discovered: Mapping[str, int] | None = None,
    shuffle: int | None = None,
    device: torch.device | str = "cpu",
) -> list[Batch]:
    """examples in Batches of at most size, built on device from subgraphs (as
    subgraph.fetch.fetch gives them for the examples). Each holds its confirmations when
    discovered is given, for the label-input variant. The examples keep their order, or
    are shuffled first by a generator seeded with shuffle when it is given."""
    if size < 1:
        raise ValueError(f"A batch needs at least one example, not {size}")
    if not examples.label.isin([0, 1]).all() or not (examples.weight >= 0).all():
        raise ValueError("Every example needs a label of 0 or 1 and a weight of at least 0")
    found: dict[tuple[str, int], Subgraph] = {
        (subgraph.seed, as_of_ms): subgraph
        for as_of_ms, batch in subgraphs.items()
        for subgraph in batch.subgraphs
    }
    order = np.arange(len(examples))
    if shuffle is not None:
        order = np.random.default_rng(shuffle).permutation(order)
    made: list[Batch] = []
    for start in range(0, len(order), size):
        chunk = examples.iloc[order[start : start + size]].sort_values("as_of_ms", kind="stable")
        keys: list[tuple[str, int]] = list(
            zip(chunk.account_id, chunk.as_of_ms.tolist(), strict=True)
        )
        missing = [key for key in keys if key not in found]
        if missing:
            raise ValueError(f"No subgraph was fetched for {missing}")
        dated = [
            replace(subgraphs[day], subgraphs=tuple(found[key] for key in keys if key[1] == day))
            for day in dict.fromkeys(day for _, day in keys)
        ]
        confirmed = None if discovered is None else since_confirmed(dated, discovered)
        made.append(
            Batch(
                tuple(chunk.account_id),
                build(dated, device=device),
                torch.tensor(chunk.label.to_numpy(), dtype=torch.long, device=device),
                torch.tensor(chunk.weight.to_numpy(), dtype=torch.float32, device=device),
                None if confirmed is None else confirmed.to(device),
            )
        )
    return made


def scaled(batches: Sequence[Batch]) -> list[Batch]:
    """batches with the weights of each label divided by their mean over all of batches,
    so that each label's weights have a mean of 1, as model.loss expects."""
    label = torch.cat([batch.label for batch in batches])
    weight = torch.cat([batch.weight for batch in batches])
    positive, unlabelled = (_mean(weight[label == value]) for value in (1, 0))
    return [
        replace(batch, weight=batch.weight / torch.where(batch.label == 1, positive, unlabelled))
        for batch in batches
    ]


def _mean(weights: Tensor) -> float:
    """The mean of weights, or 1 when there are none (a NaN mean) or they weigh nothing."""
    mean = float(weights.mean())
    return mean if mean > 0 else 1.0


@dataclass(frozen=True)
class Epoch:
    """One epoch of one prior: the mean training loss of its steps and the validation
    recall after it."""

    prior: float
    epoch: int
    loss: float
    recall: Recall


@dataclass(frozen=True)
class Trained:
    """A trained model in evaluation mode, the prior and epoch validation chose, and the
    history of every prior's epochs."""

    model: HeteroModel
    prior: float
    epoch: int
    history: tuple[Epoch, ...]

    def recall_by_prior(self) -> dict[float, Recall]:
        """Each prior's validation recall at its best epoch: how much the prior matters."""
        best: dict[float, Recall] = {}
        for epoch in self.history:
            if epoch.prior not in best or epoch.recall.score > best[epoch.prior].score:
                best[epoch.prior] = epoch.recall
        return best


def train(
    training: Sequence[Batch],
    validation: Sequence[Batch],
    model_config: ModelConfig,
    config: TrainingConfig,
    *,
    label_inputs: bool,
    seed: int,
    device: torch.device | str = "cpu",
) -> Trained:
    """The model trained on the training batches and chosen on the validation batches, as
    the module says, each prior starting from the seed. The batches hold confirmations
    exactly when the model has label inputs."""
    if not training or not validation:
        raise ValueError("Training needs training and validation batches")
    if any((batch.since_confirmed_ms is not None) != label_inputs for batch in training):
        raise ValueError("The training batches' confirmations do not match label_inputs")
    if any((batch.since_confirmed_ms is not None) != label_inputs for batch in validation):
        raise ValueError("The validation batches' confirmations do not match label_inputs")
    training = scaled(training)
    nodes, edges = dimensions()
    history: list[Epoch] = []
    chosen: _Best | None = None
    for prior in config.priors:
        torch.manual_seed(seed)
        model = HeteroModel(nodes, edges, model_config, label_inputs=label_inputs).to(device)
        optimiser = Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss = NonNegativePULoss(prior)
        best: _Best | None = None
        for epoch in range(1, config.max_epochs + 1):
            mean_loss = _epoch(model, optimiser, loss, training)
            recall = _recall(model, validation)
            history.append(Epoch(prior, epoch, mean_loss, recall))
            if best is None or recall.score > best.score:
                best = _Best(recall.score, prior, epoch, _copy(model))
            elif epoch - best.epoch >= config.patience:
                break
        if best is not None and (chosen is None or best.score > chosen.score):
            chosen = best
    assert chosen is not None
    model = HeteroModel(nodes, edges, model_config, label_inputs=label_inputs).to(device)
    model.load_state_dict(chosen.state)
    model.eval()
    return Trained(model, chosen.prior, chosen.epoch, tuple(history))


class _Best(NamedTuple):
    """The best epoch so far: its validation score, prior and number, and the weights."""

    score: float
    prior: float
    epoch: int
    state: dict[str, Tensor]


def scores(model: HeteroModel, batches: Sequence[Batch]) -> Tensor:
    """The model's score of each example of batches, in their order, in evaluation mode."""
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.data, batch.since_confirmed_ms) for batch in batches])


def _epoch(
    model: HeteroModel, optimiser: Adam, loss: NonNegativePULoss, batches: Sequence[Batch]
) -> float:
    """One step per batch, in a random order; the mean loss of the steps."""
    model.train()
    losses: list[Tensor] = []
    for index in torch.randperm(len(batches)).tolist():
        batch = batches[index]
        optimiser.zero_grad()
        logits = model(batch.data, batch.since_confirmed_ms)
        step, _ = loss(logits, batch.label, batch.weight)
        step.backward()
        optimiser.step()
        losses.append(step.detach())
    return float(torch.stack(losses).mean())


def _recall(model: HeteroModel, batches: Sequence[Batch]) -> Recall:
    def joined(tensors: list[Tensor]) -> np.ndarray:
        return torch.cat(tensors).cpu().numpy()

    return validation_recall(
        joined([batch.data.as_of_ms for batch in batches]),
        joined([batch.label for batch in batches]),
        joined([batch.weight for batch in batches]),
        joined([scores(model, batches)]),
    )


def _copy(model: HeteroModel) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def save(trained: Trained, path: Path, subgraph: SubgraphConfig) -> None:
    """Writes trained, whose batches were fetched with the settings subgraph, to path
    (MODEL_FILE in a run's directory) under the contract lock."""
    model = trained.model
    torch.save(
        {
            "version": SUBGRAPH_VERSION,
            "fingerprint": fingerprint(model.label_inputs),
            "subgraph": asdict(subgraph),
            "label_inputs": model.label_inputs,
            "model": asdict(model.config),
            "prior": trained.prior,
            "epoch": trained.epoch,
            "history": [
                (epoch.prior, epoch.epoch, epoch.loss, epoch.recall.top_1pct, epoch.recall.top_5pct)
                for epoch in trained.history
            ],
            "state": {name: value.cpu() for name, value in model.state_dict().items()},
        },
        path,
    )


def load(path: Path, subgraph: SubgraphConfig, device: torch.device | str = "cpu") -> Trained:
    """The model saved at path, on device, refused when it was built for another version of
    fetch_subgraphs or another feature layout than this code's, or on subgraphs fetched
    with other settings than subgraph, the ones its scores will be fetched with."""
    stored = torch.load(path, map_location=device, weights_only=True)
    if stored["version"] != SUBGRAPH_VERSION:
        raise ValueError(
            f"{path} was trained on fetch_subgraphs {stored['version']!r}, not "
            f"{SUBGRAPH_VERSION}: train it again"
        )
    label_inputs: bool = stored["label_inputs"]
    if stored["fingerprint"] != fingerprint(label_inputs):
        raise ValueError(f"{path} reads another feature layout than this code's: train it again")
    settings: dict[str, int] = stored["subgraph"]
    ours = asdict(subgraph)
    if settings != ours:
        other = {name: value for name, value in settings.items() if ours.get(name) != value}
        raise ValueError(
            f"{path} was trained on subgraphs fetched with other settings, {other}: fetch "
            "with them or train it again"
        )
    nodes, edges = dimensions()
    config = ModelConfig(**stored["model"])
    model = HeteroModel(nodes, edges, config, label_inputs=label_inputs).to(device)
    model.load_state_dict(stored["state"])
    model.eval()
    history = tuple(
        Epoch(prior, epoch, loss, Recall(top_1pct, top_5pct))
        for prior, epoch, loss, top_1pct, top_5pct in stored["history"]
    )
    return Trained(model, stored["prior"], stored["epoch"], history)
