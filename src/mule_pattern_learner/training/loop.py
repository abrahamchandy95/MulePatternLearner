from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
import socket
from dataclasses import dataclass
import time
from typing import Protocol, cast

import requests

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from tqdm import tqdm

from mule_pattern_learner.device import select_device
from mule_pattern_learner.training.loss import NonNegativePULoss
from mule_pattern_learner.training.metrics import ValScores, evaluate_ranking

_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_ACCOUNT: NodeType = "Account"

BatchTransform = Callable[[HeteroData], HeteroData]


def _identity(batch: HeteroData) -> HeteroData:
    return batch


_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
    TimeoutError,
    ConnectionError,
    socket.timeout,
)
_MAX_BATCH_RETRIES = 5
_RETRY_BACKOFF_S = 10.0


def _resilient_batches(
    make_loader: Callable[[], Iterator[HeteroData]],
) -> Iterator[HeteroData]:
    """Iterate a loader, surviving transient network errors per batch.

    make_loader is a zero-arg factory returning a FRESH batch iterator (each call
    re-issues the underlying TigerGraph queries from the start). When iteration
    raises a transient error mid-stream, we rebuild the loader and fast-forward
    past the batches already yielded, so no batch is reprocessed (which would
    double-count into the optimizer) or skipped (which would drop training data
    or corrupt the validation metric). The re-fetch of skipped batches is wasted
    work, but it is bounded by where the failure occurred and is the price of
    not losing the whole run. A batch that fails _MAX_BATCH_RETRIES times in a
    row re-raises -- that is a real outage, not a blip.

    NOTE: relies on the loader being deterministic across rebuilds (same seed,
    same order), which it is -- make_*_loader uses a fixed seed and shuffle=False.
    """
    delivered = 0
    attempts = 0
    while True:
        produced_this_pass = 0
        try:
            for i, batch in enumerate(make_loader()):
                if i < delivered:
                    continue  # already yielded on a previous pass; skip past it
                yield batch
                delivered += 1
                produced_this_pass += 1
                attempts = 0  # a clean delivery resets the retry budget
            return  # loader exhausted normally -> epoch complete
        except _TRANSIENT_ERRORS as err:
            attempts += 1
            if attempts > _MAX_BATCH_RETRIES:
                raise
            wait = _RETRY_BACKOFF_S * attempts
            print(
                f"    transient fetch error after {delivered} batches "
                + f"(attempt {attempts}/{_MAX_BATCH_RETRIES}); "
                + f"rebuilding loader, retrying in {wait:.0f}s: {type(err).__name__}",
                flush=True,
            )
            time.sleep(wait)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """
    Training-loop hyperparameters.

    prior:        class prior pi for the nnPU loss (true mule fraction).
    max_epochs:   hard cap on epochs.
    patience:     early-stopping patience (epochs without val-PAUC improvement).
    lr / weight_decay: AdamW settings (weight_decay is the decoupled L2 lever).
    eval_k:       cutoff for precision@k in validation.
    min_delta:    minimum val-PAUC gain to count as a significant improvement.
    positive_weight: weight on the nnPU positive_risk term. None reproduces the
        textbook objective (weight == prior); under extreme imbalance that makes
        the positive term a negligible fraction of the loss, so the optimizer
        drives every logit negative and inverts the ranking. Set it well above
        prior (e.g. 0.5) to give the positives real gradient weight. Passed
        straight through to NonNegativePULoss; see its docstring.
    """

    prior: float
    max_epochs: int = 50
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_k: int = 100
    min_delta: float = 1e-4
    positive_weight: float | None = None
    pauc_tie_epsilon: float = 2e-3
    """Half-width of the PAUC 'tie' band for PR-AUC tiebreaking (see EarlyStopper).

    Under heavy class imbalance ROC-AUC (the Proxy AUC) saturates near 1.0, so
    differences between strong epochs at the top end are dominated by validation
    noise rather than real ranking gains -- and selecting on raw PAUC can pin the
    'best' checkpoint to an early epoch that happens to win by noise while having
    worse top-of-ranking behaviour (more false positives). When two epochs are
    within pauc_tie_epsilon of the best PAUC, they are treated as a statistical
    tie on PAUC and broken by PR-AUC (average_precision), which does NOT saturate
    under imbalance and is sensitive to exactly the high-score false positives we
    pay for. PAUC remains the primary, coarse signal; PR-AUC only adjudicates
    near-ties. Set to 0.0 to recover pure-PAUC selection.
    """


class EarlyStopper:
    """
    Tracks the best validation epoch and decides when to stop.

    SELECTION is PAUC-primary with a PR-AUC tiebreaker. Higher is better for
    both (PAUC = validation Proxy AUC; PR-AUC = average_precision). Under heavy
    class imbalance PAUC saturates near 1.0, so among strong epochs its tiny
    differences are mostly validation noise -- and pure-PAUC selection can pin
    the checkpoint to an early epoch that won by noise yet has worse top-of-
    ranking behaviour (more false positives). To fix that WITHOUT abandoning
    PAUC (which PU theory endorses as robust to unlabeled-as-negative label
    corruption), an epoch is taken as the new best-to-save when EITHER:

      * its PAUC exceeds the best PAUC by more than tie_epsilon (a clear PAUC
        win -- decided on PAUC alone, exactly as before), OR
      * its PAUC is within tie_epsilon of the best PAUC (a statistical tie on a
        saturated metric) AND its PR-AUC strictly exceeds the best epoch's
        PR-AUC. PR-AUC does not saturate under imbalance and is sensitive to the
        high-score false positives, so it is the right adjudicator for ties.

    Two judgments remain deliberately separated:

      * is_best (the return of update): the save signal, per the rule above.
      * patience: only a min_delta-significant PAUC gain resets the no-improve
        counter, so stopping behaviour is driven by PAUC alone and is unchanged.
        (The tiebreaker changes WHICH saved checkpoint is best, not WHEN we stop.)
    """

    _patience: int
    _min_delta: float
    _tie_epsilon: float
    _best: float
    _best_secondary: float
    _significant_best: float
    _bad_epochs: int
    best_epoch: int

    def __init__(self, patience: int, min_delta: float, tie_epsilon: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._tie_epsilon = tie_epsilon
        self._best = float("-inf")  # best PAUC seen at a saved epoch
        self._best_secondary = float("-inf")  # PR-AUC at the saved best epoch
        self._significant_best = float("-inf")  # PAUC for patience accounting
        self._bad_epochs = 0
        self.best_epoch = -1

    @property
    def best(self) -> float:
        return self._best

    @property
    def best_secondary(self) -> float:
        """PR-AUC recorded at the currently-saved best epoch."""
        return self._best_secondary

    @property
    def bad_epochs(self) -> int:
        """Epochs since the last min_delta-significant PAUC improvement (patience progress)."""
        return self._bad_epochs

    def update(self, score: float, epoch: int, secondary: float = float("-inf")) -> bool:
        """Record an epoch and return True if it is the new best to save.

        score:     primary selection metric (validation Proxy AUC / PAUC).
        secondary: tiebreaker metric (PR-AUC / average_precision); used only when
                   score is within tie_epsilon of the current best PAUC.

        Selection (see class docstring): a clear PAUC win (score > best + eps) is
        best; otherwise a near-tie on PAUC (|score - best| <= eps, i.e.
        score >= best - eps) that strictly improves PR-AUC is also best. Patience
        is managed separately on PAUC, so a sub-delta gain can still be saved
        while the no-improve counter advances toward early stopping.

        With tie_epsilon == 0.0 this reduces to the original strict-PAUC rule
        (the near-tie branch requires score >= best AND better PR-AUC, which only
        improves the secondary at an exact PAUC plateau and never regresses PAUC).
        """
        clear_primary_win = score > self._best + self._tie_epsilon
        within_tie_band = score >= self._best - self._tie_epsilon
        tie_broken_by_secondary = within_tie_band and (secondary > self._best_secondary)

        is_best = clear_primary_win or tie_broken_by_secondary
        if is_best:
            # Track the PAUC of the saved epoch as the comparison point. On a
            # tiebreak the saved PAUC may dip by up to eps; that is the intended
            # trade (a noise-sized PAUC concession for a real PR-AUC gain).
            self._best = max(self._best, score)
            self._best_secondary = max(self._best_secondary, secondary)
            self.best_epoch = epoch

        # patience: PAUC-only, unchanged from the original.
        if score > self._significant_best + self._min_delta:
            self._significant_best = score
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        return is_best

    @property
    def should_stop(self) -> bool:
        return self._bad_epochs >= self._patience


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


def _node_store(batch: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, batch[key]))


def _edge_store(batch: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, batch[key]))


def _forward(model: Module, batch: HeteroData, device: torch.device) -> Tensor:
    x_dict: dict[NodeType, Tensor] = {_ACCOUNT: _node_store(batch, _ACCOUNT).x.to(device)}
    node_counts: dict[NodeType, int] = {
        ntype: int(_node_store(batch, ntype).n_id.shape[0]) for ntype in batch.node_types
    }
    edge_index_dict: dict[EdgeType, Tensor] = {
        etype: _edge_store(batch, etype).edge_index.to(device) for etype in batch.edge_types
    }
    edge_attr_dict: dict[EdgeType, Tensor] = {
        _HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr.to(device)
    }
    return cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))


def _seed_ids(batch: HeteroData, mapper_to_ids: Callable[[list[int]], list[str]]) -> list[str]:
    bsize = int(batch[_ACCOUNT].batch_size)  # pyright: ignore[reportAny]
    seed_int = cast(list[int], _node_store(batch, _ACCOUNT).n_id[:bsize].tolist())
    return mapper_to_ids(seed_int)


def _targets(seed_ids: Sequence[str], label_of: Mapping[str, int], device: torch.device) -> Tensor:
    return torch.tensor([label_of[s] for s in seed_ids], dtype=torch.long, device=device)


def train_epoch(
    model: Module,
    make_loader: Callable[[], Iterator[HeteroData]],
    loss_fn: NonNegativePULoss,
    optimizer: Optimizer,
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    batch_transform: BatchTransform = _identity,
    total_batches: int | None = None,
    epoch: int = 0,
    device: torch.device | None = None,
) -> float:
    """
    One training epoch. Returns the mean training loss over batches.

    For each batch: forward the whole sampled neighborhood, slice to the seed
    accounts (logits[:batch_size]), look up their pu_label targets, compute the
    nnPU loss on the seeds only, and backprop. Neighbors only inform embeddings;
    they never contribute to the loss.

    total_batches, if given, drives the progress bar's ETA (the loader is a lazy
    generator with no len, so the count must be passed in). miniters=1 because
    per-batch latency is erratic (each batch is a TigerGraph round-trip).
    """
    dev = device if device is not None else torch.device("cpu")
    _ = model.train()
    total = 0.0
    count = 0
    bar = tqdm(
        _resilient_batches(make_loader),
        total=total_batches,
        unit="batch",
        miniters=1,
        desc=f"train e{epoch}",
    )
    for raw in bar:
        batch = batch_transform(raw)
        logits = _forward(model, batch, dev)
        seeds = _seed_ids(batch, mapper_to_ids)
        seed_logits = logits[: len(seeds)]
        targets = _targets(seeds, pu_label_of, dev)

        train_loss, _ = cast("tuple[Tensor, Tensor]", loss_fn(seed_logits, targets))
        optimizer.zero_grad()
        _ = train_loss.backward()
        _ = optimizer.step()

        total += float(train_loss.item())
        count += 1
        bar.set_postfix(loss=f"{total / count:.4f}")
    return total / max(count, 1)


def validate(
    model: Module,
    make_loader: Callable[[], Iterator[HeteroData]],
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    eval_k: int,
    total_batches: int | None = None,
    epoch: int = 0,
    device: torch.device | None = None,
) -> ValScores:
    """
    Score the model over the validation seeds and return ranking metrics.

    Collects per-seed scores (sigmoid of the logit) and the seed's pu_label
    across all val batches, then evaluates ranking quality against pu_label.
    This uses only the labels the model also trains on, so it is production-valid.
    The loader must use the
    validation sampling regime (allow_val=True, allow_test=False) so no test
    account leaks into a val neighborhood.
    """
    dev = device if device is not None else torch.device("cpu")
    _ = model.eval()
    scores: list[float] = []
    labels: list[int] = []
    raw_logits: list[float] = []
    bar = tqdm(
        _resilient_batches(make_loader),
        total=total_batches,
        unit="batch",
        miniters=1,
        desc=f"  val e{epoch}",
    )
    with torch.no_grad():
        for batch in bar:
            logits = _forward(model, batch, dev)
            seeds = _seed_ids(batch, mapper_to_ids)
            seed_logits = logits[: len(seeds)]
            probs = cast(list[float], torch.sigmoid(seed_logits).tolist())
            scores.extend(probs)
            raw_logits.extend(cast(list[float], seed_logits.tolist()))
            labels.extend(pu_label_of[s] for s in seeds)

    scores_arr: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    labels_arr: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)

    logits_arr: NDArray[np.float64] = np.asarray(raw_logits, dtype=np.float64)
    pos_mask = labels_arr == 1
    pos_logits = logits_arr[pos_mask]
    unl_logits = logits_arr[~pos_mask]
    pos_mean = float(pos_logits.mean()) if pos_logits.size else float("nan")
    unl_mean = float(unl_logits.mean()) if unl_logits.size else float("nan")
    print(
        f"    logits: all[mean={logits_arr.mean():.3f} std={logits_arr.std():.3f} "
        + f"min={logits_arr.min():.3f} max={logits_arr.max():.3f}] "
        + f"pos_mean={pos_mean:.3f} unl_mean={unl_mean:.3f} "
        + f"separation={pos_mean - unl_mean:+.3f}",
        flush=True,
    )

    return evaluate_ranking(scores_arr, labels_arr, k=eval_k)


@dataclass(frozen=True, slots=True)
class EpochReport:
    epoch: int
    train_loss: float
    val_average_precision: float
    val_precision_at_k: float
    val_roc_auc: float
    is_best: bool


@dataclass(frozen=True, slots=True)
class FitResult:
    """Outcome of a training run.

    reports holds the per-epoch history. best_state_dict is a deep copy of the
    model weights at the epoch with the highest validation Proxy AUC (PAUC, the
    PU model-selection signal; deep-copied so later epochs do not mutate it), and
    best_epoch / best_val_pauc identify that epoch. After fit returns, the model
    has these best weights loaded, so it is ready to score or to save directly.
    """

    reports: list[EpochReport]
    best_state_dict: dict[str, Tensor]
    best_epoch: int
    best_val_pauc: float


def fit(
    model: Module,
    make_train_loader: Callable[[], Iterator[HeteroData]],
    make_val_loader: Callable[[], Iterator[HeteroData]],
    pu_label_of: Mapping[str, int],
    mapper_to_ids: Callable[[list[int]], list[str]],
    config: TrainConfig,
    optimizer: Optimizer | None = None,
    batch_transform: BatchTransform = _identity,
    train_batches: int | None = None,
    val_batches: int | None = None,
    device: torch.device | None = None,
    on_best: Callable[[dict[str, Tensor], int, float], None] | None = None,
) -> FitResult:
    """
    Train with early stopping on validation Proxy AUC (PAUC).

    make_train_loader / make_val_loader are zero-arg factories returning a fresh
    iterator of HeteroData batches for one epoch (fresh so each epoch reshuffles
    seeds and resamples neighborhoods). The train loader must use the training
    sampling regime (allow_val=False, allow_test=False); the val loader must use
    allow_val=True, allow_test=False. Returns one EpochReport per epoch run.

    train_batches / val_batches are the per-epoch batch counts, passed through to
    the progress bars for ETA (the loaders are lazy generators with no len).

    on_best, if given, is called the moment a new best epoch is found, with the
    best weights, epoch index, and val PAUC. Use it to persist the best
    checkpoint DURING training, so an interrupted long run still leaves the best
    model on disk rather than losing everything (the loop only returns at the
    very end).

    The optimizer defaults to AdamW(lr, weight_decay) over the model parameters;
    pass a pre-built optimizer to override.
    """
    dev = device if device is not None else select_device()
    _ = model.to(dev)
    opt: Optimizer = (
        optimizer
        if optimizer is not None
        else AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    )
    loss_fn = NonNegativePULoss(prior=config.prior, positive_weight=config.positive_weight)
    stopper = EarlyStopper(
        patience=config.patience,
        min_delta=config.min_delta,
        tie_epsilon=config.pauc_tie_epsilon,
    )

    reports: list[EpochReport] = []
    best_state: dict[str, Tensor] = copy.deepcopy(model.state_dict())
    best_epoch = -1
    best_val_pauc = float("-inf")
    for epoch in range(config.max_epochs):
        train_loss = train_epoch(
            model,
            make_train_loader,
            loss_fn,
            opt,
            pu_label_of,
            mapper_to_ids,
            batch_transform,
            total_batches=train_batches,
            epoch=epoch,
            device=dev,
        )
        val = validate(
            model,
            make_val_loader,
            pu_label_of,
            mapper_to_ids,
            config.eval_k,
            total_batches=val_batches,
            epoch=epoch,
            device=dev,
        )
        prev_best_pauc = stopper.best
        prev_best_ap = stopper.best_secondary
        is_best = stopper.update(val.roc_auc, epoch, secondary=val.average_precision)
        if is_best:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_pauc = val.roc_auc
            if on_best is not None:
                on_best(best_state, best_epoch, best_val_pauc)
        reports.append(
            EpochReport(
                epoch=epoch,
                train_loss=train_loss,
                val_average_precision=val.average_precision,
                val_precision_at_k=val.precision_at_k,
                val_roc_auc=val.roc_auc,
                is_best=is_best,
            )
        )

        if is_best:
            # Was this a clear PAUC win, or a PR-AUC tiebreak within the PAUC
            # noise band? Classify from the pre-update bests for a transparent log.
            if val.roc_auc > prev_best_pauc + config.pauc_tie_epsilon:
                marker = " *best (PAUC)"
            else:
                marker = (
                    f" *best (PR-AUC tiebreak: PAUC {val.roc_auc:.4f}~"
                    + f"{prev_best_pauc:.4f}, AP {val.average_precision:.4f}>"
                    + f"{prev_best_ap:.4f})"
                )
        else:
            marker = f"  (no gain for {stopper.bad_epochs})"
        print(
            f"epoch {epoch:>2} | train_loss {train_loss:.4f} | "
            + f"val PAUC {val.roc_auc:.4f} | "
            + f"AP {val.average_precision:.4f} | "
            + f"P@{val.k} {val.precision_at_k:.4f} | "
            + f"pos {val.num_labeled_positives}/{val.num_evaluated}{marker}",
            flush=True,
        )

        if stopper.should_stop:
            print(
                f"early stop: val PAUC has not improved for {config.patience} epochs "
                + f"(best was epoch {best_epoch}, PAUC {best_val_pauc:.4f}).",
                flush=True,
            )
            break

    _ = model.load_state_dict(best_state)
    return FitResult(
        reports=reports,
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_val_pauc=best_val_pauc,
    )
