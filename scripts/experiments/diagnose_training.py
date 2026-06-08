"""Find WHY training inverts a strong signal: instrument a few real train steps.

The hidden-mule eval proved the signal exists (device_share_cnt alone scores
AUC 0.84) yet the trained model ranks mules below random (AUC 0.43). That is a
training failure, not a data failure. Two mechanisms could cause it:

  H1 (loss): with a tiny class prior pi and few UNIQUE positives, the nnPU
     negative-risk correction may fire on most batches. When it fires, the step
     does gradient ASCENT on gamma * (-negative_risk) and DROPS the positive
     term entirely -- so the model is trained mostly by a term unrelated to
     ranking mules, collapsing logits toward a constant.
  H2 (architecture): the GATv2 message passing washes out or inverts each
     node's own device_share signal, so the model OUTPUT is worse than its
     INPUT features regardless of the loss.

This script runs a handful of REAL training batches (same forward / loss /
optimizer as loop.train_epoch) and logs, per batch:
  * fired        -- did the nnPU correction trigger (negative_risk < -beta)?
  * pos_risk     -- pi * mean positive surrogate  (the term that ranks mules up)
  * neg_risk     -- the corrected negative-risk term
  * ratio        -- pos_risk / |neg_risk|  (how swamped the positive signal is)
  * pos/unl logit mean on the batch's own seeds (collapse vs separation, live)

Read the output:
  * fired on most/all batches AND ratio << 1  -> H1 confirmed: the correction
    over-fires and starves the positive signal. Fix the loss/optimizer
    (more UNIQUE positives, lower lr, tune beta) -- NOT the architecture.
  * fired rarely, logits keep real spread, yet ordering stays backwards
    -> points at H2: instrument/fix message passing, not the loss.

Runs on the existing graph; trains only a few steps (minutes). No checkpoint is
saved and the graph is not modified.

    python scripts/experiments/diagnose_training.py
    python scripts/experiments/diagnose_training.py --steps 30
"""

from collections.abc import Iterator
import argparse
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

from mule_pattern_learner.features.nodes import (
    build_account_features,
    normalizer_from_features,
)
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.fetch import fetch_account_vertices
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.derivation import (
    derive_reference_epoch_s,
    derive_temporal_spec,
)
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.seeds import SeedPool, epoch_batches
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_BATCH_SIZE = 1024
_POSITIVES_PER_BATCH = 64
_RNG_SEED = 1337
_FEATURE_FETCH_CHUNK = 5_000
_DEFAULT_STEPS = 20
_LR = 1e-3
_WEIGHT_DECAY = 1e-4
_BETA = 0.0
_GAMMA = 1.0
# Positive-term weight: set equal to the value you pass to NonNegativePULoss in
# loop.py. None -> use the prior pi (textbook nnPU, the OLD behavior). Set to
# e.g. 0.5 to test the fix that stops positives being weighted into irrelevance.
_POSITIVE_WEIGHT: float | None = 0.5
_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_ACCOUNT: NodeType = "Account"


@dataclass(frozen=True, slots=True)
class _Args:
    steps: int


class _NodeStore(Protocol):
    n_id: Tensor
    x: Tensor
    batch_size: int


class _EdgeStore(Protocol):
    edge_index: Tensor
    edge_attr: Tensor


def _node_store(batch: HeteroData, key: NodeType) -> _NodeStore:
    return cast(_NodeStore, cast(object, batch[key]))


def _edge_store(batch: HeteroData, key: EdgeType) -> _EdgeStore:
    return cast(_EdgeStore, cast(object, batch[key]))


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(
        prog="diagnose-training",
        description="Instrument a few real training steps to locate the inversion.",
    )
    _ = parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        metavar="N",
        help=f"how many train batches to run (default {_DEFAULT_STEPS})",
    )
    ns = parser.parse_args()
    return _Args(steps=cast(int, ns.steps))


def _forward(model: MulePatternModel, batch: HeteroData) -> Tensor:
    account = _node_store(batch, _ACCOUNT)
    x_dict: dict[NodeType, Tensor] = {_ACCOUNT: account.x}
    node_counts: dict[NodeType, int] = {
        nt: int(_node_store(batch, nt).n_id.shape[0]) for nt in batch.node_types
    }
    edge_index_dict: dict[EdgeType, Tensor] = {
        et: _edge_store(batch, et).edge_index for et in batch.edge_types
    }
    edge_attr_dict: dict[EdgeType, Tensor] = {_HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr}
    return cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))


def main() -> None:
    args = _parse_args()

    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper

    spec = derive_temporal_spec(client)
    max_bins = spec.max_bins
    edge_dim = spec.edge_dim
    reference_epoch_s = derive_reference_epoch_s(client)

    train_seeds = fetch_split_seeds(client, "train")
    pu_label_of = dict(train_seeds.pu_label_of)
    train_pool = SeedPool(
        positives=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 1),
        unlabeled=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 0),
    )
    pi = train_pool.num_positives / max(train_pool.num_positives + train_pool.num_unlabeled, 1)
    print(
        f"train pos={train_pool.num_positives} (UNIQUE) "
        + f"unl={train_pool.num_unlabeled} | prior pi={pi:.6f}"
    )
    pos_weight = pi if _POSITIVE_WEIGHT is None else _POSITIVE_WEIGHT
    print(
        f"batches carry {_POSITIVES_PER_BATCH} positives sampled WITH replacement "
        + f"from {train_pool.num_positives} unique"
    )
    print(
        f"positive_weight = {pos_weight:.6f} "
        + ("(= prior pi, OLD behavior)" if _POSITIVE_WEIGHT is None else "(FIX active)")
    )

    train_ids = tuple(train_seeds.account_ids)
    feature_rows: list[Tensor] = []
    for start in range(0, len(train_ids), _FEATURE_FETCH_CHUNK):
        chunk = list(train_ids[start : start + _FEATURE_FETCH_CHUNK])
        vertices = fetch_account_vertices(client, chunk)
        feature_rows.append(build_account_features(vertices).feats)
    normalizer = normalizer_from_features(torch.cat(feature_rows, dim=0))

    fanout = NeighborFanout()
    model = MulePatternModel(account_in_dim=normalizer.mean.shape[0], edge_dim=edge_dim)
    _ = model.train()
    opt = AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)

    def make_train_loader() -> Iterator[HeteroData]:
        for seed_batch in epoch_batches(
            train_pool,
            batch_size=_BATCH_SIZE,
            positives_per_batch=_POSITIVES_PER_BATCH,
            seed=_RNG_SEED,
        ):
            loader = backend.make_loader(
                seed_ids=seed_batch,
                reference_epoch_s=reference_epoch_s,
                max_bins=max_bins,
                fanout=fanout,
                batch_size=len(seed_batch),
                shuffle=False,
                allow_val=False,
                allow_test=False,
                normalizer=normalizer,
            )
            yield from cast(Iterator[HeteroData], loader)

    print("=" * 72)
    print(f"running {args.steps} real training steps (beta={_BETA} gamma={_GAMMA})")
    print("=" * 72)
    print(
        f"{'step':>4} {'fired':>6} {'pos_risk':>10} {'neg_risk':>10} "
        + f"{'ratio':>8} {'pos_lgt':>8} {'unl_lgt':>8}"
    )

    fired_count = 0
    done = 0
    for batch in make_train_loader():
        account = _node_store(batch, _ACCOUNT)
        bsize = int(account.batch_size)
        n_id = cast("list[int]", account.n_id[:bsize].tolist())
        seeds = mapper.to_strings(_ACCOUNT, n_id)
        targets = torch.tensor([pu_label_of.get(s, 0) for s in seeds], dtype=torch.float32)

        logits = _forward(model, batch)
        seed_logits = logits[:bsize]

        # Recompute the nnPU terms separately (loss.py returns only the combined
        # value); same surrogates: l(+f)=sigmoid(-f), l(-f)=sigmoid(+f).
        positive = (targets == 1).to(seed_logits.dtype)
        unlabeled = (targets == 0).to(seed_logits.dtype)
        n_pos = torch.clamp(positive.sum(), min=1.0)
        n_unl = torch.clamp(unlabeled.sum(), min=1.0)
        l_pos = torch.sigmoid(-seed_logits)
        l_neg = torch.sigmoid(seed_logits)
        positive_risk = pos_weight * torch.sum(positive * l_pos) / n_pos
        negative_risk = (
            torch.sum(unlabeled * l_neg) / n_unl - pi * torch.sum(positive * l_neg) / n_pos
        )
        fired = bool(negative_risk.item() < -_BETA)
        if fired:
            train_loss = _GAMMA * (-negative_risk)
            fired_count += 1
        else:
            train_loss = positive_risk + negative_risk

        opt.zero_grad()
        _ = train_loss.backward()
        _ = opt.step()

        pos_l = seed_logits[positive.bool()]
        unl_l = seed_logits[unlabeled.bool()]
        pos_lm = float(pos_l.mean().item()) if pos_l.numel() else float("nan")
        unl_lm = float(unl_l.mean().item()) if unl_l.numel() else float("nan")
        pr = float(positive_risk.item())
        nr = float(negative_risk.item())
        ratio = pr / abs(nr) if nr != 0 else float("inf")
        print(
            f"{done:>4} {('YES' if fired else 'no'):>6} {pr:>10.6f} {nr:>10.6f} "
            + f"{ratio:>8.4f} {pos_lm:>8.3f} {unl_lm:>8.3f}",
            flush=True,
        )

        done += 1
        if done >= args.steps:
            break

    print("=" * 72)
    rate = fired_count / max(done, 1)
    print(f"correction fired on {fired_count}/{done} batches ({rate:.0%})")
    if rate > 0.8:
        print(
            "-> H1 LIKELY: correction over-fires; positive signal is starved. "
            + "Fix the LOSS/optimizer (more UNIQUE positives, lower lr, tune "
            + "beta), not the architecture. THIS is when generating more mules "
            + "helps -- more UNIQUE positives, not just resampling the same few."
        )
    else:
        print(
            "-> H1 weak: correction does not dominate. If ordering is still "
            + "backwards, suspect H2 (message passing). Next: probe the model "
            + "with message passing disabled vs the raw-feature 0.84 baseline."
        )


if __name__ == "__main__":
    main()
