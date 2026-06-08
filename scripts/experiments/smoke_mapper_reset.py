"""Smoke-check the per-batch mapper reset before any full training run.

Runs the REAL train-loader path (same wiring as train.py) for a few batches and
prints, per batch, how many Account nodes the shared mapper holds. With the
per-batch reset in place this count stays bounded (one batch's worth) instead of
growing every batch as it did before. It also confirms the sampler -> feature
fetch round-trip still succeeds after a reset: each yielded batch carries real
Account features, which can only be built if the mapper resolved this batch's
integer ids back to string ids correctly.

This is the cheap stand-in for a full epoch: it touches the graph for only a
handful of batches (seconds, not hours) and answers the one question a static
read could not -- does resetting the mapper each batch break the feature fetch.

    python scripts/experiments/smoke_mapper_reset.py
    python scripts/experiments/smoke_mapper_reset.py --batches 5

Read the output:
  * "account nodes in mapper" stays in the same ballpark across batches
    (does NOT climb 5k -> 10k -> 15k) -> the leak is capped.
  * every batch prints "features OK [N x F]" with no NodeIDMapperError
    -> reset-then-fetch is safe at num_workers=0 (the loader is synchronous).
A NodeIDMapperError here means the reset wiped ids an in-flight fetch needed;
stop and revisit before running a full epoch.
"""

from collections.abc import Iterable, Iterator
import argparse
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from mule_pattern_learner.features.nodes import (
    build_account_features,
    normalizer_from_features,
)
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.fetch import fetch_account_vertices
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.derivation import (
    derive_reference_epoch_s,
    derive_temporal_spec,
)
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.seeds import SeedPool, epoch_batches

_BATCH_SIZE = 512
_POSITIVES_PER_BATCH = 32
_RNG_SEED = 1337
_FEATURE_FETCH_CHUNK = 5_000
_DEFAULT_BATCHES = 3
_ACCOUNT = "Account"


@dataclass(frozen=True, slots=True)
class _Args:
    batches: int


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(
        prog="smoke-mapper-reset",
        description="Run the train loader for a few batches and watch mapper size.",
    )
    _ = parser.add_argument(
        "--batches",
        type=int,
        default=_DEFAULT_BATCHES,
        metavar="N",
        help=f"how many batches to pull (default {_DEFAULT_BATCHES})",
    )
    ns = parser.parse_args()
    return _Args(batches=cast(int, ns.batches))


def main() -> None:
    args = _parse_args()

    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper

    spec = derive_temporal_spec(client)
    max_bins = spec.max_bins
    reference_epoch_s = derive_reference_epoch_s(client)
    print(f"derived: max_bins={max_bins} reference_epoch_s={reference_epoch_s:.0f}")

    from mule_pattern_learner.training.seeds_source import fetch_split_seeds

    train_seeds = fetch_split_seeds(client, "train")
    train_pool = SeedPool(
        positives=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 1),
        unlabeled=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 0),
    )
    print(f"graph seeds: train pos={train_pool.num_positives} " + f"unl={train_pool.num_unlabeled}")

    # Fit the normalizer on the train split, exactly as train.py does, so the
    # loader path here matches training (the fetch + transform is what we test).
    train_ids = tuple(train_seeds.account_ids)
    feature_rows: list[Tensor] = []
    for start in range(0, len(train_ids), _FEATURE_FETCH_CHUNK):
        chunk = list(train_ids[start : start + _FEATURE_FETCH_CHUNK])
        vertices = fetch_account_vertices(client, chunk)
        feature_rows.append(build_account_features(vertices).feats)
    train_features = torch.cat(feature_rows, dim=0)
    normalizer = normalizer_from_features(train_features)
    print(f"normalizer fit on {train_features.shape[0]} train accounts")

    fanout = NeighborFanout()

    # Build ONE epoch's worth of seed batches, but only consume the first few.
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
            yield from cast(Iterable[HeteroData], loader)

    print("=" * 60)
    print(f"pulling {args.batches} batches (watch the mapper size)")
    print("=" * 60)

    pulled = 0
    for batch in make_train_loader():
        account_store = cast(object, batch[_ACCOUNT])
        x = cast(Tensor, getattr(account_store, "x"))
        n_mapper = mapper.num_nodes(_ACCOUNT)
        print(
            f"batch {pulled}: account nodes in mapper = {n_mapper:>6} | "
            + f"features OK [{x.shape[0]} x {x.shape[1]}]",
            flush=True,
        )
        pulled += 1
        if pulled >= args.batches:
            break

    print("=" * 60)
    print(
        "done. if the mapper count stayed bounded and every batch printed "
        + "features OK, the reset is safe -- proceed to a real run."
    )


if __name__ == "__main__":
    main()
