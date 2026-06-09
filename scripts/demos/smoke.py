"""End-to-end smoke test + timing probe for the training pipeline.

No arguments. Run this BEFORE a real training job. It exercises every moving
part once -- connect, derive temporal spec + reference epoch, read train/val
seeds + pu_label FROM THE GRAPH, auto-discover the eval parquet for the
synthetic answer key, FIT THE FEATURE NORMALIZER on the train split, build both
loaders with the strict-inductive flags AND the normalizer, run a capped number
of real-sized batches through forward -> nnPU loss -> backward, a validation
pass, best-weight capture, and a throwaway checkpoint.

This probe is configured to match train.py EXACTLY on the two settings that
decide whether the model learns:
  * positive_weight=0.5 is passed to TrainConfig (without it the loss runs the
    textbook nnPU weight = pi ~ 1e-4, which starves the positive term and
    inverts the ranking -- the probe would then mislead you into thinking the
    model cannot learn when really the FIX just was not wired in).
  * the feature normalizer is fitted and passed to BOTH loaders (without it,
    raw unstandardized features produce runaway logits, another false alarm).
If this probe and train.py ever diverge on these, the probe is lying.

    python scripts/demos/smoke.py
"""

from collections.abc import Iterable, Iterator
from itertools import islice
import math
from pathlib import Path
import time
from typing import cast

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from mule_pattern_learner.device import select_device
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
from mule_pattern_learner.training.loop import TrainConfig, fit
from mule_pattern_learner.training.seeds import SeedPool, epoch_batches
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MASKS_DIR = Path("data/masks")
_COL_ACCOUNT_ID = "account_id"
_COL_TRUE_LABEL = "true_label"
_ACCOUNT_FEATURES = 31

_BATCH_SIZE = 1024
_POSITIVES_PER_BATCH = 64
_POSITIVE_WEIGHT = 0.5
_SYNTHETIC_MULE_COUNT = 216
_FEATURE_FETCH_CHUNK = 5_000
_VERTEX_ACCOUNT = "Account"

_PROBE_TRAIN_BATCHES = 10
_PROBE_VAL_SEEDS = 1024
_PROBE_EPOCHS = 2

_FULL_MAX_EPOCHS = 30


def _find_eval_parquet(masks_dir: Path) -> Path:
    candidates = sorted(
        masks_dir.glob("pu_labels_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no pu_labels_*.parquet in {masks_dir}; run the masking step first."
        )
    return candidates[0]


def _eval_labels(frame: pd.DataFrame, column: str) -> dict[str, int]:
    ids: list[str] = frame[_COL_ACCOUNT_ID].astype(str).tolist()
    values: list[int] = frame[column].astype(int).tolist()
    return dict(zip(ids, values))


def _fmt(seconds: float) -> str:
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f} min"
    if seconds < 172800.0:
        return f"{seconds / 3600.0:.1f} hr"
    return f"{seconds / 86400.0:.1f} days"


def main() -> None:
    print("[1/7] connecting ...")
    client = Client(Settings())
    print(f"      connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper
    device = select_device()
    print(f"      device: {device}")

    print("[2/7] deriving temporal spec + reference epoch ...")
    spec = derive_temporal_spec(client)
    max_bins = spec.max_bins
    edge_dim = spec.edge_dim
    reference_epoch_s = derive_reference_epoch_s(client)
    print(f"      max_bins={max_bins} edge_dim={edge_dim} ref_epoch={reference_epoch_s:.0f}")

    print("[3/7] reading train/val seeds + pu_label from the graph ...")
    train_seeds = fetch_split_seeds(client, "train")
    val_seeds = fetch_split_seeds(client, "val")
    pu_label_of = dict(train_seeds.pu_label_of)
    pu_label_of.update(val_seeds.pu_label_of)
    train_pool = SeedPool(
        positives=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 1),
        unlabeled=tuple(a for a, y in train_seeds.pu_label_of.items() if y == 0),
    )
    val_ids_all = list(val_seeds.account_ids)

    print("[4/7] auto-discovering eval parquet (synthetic answer key) ...")
    eval_path = _find_eval_parquet(_MASKS_DIR)
    eval_frame = pd.read_parquet(eval_path)
    true_label_of = _eval_labels(eval_frame, _COL_TRUE_LABEL)
    print(f"      {eval_path.name}")

    val_pos = [a for a in val_ids_all if true_label_of.get(a, 0) == 1]
    val_neg = [a for a in val_ids_all if true_label_of.get(a, 0) == 0]
    n_neg = max(0, _PROBE_VAL_SEEDS - len(val_pos))
    val_seed_ids: tuple[str, ...] = tuple(val_pos + val_neg[:n_neg])
    print(
        f"      train pos={train_pool.num_positives} unl={train_pool.num_unlabeled} "
        + f"val_seeds={len(val_seed_ids)} (val_pos={len(val_pos)})"
    )

    # Fit the feature normalizer on the TRAIN split, exactly as train.py does,
    # and pass it to BOTH loaders below. Without this the loaders serve raw
    # unstandardized features and the logits blow up (mean ~ -70, min ~ -700),
    # a false-alarm "inversion" that has nothing to do with the model.
    print("      fitting feature normalizer on train split ...")
    train_ids = tuple(train_seeds.account_ids)
    feature_rows: list[Tensor] = []
    for start in range(0, len(train_ids), _FEATURE_FETCH_CHUNK):
        chunk = list(train_ids[start : start + _FEATURE_FETCH_CHUNK])
        vertices = fetch_account_vertices(client, chunk)
        feature_rows.append(build_account_features(vertices).feats)
    normalizer = normalizer_from_features(torch.cat(feature_rows, dim=0))

    fanout = NeighborFanout()

    def mapper_to_ids(int_ids: list[int]) -> list[str]:
        return mapper.to_strings("Account", int_ids)

    def make_train_loader() -> Iterator[HeteroData]:
        batches = epoch_batches(
            train_pool,
            batch_size=_BATCH_SIZE,
            positives_per_batch=_POSITIVES_PER_BATCH,
            seed=1337,
        )
        for seed_batch in islice(batches, _PROBE_TRAIN_BATCHES):
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

    def make_val_loader() -> Iterator[HeteroData]:
        loader = backend.make_loader(
            seed_ids=val_seed_ids,
            reference_epoch_s=reference_epoch_s,
            max_bins=max_bins,
            fanout=fanout,
            batch_size=_BATCH_SIZE,
            shuffle=False,
            allow_val=True,
            allow_test=False,
            normalizer=normalizer,
        )
        yield from cast(Iterable[HeteroData], loader)

    print("[5/7] building model ...")
    model = MulePatternModel(account_in_dim=_ACCOUNT_FEATURES, edge_dim=edge_dim)
    raw_count = client.conn.getVertexCount(_VERTEX_ACCOUNT, realtime=True)
    if not isinstance(raw_count, int):
        raise TypeError(f"expected int account count, got {type(raw_count).__name__}")
    prior = _SYNTHETIC_MULE_COUNT / raw_count
    config = TrainConfig(
        prior=prior,
        max_epochs=_PROBE_EPOCHS,
        patience=99,
        eval_k=50,
        positive_weight=_POSITIVE_WEIGHT,
    )

    print(
        f"[6/7] timing {_PROBE_TRAIN_BATCHES} real-sized train batches x {_PROBE_EPOCHS} "
        + "epochs (watch the tqdm rate) ..."
    )
    start = time.perf_counter()
    result = fit(
        model=model,
        make_train_loader=make_train_loader,
        make_val_loader=make_val_loader,
        pu_label_of=pu_label_of,
        mapper_to_ids=mapper_to_ids,
        config=config,
        device=device,
    )
    elapsed = time.perf_counter() - start
    for r in result.reports:
        print(
            f"      epoch {r.epoch}: train_loss={r.train_loss:.4f} "
            + f"val_AP={r.val_average_precision:.4f}"
        )

    print("[7/7] saving a throwaway checkpoint ...")
    out_path = Path("models") / "smoke_test_checkpoint.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": result.best_state_dict, "edge_dim": edge_dim}, out_path)
    print(f"      wrote {out_path}")

    probe_val_batches = max(1, math.ceil(len(val_seed_ids) / _BATCH_SIZE))
    batches_run = (_PROBE_TRAIN_BATCHES + probe_val_batches) * _PROBE_EPOCHS
    per_batch = elapsed / max(batches_run, 1)

    unlabeled_per_batch = _BATCH_SIZE - _POSITIVES_PER_BATCH
    full_train_batches = max(1, math.ceil(train_pool.num_unlabeled / unlabeled_per_batch))
    full_val_batches = max(1, math.ceil(len(val_ids_all) / _BATCH_SIZE))
    per_epoch = (full_train_batches + full_val_batches) * per_batch

    print("\n" + "=" * 60)
    print("TIMING ESTIMATE (from measured per-batch cost)")
    print("=" * 60)
    print(f"  device                : {device}")
    print(f"  measured per batch    : {per_batch:.2f}s ({batches_run} batches in {_fmt(elapsed)})")
    print(f"  full train batches    : {full_train_batches}")
    print(f"  full val batches      : {full_val_batches}")
    print(f"  est. per epoch        : {_fmt(per_epoch)}")
    print(f"  est. {_FULL_MAX_EPOCHS} epochs (max)   : {_fmt(per_epoch * _FULL_MAX_EPOCHS)}")
    print(f"  est. ~10 epochs (early stop): {_fmt(per_epoch * 10)}")
    print("=" * 60)
    print("Note: probe batches start cold (connection warmup); the steady-state")
    print("tqdm rate is the more reliable figure if it differs from the average.")


if __name__ == "__main__":
    main()
