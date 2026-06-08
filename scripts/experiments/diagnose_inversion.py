"""Diagnose WHY hidden-mule AUC came out below 0.5 (ranking inverted).

The eval, prior, and nnPU loss are all sign-correct (verified by reading them),
so a sub-random AUC is not a wiring bug -- the trained model genuinely orders
hidden mules below random accounts. This script finds the cause without
retraining or regenerating, using the test split's answer key (true_label).

It loads the saved checkpoint and scores the test split exactly as
evaluate_hidden.py does, but instead of recall@k it reports:

  1. CLASS-CONDITIONAL LOGITS -- mean/std of the model's logit for true mules
     vs non-mules. Two diagnoses fall out:
       * means nearly equal (gap ~ 0)   -> COLLAPSE: model says "negative" for
         everyone; ranking is noise. Cause is in training dynamics, not features.
       * mule mean clearly LOWER         -> INVERSION: model actively scores
         mules as less mule-like. Cause is a feature whose sign the model learned
         backwards, or a feature that genuinely runs opposite to muleness.

  2. PER-FEATURE CORRELATION with true_label (on standardized features, the same
     ones the model sees). Features with strongly NEGATIVE correlation are the
     anti-signal suspects. If the largest-magnitude correlations are all small
     (|r| < ~0.1), the generator is not producing separable mules -- and NO
     amount of extra data fixes that; the data itself lacks signal.

  3. BEST SINGLE-FEATURE AUC -- the AUC you'd get ranking by the one most
     mule-correlated feature alone. If this is comfortably > 0.5 while the
     model's AUC is < 0.5, the SIGNAL EXISTS and the model inverted/missed it
     (a training problem). If even the best feature is ~0.5, the signal is
     absent (a data problem). This is the single most decision-relevant number:
     it tells you whether to fix training or fix the generator.

    python scripts/experiments/diagnose_inversion.py
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from tqdm import tqdm

from mule_pattern_learner.features.nodes import (
    FeatureNormalizer,
    account_feature_names,
)
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MASKS_DIR = Path("data/masks")
_MODELS_DIR = Path("models")
_COL_ACCOUNT_ID = "account_id"
_COL_TRUE_LABEL = "true_label"
_EVAL_SPLIT = "test"
_BATCH_SIZE = 1024
_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")
_ACCOUNT: NodeType = "Account"
_SMALL_R = 0.10


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


def _find_latest(models_dir: Path, pattern: str) -> Path:
    candidates = sorted(
        models_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no {pattern} in {models_dir}.")
    return candidates[0]


def _labels(frame: pd.DataFrame, column: str) -> dict[str, int]:
    ids: list[str] = frame[_COL_ACCOUNT_ID].astype(str).tolist()
    values: list[int] = frame[column].astype(int).tolist()
    return dict(zip(ids, values))


def _auc(scores: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    # Mann-Whitney U form of ROC-AUC; no sklearn dependency.
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[labels == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper

    ckpt_path = _find_latest(_MODELS_DIR, "mule_model_*.pt")
    checkpoint = cast(dict[str, object], torch.load(ckpt_path, weights_only=False))
    edge_dim = cast(int, checkpoint["edge_dim"])
    max_bins = cast(int, checkpoint["max_bins"])
    reference_epoch_s = cast(float, checkpoint["reference_epoch_s"])
    account_in_dim = cast(int, checkpoint["account_in_dim"])
    state_dict = cast("dict[str, torch.Tensor]", checkpoint["model_state_dict"])
    normalizer = FeatureNormalizer(
        mean=cast(torch.Tensor, checkpoint["feature_mean"]),
        std=cast(torch.Tensor, checkpoint["feature_std"]),
    )
    print(f"checkpoint: {ckpt_path.name}")

    model = MulePatternModel(account_in_dim=account_in_dim, edge_dim=edge_dim)
    _ = model.load_state_dict(state_dict)
    _ = model.eval()

    split_seeds = fetch_split_seeds(client, _EVAL_SPLIT)
    seed_ids = split_seeds.account_ids
    print(f"{_EVAL_SPLIT} split: {len(seed_ids)} accounts")

    eval_path = _find_latest(_MASKS_DIR, "pu_labels_*.parquet")
    frame = pd.read_parquet(eval_path)
    true_label_of = _labels(frame, _COL_TRUE_LABEL)

    fanout = NeighborFanout()
    loader = backend.make_loader(
        seed_ids=seed_ids,
        reference_epoch_s=reference_epoch_s,
        max_bins=max_bins,
        fanout=fanout,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        allow_val=True,
        allow_test=True,
        normalizer=normalizer,
    )

    logit_rows: list[float] = []
    feat_rows: list[NDArray[np.float64]] = []
    seen_ids: list[str] = []
    bar = tqdm(cast(Iterable[HeteroData], loader), unit="batch", desc="  score")
    with torch.no_grad():
        for batch in bar:
            account = _node_store(batch, _ACCOUNT)
            bsize = int(account.batch_size)
            n_id = cast("list[int]", account.n_id[:bsize].tolist())
            seeds = mapper.to_strings(_ACCOUNT, n_id)
            x_dict: dict[NodeType, Tensor] = {_ACCOUNT: account.x}
            node_counts: dict[NodeType, int] = {
                nt: int(_node_store(batch, nt).n_id.shape[0]) for nt in batch.node_types
            }
            edge_index_dict: dict[EdgeType, Tensor] = {
                et: _edge_store(batch, et).edge_index for et in batch.edge_types
            }
            edge_attr_dict: dict[EdgeType, Tensor] = {
                _HAS_PAID: _edge_store(batch, _HAS_PAID).edge_attr
            }
            logits = cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))
            logit_rows.extend(cast("list[float]", logits[:bsize].tolist()))
            # the seed rows' standardized features, as the model saw them
            feat_rows.append(account.x[:bsize].detach().cpu().numpy().astype(np.float64))
            seen_ids.extend(seeds)

    logits_arr: NDArray[np.float64] = np.asarray(logit_rows, dtype=np.float64)
    feats_arr: NDArray[np.float64] = np.concatenate(feat_rows, axis=0)
    true_arr: NDArray[np.int64] = np.asarray([true_label_of[s] for s in seen_ids], dtype=np.int64)

    is_pos = true_arr == 1
    is_neg = true_arr == 0
    n_pos = int(is_pos.sum())

    print("\n" + "=" * 60)
    print("1. CLASS-CONDITIONAL LOGITS")
    print("=" * 60)
    pos_mean = float(logits_arr[is_pos].mean()) if n_pos else float("nan")
    neg_mean = float(logits_arr[is_neg].mean())
    pos_std = float(logits_arr[is_pos].std()) if n_pos else float("nan")
    neg_std = float(logits_arr[is_neg].std())
    print(f"  true mules ({n_pos:>4}): logit mean={pos_mean:+.4f} std={pos_std:.4f}")
    print(f"  non-mules  ({int(is_neg.sum()):>4}): logit mean={neg_mean:+.4f} std={neg_std:.4f}")
    gap = pos_mean - neg_mean
    print(f"  gap (mule - non) = {gap:+.4f}")
    if abs(gap) < 0.05:
        print("  -> COLLAPSE: logits ~identical across classes; ranking is noise.")
    elif gap < 0:
        print("  -> INVERSION: mules score LOWER. Model learned the signal backwards.")
    else:
        print("  -> mules score higher (expected direction); inversion is elsewhere.")

    print("\n" + "=" * 60)
    print("2. PER-FEATURE CORRELATION with true_label (standardized features)")
    print("=" * 60)
    names = account_feature_names()
    n_feat = feats_arr.shape[1]
    corrs: list[tuple[str, float]] = []
    for j in range(n_feat):
        col = feats_arr[:, j]
        if float(col.std()) == 0.0:
            r = 0.0
        else:
            r = float(np.corrcoef(col, true_arr.astype(np.float64))[0, 1])
        name = names[j] if j < len(names) else f"feat_{j}"
        corrs.append((name, r))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    for name, r in corrs[:12]:
        flag = "  <-- anti-signal" if r < -_SMALL_R else ""
        print(f"  {name:<28} r={r:+.4f}{flag}")
    strongest = abs(corrs[0][1]) if corrs else 0.0

    print("\n" + "=" * 60)
    print("3. BEST SINGLE-FEATURE AUC vs the model's AUC")
    print("=" * 60)
    model_auc = _auc(logits_arr, true_arr)
    best_name, best_r = corrs[0] if corrs else ("none", 0.0)
    best_j = names.index(best_name) if best_name in names else 0
    feat_dir = feats_arr[:, best_j] * (1.0 if best_r >= 0 else -1.0)
    feat_auc = _auc(feat_dir.astype(np.float64), true_arr)
    print(f"  model logit AUC            : {model_auc:.4f}")
    print(f"  best single-feature AUC    : {feat_auc:.4f}  (via {best_name})")
    print("-" * 60)
    if strongest < _SMALL_R:
        print("  VERDICT: no feature separates mules (all |r| small). The signal")
        print("  is ABSENT in the data -> this is a GENERATOR problem. More data at")
        print("  the same generator settings will NOT help; mules are not distinct.")
    elif feat_auc > 0.55 and model_auc < 0.5:
        print("  VERDICT: signal EXISTS (a feature ranks mules > 0.55) but the model")
        print("  scores below random -> this is a TRAINING problem, not a data one.")
        print("  Fixing training beats regenerating; scaling now would waste hours.")
    else:
        print("  VERDICT: mixed. Inspect the correlations above before deciding.")


if __name__ == "__main__":
    main()
