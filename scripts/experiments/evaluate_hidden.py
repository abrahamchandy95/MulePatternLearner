"""Synthetic-only evaluation: does the trained model find HIDDEN mules?

SYNTHETIC DATA ONLY. This step needs the answer key (true_label / bucket) from
the masking parquet, which does not exist in production -- so this is separate
from training, run after it. It loads the latest checkpoint, scores a split's
accounts, and reports generalization to mules that were never labelled.

No config files. Loads the most recent models/mule_model_*.pt and scores the
TEST split (the held-out dark rings -- the hardest generalization test) against
the answer key for the REQUESTED reveal prevalence, printing hidden-recall@k
plus AP/AUC against true labels.

--reveal-prevalence is required and must match the prevalence the loaded
checkpoint was trained at: the checkpoint does not record its own prevalence, so
scoring it against a different-prevalence answer key (a different mule
population and split) yields a meaningless but plausible-looking number. The
answer-key path is built from the prevalence, never auto-discovered.

    python scripts/.../evaluate_hidden.py --reveal-prevalence 0.2
"""

from collections.abc import Iterable
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from tqdm import tqdm

from mule_pattern_learner.features.nodes import FeatureNormalizer
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.model import MulePatternModel
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.metrics import (
    evaluate_hidden,
    evaluate_summary,
    format_summary_report,
    save_generalization_chart,
    save_summary_chart,
)
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MASKS_DIR = Path("data/masks")
_MODELS_DIR = Path("models")
# Summary report (gains/lift table + chart) is written here, next to the
# checkpoints it evaluates. Created on demand.
_EVAL_OUT_DIR = Path("models/eval")
_COL_ACCOUNT_ID = "account_id"
_COL_TRUE_LABEL = "true_label"
_COL_BUCKET = "bucket"

# Which split to evaluate generalization on. test holds the dark rings (mules
# whose whole ring was hidden) -- the strongest test of finding unseen mules.
_EVAL_SPLIT = "test"
_EVAL_K = 100
# Eval scoring is read-only: batch size affects only memory / speed, never the
# scores or metrics. The test split holds the dark rings -- the densest, hubbiest
# accounts in the graph -- so neighbor-sampled subgraphs here are the largest
# anywhere in the pipeline. 256 keeps peak memory well under the OOM ceiling that
# killed training; raise it only after watching RSS on a completing run.
_BATCH_SIZE = 256

# Capacity sweep for the summary report's gains/lift chart. A dense range of
# alert rates (fractions of accounts reviewed) brackets the realistic AML band
# from a true-bank alert rate (~0.05%) up to a generous review budget (5%), so
# the curve is smooth and each viewer reads off their own operating point. The
# printed table uses a coarse subset for legibility; the chart uses the full
# range.
_SUMMARY_ALERT_RATES_TABLE: tuple[float, ...] = (0.0005, 0.001, 0.005, 0.01, 0.02)
_SUMMARY_ALERT_RATES_CHART: tuple[float, ...] = tuple(
    round(0.0005 + (0.05 - 0.0005) * i / 59, 6) for i in range(60)
)


_HAS_PAID: EdgeType = ("Account", "HAS_PAID", "Account")


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


def _find_latest_checkpoint(models_dir: Path) -> Path:
    candidates = sorted(
        models_dir.glob("mule_model_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no mule_model_*.pt in {models_dir}; train first.")
    return candidates[0]


def _eval_parquet_path(masks_dir: Path, seed: int, reveal_prevalence: float) -> Path:
    # Build the EXACT answer-key path for the requested seed and reveal
    # prevalence, matching masking.py's naming (pu_labels_seed{seed}_p{prev}).
    # This deliberately replaces the old latest-by-mtime auto-discovery: that
    # silently paired whatever parquet was written most recently with the loaded
    # checkpoint, so re-masking at a new prevalence (e.g. p0.2) caused a model
    # trained at p0.1 to be scored against a p0.2 answer key -- a different mule
    # population and split, producing a meaningless but plausible-looking number.
    # The checkpoint does not record its prevalence, so the match cannot be
    # auto-verified; requiring the prevalence explicitly, and failing if the file
    # is absent, makes the mismatch impossible to hit silently.
    path = masks_dir / f"pu_labels_seed{seed}_p{reveal_prevalence}.parquet"
    if not path.exists():
        available = sorted(p.name for p in masks_dir.glob("pu_labels_*.parquet"))
        raise FileNotFoundError(
            f"answer key {path.name} not found in {masks_dir}. "
            + f"available: {available}. Pass --reveal-prevalence / --seed matching "
            + "the prevalence the loaded checkpoint was TRAINED at; scoring a model "
            + "against a different-prevalence key gives meaningless results."
        )
    return path


def _labels(frame: pd.DataFrame, column: str) -> dict[str, int]:
    ids: list[str] = frame[_COL_ACCOUNT_ID].astype(str).tolist()
    values: list[int] = frame[column].astype(int).tolist()
    return dict(zip(ids, values))


@dataclass(frozen=True, slots=True)
class _Args:
    reveal_prevalence: float
    seed: int


def _parse_args() -> _Args:
    # reveal-prevalence is REQUIRED (no default): the answer key must match the
    # prevalence the loaded checkpoint was trained at, and the checkpoint does
    # not record its own prevalence, so there is no safe default to guess. Seed
    # defaults to the project seed (masking's default), overridable for parity.
    p = argparse.ArgumentParser(description="Synthetic hidden-mule generalization eval.")
    _ = p.add_argument(
        "--reveal-prevalence",
        type=float,
        required=True,
        help="reveal prevalence of the answer key to score against; MUST match "
        + "the prevalence the loaded checkpoint was trained at.",
    )
    _ = p.add_argument("--seed", type=int, default=1337, help="masking seed (default 1337).")
    ns = p.parse_args()
    return _Args(
        reveal_prevalence=cast(float, ns.reveal_prevalence),
        seed=cast(int, ns.seed),
    )


def main() -> None:
    args = _parse_args()
    client = Client(Settings())
    print(f"connected: {client.graphname}")
    backend = TigerGraphRemoteBackend(client)
    mapper = backend.mapper

    # load the checkpoint + its metadata (dims/recency it was trained with)
    ckpt_path = _find_latest_checkpoint(_MODELS_DIR)
    checkpoint = cast(dict[str, object], torch.load(ckpt_path, weights_only=False))
    edge_dim = cast(int, checkpoint["edge_dim"])
    max_bins = cast(int, checkpoint["max_bins"])
    reference_epoch_s = cast(float, checkpoint["reference_epoch_s"])
    account_in_dim = cast(int, checkpoint["account_in_dim"])
    state_dict = cast("dict[str, torch.Tensor]", checkpoint["model_state_dict"])
    # Rebuild the training-fit feature standardization so eval scores features
    # exactly as training saw them.
    normalizer = FeatureNormalizer(
        mean=cast(torch.Tensor, checkpoint["feature_mean"]),
        std=cast(torch.Tensor, checkpoint["feature_std"]),
    )
    print(
        f"checkpoint: {ckpt_path.name} (best_val_pauc={cast(float, checkpoint['best_val_pauc']):.4f})"
    )

    model = MulePatternModel(account_in_dim=account_in_dim, edge_dim=edge_dim)
    _ = model.load_state_dict(state_dict)
    _ = model.eval()

    # seeds for the eval split, from the graph
    split_seeds = fetch_split_seeds(client, _EVAL_SPLIT)
    seed_ids = split_seeds.account_ids
    print(f"{_EVAL_SPLIT} split: {len(seed_ids)} accounts")

    # answer key from the parquet (synthetic only); path built from the
    # explicitly-requested prevalence + seed, never auto-discovered by mtime.
    eval_path = _eval_parquet_path(_MASKS_DIR, args.seed, args.reveal_prevalence)
    frame = pd.read_parquet(eval_path)
    true_label_of = _labels(frame, _COL_TRUE_LABEL)
    bucket_of = _labels(frame, _COL_BUCKET)
    print(f"answer key: {eval_path.name}")

    fanout = NeighborFanout()

    def mapper_to_ids(int_ids: list[int]) -> list[str]:
        return mapper.to_strings("Account", int_ids)

    # test-time sampling may use the whole graph (allow_val=True, allow_test=True)
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

    scores: list[float] = []
    seen_ids: list[str] = []
    eval_batches = max(1, math.ceil(len(seed_ids) / _BATCH_SIZE))
    bar = tqdm(
        cast(Iterable[HeteroData], loader),
        total=eval_batches,
        unit="batch",
        miniters=1,
        desc="  eval",
    )
    with torch.no_grad():
        for batch in bar:
            # A degenerate tail batch can come back without the Account node type
            # at all (no seeds survived sampling); accessing batch["Account"]
            # would raise KeyError: 'Account'. Skip it -- there is nothing to
            # score and it contributes no seeds to the metrics. (The edge-attr
            # guard below handles the related sparse-batch case for HAS_PAID.)
            if "Account" not in batch.node_types:
                continue
            account = _node_store(batch, "Account")
            bsize = int(account.batch_size)
            n_id = cast("list[int]", account.n_id[:bsize].tolist())
            seeds = mapper_to_ids(n_id)
            x_dict: dict[NodeType, Tensor] = {"Account": account.x}
            node_counts: dict[NodeType, int] = {
                nt: int(_node_store(batch, nt).n_id.shape[0]) for nt in batch.node_types
            }
            edge_index_dict: dict[EdgeType, Tensor] = {
                et: _edge_store(batch, et).edge_index for et in batch.edge_types
            }
            # HAS_PAID edge features only exist when the batch sampled HAS_PAID
            # edges. A sparse tail batch (isolated accounts) may have none,
            # leaving the edge store without edge_attr; forcing the key raises
            # AttributeError. HeteroConv runs the HAS_PAID conv only when the
            # type is present in edge_index_dict, so omitting it here is correct.
            edge_attr_dict: dict[EdgeType, Tensor] = {}
            if _HAS_PAID in batch.edge_types:
                edge_attr_dict[_HAS_PAID] = _edge_store(batch, _HAS_PAID).edge_attr
            logits = cast(Tensor, model(x_dict, edge_index_dict, edge_attr_dict, node_counts))
            probs = cast("list[float]", torch.sigmoid(logits[:bsize]).tolist())
            scores.extend(probs)
            seen_ids.extend(seeds)

    scores_arr: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    true_arr: NDArray[np.int64] = np.asarray([true_label_of[s] for s in seen_ids], dtype=np.int64)
    bucket_arr: NDArray[np.int64] = np.asarray([bucket_of[s] for s in seen_ids], dtype=np.int64)

    result = evaluate_hidden(scores_arr, true_arr, bucket_arr, k=_EVAL_K)
    print("\n" + "=" * 60)
    print(f"HIDDEN-MULE GENERALIZATION on the {_EVAL_SPLIT} split (synthetic)")
    print("=" * 60)
    print(f"  accounts scored        : {len(seen_ids)}")
    print(f"  true mules             : {result.num_true_positives}")
    print(f"  hidden mules           : {result.num_hidden_positives}")
    print(f"  hidden_recall@{result.k:<4}    : {result.hidden_recall_at_k:.4f}")
    print(f"  AP (vs true labels)    : {result.average_precision_true:.4f}")
    print(f"  AUC (vs true labels)   : {result.roc_auc_true:.4f}")
    print("=" * 60)
    print("hidden_recall@k is the headline: of mules the model was NEVER told")
    print("about, the fraction it ranked in the top-k. High = it learned the")
    print("pattern, not the few labelled examples.")

    # ── Summary report: gains/lift across review capacity + chart ──
    # The single-k recall above answers "did it find hidden mules at one cutoff".
    # The summary adds the operating curve an analyst team actually faces: recall,
    # precision, and lift across a sweep of alert rates, plus PR-AUC as the
    # imbalance-robust headline (ROC-AUC reported but de-emphasised). Precision
    # and lift are trustworthy ONLY because this is synthetic (true-label answer
    # key); on real PU data they are biased and only recall / PR-AUC carry over.
    _EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = evaluate_summary(scores_arr, true_arr, alert_rates=_SUMMARY_ALERT_RATES_TABLE)
    print("\n" + format_summary_report(summary))

    # write the printed table to a text file alongside the chart
    table_path = _EVAL_OUT_DIR / f"summary_{_EVAL_SPLIT}.txt"
    _ = table_path.write_text(format_summary_report(summary) + "\n")

    # the chart uses the dense sweep for smooth curves; the provenance note
    # records which checkpoint produced the scores so the figure is never
    # mistaken for a different run's results.
    summary_chart = evaluate_summary(scores_arr, true_arr, alert_rates=_SUMMARY_ALERT_RATES_CHART)
    chart_path = _EVAL_OUT_DIR / f"summary_{_EVAL_SPLIT}.png"
    _ = save_summary_chart(
        summary_chart,
        str(chart_path),
        title=f"Mule detection - gains across review capacity ({_EVAL_SPLIT} split)",
        note=f"scores from checkpoint {ckpt_path.name}; synthetic answer key {eval_path.name}",
    )
    print(f"\nsummary report -> {table_path}")
    print(f"gains/lift chart -> {chart_path}")

    # ── Generalization chart: the hidden-vs-revealed story ──
    # The gains chart above is the operating-point (staffing) view against TRUE
    # labels and does NOT separate hidden mules from revealed ones. This chart
    # answers the project's thesis directly -- did the model rank mules it was
    # NEVER labelled (hidden) alongside the ones it saw (revealed)? -- via three
    # panels: score distribution by bucket, hidden-recall vs. review depth
    # against the achievable ceiling, and the precision-recall curve. Same
    # provenance note so the figure is tied to its checkpoint.
    generalization_path = _EVAL_OUT_DIR / f"generalization_{_EVAL_SPLIT}.png"
    _ = save_generalization_chart(
        scores_arr,
        true_arr,
        bucket_arr,
        str(generalization_path),
        primary_k=_EVAL_K,
        title=f"Mule detection - generalization to hidden mules ({_EVAL_SPLIT} split)",
        note=f"scores from checkpoint {ckpt_path.name}; synthetic answer key {eval_path.name}",
    )
    print(f"generalization chart -> {generalization_path}")


if __name__ == "__main__":
    main()
