"""Reproduce the eval tail-batch KeyError in minutes, with a full traceback.

The full eval crashes on the LAST batch (the 138-account tail of the 91018-account
test split) after ~6 hours of fetching all 356 batches. This script builds a loader
over ONLY the tail accounts, so it reaches the crashing batch in one or two
fetches (minutes), and prints the COMPLETE traceback (tqdm disabled, bare
try/except) so we finally see the exact file and line that raises KeyError:
'Account'.

Run:
    python repro_tail.py --reveal-prevalence 0.2
"""

import argparse
import traceback
from typing import cast

import torch

from mule_pattern_learner.features.nodes import FeatureNormalizer
from mule_pattern_learner.pyg.backend import TigerGraphRemoteBackend
from mule_pattern_learner.pyg.neighbors import NeighborFanout
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.training.seeds_source import fetch_split_seeds

_MODELS_DIR_GLOB = "models/mule_model_*.pt"
_EVAL_SPLIT = "test"
_BATCH_SIZE = 256
# how many of the LAST accounts to include; >= the tail size (138) so the loader's
# final batch is exactly the tail batch that crashes, reached in ~1-2 fetches.
_TAIL_N = 200


def _latest_ckpt() -> str:
    from pathlib import Path

    cands = sorted(Path("models").glob("mule_model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("no checkpoint found")
    return str(cands[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    _ = p.add_argument("--reveal-prevalence", type=float, required=True)
    _ = p.add_argument("--seed", type=int, default=1337)
    ns = p.parse_args()
    _ = ns.reveal_prevalence  # only needed for parity; repro does not score

    client = Client(Settings())
    print(f"connected: {client.graphname}", flush=True)
    backend = TigerGraphRemoteBackend(client)

    ckpt_path = _latest_ckpt()
    checkpoint = cast(dict[str, object], torch.load(ckpt_path, weights_only=False))
    max_bins = cast(int, checkpoint["max_bins"])
    reference_epoch_s = cast(float, checkpoint["reference_epoch_s"])
    normalizer = FeatureNormalizer(
        mean=cast(torch.Tensor, checkpoint["feature_mean"]),
        std=cast(torch.Tensor, checkpoint["feature_std"]),
    )
    print(f"checkpoint: {ckpt_path}", flush=True)

    split_seeds = fetch_split_seeds(client, _EVAL_SPLIT)
    all_ids = split_seeds.account_ids
    tail_ids = all_ids[-_TAIL_N:]
    print(
        f"{_EVAL_SPLIT} split has {len(all_ids)} accounts; using last {len(tail_ids)}", flush=True
    )
    print(f"(full-run tail batch is the final {len(all_ids) % _BATCH_SIZE} accounts)", flush=True)

    fanout = NeighborFanout()
    loader = backend.make_loader(
        seed_ids=tail_ids,
        reference_epoch_s=reference_epoch_s,
        max_bins=max_bins,
        fanout=fanout,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        allow_val=True,
        allow_test=True,
        normalizer=normalizer,
    )

    # NO tqdm. Iterate raw so any exception prints its FULL traceback.
    print("iterating tail batches (no tqdm) ...", flush=True)
    i = 0
    it = iter(loader)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            print(f"done: {i} batches, no crash", flush=True)
            break
        except Exception:
            print(f"\n=== CRASH on batch index {i} -- FULL TRACEBACK BELOW ===", flush=True)
            traceback.print_exc()
            print("=== END TRACEBACK ===", flush=True)
            raise
        # report what the batch actually contains, so we see the tail batch's shape
        node_types = list(batch.node_types)
        edge_types = [str(e) for e in batch.edge_types]
        print(f"batch {i}: node_types={node_types} edge_types={edge_types}", flush=True)
        i += 1


if __name__ == "__main__":
    main()
