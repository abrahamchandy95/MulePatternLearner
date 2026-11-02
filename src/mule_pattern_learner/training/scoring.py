"""Scoring with a saved model (docs/design.md, "Training and selection", "Which model scores",
and "Commands", mule score).

mule score uses the full model of the seed with the best selection score, never one chosen by
test: of the model files of the full model, the one whose stored selection score is highest,
the lowest seed among equals. A model file built under another contract lock than the current
query versions and settings is refused, and its differing fields named (training.contract).

The scored keys are built into batches by the same builder and settings as training's, and
each logit is ranked as a raw logit in float64 within its week, highest first, since a
float32 sigmoid tied every logit above about 17.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from ..subgraph.switches import Switch
from .batches import ScoringBatches
from .contract import Lock
from .trainer import Saved, load_trained, logits


def best_seed(paths: Sequence[Path], lock: Lock, device: torch.device | str = "cpu") -> Saved:
    """Of the model files at paths, all under lock and of one ranker, the one with the best
    selection score, the lowest seed among equals."""
    if not paths:
        raise ValueError("mule score needs at least one model file")
    saved = [load_trained(path, lock, device) for path in paths]
    if len({model.ranker for model in saved}) != 1:
        raise ValueError("The model files to choose among are of different rankers")
    if len({model.seed for model in saved}) != len(saved):
        raise ValueError("Two model files to choose among are of one seed")
    return min(saved, key=lambda model: (-model.score, model.seed))


def score(
    saved: Saved,
    batches: ScoringBatches,
    switch: Switch | None = None,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """The logit of every key of batches under the model, and its rank within its week (1 the
    highest; tied logits share the best rank), by week and account."""
    keys, found = logits(saved.model, batches, switch, torch.device(device))
    frame = pd.DataFrame(
        {
            "t_ms": pd.Series([t_ms for t_ms, _ in keys], dtype="int64"),
            "account_id": pd.Series([account for _, account in keys], dtype="str"),
            "logit": pd.Series(found, dtype="float64"),
        }
    )
    frame["rank"] = frame.groupby("t_ms")["logit"].rank(method="min", ascending=False)
    frame["rank"] = frame["rank"].astype("int64")
    return frame.sort_values(["t_ms", "account_id"]).reset_index(drop=True)
