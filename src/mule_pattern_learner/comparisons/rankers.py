"""The two XGBoost comparisons (docs/design.md, "Comparisons"): XGBoost on the scored account's
own row, which asks what graph learning adds at all, and XGBoost on the graph features, which
asks what message passing adds beyond hand-made graph features (comparisons.features).

- Each trains on the GNN's training examples, with their labels and weights, which come with
  the vectors in the order of their keys (comparisons.features), and is scored on the same
  selection Mondays from the same keys.
- It learns identified (label 1) against unlabelled (0), with each label's weights rescaled to
  equal totals: when labels are selected completely at random, a classifier of labelled
  against unlabelled ranks like one of positive against negative (Elkan and Noto, 2008). Each
  label's weights total half the number of examples, so their mean is 1 and XGBoost's
  defaults, which read the summed weight of a leaf, keep their usual scale.
- It uses XGBoost's default settings, with at most MAX_ROUNDS rounds, and is trained once, with
  no seed (the owner, 24 October 2026): at those defaults (every row and column used by every
  tree) a seed changes no tree, five seeds gave byte-identical boosters, so a ranker is one
  model with no seed spread, and the GNN's five seeds are compared with it. The round count is
  chosen among 10, 20, ..., 1,000 by the selection score on the selection world
  (training.selection), as the GNN chooses its epochs, a larger count only when strictly better.
- An account's score is the sum of the outputs of the trees up to the chosen round, in
  float64: XGBoost's margin without its base score, which is the same for every account and so
  changes no ranking (comparisons.boosting).
- Every XGBoost fit and prediction runs in a Python process of its own that never imports torch
  (comparisons.process), so this module, which imports torch to save and load a ranker's file,
  never imports xgboost: a booster is its saved bytes here.

A ranker's file, named for the ranker alone (model_file), holds its booster, its round count,
its selection score and what it found, under the contract lock with its columns among the
locked fields (training.contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

from ..config import SubgraphConfig
from ..training import contract
from ..training.contract import Lock
from ..training.selection import Found
from .features import GRAPH_COLUMNS, OWN_COLUMNS, Vectors
from .process import choose, fit, scores
from .rounds import MAX_ROUNDS, ROUND_STEP

# The two rankers, by name.
OWN = "xgboost_own"
GRAPH = "xgboost_graph"
COLUMNS = {OWN: OWN_COLUMNS, GRAPH: GRAPH_COLUMNS}


def matrix(name: str, vectors: Vectors) -> NDArray[np.float32]:
    """The vectors a ranker reads."""
    if name not in COLUMNS:
        raise ValueError(f"No XGBoost ranker is named {name!r}")
    return vectors.own if name == OWN else vectors.graph


@dataclass(frozen=True)
class Trees:
    """A trained XGBoost ranker: its name and booster (as XGBoost saves it), the round count
    selection chose, its score and what it found, and the score of every round count tried.
    It is one model: there is no seed."""

    name: str
    booster: bytes
    rounds: int
    score: float
    found: Found
    history: tuple[tuple[int, float], ...]

    def scores(self, vectors: Vectors) -> NDArray[np.float64]:
        """The score of every example of vectors at the chosen round count."""
        return scores(self.booster, matrix(self.name, vectors), COLUMNS[self.name], self.rounds)


def train(name: str, training: Vectors, selection: Vectors, keys: pd.DataFrame) -> Trees:
    """The ranker of name trained once on the training vectors with the labels and weights
    they carry, its round count chosen on the selection vectors of the ranked selection keys
    (in their order)."""
    x_training, x_selection = matrix(name, training), matrix(name, selection)
    if not training.labelled:
        raise ValueError(
            "The training vectors carry no labels and weights: they are not of training batches"
        )
    expected = list(zip(keys["t_ms"].tolist(), keys["account_id"].tolist(), strict=True))
    if list(selection.keys) != [(int(t), str(a)) for t, a in expected]:
        raise ValueError("The selection vectors are not those of the selection keys")
    booster = fit(x_training, training.labels, training.weights, COLUMNS[name])
    rounds, chosen, history = choose(booster, x_selection, COLUMNS[name], keys)
    return Trees(name, booster, rounds, chosen.score, chosen, history)


def lock(name: str, subgraph: SubgraphConfig) -> Lock:
    """The contract lock of an XGBoost ranker: the fields of every ranker's lock but a seed
    count, which it has none of, with the community flag set when its vectors hold the community
    block, and its columns and round counts."""
    return contract.lock(
        name,
        community=name == GRAPH,
        subgraph=subgraph,
        extra={"columns": COLUMNS[name], "max_rounds": MAX_ROUNDS, "round_step": ROUND_STEP},
    )


def model_file(folder: Path, name: str) -> Path:
    """The model file of an XGBoost ranker in a run's results folder: named for the ranker
    alone, since it is trained once and has no seed."""
    return folder / f"{name}.pt"


def save(trees: Trees, path: Path, current: Lock) -> None:
    """Write a ranker's file under its lock."""
    contract.save(
        path,
        current,
        {
            "ranker": trees.name,
            "booster": trees.booster,
            "rounds": trees.rounds,
            "score": trees.score,
            "accounts": list(trees.found.accounts),
            "weights": torch.from_numpy(trees.found.weights.copy()),
            "shares": torch.from_numpy(trees.found.shares.copy()),
            "history": [list(entry) for entry in trees.history],
        },
    )


def load(path: Path, current: Lock) -> Trees:
    """A ranker's file, refused when its lock is not current."""
    stored = contract.load(path, current)
    found_back = Found(
        tuple(stored["accounts"]),
        stored["weights"].numpy().astype(np.float64),
        stored["shares"].numpy().astype(np.float64),
    )
    history = tuple((int(rounds), float(score)) for rounds, score in stored["history"])
    return Trees(
        stored["ranker"],
        bytes(stored["booster"]),
        stored["rounds"],
        stored["score"],
        found_back,
        history,
    )
