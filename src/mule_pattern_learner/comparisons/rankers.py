"""The two XGBoost comparisons (docs/design.md, "Comparisons"): XGBoost on the scored account's
own row, which asks what graph learning adds at all, and XGBoost on the graph features, which
asks what message passing adds beyond hand-made graph features (comparisons.features).

- Each trains on the GNN's training examples, with their labels and weights, and is scored on
  the same selection Mondays from the same keys.
- It learns identified (label 1) against unlabelled (0), with each label's weights rescaled to
  equal totals: when labels are selected completely at random, a classifier of labelled
  against unlabelled ranks like one of positive against negative (Elkan and Noto, 2008). Each
  label's weights total half the number of examples, so their mean is 1 and XGBoost's
  defaults, which read the summed weight of a leaf, keep their usual scale.
- It uses XGBoost's default settings, random_state the model seed, with at most MAX_ROUNDS
  rounds; the round count is chosen among 10, 20, ..., 1,000 by the selection score on the
  selection world (training.selection), as the GNN chooses its epochs, a larger count only
  when strictly better.
- An account's score is the sum of the outputs of the trees up to the chosen round, in
  float64: XGBoost's margin without its base score, which is the same for every account and so
  changes no ranking. The sum is taken ROUND_STEP rounds at a time (XGBoost's iteration_range),
  so one pass over the trees scores every round count of the choice, and every scoring sums
  the same blocks in the same order.

A ranker's file holds its booster, its seed, its round count, its selection score and what it
found, under the contract lock with its columns among the locked fields (training.contract).

On macOS the pip wheels of torch and XGBoost each bring an OpenMP runtime of their own, and
XGBoost crashes the process when it runs beside torch's runtime with more than one thread, so
there every function that runs XGBoost refuses to unless OMP_NUM_THREADS is 1, set before torch
is imported. One thread changes no result, only the time XGBoost takes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
import xgboost

from ..config import SubgraphConfig, TrainingConfig
from ..training import contract
from ..training.contract import Lock
from ..training.selection import Found, found
from .features import GRAPH_COLUMNS, OWN_COLUMNS, Vectors

# The two rankers, by name.
OWN = "xgboost_own"
GRAPH = "xgboost_graph"
COLUMNS = {OWN: OWN_COLUMNS, GRAPH: GRAPH_COLUMNS}
# The rounds trained, and the step of the round counts the choice tries.
MAX_ROUNDS = 1_000
ROUND_STEP = 10


def require_one_openmp_thread() -> None:
    """Refuse to run XGBoost on macOS unless OMP_NUM_THREADS is 1 (see the module's
    docstring)."""
    if sys.platform == "darwin" and os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError(
            "On macOS, XGBoost crashes beside torch's OpenMP runtime with more than one thread: "
            "set OMP_NUM_THREADS=1 before the run starts"
        )


def matrix(name: str, vectors: Vectors) -> NDArray[np.float64]:
    """The vectors a ranker reads."""
    if name not in COLUMNS:
        raise ValueError(f"No XGBoost ranker is named {name!r}")
    return vectors.own if name == OWN else vectors.graph


def equal_totals(labels: NDArray[np.integer], weights: NDArray[np.floating]) -> NDArray[np.float64]:
    """The weights rescaled so that each label's total is half the number of examples."""
    found = np.asarray(weights, dtype=np.float64).copy()
    for label in (0, 1):
        chosen = labels == label
        if not chosen.any():
            raise ValueError(f"The training examples hold no label {label}")
        found[chosen] *= (len(found) / 2) / found[chosen].sum()
    return found


def fit(
    x: NDArray[np.float64],
    labels: NDArray[np.integer],
    weights: NDArray[np.floating],
    seed: int,
    columns: Sequence[str],
) -> xgboost.Booster:
    """XGBoost with its default settings, random_state seed, MAX_ROUNDS rounds, on x with each
    label's weights rescaled to equal totals."""
    require_one_openmp_thread()
    if x.shape[1] != len(columns):
        raise ValueError(f"The vectors are {x.shape[1]} wide, not {len(columns)}")
    model = xgboost.XGBClassifier(n_estimators=MAX_ROUNDS, random_state=seed)
    frame = pd.DataFrame(x, columns=list(columns))
    model.fit(frame, np.asarray(labels), sample_weight=equal_totals(labels, weights))
    return model.get_booster()


def margins(
    booster: xgboost.Booster, x: NDArray[np.float64], columns: Sequence[str], rounds: int
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """For each round count of ROUND_STEP, 2 ROUND_STEP, ... up to rounds, the sum in float64
    of every account's tree outputs up to it."""
    require_one_openmp_thread()
    if rounds % ROUND_STEP or not 0 < rounds <= booster.num_boosted_rounds():
        raise ValueError(f"A round count is a multiple of {ROUND_STEP} up to the trees trained")
    data = xgboost.DMatrix(pd.DataFrame(x, columns=list(columns)), base_margin=np.zeros(len(x)))
    total = np.zeros(len(x))
    for start in range(0, rounds, ROUND_STEP):
        block = booster.predict(
            data, output_margin=True, iteration_range=(start, start + ROUND_STEP)
        )
        total = total + block.astype(np.float64)
        yield start + ROUND_STEP, total


def scores(
    booster: xgboost.Booster, x: NDArray[np.float64], columns: Sequence[str], rounds: int
) -> NDArray[np.float64]:
    """Every account's score at a round count."""
    found = np.zeros(len(x))
    for _, found in margins(booster, x, columns, rounds):
        pass
    return found


@dataclass(frozen=True)
class Trees:
    """A trained XGBoost ranker: its name, seed and booster, the round count selection chose,
    its score and what it found, and the score of every round count tried."""

    name: str
    seed: int
    booster: xgboost.Booster
    rounds: int
    score: float
    found: Found
    history: tuple[tuple[int, float], ...]

    def scores(self, vectors: Vectors) -> NDArray[np.float64]:
        """The score of every example of vectors at the chosen round count."""
        return scores(self.booster, matrix(self.name, vectors), COLUMNS[self.name], self.rounds)


def choose(
    booster: xgboost.Booster,
    x: NDArray[np.float64],
    columns: Sequence[str],
    keys: pd.DataFrame,
) -> tuple[int, Found, tuple[tuple[int, float], ...]]:
    """The round count with the best selection score on the ranked selection keys (x a row per
    key, in their order), a larger count only when strictly better; what it found; and the
    score of every count tried."""
    best: tuple[int, Found] | None = None
    history: list[tuple[int, float]] = []
    for rounds, total in margins(booster, x, columns, MAX_ROUNDS):
        result = found(keys, total)
        history.append((rounds, result.score))
        if best is None or result.score > best[1].score:
            best = (rounds, result)
    assert best is not None
    return best[0], best[1], tuple(history)


def train(
    name: str,
    training: Vectors,
    labels: NDArray[np.integer],
    weights: NDArray[np.floating],
    selection: Vectors,
    keys: pd.DataFrame,
    seed: int,
) -> Trees:
    """The ranker of name trained on the training vectors with their labels and weights, its
    round count chosen on the selection vectors of the ranked selection keys (in their
    order)."""
    x_training, x_selection = matrix(name, training), matrix(name, selection)
    expected = list(zip(keys["t_ms"].tolist(), keys["account_id"].tolist(), strict=True))
    if list(selection.keys) != [(int(t), str(a)) for t, a in expected]:
        raise ValueError("The selection vectors are not those of the selection keys")
    booster = fit(x_training, labels, weights, seed, COLUMNS[name])
    rounds, chosen, history = choose(booster, x_selection, COLUMNS[name], keys)
    return Trees(name, seed, booster, rounds, chosen.score, chosen, history)


def lock(name: str, subgraph: SubgraphConfig, training: TrainingConfig) -> Lock:
    """The contract lock of an XGBoost ranker: the fields of every ranker's lock, with the
    community flag set when its vectors hold the community block, and its columns and round
    counts."""
    return contract.lock(
        name,
        community=name == GRAPH,
        subgraph=subgraph,
        training=training,
        extra={"columns": COLUMNS[name], "max_rounds": MAX_ROUNDS, "round_step": ROUND_STEP},
    )


def save(trees: Trees, path: Path, current: Lock) -> None:
    """Write a ranker's file under its lock."""
    contract.save(
        path,
        current,
        {
            "ranker": trees.name,
            "seed": trees.seed,
            "booster": bytes(trees.booster.save_raw("ubj")),
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
    booster = xgboost.Booster()
    booster.load_model(bytearray(stored["booster"]))
    found_back = Found(
        tuple(stored["accounts"]),
        stored["weights"].numpy().astype(np.float64),
        stored["shares"].numpy().astype(np.float64),
    )
    history = tuple((int(rounds), float(score)) for rounds, score in stored["history"])
    return Trees(
        stored["ranker"],
        stored["seed"],
        booster,
        stored["rounds"],
        stored["score"],
        found_back,
        history,
    )
