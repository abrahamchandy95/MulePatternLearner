"""XGBoost, in a Python process of its own (docs/design.md, "Comparisons").

torch and XGBoost each bring an OpenMP runtime (on macOS, torch 2.12's own and the one of
XGBoost 3.2.0), and a process that holds both crashes in XGBoost's threads. The owner's answer
is a separate process, not a limit on the threads: every XGBoost fit and prediction, of the two
comparisons (comparisons.rankers) and of the clock probe (evaluation.clock_probe), runs here, in
a child that comparisons.process starts and that never imports torch, at XGBoost's full thread
count. This is the only module that imports xgboost, and it imports nothing that imports torch
(a test checks both in fresh interpreters, and an import contract checks the second).

The child is started as `python -m mule_pattern_learner.comparisons.boosting OPERATION INPUTS
OUTPUTS`: INPUTS is a folder of the arrays (.npy) and tables (.parquet) the operation reads,
named as OPERATIONS below read them, and OUTPUTS the folder it writes its arrays into. The
functions are ordinary functions too, which a process without torch may call directly."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import xgboost

from ..training.selection import Found
from .rounds import MAX_ROUNDS, ROUND_STEP, check_rounds, choose_rounds

type Arrays = dict[str, NDArray[Any]]
type Tables = dict[str, pd.DataFrame]


def fit(
    x: NDArray[np.floating],
    labels: NDArray[np.integer],
    weights: NDArray[np.floating],
    columns: Sequence[str],
) -> bytes:
    """XGBoost with its default settings and MAX_ROUNDS rounds on x, one label per row,
    each row with its weight: the booster as XGBoost saves it (UBJSON)."""
    if x.shape[1] != len(columns):
        raise ValueError(f"The vectors are {x.shape[1]} wide, not {len(columns)}")
    model = xgboost.XGBClassifier(n_estimators=MAX_ROUNDS)
    model.fit(pd.DataFrame(x, columns=list(columns)), np.asarray(labels), sample_weight=weights)
    return bytes(model.get_booster().save_raw("ubj"))


def _booster(raw: bytes) -> xgboost.Booster:
    booster = xgboost.Booster()
    booster.load_model(bytearray(raw))
    return booster


def margins(
    raw: bytes, x: NDArray[np.floating], columns: Sequence[str], rounds: int
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """For each round count of ROUND_STEP, 2 ROUND_STEP, ... up to rounds, the sum in float64
    of every row's tree outputs up to it: XGBoost's margin without its base score, which is
    the same for every row and so changes no ranking. The sum is taken ROUND_STEP rounds at a
    time (XGBoost's iteration_range), so one pass over the trees gives every round count of the
    choice, and every scoring sums the same blocks in the same order."""
    booster = _booster(raw)
    check_rounds(rounds, booster.num_boosted_rounds())
    data = xgboost.DMatrix(pd.DataFrame(x, columns=list(columns)), base_margin=np.zeros(len(x)))
    total = np.zeros(len(x))
    for start in range(0, rounds, ROUND_STEP):
        block = booster.predict(
            data, output_margin=True, iteration_range=(start, start + ROUND_STEP)
        )
        total = total + block.astype(np.float64)
        yield start + ROUND_STEP, total


def scores(
    raw: bytes, x: NDArray[np.floating], columns: Sequence[str], rounds: int
) -> NDArray[np.float64]:
    """Every row's score at a round count: margins' last sum."""
    found = np.zeros(len(x))
    for _, found in margins(raw, x, columns, rounds):
        pass
    return found


def choose(
    raw: bytes, x: NDArray[np.floating], columns: Sequence[str], keys: pd.DataFrame
) -> tuple[int, Found, tuple[tuple[int, float], ...]]:
    """The round count with the best selection score on the ranked selection keys (x a row per
    key, in their order), a larger count only when strictly better; what it found; and the
    score of every count tried (rounds.choose_rounds)."""
    return choose_rounds(margins(raw, x, columns, MAX_ROUNDS), keys)


def fold_scores(
    values: NDArray[np.float64],
    positive: NDArray[np.bool_],
    fold: NDArray[np.int64],
    seed: int,
) -> NDArray[np.float64]:
    """Each row's margin from XGBoost with its default settings, random_state seed, trained on
    the rows of the other folds: the clock probe's out-of-fold scores (evaluation.clock_probe)."""
    out = np.zeros(len(values))
    for held in np.unique(fold):
        train = fold != held
        model = xgboost.XGBClassifier(random_state=seed)
        model.fit(values[train], positive[train].astype(np.int64))
        out[~train] = np.asarray(
            model.predict(values[~train], output_margin=True), dtype=np.float64
        )
    return out


def _fit(arrays: Arrays, tables: Tables) -> Arrays:
    raw = fit(arrays["x"], arrays["labels"], arrays["weights"], arrays["columns"].tolist())
    return {"booster": np.frombuffer(raw, dtype=np.uint8)}


def _scores(arrays: Arrays, tables: Tables) -> Arrays:
    raw = arrays["booster"].tobytes()
    return {"scores": scores(raw, arrays["x"], arrays["columns"].tolist(), int(arrays["rounds"]))}


def _choose(arrays: Arrays, tables: Tables) -> Arrays:
    rounds, found, history = choose(
        arrays["booster"].tobytes(), arrays["x"], arrays["columns"].tolist(), tables["keys"]
    )
    return {
        "rounds": np.asarray(rounds),
        "accounts": np.array(found.accounts, dtype=np.str_),
        "weights": found.weights,
        "shares": found.shares,
        "history_rounds": np.array([count for count, _ in history], dtype=np.int64),
        "history_scores": np.array([score for _, score in history], dtype=np.float64),
    }


def _fold_scores(arrays: Arrays, tables: Tables) -> Arrays:
    return {
        "scores": fold_scores(
            arrays["values"], arrays["positive"], arrays["fold"], int(arrays["seed"])
        )
    }


# What the child can be asked, by name; comparisons.process asks for each by it.
OPERATIONS: dict[str, Callable[[Arrays, Tables], Arrays]] = {
    "fit": _fit,
    "scores": _scores,
    "choose": _choose,
    "fold_scores": _fold_scores,
}


def main(arguments: Sequence[str]) -> int:
    """Run the operation named by the first argument on the files of the folder named by the
    second, writing its arrays into the third; the process's exit status."""
    if len(arguments) != 3 or arguments[0] not in OPERATIONS:
        sys.stderr.write(
            f"usage: python -m {__package__}.boosting OPERATION INPUTS OUTPUTS, OPERATION one of "
            f"{', '.join(OPERATIONS)}\n"
        )
        return 2
    operation, inputs, outputs = arguments[0], Path(arguments[1]), Path(arguments[2])
    arrays = {path.stem: np.load(path, allow_pickle=False) for path in sorted(inputs.glob("*.npy"))}
    tables = {path.stem: pd.read_parquet(path) for path in sorted(inputs.glob("*.parquet"))}
    for name, array in OPERATIONS[operation](arrays, tables).items():
        np.save(outputs / f"{name}.npy", array, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
