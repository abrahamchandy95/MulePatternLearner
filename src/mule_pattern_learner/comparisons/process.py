"""Starting the process XGBoost runs in (docs/design.md, "Comparisons"; comparisons.boosting says
why it is a process of its own).

Each function here takes what the function of the same name in comparisons.boosting takes and
gives what it gives, but runs it in a child Python process: the arrays and tables go into a
temporary folder as .npy and Parquet files, the child (a fresh interpreter, started as
`python -m` of comparisons.boosting) reads them and writes its arrays back, and the parent
reads those. The child never imports torch, so it runs at XGBoost's full thread count beside a
parent that has torch loaded. A child that fails raises XGBoostProcessError, with the end of its
error output.

This module imports neither xgboost nor torch, so a process that holds torch may import it, and
a process without torch may call it too (the tests do both). The folder of a job is made under
the system's temporary folder (TMPDIR) and removed afterwards, even when the child fails."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..training.selection import Found
from .rounds import check_rounds, equal_totals

# The module the child runs, and how many lines of its error output a failure shows.
MODULE = "mule_pattern_learner.comparisons.boosting"
ERROR_LINES = 12


class XGBoostProcessError(RuntimeError):
    """The process XGBoost runs in failed."""


def error_lines(errors: str) -> list[str]:
    """The lines of a child's error output that say what went wrong: the last exception of
    its last traceback (the line that names it and the lines after it), up to ERROR_LINES; the
    last lines when it printed no traceback."""
    lines = [line for line in errors.splitlines() if line.strip()]
    for start in range(len(lines) - 1, -1, -1):
        if lines[start].startswith("Traceback (most recent call last)"):
            first = start + 1
            while first < len(lines) and lines[first].startswith(" "):
                first += 1
            return lines[first : first + ERROR_LINES]
    return lines[-ERROR_LINES:]


def exit_message(operation: str, code: int, errors: str) -> str:
    """Why a child that ran operation failed: how it ended (an exit status, or the signal that
    killed it when code is negative) and what its error output says."""
    if code < 0:
        try:
            how = f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            how = f"was killed by signal {-code}"
    else:
        how = f"exited with status {code}"
    shown = "\n".join(error_lines(errors)) or "it printed no error"
    return f"The XGBoost process for {operation} {how}:\n{shown}"


def run(
    operation: str,
    arrays: Mapping[str, NDArray[Any]],
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, NDArray[Any]]:
    """Run an operation of comparisons.boosting.OPERATIONS in a child process on arrays and
    tables, each written to a file named for its key; the arrays it gives back by name."""
    with tempfile.TemporaryDirectory(prefix="mule-xgboost-") as name:
        folder = Path(name)
        inputs, outputs = folder / "in", folder / "out"
        inputs.mkdir()
        outputs.mkdir()
        for key, array in arrays.items():
            np.save(inputs / f"{key}.npy", array, allow_pickle=False)
        for key, table in (tables or {}).items():
            table.to_parquet(inputs / f"{key}.parquet", index=False)
        # Run from the job's own folder, so no module of the caller's folder is picked up.
        done = subprocess.run(
            [sys.executable, "-m", MODULE, operation, str(inputs), str(outputs)],
            cwd=folder,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise XGBoostProcessError(exit_message(operation, done.returncode, done.stderr))
        return {path.stem: np.load(path, allow_pickle=False) for path in outputs.glob("*.npy")}


def fit(
    x: NDArray[np.floating],
    labels: NDArray[np.integer],
    weights: NDArray[np.floating],
    columns: Sequence[str],
) -> bytes:
    """XGBoost with its default settings on x, one label per row, the weight of each label
    rescaled to equal totals (rounds.equal_totals), in a child process: the booster."""
    if x.shape[1] != len(columns):
        raise ValueError(f"The vectors are {x.shape[1]} wide, not {len(columns)}")
    found = run(
        "fit",
        {
            "x": x,
            "labels": np.asarray(labels),
            "weights": equal_totals(np.asarray(labels), weights),
            "columns": np.array(list(columns), dtype=np.str_),
        },
    )
    return found["booster"].tobytes()


def scores(
    booster: bytes, x: NDArray[np.floating], columns: Sequence[str], rounds: int
) -> NDArray[np.float64]:
    """Every row's score at a round count, from a child process (boosting.scores)."""
    check_rounds(rounds)
    found = run(
        "scores",
        {
            "booster": np.frombuffer(booster, dtype=np.uint8),
            "x": x,
            "columns": np.array(list(columns), dtype=np.str_),
            "rounds": np.asarray(rounds),
        },
    )
    return found["scores"].astype(np.float64)


def choose(
    booster: bytes, x: NDArray[np.floating], columns: Sequence[str], keys: pd.DataFrame
) -> tuple[int, Found, tuple[tuple[int, float], ...]]:
    """The round count with the best selection score on the ranked selection keys, chosen in
    a child process (boosting.choose): the count, what it found and the score of every count
    tried. Only the keys' Monday, account, class and weight are read."""
    found = run(
        "choose",
        {
            "booster": np.frombuffer(booster, dtype=np.uint8),
            "x": x,
            "columns": np.array(list(columns), dtype=np.str_),
        },
        {"keys": keys.loc[:, ["t_ms", "account_id", "positive", "weight"]]},
    )
    history = tuple(
        zip(found["history_rounds"].tolist(), found["history_scores"].tolist(), strict=True)
    )
    chosen = Found(
        tuple(str(account) for account in found["accounts"].tolist()),
        found["weights"].astype(np.float64),
        found["shares"].astype(np.float64),
    )
    return int(found["rounds"]), chosen, history


def fold_scores(
    values: NDArray[np.float64],
    positive: NDArray[np.bool_],
    fold: NDArray[np.int64],
    seed: int,
) -> NDArray[np.float64]:
    """Each row's margin from XGBoost trained on the other folds, from a child process
    (boosting.fold_scores)."""
    found = run(
        "fold_scores",
        {
            "values": values,
            "positive": positive,
            "fold": fold,
            "seed": np.asarray(seed),
        },
    )
    return found["scores"].astype(np.float64)
