"""The batches training and scoring read, built from the cache (docs/design.md, "Training and
selection": batches are built once, shuffled across Mondays and labels, and reused).

- The training batches of a world are built once from its training examples (data.examples'
  training table: t_ms, account_id, label, weight) and serve every seed, prior, epoch and
  ablation. Their examples are shuffled once, by a generator of the fixed SHUFFLE_SEED, so
  that every batch mixes Mondays and labels; the model seed then orders the batches in each
  epoch (training.trainer). So the seeds vary only the initial weights, dropout and batch
  order, and the batches themselves are the same for every seed.
- The scoring batches of a world (selection, test, mule score) are built anew each time they
  are read, a Monday at a time in the order of their keys, since a whole world's examples
  do not fit in memory. They are built by the same builder with the same settings as the
  training batches, without labels and weights.

Every key is one the caller computed itself, from its own examples table: the cache is only
asked for the rows of those keys (the Keys chosen by their reader guard).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import math

import numpy as np
import pandas as pd

from ..subgraph.build import Batch, build, read_rows
from ..subgraph.cache import Cache

# The seed of the one shuffle of a world's training examples into batches.
SHUFFLE_SEED = 0
# The columns of the training examples, and of the keys of a scoring run.
TRAINING_COLUMNS = ("t_ms", "account_id", "label", "weight")
KEY_COLUMNS = ("t_ms", "account_id")
# The examples of one Monday whose rows a scoring run reads from the cache at a time.
READ_GROUP = 4_096


def _check_keys(keys: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [column for column in columns if column not in keys.columns]
    if missing:
        raise ValueError(f"The {what} lack the columns {missing}")
    if keys.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"The {what} hold an (account, Monday) twice")


def training_batches(cache: Cache, world: str, examples: pd.DataFrame, size: int) -> list[Batch]:
    """The training batches of a world's examples, at most size examples each: the examples
    shuffled once by SHUFFLE_SEED, each batch with its labels and weights."""
    if size < 1:
        raise ValueError(f"A batch holds at least one example, not {size}")
    _check_keys(examples, TRAINING_COLUMNS, "training examples")
    if examples.empty:
        raise ValueError("Training needs at least one example")
    labels = examples["label"].to_numpy()
    weights = examples["weight"].to_numpy(dtype=np.float64)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("A training label is 0 (unlabelled) or 1 (positive)")
    if not all(math.isfinite(weight) and weight > 0 for weight in weights):
        raise ValueError("A training weight is a finite number above 0")
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(examples))
    t_ms = examples["t_ms"].to_numpy(dtype=np.int64)
    accounts = examples["account_id"].to_numpy()
    keys = [(int(t_ms[index]), str(accounts[index])) for index in order]
    rows = read_rows(cache, keys)
    found: list[Batch] = []
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        found.append(
            build(
                rows[start : start + size],
                world,
                labels=[int(labels[index]) for index in chunk],
                weights=[float(weights[index]) for index in chunk],
            )
        )
    return found


class ScoringBatches:
    """The scoring batches of keys (t_ms, account_id) of a world, at most size examples each,
    built anew on every pass, Monday by Monday in increasing order, each Monday's accounts in
    increasing order. keys holds them in that order; the scores of a pass follow it."""

    def __init__(self, cache: Cache, world: str, keys: pd.DataFrame, size: int) -> None:
        if size < 1:
            raise ValueError(f"A batch holds at least one example, not {size}")
        _check_keys(keys, KEY_COLUMNS, "scoring keys")
        if keys.empty:
            raise ValueError("A scoring run needs at least one key")
        ordered = keys.loc[:, list(KEY_COLUMNS)].astype({"t_ms": "int64", "account_id": "str"})
        self.keys = ordered.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        self.cache = cache
        self.world = world
        self.size = size

    def __len__(self) -> int:
        """The examples of a pass."""
        return len(self.keys)

    def __iter__(self) -> Iterator[Batch]:
        t_ms = self.keys["t_ms"].to_numpy(dtype=np.int64)
        accounts = [str(account) for account in self.keys["account_id"]]
        for day in np.unique(t_ms).tolist():
            rows = np.flatnonzero(t_ms == day)
            for start in range(0, len(rows), READ_GROUP):
                group = [(day, accounts[row]) for row in rows[start : start + READ_GROUP]]
                found = read_rows(self.cache, group)
                for first in range(0, len(found), self.size):
                    yield build(found[first : first + self.size], self.world)
