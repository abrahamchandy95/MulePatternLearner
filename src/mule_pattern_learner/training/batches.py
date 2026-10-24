"""The batches training and scoring read, built from the cache (docs/design.md, "Training and
selection": batches are built once, shuffled across Mondays and labels, and reused).

- The training batches of a world are built once from its training examples (data.examples'
  training table: t_ms, account_id, label, weight) and serve every seed, prior, epoch and
  ablation. Their examples are shuffled once, by a generator of the shuffle seed (config's
  training.shuffle_seed), so that every batch mixes Mondays and labels; the model seed then
  orders the batches in each epoch (training.trainer). So the seeds vary only the initial
  weights, dropout and batch order, and the batches themselves are the same for every seed.
- The scoring batches of a world (selection, test, mule score) are built anew each time they
  are read, a Monday at a time in the order of their keys, since a whole world's examples
  do not fit in memory. They are built by the same builder with the same settings as the
  training batches, without labels and weights. Selection reads them once per epoch of every
  prior, seed and ablation (training.trainer.selection_passes), so the time spent reading rows
  and building is counted (ScoringBatches.seconds over ScoringBatches.built), for mule bench
  to multiply by the passes before whole-world selection is accepted.
- Proof of concept (25 October 2026): with keep, the first whole pass keeps the batches it
  built and every later pass replays them, so each example of a world is built once for every
  epoch, prior, seed and ablation, and for the test world's every run (docs/design.md,
  "Proof-of-concept settings (25 October 2026)"). The kept batches are the same ones a fresh
  pass builds, and they are held in memory, so keep is for the sampled worlds of the proof of
  concept; the full design's whole worlds do not fit, and keep is off for them.

Every key is one the caller computed itself, from its own examples table: the cache is only
asked for the rows of those keys (the Keys chosen by their reader guard).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import math
import time

import numpy as np
import pandas as pd

from ..subgraph.build import Batch, build, read_rows
from ..subgraph.cache import Cache

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


def training_batches(
    cache: Cache, world: str, examples: pd.DataFrame, size: int, *, seed: int
) -> list[Batch]:
    """The training batches of a world's examples, at most size examples each: the examples
    shuffled once by a generator of seed (config's training.shuffle_seed), each batch with its
    labels and weights."""
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
    order = np.random.default_rng(seed).permutation(len(examples))
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
    increasing order, or, with keep, built by the first whole pass and replayed by every later
    one. keys holds them in that order; the scores of a pass follow it."""

    def __init__(
        self, cache: Cache, world: str, keys: pd.DataFrame, size: int, *, keep: bool = False
    ) -> None:
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
        self.keep = keep
        # The batches of the first whole pass, when keep says to hold them.
        self._kept: list[Batch] | None = None
        # The examples built over every pass so far, and the seconds spent reading their rows
        # and building them (not the consumer's own time between batches).
        self.built = 0
        self.seconds = 0.0

    def __len__(self) -> int:
        """The examples of a pass."""
        return len(self.keys)

    def __iter__(self) -> Iterator[Batch]:
        if self._kept is not None:
            yield from self._kept
            return
        # A pass that stops before its end keeps nothing, so a kept pass is always whole.
        held: list[Batch] = []
        t_ms = self.keys["t_ms"].to_numpy(dtype=np.int64)
        accounts = [str(account) for account in self.keys["account_id"]]
        # The keys run Monday after Monday, so each Monday's examples are one slice of them.
        starts = np.flatnonzero(np.r_[True, t_ms[1:] != t_ms[:-1]])
        for first, last in zip(starts.tolist(), np.r_[starts[1:], len(t_ms)].tolist(), strict=True):
            day = int(t_ms[first])
            for start in range(first, last, READ_GROUP):
                group = [
                    (day, account) for account in accounts[start : min(start + READ_GROUP, last)]
                ]
                began = time.perf_counter()
                found = read_rows(self.cache, group)
                self.seconds += time.perf_counter() - began
                for chunk in range(0, len(found), self.size):
                    began = time.perf_counter()
                    batch = build(found[chunk : chunk + self.size], self.world)
                    self.seconds += time.perf_counter() - began
                    self.built += len(batch.keys)
                    if self.keep:
                        held.append(batch)
                    yield batch
        if self.keep:
            self._kept = held
