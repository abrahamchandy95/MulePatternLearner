"""The examples of docs/design.md ("Training examples", and "Evaluation" for the test
window): (account, as-of time) pairs on a window's weekly dates, each with a label and a
weight, one frame per window with the columns of COLUMNS.

- Training. On each date, a revealed mule is positive (label 1) from its onset until its
  discovery if its account is open; each mule weighs 1 in all, shared equally among its
  dates. Unlabelled accounts (label 0) are drawn from the other accounts open that date,
  unlabelled_per_positive per positive (all of them when there are fewer), and share what
  the date's positives weigh, so both labels weigh the same on every date and the date
  tells nothing. A revealed mule's dates in the 30 days before its onset are dropped, and
  on its earlier dates it may be drawn as unlabelled.
- Validation and test. On each date, every mule active that date is positive with weight
  1: the revealed mules for validation, the answer key's for test. The other open accounts
  are a uniform sample of scored_sample of them, each weighing the accounts it stands for.
  A mule discovered by the date is not ranked.

A mule is active from its onset until its discovery, if it has one. Every draw depends only
on the seed, the date and the accounts it draws from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..tigergraph.executor import QueryExecutor, account_pages
from ..tigergraph.queries import (
    LIST_POPULATION,
    Phase,
    PopulationAccount,
    parse_population,
    population_params,
)
from .frames import frame
from .grid import DAY_MS, Window
from .supervision import read_supervision

POSITIVE, UNLABELLED = 1, 0
# The days before a revealed mule's onset on which it is no example.
PRE_ONSET_DAYS = 30
# The columns of the examples and their types.
COLUMNS = {
    "as_of_ms": "int64",
    "account_id": "str",
    "split": "int64",
    "label": "int64",
    "weight": "float64",
}


def read_population(executor: QueryExecutor, as_of_ms: int, phase: Phase) -> pd.DataFrame:
    """The accounts open at as_of_ms of the customers phase sees, accounts with no payments
    among them, one row each by account: account_id, account_type, opened_ms and split."""
    pages = account_pages(executor, LIST_POPULATION, population_params(as_of_ms, phase))
    return frame(PopulationAccount, [row for page in pages for row in parse_population(page)])


def training_examples(executor: QueryExecutor, window: Window, config: DataConfig) -> pd.DataFrame:
    """The training window's examples, from the mules revealed before its labels' cutoff."""
    revealed = read_supervision(executor, _cutoff(window), window.phase)
    # From then on a revealed mule is never unlabelled: it is dropped, then positive.
    labelled_from = revealed.mule_onset_ms - PRE_ONSET_DAYS * DAY_MS
    parts: list[pd.DataFrame] = []
    for as_of_ms in window.as_of_dates():
        population = read_population(executor, as_of_ms, window.phase)
        positive = population.account_id.isin(_active(revealed, as_of_ms))
        labelled = revealed.account_id[labelled_from <= as_of_ms]
        pool = population[~population.account_id.isin(labelled)]
        drawn = _draw(pool, config.unlabelled_per_positive * int(positive.sum()), config, as_of_ms)
        parts += [
            _rows(population[positive], as_of_ms, POSITIVE, 0.0),
            _rows(drawn, as_of_ms, UNLABELLED, 0.0),
        ]
    return _weighed(_joined(parts))


def _weighed(examples: pd.DataFrame) -> pd.DataFrame:
    """Training examples with their weights: each mule's positives share 1, and each date's
    unlabelled share what the date's positives weigh."""
    positive = examples.label == POSITIVE
    dates = examples[positive].groupby("account_id").as_of_ms.transform("size")
    examples.loc[positive, "weight"] = 1.0 / dates
    weighs = examples[positive].groupby("as_of_ms").weight.sum()
    unlabelled = examples[~positive]
    drawn = unlabelled.groupby("as_of_ms").as_of_ms.transform("size")
    examples.loc[~positive, "weight"] = unlabelled.as_of_ms.map(weighs) / drawn
    return examples


def validation_examples(
    executor: QueryExecutor, window: Window, config: DataConfig
) -> pd.DataFrame:
    """The validation window's examples, from the mules revealed before its labels' cutoff."""
    revealed = read_supervision(executor, _cutoff(window), window.phase)
    return _scored(executor, window, revealed, config)


def evaluation_examples(
    executor: QueryExecutor, window: Window, truth: pd.DataFrame, config: DataConfig
) -> pd.DataFrame:
    """The test window's examples, from the mules of the answer key, truth (as
    data.answer_key.read_answer_key gives it). Every account they rank must have a known
    truth, 0 or 1, and every mule an onset not after its discovery."""
    mules = truth[truth.is_mule == 1]
    late = (mules.discovered_ms > 0) & (mules.discovered_ms < mules.mule_onset_ms)
    wrong = mules.account_id[(mules.mule_onset_ms <= 0) | late]
    if not wrong.empty:
        raise ValueError(f"The answer key gives {', '.join(wrong)} no onset by their discovery")
    known = set(truth.account_id[truth.is_mule >= 0])
    return _scored(executor, window, mules, config, known)


def _cutoff(window: Window) -> int:
    if window.labels_before_ms is None:
        raise ValueError("The window has no labels' cutoff: it uses the full answer key")
    return window.labels_before_ms


def _active(mules: pd.DataFrame, as_of_ms: int) -> pd.Series:
    """The accounts of mules (with mule_onset_ms, and discovered_ms or 0 if never found)
    active at as_of_ms: begun by it and not discovered by it."""
    found = mules.discovered_ms
    active = (mules.mule_onset_ms <= as_of_ms) & ((found == 0) | (as_of_ms < found))
    return mules.account_id[active]


def _scored(
    executor: QueryExecutor,
    window: Window,
    mules: pd.DataFrame,
    config: DataConfig,
    known: set[str] | None = None,
) -> pd.DataFrame:
    """The active mules and a weighted sample of the other open accounts on each date of
    window; every account must be in known, when it is given."""
    discovered = mules.discovered_ms
    parts: list[pd.DataFrame] = []
    for as_of_ms in window.as_of_dates():
        population = read_population(executor, as_of_ms, window.phase)
        if known is not None and not population.account_id.isin(known).all():
            raise ValueError(f"The answer key does not know every account open at {as_of_ms}")
        active = population.account_id.isin(_active(mules, as_of_ms))
        found = mules.account_id[(discovered > 0) & (discovered <= as_of_ms)]
        others = population[~active & ~population.account_id.isin(found)]
        drawn = _draw(others, config.scored_sample, config, as_of_ms)
        parts += [
            _rows(population[active], as_of_ms, POSITIVE, 1.0),
            _rows(drawn, as_of_ms, UNLABELLED, len(others) / max(len(drawn), 1)),
        ]
    return _joined(parts)


def _draw(accounts: pd.DataFrame, size: int, config: DataConfig, as_of_ms: int) -> pd.DataFrame:
    """size of accounts drawn uniformly without replacement, or all of them if there are no
    more, by a generator seeded with the seed and the date."""
    ordered = accounts.sort_values("account_id")
    if size >= len(ordered):
        return ordered
    chosen = np.random.default_rng([config.seed, as_of_ms]).choice(len(ordered), size, False)
    return ordered.iloc[np.sort(chosen)]


def _rows(accounts: pd.DataFrame, as_of_ms: int, label: int, weight: float) -> pd.DataFrame:
    """The examples of accounts at as_of_ms, each with label and weight."""
    return pd.DataFrame(
        {
            "as_of_ms": as_of_ms,
            "account_id": accounts.account_id.to_numpy(),
            "split": accounts.split.to_numpy(),
            "label": label,
            "weight": weight,
        }
    ).astype(COLUMNS)


def _joined(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """The examples of parts, at least one, in one frame by date and account."""
    examples = pd.concat(parts, ignore_index=True)
    return examples.sort_values(["as_of_ms", "account_id"], ignore_index=True)
