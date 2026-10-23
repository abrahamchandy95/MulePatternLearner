"""Test scoring and the rankings of the test world (docs/design.md, "Test world", "Rankings" and
"Metrics").

Every ranker scores the same test keys from the cache, after selection is frozen: the GNN and
its ablations from their model files (training.scoring), the two XGBoost comparisons from
theirs (comparisons.rankers), each key's logit in float64. The test keys are those the
evaluation reads itself: every scorable account on every Monday (the examples' test table), or
under the fallback the stored fallback list with its weights (evaluation.fallback).

On each test Monday the ranking holds every scored account but the true mules outside their
activity: positives are the true mules active that Monday, the rest non-mules. A mule is
flagged on a Monday when it ranks within that Monday's top 1% or top 5% of the ranked weight,
tie-aware (metrics.MondayRanking): a flag is a share, the chance of being within the budget
when tied logits are ordered at random. A true mule on its early-warning Monday, or after its
activity, is not ranked; its share is read against that Monday's ranked rows (against).

What one run (a ranker's seed) found, beside the headline metrics of the main ranking:

- per Monday, recall and precision at each budget, and pooled over Mondays (the weight found
  on every Monday over every positive's weight, or over every budget's size);
- ROC AUC and PR AUC over every ranked account-Monday;
- per true mule: the chance it is flagged on any active Monday, the expected days from its
  onset to its first flag while active, its median rank over its active Mondays, its share on
  its early-warning Monday, and the chance it is flagged after its activity;
- early warning: the mean share on the early-warning Mondays, and among the mules whose ring
  link is visible that Monday (a node of the mule's own that another account of its ring
  reaches that Monday, evaluation.development_checks.own_nodes);
- after activity: the mean share of the account-Mondays after mule_last_ms + 7 days;
- distinct accounts alerted over the scored year at each budget: the expected number of
  ranked accounts flagged on at least one Monday, which beside the share of mules ever flagged
  shows a list that churns. It is defined only when every key is scored whole, at weight 1. The
  fallback list samples each (account, Monday) of the non-mules alone, at one in ten, so an
  account is in it on about five of its 52 Mondays, and the chance of a flag over those rows is
  the chance over five Mondays, not 52: the count would come out low and reward the lists that
  churn most (an account flagged on 1% of its Mondays at random would add 0.049 and not 0.407).
  Under weighted keys it is NaN.

Lift is never computed.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

from ..comparisons import rankers
from ..comparisons.features import vectors_of
from ..metrics import TEST_BUDGETS, MondayRanking, budget_name, ever_flagged, first_flag
from ..subgraph.build import read_rows
from ..subgraph.cache import Cache
from ..subgraph.switches import Switch
from ..training.batches import ScoringBatches
from ..training.contract import Lock
from ..training.scoring import score
from ..training.trainer import load_trained
from .answer_key import ACTIVE, AFTER, EARLY_WARNING, NON_MULE, AnswerKey, statuses
from .development_checks import own_nodes
from .label_checks import JOINED_COLUMNS, truth_join_columns
from .uncertainty import Ranked

DAY_MS = 86_400_000
# The columns of a test frame.
COLUMNS = (
    "t_ms",
    "account_id",
    "logit",
    "weight",
    "status",
    "mule_ring_id",
    "mule_onset_ms",
    "mule_last_ms",
    "opened_ms",
    "identified",
)
# The headline metrics of a ranking, in the order of headline's values.
HEADLINE = (
    *(f"recall_at_{budget_name(b)}" for b in TEST_BUDGETS),
    *(f"precision_at_{budget_name(b)}" for b in TEST_BUDGETS),
    "roc_auc",
    "pr_auc",
)


def score_gnn(
    path: Path,
    lock: Lock,
    batches: ScoringBatches,
    switch: Switch | None = None,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """The logit of every key of batches under the GNN model file at path (refused under
    another lock), by Monday and account."""
    found = score(load_trained(path, lock, device), batches, switch, device)
    return found.loc[:, ["t_ms", "account_id", "logit"]]


def score_trees(path: Path, lock: Lock, batches: ScoringBatches) -> pd.DataFrame:
    """The logit of every key of batches under the XGBoost ranker file at path, its margin
    summed in float64 up to its chosen rounds, by Monday and account."""
    trees = rankers.load(path, lock)
    vectors = vectors_of(batches)
    return pd.DataFrame(
        {
            "t_ms": pd.Series([t_ms for t_ms, _ in vectors.keys], dtype="int64"),
            "account_id": pd.Series([account for _, account in vectors.keys], dtype="str"),
            "logit": pd.Series(trees.scores(vectors), dtype="float64"),
        }
    ).sort_values(["t_ms", "account_id"], ignore_index=True)


def evaluation_frame(
    scores: pd.DataFrame, keys: pd.DataFrame, key: AnswerKey, identified: Collection[str]
) -> pd.DataFrame:
    """Every scored test key, by Monday and account, with its logit, its weight (keys:
    t_ms, account_id, weight), what it is that Monday (answer_key.status), its truth and
    whether the bank identified it. Every key needs a logit and a truth."""
    joined = keys.loc[:, ["t_ms", "account_id", "weight"]].merge(
        scores.loc[:, ["t_ms", "account_id", "logit"]],
        on=["t_ms", "account_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["logit"].isna().any() or len(scores) != len(keys):
        raise ValueError("The scores are not those of the test keys")
    if tuple(joined.columns) != JOINED_COLUMNS:
        raise ValueError(
            f"The test keys and scores join into {list(joined.columns)}, not the "
            f"{list(JOINED_COLUMNS)} that mule audit checks against the truth"
        )
    truth = key.frame()
    shared = truth_join_columns(joined.columns, truth)
    if shared:
        raise ValueError(f"The test keys share {list(shared)} with the truth")
    frame = joined.merge(truth, on="account_id", how="left", validate="many_to_one")
    if frame["is_mule"].isna().any():
        raise ValueError("A test key has no truth in the answer key")
    frame["status"] = statuses(
        frame["t_ms"].to_numpy(dtype=np.int64),
        frame["is_mule"].to_numpy(dtype=np.int64),
        frame["mule_onset_ms"].to_numpy(dtype=np.int64),
        frame["mule_last_ms"].to_numpy(dtype=np.int64),
    )
    frame["identified"] = frame["account_id"].isin(set(identified))
    frame = frame.loc[:, list(COLUMNS)].astype({"logit": "float64", "weight": "float64"})
    return frame.sort_values(["t_ms", "account_id"], ignore_index=True)


@dataclass(frozen=True)
class Ranking:
    """A ranking of rows of a test frame: the rows (non-mules, and the positives chosen), which
    are positive, and their Monday ranking and pooled ranking."""

    frame: pd.DataFrame
    positive: NDArray[np.bool_]
    order: MondayRanking
    pooled: Ranked

    @classmethod
    def of(cls, frame: pd.DataFrame, positive: NDArray[np.bool_]) -> Ranking:
        """The ranking of the non-mule rows of frame and the rows positive marks."""
        kept = positive | (frame["status"] == NON_MULE).to_numpy()
        rows = frame.loc[kept].reset_index(drop=True)
        chosen = positive[kept]
        logits = rows["logit"].to_numpy(dtype=np.float64)
        return cls(
            rows,
            chosen,
            MondayRanking(rows["t_ms"].to_numpy(dtype=np.int64), logits),
            Ranked(logits, chosen),
        )

    @property
    def weight(self) -> NDArray[np.float64]:
        return self.frame["weight"].to_numpy(dtype=np.float64)

    def headline(self, weight: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """The headline metrics (HEADLINE) under weight, each row's weight by default: pooled
        recall and precision at each budget, ROC AUC and PR AUC; NaN where undefined."""
        used = self.weight if weight is None else weight
        shares = self.order.shares(used)
        found, positives, sizes = self.order.budget_found(shares, self.positive, used)
        total = positives.sum()
        recall = found.sum(axis=0) / total if total > 0 else np.full(len(TEST_BUDGETS), np.nan)
        precision = found.sum(axis=0) / sizes.sum(axis=0)
        return np.r_[
            recall, precision, self.pooled.roc_auc(used), self.pooled.average_precision(used)
        ]

    def per_monday(self) -> pd.DataFrame:
        """Recall and precision at each budget on each Monday, with its positives' weight."""
        weight = self.weight
        found, positives, sizes = self.order.budget_found(
            self.order.shares(weight), self.positive, weight
        )
        table = pd.DataFrame({"t_ms": self.order.mondays, "positives": positives})
        for column, budget in enumerate(TEST_BUDGETS):
            name = budget_name(budget)
            table[f"recall_at_{name}"] = np.divide(
                found[:, column],
                positives,
                out=np.full(len(positives), np.nan),
                where=positives > 0,
            )
            table[f"precision_at_{name}"] = found[:, column] / sizes[:, column]
        return table


def ring_links(
    cache: Cache, frame: pd.DataFrame, rows: Iterable[tuple[int, str]]
) -> frozenset[tuple[int, str]]:
    """Of rows, (Monday, mule account), those whose ring link is visible that Monday: a device
    or IP of the mule's own (its owner-enrolment or seen edges) that the own edges of another
    account of its ring scored that Monday reach too, read from the cache."""
    members = frame.loc[frame["mule_ring_id"] >= 0, ["t_ms", "account_id", "mule_ring_id"]]
    ring_of = {
        (int(t_ms), str(account)): int(ring)
        for t_ms, account, ring in members.itertuples(index=False, name=None)
    }
    asked = sorted(row for row in rows if row in ring_of)
    found: set[tuple[int, str]] = set()
    for t_ms in sorted({day for day, _ in asked}):
        mine = [account for day, account in asked if day == t_ms]
        rings = {ring_of[(t_ms, account)] for account in mine}
        mates = sorted(
            account for (day, account), ring in ring_of.items() if day == t_ms and ring in rings
        )
        nodes = {
            example.scored: set(own_nodes(example).nodes)
            for example in read_rows(cache, [(t_ms, account) for account in mates])
        }
        for account in mine:
            ring = ring_of[(t_ms, account)]
            others = [
                nodes[mate] for mate in mates if mate != account and ring_of[(t_ms, mate)] == ring
            ]
            if nodes[account] & set[tuple[str, str]]().union(*others):
                found.add((t_ms, account))
    return frozenset(found)


@dataclass(frozen=True)
class RunMetrics:
    """What one run found on the test world: its headline metrics and the others of the
    module's docstring by name, its per-Monday table, its table of true mules, and the ranking
    they were found from."""

    values: Mapping[str, float]
    per_monday: pd.DataFrame
    mules: pd.DataFrame
    ranking: Ranking


def mule_table(
    frame: pd.DataFrame, ranking: Ranking, links: Collection[tuple[int, str]]
) -> pd.DataFrame:
    """Per true mule with a scored Monday: its ring and identification, its chance of a flag
    while active at each budget, the expected days to its first flag at each budget, its median
    rank fraction over its active Mondays, its share on its early-warning Monday, whether its
    ring link was visible then, and its chance of a flag after its activity.

    Each mule's rows are found once, by account, in the active rows of the ranking and in the
    rows outside it, and not by a scan of either for every mule."""
    mules = frame[frame["status"] != NON_MULE]
    active = ranking.positive
    weight = ranking.weight
    shares = ranking.order.shares(weight)[active]
    fractions = ranking.order.rank_fraction(weight)[active]
    outside = mules[mules["status"].isin([EARLY_WARNING, AFTER])]
    outside_shares, _ = ranking.order.against(
        weight,
        outside["t_ms"].to_numpy(dtype=np.int64),
        outside["logit"].to_numpy(dtype=np.float64),
    )
    ranked = ranking.frame[active]
    ranked_t = ranked["t_ms"].to_numpy(dtype=np.int64)
    ranked_rows = ranked.groupby("account_id", sort=False).indices
    outside_t = outside["t_ms"].to_numpy(dtype=np.int64)
    outside_status = outside["status"].to_numpy()
    outside_rows = outside.groupby("account_id", sort=False).indices
    nothing = np.zeros(0, dtype=np.intp)
    rows: list[dict[str, object]] = []
    for account, held in mules.groupby("account_id", sort=True):
        first = held.iloc[0]
        row: dict[str, object] = {
            "account_id": account,
            "mule_ring_id": int(first["mule_ring_id"]),
            "identified": bool(first["identified"]),
            "opened_ms": int(first["opened_ms"]),
        }
        chosen = ranked_rows.get(account, nothing)
        days = (ranked_t[chosen] - int(first["mule_onset_ms"])) / DAY_MS
        mine = shares[chosen]
        row["active_mondays"] = len(chosen)
        for column, budget in enumerate(TEST_BUDGETS):
            ever, expected = first_flag(days, mine[:, column]) if len(days) else (0.0, np.nan)
            row[f"ever_at_{budget_name(budget)}"] = ever
            row[f"first_flag_days_at_{budget_name(budget)}"] = expected
        row["median_rank"] = float(np.median(fractions[chosen])) if len(chosen) else float("nan")
        mine_outside = outside_rows.get(account, nothing)
        warned = mine_outside[outside_status[mine_outside] == EARLY_WARNING]
        after = mine_outside[outside_status[mine_outside] == AFTER]
        for column, budget in enumerate(TEST_BUDGETS):
            name = budget_name(budget)
            row[f"early_warning_at_{name}"] = (
                float(outside_shares[warned[0], column]) if len(warned) else np.nan
            )
            row[f"after_at_{name}"] = float(
                ever_flagged(outside_shares[after, column], np.zeros(len(after), dtype=np.intp), 1)[
                    0
                ]
            )
        row["early_warning_monday"] = int(outside_t[warned[0]]) if len(warned) else 0
        row["ring_link_visible"] = len(warned) > 0 and (int(outside_t[warned[0]]), account) in links
        rows.append(row)
    return pd.DataFrame(rows)


def run_metrics(frame: pd.DataFrame, links: Collection[tuple[int, str]]) -> RunMetrics:
    """What one run's test frame shows, as the module says; links are the early-warning
    rows whose ring link is visible (ring_links)."""
    ranking = Ranking.of(frame, (frame["status"] == ACTIVE).to_numpy())
    values: dict[str, float] = dict(zip(HEADLINE, ranking.headline().tolist(), strict=True))
    mules = mule_table(frame, ranking, links)
    scored = mules[mules["active_mondays"] > 0]
    warned = mules[mules["early_warning_monday"] > 0]
    weight = ranking.weight
    shares = ranking.order.shares(weight)
    # Distinct accounts alerted are counted only when every key is scored whole, at weight 1.
    whole = bool((weight == 1.0).all())
    account_index, accounts = (
        pd.factorize(ranking.frame["account_id"]) if whole else (np.zeros(0, dtype=np.intp), [])
    )
    after = frame[frame["status"] == AFTER]
    after_shares, _ = ranking.order.against(
        weight, after["t_ms"].to_numpy(dtype=np.int64), after["logit"].to_numpy(dtype=np.float64)
    )
    for column, budget in enumerate(TEST_BUDGETS):
        name = budget_name(budget)
        values[f"ever_flagged_at_{name}"] = float(scored[f"ever_at_{name}"].mean())
        flagged = scored[scored[f"ever_at_{name}"] > 0]
        days = flagged[f"first_flag_days_at_{name}"].to_numpy(dtype=np.float64)
        chance = flagged[f"ever_at_{name}"].to_numpy(dtype=np.float64)
        values[f"first_flag_mean_days_at_{name}"] = (
            float(days @ chance / chance.sum()) if len(days) else float("nan")
        )
        values[f"first_flag_median_days_at_{name}"] = (
            float(np.median(days)) if len(days) else float("nan")
        )
        values[f"early_warning_at_{name}"] = float(warned[f"early_warning_at_{name}"].mean())
        linked = warned[warned["ring_link_visible"]]
        values[f"early_warning_ring_link_at_{name}"] = float(
            linked[f"early_warning_at_{name}"].mean()
        )
        values[f"after_activity_at_{name}"] = (
            float(after_shares[:, column].mean()) if len(after) else float("nan")
        )
        values[f"accounts_alerted_at_{name}"] = (
            float(ever_flagged(shares[:, column], account_index, len(accounts)).sum())
            if whole
            else float("nan")
        )
    values["mules_scored"] = float(len(scored))
    values["early_warning_mules"] = float(len(warned))
    values["early_warning_ring_link_mules"] = float(warned["ring_link_visible"].sum())
    values["after_activity_account_mondays"] = float(len(after))
    values["ranked_account_mondays"] = float(len(ranking.frame))
    return RunMetrics(values, ranking.per_monday(), mules, ranking)


def scored_keys(frame: pd.DataFrame, statuses_kept: Sequence[str]) -> list[tuple[int, str]]:
    """The (Monday, account) of the rows of a test frame of the statuses given."""
    chosen = frame[frame["status"].isin(list(statuses_kept))]
    return [
        (int(t_ms), str(account))
        for t_ms, account in zip(chosen["t_ms"], chosen["account_id"], strict=True)
    ]
