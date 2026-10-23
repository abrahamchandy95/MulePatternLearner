"""Identification bias (docs/design.md, "Identification bias").

Victim reports may identify more often the mules whose victims bank here, so
new_internal_payers_* could learn who gets caught rather than what mules do. Evaluation reports:

- identified against hidden mules by victim origin: each true mule's new payers over 30 days,
  internal and external, summed over its active Mondays from its own B2 rows, and the internal
  share of them (victim_origin, origin_table);
- recall of hidden mules by that share, in bins of a quarter (hidden_recall_by_origin): the
  mean chance of a flag of the hidden mules of each bin, and of those with no new payer;
- how well the features separate identified from hidden mules: each audited column's ROC AUC
  over the true mules' active account-Mondays, the identified as the positive class
  (identification_separation);
- on the development rotation only, for seed 0: the correlation of the selection score with
  the recall of all true mules of the selection world, per prior and epoch, from every
  epoch's kept model rescored on the selection world (rescore_epochs, selection_correlation),
  reported and never used to select.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import ModelConfig
from ..metrics import TEST_BUDGETS, budget_name
from ..model.hetero import HeteroModel
from ..subgraph.build import build, read_rows
from ..subgraph.cache import COUNTERPARTIES, Cache
from ..subgraph.layouts import LAYOUT
from ..training import contract
from ..training.batches import ScoringBatches
from ..training.contract import Lock
from ..training.trainer import logits
from .answer_key import ACTIVE, AnswerKey
from .fallback import by_monday
from .feature_audit import COLUMNS, audit_columns
from .rankings import Ranking, evaluation_frame
from .uncertainty import Ranked

# The bins of the internal share of new payers.
ORIGIN_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)
NO_NEW_PAYER = "no new payer"
# The examples built into one batch at a time.
BATCH = 512


def _active(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"] == ACTIVE]


def victim_origin(cache: Cache, frame: pd.DataFrame) -> pd.DataFrame:
    """Per true mule with an active Monday: whether the bank identified it, its new internal
    and external payers over 30 days summed over its active Mondays, from its own B2 rows, and
    the internal share of them (NaN with none)."""
    active = _active(frame)
    sums: dict[str, list[int]] = {}
    for t_ms, accounts in by_monday(active).items():
        for row in cache.rows(COUNTERPARTIES, t_ms, accounts):
            held = sums.setdefault(row.account_id, [0, 0])
            held[0] += row.new_internal_payers_30d
            held[1] += row.new_external_payers_30d
    identified = active.groupby("account_id")["identified"].first()
    table = pd.DataFrame(
        [
            (account, bool(identified[account]), internal, external)
            for account, (internal, external) in sorted(sums.items())
        ],
        columns=["account_id", "identified", "new_internal_payers", "new_external_payers"],
    )
    total = table["new_internal_payers"] + table["new_external_payers"]
    table["internal_share"] = np.where(
        total > 0, table["new_internal_payers"] / total.clip(lower=1), np.nan
    )
    return table


def origin_table(origin: pd.DataFrame) -> pd.DataFrame:
    """Identified against hidden mules by victim origin: per class, the mules, those with a new
    payer and their mean internal share."""
    rows: list[tuple[str, int, int, float]] = []
    for name, chosen in (("identified", origin["identified"]), ("hidden", ~origin["identified"])):
        held = origin[chosen]
        shares = held["internal_share"].dropna()
        rows.append((name, len(held), len(shares), float(shares.mean()) if len(shares) else np.nan))
    return pd.DataFrame(rows, columns=["class", "mules", "with_new_payers", "mean_internal_share"])


def origin_bin(share: float) -> str:
    """The bin of an internal share: a quarter of [0, 1], the last closed, or no new payer."""
    if np.isnan(share):
        return NO_NEW_PAYER
    for low, high in zip(ORIGIN_BINS, ORIGIN_BINS[1:], strict=False):
        if share < high or high == ORIGIN_BINS[-1]:
            return f"[{low:g}, {high:g}{']' if high == ORIGIN_BINS[-1] else ')'}"
    raise ValueError(f"An internal share lies in [0, 1], got {share}")


def hidden_recall_by_origin(origin: pd.DataFrame, mules: pd.DataFrame) -> pd.DataFrame:
    """Recall of hidden mules by the internal share of their new payers: per bin of a quarter
    (and for those with no new payer), the hidden mules and their mean chance of a flag while
    active at each budget, from a run's mule table (rankings.mule_table)."""
    hidden = origin[~origin["identified"]].merge(
        mules.drop(columns=["identified"]), on="account_id", how="inner", validate="one_to_one"
    )
    hidden = hidden.assign(bin=[origin_bin(float(share)) for share in hidden["internal_share"]])
    columns = [f"ever_at_{budget_name(budget)}" for budget in TEST_BUDGETS]
    grouped = hidden.groupby("bin", sort=True)
    table = grouped[columns].mean().reset_index()
    table.insert(1, "mules", grouped.size().to_numpy())
    return table


def identification_separation(cache: Cache, frame: pd.DataFrame, world: str) -> pd.DataFrame:
    """How well each audited column separates identified from hidden true mules over their
    active account-Mondays: its ROC AUC with the identified as the positive class (NaN without
    both classes)."""
    active = _active(frame)
    keys = [
        (int(t_ms), str(account))
        for t_ms, account in zip(active["t_ms"], active["account_id"], strict=True)
    ]
    examples = read_rows(cache, keys) if keys else []
    rows = [
        audit_columns(build(examples[start : start + BATCH], world))
        for start in range(0, len(examples), BATCH)
    ]
    matrix = np.concatenate(rows) if rows else np.zeros((0, len(COLUMNS)))
    positive = active["identified"].to_numpy(dtype=bool)
    weight = np.ones(len(positive))
    return pd.DataFrame(
        {
            "column": COLUMNS,
            "roc_auc": [
                Ranked(matrix[:, column], positive).roc_auc(weight)
                for column in range(len(COLUMNS))
            ],
        }
    )


def rescore_epochs(
    paths: Sequence[Path],
    lock: Lock,
    batches: ScoringBatches,
    keys: pd.DataFrame,
    key: AnswerKey,
    identified: Collection[str],
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Every kept epoch's model (training.trainer keeps them under lock for seed 0 of the
    development rotation) rescored on the selection world: its prior's multiple, its epoch,
    its selection score and the pooled recall of every true mule of the selection world at
    each budget. keys are the selection world's scored keys with their weights, batches their
    scoring batches, key its answer key and identified its identified mules."""
    target = torch.device(device)
    rows: list[dict[str, object]] = []
    for path in paths:
        stored = contract.load(path, lock, target)
        model = HeteroModel(LAYOUT, ModelConfig(**stored["model_config"])).to(target)
        model.load_state_dict(stored["state"])
        found_keys, found = logits(model, batches, None, target)
        scores = pd.DataFrame(
            {
                "t_ms": pd.Series([t_ms for t_ms, _ in found_keys], dtype="int64"),
                "account_id": pd.Series([account for _, account in found_keys], dtype="str"),
                "logit": pd.Series(found, dtype="float64"),
            }
        )
        frame = evaluation_frame(scores, keys, key, identified)
        recall = Ranking.of(frame, (frame["status"] == ACTIVE).to_numpy()).headline()
        row: dict[str, object] = {
            "prior": float(stored["prior"][0]),
            "epoch": int(stored["epoch"]),
            "selection_score": float(stored["score"]),
        }
        for column, budget in enumerate(TEST_BUDGETS):
            row[f"recall_at_{budget_name(budget)}"] = float(recall[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["prior", "epoch"], ignore_index=True)


def selection_correlation(epochs: pd.DataFrame) -> pd.DataFrame:
    """Per prior, the correlation (Pearson) over its epochs of the selection score with the
    mean of the recalls of every true mule at the budgets; NaN with fewer than two epochs or
    no spread."""
    recall = [f"recall_at_{budget_name(budget)}" for budget in TEST_BUDGETS]
    rows: list[tuple[float, int, float]] = []
    for prior, held in epochs.groupby("prior", sort=True):
        score = held["selection_score"].to_numpy(dtype=np.float64)
        mean = held[recall].to_numpy(dtype=np.float64).mean(axis=1)
        defined = len(held) > 1 and score.std() > 0 and mean.std() > 0
        rows.append(
            (
                float(str(prior)),
                len(held),
                float(np.corrcoef(score, mean)[0, 1]) if defined else float("nan"),
            )
        )
    return pd.DataFrame(rows, columns=["prior", "epochs", "correlation"])
