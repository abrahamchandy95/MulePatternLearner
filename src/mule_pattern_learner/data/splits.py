"""Split dates, the cutoff clocks they resolve to, and the context keys of a split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..contract.clock import cutoff_ms, timestamp
from ..contract.graph_schema import SPLIT_PHASE, ContextKey, context_scope
from ..tigergraph.executor import QueryExecutor, checked_rows, printed


def validate_dates(config: dict[str, Any]) -> None:
    dates = config["dates"]
    if not all(dates.get(split) for split in ("train", "validation", "test")):
        raise ValueError("Three nonempty chronological splits are required")
    clocks = {split: [timestamp(date) for date in dates[split]] for split in dates}
    if (
        not max(clocks["train"])
        < min(clocks["validation"])
        <= max(clocks["validation"])
        < min(clocks["test"])
    ):
        raise ValueError("Train, validation and test cutoffs overlap or are out of order")


def eligible_mask(accounts: pd.DataFrame, split: str, date: str) -> np.ndarray:
    """Rows of split whose account existed before the cutoff date."""
    return (accounts["split"].to_numpy() == split) & (
        accounts["first_seen_ts_ms"].to_numpy() < timestamp(date)
    )


def marginal_mask(accounts: pd.DataFrame) -> np.ndarray:
    """Rows of the label-blind reservoir (the observed positives outside it are not)."""
    return accounts["in_marginal"].to_numpy(bool)


def sample_keys(accounts: pd.DataFrame, date: str, manifest: dict[str, Any]) -> list[ContextKey]:
    ms = cutoff_ms(date)
    seq = int(manifest["cutoff_seqs"][date])
    scope = context_scope(manifest["config"])
    return [
        ContextKey("Account", str(row.account_id), seq, ms, scope, SPLIT_PHASE[str(row.split)])
        for row in accounts.itertuples(index=False)
    ]


def resolve_cutoffs(executor: QueryExecutor, dates: list[str]) -> dict[str, int]:
    """ContextKey cutoff_seq per date: one past the last event visible at date - 1 ms."""
    result = checked_rows(
        executor.run(
            "temporal_training_cutoffs",
            {"cutoff_times": [cutoff_ms(date) for date in dates]},
            timeout_s=900.0,
        )
    )
    clocks = printed(result, "last_visible_seqs")
    cutoffs = {}
    for date in dates:
        last = int(clocks[str(cutoff_ms(date))])
        if last <= 0:
            raise ValueError(f"No events or entities are visible before {date}")
        cutoffs[date] = last + 1
    return cutoffs


def resolve_cutoff(executor: QueryExecutor, date: str) -> tuple[int, int]:
    """The (cutoff_seq, cutoff_ms) of an unprepared scoring date."""
    return resolve_cutoffs(executor, [date])[date], cutoff_ms(date)
