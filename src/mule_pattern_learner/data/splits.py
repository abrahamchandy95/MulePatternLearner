"""The cutoff clocks split dates resolve to, and the context keys of a split.

The dates themselves are checked where they are configured (config.SplitDates).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..contract.clock import cutoff_ms, timestamp
from ..contract.graph_schema import SPLIT_PHASE, ContextKey, context_scope
from .ports import CutoffReader


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
    scope = context_scope(manifest["source"]["scope_id"])
    return [
        ContextKey("Account", str(row.account_id), seq, ms, scope, SPLIT_PHASE[str(row.split)])
        for row in accounts.itertuples(index=False)
    ]


def resolve_cutoffs(reader: CutoffReader, dates: list[str]) -> dict[str, int]:
    """ContextKey cutoff_seq per date: one past the last event visible at date - 1 ms."""
    clocks = reader.last_visible_seqs([cutoff_ms(date) for date in dates])
    cutoffs = {}
    for date in dates:
        last = clocks[cutoff_ms(date)]
        if last <= 0:
            raise ValueError(f"No events or entities are visible before {date}")
        cutoffs[date] = last + 1
    return cutoffs


def resolve_cutoff(reader: CutoffReader, date: str) -> tuple[int, int]:
    """The (cutoff_seq, cutoff_ms) of an unprepared scoring date."""
    return resolve_cutoffs(reader, [date])[date], cutoff_ms(date)
