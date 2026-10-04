"""When each audited mule was active, beside where the run's ranking put it.

A mule's activity, as the graph records it, is its fraud-labelled Zelle inflows: the
payments marked as scams, which the label reveal reads as victim reports. They come
from the reveal's inputs (tigergraph.oracle.REVEAL_INPUTS_QUERY, read-only). At each
audited split's cutoff, a mule is (TIMINGS):
- recent: its last fraud inflow before the cutoff came at most 30 days before it;
- earlier: 31 to 90 days before;
- long_before: more than 90 days before;
- not_yet: its fraud inflows all came at or after the cutoff;
- none: it received no fraud-labelled Zelle inflow, so its scam payments, if any,
  came on other rails.

For the hidden mules (the revealed ones removed, as the audit ranks them) and then the
revealed ones (the hidden removed), per timing: the mules, how many rank in the top 1, 5
and 10% of the population left, their ROC AUC against the non-mules and their median
population rank. A ranking that finds the recent mules and misses those active long
before the cutoff weighs only the latest of an account's history. A not_yet mule had
received no scam payment by the cutoff, so missing it says little about the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import DIAGNOSTIC_TABLES
from ..contract.clock import timestamp
from ..metrics import REVIEW_BUDGETS, budget_name, roc_auc
from .subgroups import population_rank, share_above

COLUMNS = DIAGNOSTIC_TABLES["activity_timing"]
# The timings, in report order, and the days that bound the recent and earlier ones.
TIMINGS = ("recent", "earlier", "long_before", "not_yet", "none")
RECENT_DAYS, EARLIER_DAYS = 30, 90
DAY_MS = 86_400_000


def fraud_inflows(inputs: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Each internal mule's fraud-labelled inflow times (ms), by account id, in order.

    ``inputs`` are the rows the reveal's inputs query prints; each inflow is
    "<event_seq>:<ms>".
    """
    rows = next((row["M"] for row in inputs if "M" in row), None)
    if rows is None:
        raise ValueError("The reveal's inputs hold no mules result")
    return {
        str(row["attributes"]["M.id"]): sorted(
            int(inflow.split(":")[1]) for inflow in row["attributes"]["M.@inflows"]
        )
        for row in rows
    }


def timing(inflows: Sequence[int], cutoff_ms: int) -> str:
    """A mule's timing at a cutoff (TIMINGS) from its fraud inflow times."""
    if not inflows:
        return "none"
    before = [ms for ms in inflows if ms < cutoff_ms]
    if not before:
        return "not_yet"
    days = (cutoff_ms - max(before)) / DAY_MS
    if days <= RECENT_DAYS:
        return "recent"
    if days <= EARLIER_DAYS:
        return "earlier"
    return "long_before"


def split_rows(
    split: str, frame: pd.DataFrame, cutoff_ms: int, inflows: Mapping[str, Sequence[int]]
) -> list[tuple[Any, ...]]:
    """The activity-timing rows of one audited split."""
    y = frame.is_mule.to_numpy(np.int64)
    score = frame.score.to_numpy(np.float64)
    weight = 1 / frame.inclusion_probability.to_numpy(np.float64)
    revealed = frame.revealed.to_numpy(bool)
    ids = frame.account_id.astype(str).to_numpy()
    missing = [account for account in ids[y == 1] if account not in inflows]
    if missing:
        raise ValueError(
            f"{len(missing)} {split} mules are not among the reveal's inputs, such as {missing[:3]}"
        )
    kinds = np.array(
        [timing(inflows[a], cutoff_ms) if m == 1 else "" for a, m in zip(ids, y, strict=True)]
    )
    records: list[tuple[Any, ...]] = []
    for subset, chosen in (("hidden", ~revealed), ("revealed", revealed)):
        keep = (y == 0) | ((y == 1) & chosen)
        yk, sk, wk, kk = y[keep], score[keep], weight[keep], kinds[keep]
        share = share_above(sk, wk)
        for name in TIMINGS:
            mules = (yk == 1) & (kk == name)
            records.append((split, subset, name, "mules", float(mules.sum())))
            if not mules.any():
                continue
            for fraction in REVIEW_BUDGETS:
                found = float((mules & (share <= fraction)).sum())
                records.append((split, subset, name, f"in_top_{budget_name(fraction)}", found))
            pick = (yk == 0) | mules
            auc = roc_auc(yk[pick], sk[pick], wk[pick])
            if auc is not None:
                records.append((split, subset, name, "roc_auc", float(auc)))
            ranks = population_rank(sk[pick], wk[pick], yk[pick])
            records.append((split, subset, name, "median_population_rank", float(np.median(ranks))))
    return records


def activity_timing(
    audits: Mapping[str, pd.DataFrame],
    dates: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """The activity-timing table of a run's scored audit samples, by split.

    ``dates`` are each audited split's cutoff and ``inputs`` the reveal's inputs.
    """
    inflows = fraud_inflows(inputs)
    records = [
        row
        for split, frame in audits.items()
        for row in split_rows(split, frame, timestamp(dates[split]), inflows)
    ]
    return pd.DataFrame(records, columns=list(COLUMNS))
