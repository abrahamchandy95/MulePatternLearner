"""The audit sample: the account-Mondays mule audit's clock probe, feature audit and development
checks read (docs/design.md, the Clock guard and "The feature audit").

- Every active true mule on every scored Monday it is scorable and active, each weighing 1.
- A keyed sample of audit.nonmule_sample of the non-mule account-Mondays (is_mule 0, scorable
  on that Monday): those with the lowest keyed hash of (account, Monday) over the whole scored
  year (data.draw.draw_hash, the hash of the training draw), each weighing the non-mule
  account-Mondays over the sample's size, so that the sample stands for every one of them.

True mules outside their activity are in neither class, as in every ranking. An account the
answer key lacks is left out; the label checks fail on it. Each account-Monday carries its
cluster of the ring bootstrap (evaluation.uncertainty): its ring, its own account for a
ringless mule, and its own account for a non-mule.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import heapq

import numpy as np
import pandas as pd

from ..data.draw import draw_hash
from .answer_key import ACTIVE, NON_MULE, AnswerKey, status
from .uncertainty import cluster_keys

COLUMNS = ("t_ms", "account_id", "positive", "ring", "weight", "cluster")


def audit_sample(
    key: AnswerKey, populations: Mapping[int, Sequence[str]], size: int, draw_key: bytes
) -> pd.DataFrame:
    """The audit sample, by Monday and account: t_ms, account_id, positive (an active true
    mule), ring, weight and cluster."""

    def non_mules() -> Iterator[tuple[int, str]]:
        for t_ms, accounts in populations.items():
            for account in accounts:
                truth = key.by_id.get(account)
                if truth is not None and status(truth, t_ms) == NON_MULE:
                    yield t_ms, account

    total = sum(1 for _ in non_mules())
    hashed = ((draw_hash(draw_key, account, t_ms), t_ms, account) for t_ms, account in non_mules())
    drawn = heapq.nsmallest(size, hashed)
    weight = total / len(drawn) if drawn else 0.0
    rows: list[tuple[int, str, bool, int, float]] = [
        (t_ms, account, False, -1, weight) for _, t_ms, account in drawn
    ]
    for t_ms, accounts in populations.items():
        for account in accounts:
            truth = key.by_id.get(account)
            if truth is not None and status(truth, t_ms) == ACTIVE:
                rows.append((t_ms, account, True, truth.mule_ring_id, 1.0))
    found = pd.DataFrame(rows, columns=list(COLUMNS[:-1])).astype(
        {"t_ms": "int64", "account_id": "str", "positive": "bool", "ring": "int64"}
    )
    found["weight"] = found["weight"].astype("float64")
    found["cluster"] = cluster_keys(
        found["account_id"].tolist(),
        found["positive"].to_numpy(dtype=np.int64),
        found["ring"].to_numpy(dtype=np.int64),
    )
    return found.sort_values(["t_ms", "account_id"]).reset_index(drop=True)
