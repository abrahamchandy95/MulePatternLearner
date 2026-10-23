"""The groups of true mules evaluation reports apart (docs/design.md, "Groups"). Each group is
ranked against the non-mules with every other true mule removed, as an investigator removes
known cases:

- hidden_ring: mules of rings the bank never surfaced (no account of the ring identified), the
  headline;
- surfaced_unidentified: the unidentified mules of rings that surfaced;
- identified: the mules the bank identified (pu_label 1 in the test world, which the model
  never saw either);
- no_ring: mules of no ring (account-takeover drops and impostor payees);
- new_account: mules whose account opened within 90 days of t (opened_ms > t - 90 days, from
  the answer key), purpose-opened mules being the hard case;
- no_payment: mules scored with no payment in the lookback (their B3 row's in_90d and out_90d
  both 0), on an active Monday or on their early-warning Monday, which only links to other
  mules can reveal: the owner's Account Z, reported beside the enrolment ceiling (72% of
  device and 61% of IP associations are on file);
- hop_1, hop_2, hop_beyond: mules at one edge, at two edges or not within two hops of the
  nearest identified mule in their own subgraph that Monday (evaluation.hops).

Every group but no_payment holds active account-Mondays only. A group's metrics are its pooled
recall and precision at each budget, ROC AUC and PR AUC, the chance of each of its mules being
flagged on any of its Mondays in the group, averaged, and its mules and account-Mondays.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..metrics import TEST_BUDGETS, budget_name, ever_flagged
from ..subgraph.cache import ACTIVITY, Cache
from .answer_key import ACTIVE, EARLY_WARNING, AnswerKey
from .rankings import HEADLINE, Ranking

DAY_MS = 86_400_000
# The days within which an account opened makes a mule's group new_account.
NEW_ACCOUNT_DAYS = 90
GROUPS = (
    "hidden_ring",
    "surfaced_unidentified",
    "identified",
    "no_ring",
    "new_account",
    "no_payment",
    "hop_1",
    "hop_2",
    "hop_beyond",
)
# The share of a ring's device and of its IP associations on file, which bounds what no_payment
# can find through enrolment (docs/design.md, "Groups").
ENROLMENT_CEILING = {"device": 0.72, "ip": 0.61}


def surfaced_rings(key: AnswerKey, identified: Collection[str]) -> frozenset[int]:
    """The rings with an account the bank identified."""
    return frozenset(
        truth.mule_ring_id
        for truth in key.mules
        if truth.mule_ring_id >= 0 and truth.account_id in identified
    )


def quiet_rows(cache: Cache, keys: Iterable[tuple[int, str]]) -> frozenset[tuple[int, str]]:
    """Of keys, (Monday, account), those with no payment in the lookback, by their B3 rows in
    the cache: no payment received and none sent in the last 90 days."""
    by_monday: dict[int, list[str]] = {}
    for t_ms, account in keys:
        by_monday.setdefault(t_ms, []).append(account)
    found: set[tuple[int, str]] = set()
    for t_ms, accounts in sorted(by_monday.items()):
        for row in cache.rows(ACTIVITY, t_ms, accounts):
            if row.in_90d == 0 and row.out_90d == 0:
                found.add((t_ms, row.account_id))
    return frozenset(found)


def group_rows(
    frame: pd.DataFrame,
    surfaced: Collection[int],
    no_payment: Collection[tuple[int, str]],
    hops: Mapping[tuple[int, str], int | None],
) -> dict[str, NDArray[np.bool_]]:
    """Each group's rows of a test frame: no_payment the (Monday, account) rows with no payment
    in the lookback, hops each active row's hop distance."""
    status = frame["status"].to_numpy()
    active = status == ACTIVE
    ring = frame["mule_ring_id"].to_numpy(dtype=np.int64)
    identified = frame["identified"].to_numpy(dtype=bool)
    in_surfaced = np.isin(ring, np.array(sorted(surfaced), dtype=np.int64))
    t_ms = frame["t_ms"].to_numpy(dtype=np.int64)
    keys = list(zip(t_ms.tolist(), frame["account_id"].tolist(), strict=True))
    quiet = np.array([key in no_payment for key in keys], dtype=bool)
    distance = [
        hops.get(key, None) if mule else None for key, mule in zip(keys, active, strict=True)
    ]
    return {
        "hidden_ring": active & (ring >= 0) & ~in_surfaced,
        "surfaced_unidentified": active & (ring >= 0) & in_surfaced & ~identified,
        "identified": active & identified,
        "no_ring": active & (ring < 0),
        "new_account": active
        & (frame["opened_ms"].to_numpy(dtype=np.int64) > t_ms - NEW_ACCOUNT_DAYS * DAY_MS),
        "no_payment": (active | (status == EARLY_WARNING)) & quiet,
        "hop_1": active & np.array([found == 1 for found in distance], dtype=bool),
        "hop_2": active & np.array([found == 2 for found in distance], dtype=bool),
        "hop_beyond": active & np.array([found is None for found in distance], dtype=bool),
    }


def group_names(frame: pd.DataFrame, rows: Mapping[str, NDArray[np.bool_]]) -> dict[str, str]:
    """The groups of each account, by account: the names of the groups that hold any of its
    rows, in the order of rows, joined by commas; an account in none is not a key. Each group's
    accounts are read once from its rows, not looked for again for every account."""
    found: dict[str, list[str]] = {}
    for name, chosen in rows.items():
        for account in frame.loc[chosen, "account_id"].unique().tolist():
            found.setdefault(str(account), []).append(name)
    return {account: ",".join(names) for account, names in found.items()}


def group_metrics(frame: pd.DataFrame, rows: Mapping[str, NDArray[np.bool_]]) -> pd.DataFrame:
    """Each group's metrics, a row per group and metric: the headline metrics of its ranking
    against the non-mules, the mean chance of a flag of its mules, and its sizes; beside
    no_payment, the enrolment ceiling."""
    found: list[tuple[str, str, float]] = [
        ("no_payment", f"enrolment_ceiling_{kind}", share)
        for kind, share in ENROLMENT_CEILING.items()
    ]
    for group in GROUPS:
        chosen = rows[group]
        mules = int(frame.loc[chosen, "account_id"].nunique())
        found += [(group, "mules", float(mules)), (group, "account_mondays", float(chosen.sum()))]
        if not chosen.any():
            continue
        ranking = Ranking.of(frame, chosen)
        found += [
            (group, name, value)
            for name, value in zip(HEADLINE, ranking.headline().tolist(), strict=True)
        ]
        shares = ranking.order.shares(ranking.weight)[ranking.positive]
        accounts, index = np.unique(
            ranking.frame.loc[ranking.positive, "account_id"].to_numpy(), return_inverse=True
        )
        for column, budget in enumerate(TEST_BUDGETS):
            ever = ever_flagged(shares[:, column], index.astype(np.intp), len(accounts))
            found.append((group, f"ever_flagged_at_{budget_name(budget)}", float(ever.mean())))
    return pd.DataFrame(found, columns=["group", "metric", "value"])
