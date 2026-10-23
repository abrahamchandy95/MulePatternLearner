"""The missed-mule report (docs/design.md, "The missed-mule report").

The owner: "we need to score every one of the true mules that were hidden from the model. the
model needs to be able to identify all of them, or atleast score them high. if not, we need to
investigate why." Every true mule of the test world was hidden from the model, identified or
not, since the model never reads the test world's labels; the report reads the run of the model
mule score uses, the GNN of the best selection score.

- Who is in it: every true mule with an active Monday in the ranking that did not score high:
  missed when it is never in the top 5% on any of its active Mondays (its chance of a flag at
  5% is 0), and weak when its median rank over its active Mondays is outside the top 5%
  though it was flagged on some. 5% is audit.report_budget, which the owner set.
- What it holds, written into the run's results folder (missed/):
  - mules.parquet: each one, missed or weak, with its groups, ring, identification, median
    rank and chance of a flag;
  - rows.parquet: its decoded feature rows on each active Monday, long, a row per column:
    the audited columns of its built example (evaluation.feature_audit: its own row with
    its degrees, its customer's, its devices' and IPs' and its edges' aggregates and its
    community block), each value raw, before encoding;
  - subgraph_<kind>.parquet: its subgraph on each active Monday, the nodes and edges with
    their fields as the cache holds them, a file per kind of row;
  - ring_mates.parquet: on each Monday, how each other scored account of its ring scored
    (its logit and rank, its status) and whether the bank identified it.

The report is read by nothing else; each mule is explained by a person as a bug or a missing
feature, on development worlds (the Test worlds stay unseen guard).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pyarrow as pa

from ..metrics import budget_name
from ..subgraph.build import build, read_rows
from ..subgraph.cache import Cache
from .answer_key import NON_MULE
from .feature_audit import COLUMNS, audit_columns
from .groups import group_names
from .rankings import Ranking
from .store import write_parquet

MISSED = "missed"
WEAK = "weak"
FOLDER = "missed"


def missed_mules(mules: pd.DataFrame, budget: float) -> pd.DataFrame:
    """The missed and weak mules of a run's mule table (rankings.mule_table), with their
    outcome, of those with an active Monday."""
    name = budget_name(budget)
    scored = mules[mules["active_mondays"] > 0].copy()
    never = scored[f"ever_at_{name}"] <= 0
    weak = ~never & (scored["median_rank"] > budget)
    scored["outcome"] = np.where(never, MISSED, np.where(weak, WEAK, ""))
    return scored[scored["outcome"] != ""].reset_index(drop=True)


def _write(folder: Path, name: str, frame: pd.DataFrame, marks: Mapping[str, str | bool]) -> None:
    marked = frame.assign(**marks)
    write_parquet(folder / f"{name}.parquet", pa.Table.from_pandas(marked, preserve_index=False))


def write_report(
    folder: Path,
    cache: Cache,
    world: str,
    frame: pd.DataFrame,
    reported: pd.DataFrame,
    groups: Mapping[str, NDArray[np.bool_]],
    marks: Mapping[str, str | bool],
) -> None:
    """Write the report of the reported mules (missed_mules) into folder/missed: frame is the
    run's test frame, groups each group's rows of it (groups.group_rows), and marks the
    columns every table carries (the rotation, whether it is development only)."""
    target = folder / FOLDER
    accounts = set(reported["account_id"].tolist())
    table = reported.copy()
    names = group_names(frame, groups)
    table["groups"] = [names.get(account, "") for account in table["account_id"]]
    _write(target, "mules", table, marks)
    active = frame[(frame["status"] == "active") & frame["account_id"].isin(accounts)]
    keys = [
        (int(t_ms), str(account))
        for t_ms, account in zip(active["t_ms"], active["account_id"], strict=True)
    ]
    examples = read_rows(cache, keys) if keys else []
    decoded: list[tuple[str, int, str, float]] = []
    kinds: dict[str, list[dict[str, object]]] = {}
    for example in examples:
        values = audit_columns(build([example], world))[0]
        decoded += [
            (example.scored, example.t_ms, column, float(value))
            for column, value in zip(COLUMNS, values.tolist(), strict=True)
        ]
        held = {
            "accounts": [*example.payments.accounts, *example.identity.accounts],
            "payments": example.payments.payments,
            "owned_by": example.payments.owned_by,
            "customers": example.identity.customers,
            "devices": example.identity.devices,
            "ips": example.identity.ips,
            "enrolled": example.identity.enrolled,
            "owner_enrolled": example.identity.owner_enrolled,
            "seen": example.identity.seen,
            "money_flow": list(example.flows.values()),
            "tracked_inflows": list(example.inflows.values()),
            "counterparties": list(example.counterparties.values()),
            "activity": list(example.activity.values()),
        }
        for kind, rows in held.items():
            kinds.setdefault(kind, []).extend(
                {"mule_id": example.scored, "t_ms": example.t_ms, **asdict(row)} for row in rows
            )
    columns = ["account_id", "t_ms", "column", "value"]
    _write(target, "rows", pd.DataFrame(decoded, columns=columns), marks)
    for kind, rows in kinds.items():
        _write(target, f"subgraph_{kind}", pd.DataFrame(rows), marks)
    _write(target, "ring_mates", ring_mates(frame, reported), marks)


def ring_mates(frame: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    """For each reported mule of a ring, every other scored account of its ring on each
    Monday: its status, logit and rank fraction in that Monday's ranking (against it, when it
    is a mule outside its activity), and whether the bank identified it."""
    ranking = Ranking.of(frame, (frame["status"] == "active").to_numpy())
    weight = ranking.weight
    # The rows of the true mules, by ring, found once: the frame holds every scored account
    # and each reported mule needs its ring's rows only.
    mule_rows = frame[frame["status"] != NON_MULE]
    rings = {int(str(ring)): held for ring, held in mule_rows.groupby("mule_ring_id")}
    rows: list[pd.DataFrame] = []
    for account, ring in zip(reported["account_id"], reported["mule_ring_id"], strict=True):
        if int(ring) < 0 or int(ring) not in rings:
            continue
        ring_rows = rings[int(ring)]
        mates = ring_rows[ring_rows["account_id"] != account]
        if mates.empty:
            continue
        _, fractions = ranking.order.against(
            weight,
            mates["t_ms"].to_numpy(dtype=np.int64),
            mates["logit"].to_numpy(dtype=np.float64),
        )
        rows.append(
            pd.DataFrame(
                {
                    "account_id": account,
                    "mate_id": mates["account_id"].to_numpy(),
                    "t_ms": mates["t_ms"].to_numpy(),
                    "status": mates["status"].to_numpy(),
                    "logit": mates["logit"].to_numpy(),
                    "rank": fractions,
                    "identified": mates["identified"].to_numpy(),
                }
            )
        )
    columns = ["account_id", "mate_id", "t_ms", "status", "logit", "rank", "identified"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
