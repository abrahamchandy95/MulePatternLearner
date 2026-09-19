"""External account supervision, grouped holdouts and nested positive masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import stable_score as stable_score

import numpy as np
from numpy.typing import NDArray
import pandas as pd


def account_groups(nodes: pd.DataFrame, associations: pd.DataFrame) -> dict[int, str]:
    """Group co-owned accounts through parties, including repeated tenures.

    Full-extract ownership is allowed solely for conservative split isolation;
    those future links never become model features before their valid start.
    """
    accounts = nodes[nodes["node_type"] == "Account"]
    parent = {int(i): int(i) for i in nodes["node_index"]}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = associations[associations["relation"] == "Party_Owns_Account"]
    for src, dst in links[["src", "dst"]].itertuples(index=False, name=None):
        a, b = root(int(src)), root(int(dst))
        parent[max(a, b)] = min(a, b)
    ids = nodes.set_index("node_index")["id"].to_dict()
    return {int(i): str(ids[root(int(i))]) for i in accounts["node_index"]}


def split_accounts(
    nodes: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    seed: int = 42,
    test_groups: set[str] | None = None,
) -> pd.DataFrame:
    accounts = nodes[
        (nodes["node_type"] == "Account")
        & ~nodes["is_external"].astype(bool)
        & (nodes["subtype"] == "deposit")
    ].copy()
    groups = account_groups(nodes, associations)
    accounts["group_id"] = accounts["node_index"].map(groups)
    scores = accounts["group_id"].map(lambda value: stable_score(str(value), seed, "split"))
    accounts["split"] = np.where(
        scores < 0.7, "train", np.where(scores < 0.85, "validation", "test")
    )
    if test_groups:
        accounts.loc[accounts["group_id"].isin(test_groups), "split"] = "test"
    return accounts[["node_index", "id", "group_id", "split", "first_seen_ts_ms"]].reset_index(
        drop=True
    )


def reveal_mask(accounts: pd.DataFrame, fraction: float, seed: int) -> NDArray[np.bool_]:
    if not 0 <= fraction <= 1:
        raise ValueError("Reveal fraction must be in [0,1]")
    # Selection is independent of truth, time, split and ring. Fractions are nested.
    # Applied to positives only by the training routine; an account stays hidden
    # at every cutoff. Never force-reveal a positive when a low budget yields zero.
    return (
        accounts["id"]
        .map(lambda value: stable_score(str(value), seed, "reveal") < fraction)
        .to_numpy(bool)
    )


def read_labels(path: Path, nodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_parquet(path)
    required = {"account_id", "target", "effective_ts_ms", "available_ts_ms", "ring_id"}
    if not required <= set(labels.columns):
        raise ValueError(f"Missing label fields: {sorted(required - set(labels.columns))}")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("target_definition") not in {
        "confirmed_mule",
        "synthetic_laundering_intermediary",
        "account_mule",
    }:
        raise ValueError("Target must be explicitly defined in label metadata")
    if not metadata.get("complete_negative_ground_truth", False):
        raise ValueError(
            "Evaluation requires explicit negative ground truth; unreviewed accounts are not negatives"
        )
    if labels["account_id"].duplicated().any():
        raise ValueError(
            "Labels require one record per account; adjudicate multiple memberships before import"
        )
    if not labels["target"].isin([0, 1]).all():
        raise ValueError("Evaluation targets must be binary")
    if (labels["available_ts_ms"] < labels["effective_ts_ms"]).any():
        raise ValueError("A label cannot be available before its effective time")
    mapping = nodes[nodes["node_type"] == "Account"].set_index("id")["node_index"]
    labels["node_index"] = labels["account_id"].map(mapping)
    if labels["node_index"].isna().any():
        raise ValueError("Label account absent from staged graph")
    return labels, metadata


def labels_at_cutoff(
    accounts: pd.DataFrame, labels: pd.DataFrame, cutoff_ms: int, reveal: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    lookup = labels.set_index("node_index").reindex(accounts["node_index"])
    truth = np.full(len(accounts), -1, dtype=np.int64)
    explicit_negative = lookup["target"].eq(0).to_numpy()
    known_positive = lookup["target"].eq(1).to_numpy()
    # This target is participation by the cutoff, not eventual participation.
    active_positive = known_positive & (lookup["effective_ts_ms"].to_numpy() < cutoff_ms)
    truth[explicit_negative] = 0
    truth[known_positive] = 0
    truth[active_positive] = 1
    available = active_positive & (lookup["available_ts_ms"].to_numpy() < cutoff_ms)
    observed = available & reveal & accounts["split"].eq("train").to_numpy()
    return truth, observed


def graph_reveal_mask(accounts: pd.DataFrame, labels: pd.DataFrame) -> NDArray[np.bool_]:
    """Use the explicit persisted mask, with truth only to validate its contract."""
    required = {"graph_pu_label", "graph_is_mule_masked"}
    if not required <= set(labels.columns):
        raise ValueError("Graph-mask mode requires staged Account mask fields")
    expected = labels["target"].eq(1) & ~labels["graph_is_mule_masked"].astype(bool)
    if not np.array_equal(labels["graph_pu_label"].to_numpy(), expected.to_numpy(np.int64)):
        raise ValueError("Stored Account mask and pu_label disagree")
    pu = accounts["node_index"].map(labels.set_index("node_index")["graph_pu_label"]).fillna(0)
    return pu.eq(1).to_numpy(bool)


def forecast_targets(
    stage: Path, accounts: pd.DataFrame, cutoff_ms: int, horizon_ms: int
) -> NDArray[np.int64]:
    """Label-free pretraining: any outgoing Zelle payment in the next horizon.

    Future events are targets only. This task is not mule detection.
    """
    manifest = json.loads((stage / "manifest.json").read_text())
    # Require evidence through the complete target horizon, not just its start.
    if cutoff_ms + horizon_ms > int(manifest["max_ts_ms"]):
        raise ValueError("Forecast horizon exceeds source coverage")
    events = pd.read_parquet(stage / "events.parquet", columns=["sender", "event_ts_ms", "rail"])
    future = events[
        (events["event_ts_ms"] >= cutoff_ms)
        & (events["event_ts_ms"] < cutoff_ms + horizon_ms)
        & (events["rail"] == 1)
    ]
    return accounts["node_index"].isin(future["sender"].unique()).to_numpy().astype(np.int64)
