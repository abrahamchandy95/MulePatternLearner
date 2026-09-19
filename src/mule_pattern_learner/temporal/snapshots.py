"""Causal, bounded heterogeneous neighborhoods built from staged event history."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .staging import ASSOCIATIONS, ENTITY_COLUMNS, RAILS, digest

DAY_MS = 86_400_000
SNAPSHOT_VERSION = 2


@dataclass
class Snapshot:
    x: NDArray[np.float32]
    neighbors: NDArray[np.int32]
    relation: NDArray[np.int16]
    rail: NDArray[np.int8]
    edge: NDArray[np.float32]
    age_ms: NDArray[np.int64]
    gap_ms: NDArray[np.int64]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Snapshot:
        arrays = {
            name: np.load(path / f"{name}.npy", mmap_mode="r")
            for name in ("x", "neighbors", "relation", "rail", "edge", "age_ms", "gap_ms")
        }
        return cls(**arrays, metadata=json.loads((path / "metadata.json").read_text()))


def visible(start: NDArray[np.int64], end: NDArray[np.int64], seed_seq: int) -> NDArray[np.bool_]:
    return (start <= seed_seq) & ((end == 0) | (seed_seq < end))


def _last_per_source(
    src: NDArray[np.int64], seq: NDArray[np.int64], count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return at most count positions per source, with newest rank zero."""
    if not len(src):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.lexsort((seq, src))
    ordered = src[order]
    end = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(order)]
    lengths = np.diff(np.r_[0, end])
    ranks = np.repeat(end, lengths) - 1 - np.arange(len(order))
    keep = ranks < count
    return order[keep], ranks[keep]


def build_snapshot(
    stage: Path,
    output: Path,
    cutoff_ms: int,
    *,
    history_days: int = 90,
    slots: int = 32,
    per_relation: int = 4,
) -> dict[str, Any]:
    stage_hash = digest(stage / "manifest.json")
    if output.exists():
        metadata = json.loads((output / "metadata.json").read_text())
        if metadata.get("snapshot_version") != SNAPSHOT_VERSION:
            raise ValueError("Snapshot implementation changed; rebuild into a new directory")
        if (
            metadata["stage_sha256"],
            metadata["cutoff_ms"],
            metadata["history_days"],
            metadata["slots"],
            metadata["per_relation"],
        ) != (stage_hash, cutoff_ms, history_days, slots, per_relation):
            raise ValueError("Snapshot configuration mismatch")
        return metadata
    events = pd.read_parquet(stage / "events.parquet")
    nodes = pd.read_parquet(stage / "nodes.parquet")
    associations = pd.read_parquet(stage / "associations.parquet")
    n = int(nodes["node_index"].max()) + 1
    # Cutoff is an exclusive timestamp boundary. All observations in a tied
    # timestamp group stay together; arbitrary within-second ordering is unused.
    prefix_size = int(np.searchsorted(events["event_ts_ms"].to_numpy(), cutoff_ms, side="left"))
    history = events.iloc[:prefix_size]
    if not len(history):
        raise ValueError("No payments before requested cutoff")
    # Last witnessed sequence strictly before the timestamp boundary. Using the
    # next payment's sequence would expose associations beginning in the future.
    past_nodes = nodes[nodes["first_seen_ts_ms"] < cutoff_ms]
    seed_seq = max(
        int(history["event_seq"].max()),
        int(past_nodes["first_seen_seq"].max()) if len(past_nodes) else 0,
    )
    idx = nodes["node_index"].to_numpy(np.int64)
    exists = np.zeros(n, dtype=bool)
    exists[idx] = (nodes["first_seen_seq"].to_numpy() <= seed_seq) & (
        nodes["first_seen_ts_ms"].to_numpy() < cutoff_ms
    )
    associations = associations[
        exists[associations["src"].to_numpy(np.int64)]
        & exists[associations["dst"].to_numpy(np.int64)]
    ]
    columns: list[NDArray[np.float32]] = []
    names: list[str] = []

    def feature(name: str, values: NDArray[Any]) -> None:
        values = np.asarray(values, dtype=np.float32)
        values[~exists] = 0
        columns.append(values)
        names.append(name)

    for node_type in ENTITY_COLUMNS:
        vector = np.zeros(n)
        vector[idx] = (nodes["node_type"] == node_type).to_numpy()
        feature("type_" + node_type, vector)
    # Fixed vocabulary; no full-window fitting or identifier embeddings.
    for subtype in ("deposit", "credit", "brokerage", "phone", "email", "handle"):
        vector = np.zeros(n)
        vector[idx] = (nodes["subtype"] == subtype).to_numpy()
        feature("subtype_" + subtype, vector)
    external = np.zeros(n)
    external[idx] = nodes["is_external"].astype(bool)
    feature("external", external)
    age = np.zeros(n)
    age[idx] = np.log1p(
        np.maximum(0, cutoff_ms - nodes["first_seen_ts_ms"].to_numpy(np.int64)) / DAY_MS
    )
    feature("entity_age_log_days", age)

    for days in dict.fromkeys((1, 7, 30, history_days)):
        window = history[history["event_ts_ms"] >= cutoff_ms - days * DAY_MS]
        src = window["sender"].to_numpy(np.int64)
        dst = window["counterparty"].to_numpy(np.int64)
        amount = window["amount"].to_numpy(np.float64) * window["amount_present"].to_numpy(bool)
        count_out = np.bincount(src, minlength=n).astype(float)
        count_in = np.bincount(dst, minlength=n).astype(float)
        amount_out = np.bincount(src, weights=amount, minlength=n)
        amount_in = np.bincount(dst, weights=amount, minlength=n)
        for label, arr in (
            ("out_count", count_out),
            ("in_count", count_in),
            ("out_amount", amount_out),
            ("in_amount", amount_in),
        ):
            feature(f"{label}_{days}d", np.log1p(arr))
        pairs = np.unique(src * n + dst)
        unique_out = np.bincount(pairs // n, minlength=n)
        unique_in = np.bincount(pairs % n, minlength=n)
        feature(f"out_peers_{days}d", np.log1p(unique_out))
        feature(f"in_peers_{days}d", np.log1p(unique_in))
        feature(
            f"out_repeat_fraction_{days}d",
            np.where(count_out > 0, 1 - unique_out / np.maximum(count_out, 1), 0),
        )
        feature(f"flow_balance_{days}d", (amount_out - amount_in) / (1 + amount_out + amount_in))
        # Label-free shared-entity activity, including tokens, devices and IPs.
        for role in ("device", "ip", "sender_token", "recipient_token"):
            role_ids = window[role].to_numpy(np.int64)
            feature(
                f"{role}_activity_{days}d",
                np.log1p(np.bincount(role_ids[role_ids > 0], minlength=n)),
            )
        for rail_i, rail_name in enumerate(RAILS):
            selected = window["rail"].to_numpy() == rail_i
            feature(
                f"rail_{rail_name}_{days}d",
                np.log1p(np.bincount(np.r_[src[selected], dst[selected]], minlength=n)),
            )
    for side in ("sender", "counterparty"):
        last = np.zeros(n, dtype=np.int64)
        np.maximum.at(
            last, history[side].to_numpy(np.int64), history["event_ts_ms"].to_numpy(np.int64)
        )
        feature(side + "_history_present", last > 0)
        feature(
            side + "_recency_log_seconds",
            np.where(last > 0, np.log1p(np.maximum(0, cutoff_ms - last) / 1000), 0),
        )
    starts = associations["valid_from_seq"].to_numpy(np.int64)
    ends = associations["valid_to_seq"].to_numpy(np.int64)
    active = visible(starts, ends, seed_seq)
    for rel in ASSOCIATIONS:
        rel_mask = (associations["relation"] == rel).to_numpy()
        for suffix, mask in (
            ("active", active & rel_mask),
            ("ended", (ends > 0) & (ends <= seed_seq) & rel_mask),
        ):
            endpoints = associations.loc[mask, ["src", "dst"]].to_numpy(np.int64).reshape(-1)
            feature(rel + "_" + suffix, np.log1p(np.bincount(endpoints, minlength=n)))
    x = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite snapshot feature")

    edge_parts = []
    recent = history[history["event_ts_ms"] >= cutoff_ms - history_days * DAY_MS]

    def add_event_relation(src_col: str, dst_col: str, relation: int) -> None:
        valid = (recent[src_col].to_numpy() > 0) & (recent[dst_col].to_numpy() > 0)
        frame = recent.loc[valid]
        src = frame[src_col].to_numpy(np.int64)
        selected, ranks = _last_per_source(src, frame["event_seq"].to_numpy(np.int64), per_relation)
        frame = frame.iloc[selected]
        edge_parts.append(
            pd.DataFrame(
                {
                    "src": frame[src_col].to_numpy(),
                    "dst": frame[dst_col].to_numpy(),
                    "relation": relation,
                    "rank": ranks,
                    "rail": frame["rail"].to_numpy(),
                    "amount": np.log1p(frame["amount"].to_numpy()),
                    "amount_present": frame["amount_present"].to_numpy(),
                    "gap_present": frame["gap_present"].to_numpy(),
                    "time_present": 1,
                    "age_ms": cutoff_ms - frame["event_ts_ms"].to_numpy(np.int64),
                    "gap_ms": frame["pair_gap_ms"].to_numpy(np.int64),
                }
            )
        )

    for relation, (src_col, dst_col) in enumerate(
        (
            ("sender", "counterparty"),
            ("counterparty", "sender"),
            ("sender", "sender_token"),
            ("sender_token", "sender"),
            ("recipient", "recipient_token"),
            ("recipient_token", "recipient"),
            ("sender", "device"),
            ("device", "sender"),
            ("sender", "ip"),
            ("ip", "sender"),
        )
    ):
        add_event_relation(src_col, dst_col, relation)
    # Exact witnessed boundary times only. Sequence gaps never stand in for time.
    witness = (
        pd.concat(
            [
                events[["event_seq", "event_ts_ms"]],
                nodes[["first_seen_seq", "first_seen_ts_ms"]].rename(
                    columns={"first_seen_seq": "event_seq", "first_seen_ts_ms": "event_ts_ms"}
                ),
            ]
        )
        .drop_duplicates("event_seq")
        .set_index("event_seq")["event_ts_ms"]
    )
    for rel_i, rel in enumerate(ASSOCIATIONS):
        frame = associations.loc[active & (associations["relation"] == rel).to_numpy()]
        for reverse in (0, 1):
            src_col, dst_col = ("src", "dst") if reverse == 0 else ("dst", "src")
            selected, ranks = _last_per_source(
                frame[src_col].to_numpy(np.int64),
                frame["valid_from_seq"].to_numpy(np.int64),
                per_relation,
            )
            f = frame.iloc[selected]
            timestamps = f["valid_from_seq"].map(witness).fillna(0).to_numpy(np.int64)
            time_present = (timestamps > 0) & (timestamps < cutoff_ms)
            edge_parts.append(
                pd.DataFrame(
                    {
                        "src": f[src_col].to_numpy(),
                        "dst": f[dst_col].to_numpy(),
                        "relation": 10 + 2 * rel_i + reverse,
                        "rank": ranks,
                        "rail": 0,
                        "amount": 0,
                        "amount_present": False,
                        "gap_present": False,
                        "time_present": time_present,
                        "age_ms": np.where(time_present, cutoff_ms - timestamps, 0),
                        "gap_ms": 0,
                    }
                )
            )
    edges = pd.concat(edge_parts, ignore_index=True)
    edges = edges[exists[edges["src"].to_numpy(np.int64)] & exists[edges["dst"].to_numpy(np.int64)]]
    # Round-robin relation coverage before taking a second-hop fanout prefix.
    edges = edges.sort_values(["src", "rank", "relation", "dst"], kind="stable")
    edges["slot"] = edges.groupby("src", sort=False).cumcount()
    edges = edges[edges["slot"] < slots]
    row = edges["src"].to_numpy(np.int64)
    col = edges["slot"].to_numpy(np.int64)
    arrays = {
        "x": x,
        "neighbors": np.zeros((n, slots), np.int32),
        "relation": np.zeros((n, slots), np.int16),
        "rail": np.zeros((n, slots), np.int8),
        "edge": np.zeros((n, slots, 4), np.float32),
        "age_ms": np.zeros((n, slots), np.int64),
        "gap_ms": np.zeros((n, slots), np.int64),
    }
    arrays["neighbors"][row, col] = edges["dst"].to_numpy(np.int32)
    arrays["relation"][row, col] = edges["relation"].to_numpy(np.int16)
    arrays["rail"][row, col] = edges["rail"].to_numpy(np.int8)
    arrays["edge"][row, col] = edges[
        ["amount", "amount_present", "gap_present", "time_present"]
    ].to_numpy(np.float32)
    arrays["age_ms"][row, col] = edges["age_ms"].to_numpy(np.int64)
    arrays["gap_ms"][row, col] = edges["gap_ms"].to_numpy(np.int64)
    work = output.with_name(output.name + ".building")
    work.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(work / f"{name}.npy", array)
    metadata = {
        "snapshot_version": SNAPSHOT_VERSION,
        "stage_sha256": stage_hash,
        "cutoff_ms": cutoff_ms,
        "seed_seq": seed_seq,
        "history_days": history_days,
        "slots": slots,
        "per_relation": per_relation,
        "feature_names": names,
        "feature_dim": x.shape[1],
        "relations": 24,
        "history_events": len(history),
        "sampled_edges": len(edges),
        "min_event_age_ms": int(edges.loc[edges["time_present"] == 1, "age_ms"].min())
        if edges["time_present"].any()
        else None,
        "root_semantics": "account state at cutoff; both message-passing hops use this cutoff",
    }
    (work / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    work.rename(output)
    return metadata
