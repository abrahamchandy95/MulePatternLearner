"""Stage the immutable loader export; only allowlisted facts enter model data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import digest as digest

import numpy as np
import pandas as pd

from .account_labels import ACCOUNT_FACT_COLUMNS, ACCOUNT_LOAD_COLUMNS, write_account_labels

ENTITY_COLUMNS = {
    "Party": ["id", "party_type", "first_seen_seq", "first_seen_ts_ms"],
    "Account": ACCOUNT_FACT_COLUMNS,
    "Token": ["token_id", "first_seen_seq", "first_seen_ts_ms", "token_kind", "token_network"],
    "Device": ["id", "device_type", "first_seen_seq", "first_seen_ts_ms"],
    "IP": ["id", "first_seen_seq", "first_seen_ts_ms"],
    "Address": ["address_id", "first_seen_seq", "first_seen_ts_ms", "country_code"],
}
EVENT_COLUMNS = {
    "Payment_Transaction": [
        "transaction_id",
        "amount",
        "currency",
        "payment_rail",
        "channel",
        "event_time",
        "event_ts_ms",
        "event_seq",
        "amount_present",
    ],
    "Zelle_Transfer": [
        "transfer_id",
        "event_time",
        "event_ts_ms",
        "event_seq",
        "amount",
        "amount_present",
        "currency",
        "channel",
        "fraud_label",
        "label_known",
        "label_available_seq",
        "label_available_ts_ms",
    ],
}
ASSOCIATIONS = {
    "Party_Owns_Account": ("Party", "Account"),
    "Party_Uses_Token": ("Party", "Token"),
    "Token_Bound_To_Account": ("Token", "Account"),
    "Party_Uses_Device": ("Party", "Device"),
    "Account_Uses_Device": ("Account", "Device"),
    "Party_Uses_IP": ("Party", "IP"),
    "Party_Has_Address": ("Party", "Address"),
}
RAILS = ("unknown", "zelle", "ach", "card", "cash", "check", "internal")


def read_dataset(
    manifest: dict[str, Any], name: str, columns: list[str], selected: list[str] | None = None
) -> pd.DataFrame:
    # No labels are even parsed when reading payment features.
    parts = [
        pd.read_csv(
            s["path"], sep="|", header=None, names=columns, usecols=selected, keep_default_na=False
        )
        for s in manifest["datasets"][name]["shards"]
        if s["rows"]
    ]
    if not parts:
        return pd.DataFrame(columns=selected or columns)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != manifest["datasets"][name]["rows"]:
        raise ValueError(f"Row-count mismatch for {name}")
    return frame


def stage(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    if output.exists():
        report = json.loads((output / "manifest.json").read_text())
        if report["source_manifest_sha256"] != manifest_hash:
            raise ValueError("Existing stage belongs to a different source snapshot")
        return report
    verification_path = manifest_path.parent / "tigergraph_verification.json"
    verification = json.loads(verification_path.read_text())
    if not verification.get("passed") or verification["manifest_sha256"] != manifest_hash:
        raise ValueError("A matching successful TigerGraph loader verification is required")
    for dataset in manifest["datasets"].values():
        for shard in dataset["shards"]:
            if digest(Path(shard["path"])) != shard["sha256"]:
                raise ValueError("Loader shard checksum mismatch")
    work = output.with_name(output.name + ".building")
    if work.exists():
        raise FileExistsError(f"Incomplete staging directory: {work}")
    work.mkdir(parents=True)
    frames = []
    label_count = 0
    for node_type, columns in ENTITY_COLUMNS.items():
        if node_type == "Account":
            widths = set()
            for shard in manifest["datasets"][node_type]["shards"]:
                if shard["rows"]:
                    with Path(shard["path"]).open(newline="") as stream:
                        widths.add(len(next(csv.reader(stream, delimiter="|"))))
            if widths == {len(ACCOUNT_LOAD_COLUMNS)}:
                supervision = read_dataset(manifest, node_type, ACCOUNT_LOAD_COLUMNS)
                label_count = write_account_labels(
                    supervision, work / "account_labels.parquet", manifest_hash
                )
                frame = supervision[ACCOUNT_FACT_COLUMNS].copy()
            elif widths <= {len(ACCOUNT_FACT_COLUMNS)}:
                frame = read_dataset(manifest, node_type, ACCOUNT_FACT_COLUMNS)
            else:
                raise ValueError(
                    "Account export must use a uniform five-column legacy or fifteen-column supervised contract"
                )
        else:
            frame = read_dataset(manifest, node_type, columns).rename(columns={columns[0]: "id"})
        frame["node_type"] = node_type
        frame["subtype"] = frame.get(
            {
                "Account": "account_type",
                "Token": "token_kind",
                "Device": "device_type",
                "Party": "party_type",
            }.get(node_type, "_missing"),
            "unknown",
        )
        if "is_external" not in frame:
            frame["is_external"] = False
        frames.append(
            frame[
                ["id", "node_type", "subtype", "is_external", "first_seen_seq", "first_seen_ts_ms"]
            ]
        )
    nodes = (
        pd.concat(frames, ignore_index=True).sort_values(["node_type", "id"]).reset_index(drop=True)
    )
    nodes.index += 1  # zero is a padding node, never a real identity
    nodes.insert(0, "node_index", nodes.index)
    if nodes["id"].duplicated().any():
        raise ValueError("Entity IDs must be domain-separated")
    mapping = dict(zip(nodes["id"], nodes["node_index"]))
    nodes.to_parquet(work / "nodes.parquet", index=False)
    events = []
    for kind, columns in EVENT_COLUMNS.items():
        key = columns[0]
        selected = [key, "amount", "currency", "event_seq", "event_ts_ms", "amount_present"]
        if kind == "Payment_Transaction":
            selected.append("payment_rail")
        frame = read_dataset(manifest, kind, columns, selected).rename(columns={key: "event_id"})
        frame["payment_rail"] = frame.get("payment_rail", "zelle")
        frame["rail"] = (
            frame["payment_rail"]
            .map({name: i for i, name in enumerate(RAILS)})
            .fillna(0)
            .astype(np.int8)
        )
        role_prefix = "Transfer" if kind == "Zelle_Transfer" else "Transaction"
        frame = frame.set_index("event_id")
        for role, name in [
            ("From_Account", "sender"),
            ("To_Account", "recipient"),
            ("From_Token", "sender_token"),
            ("To_Token", "recipient_token"),
            ("Used_Device", "device"),
            ("Used_IP", "ip"),
        ]:
            table = read_dataset(
                manifest, role_prefix + "_" + role, ["from_id", "to_id", "event_ts_ms", "event_seq"]
            )
            if table["from_id"].duplicated().any():
                raise ValueError(f"Ambiguous event role {role}")
            indices = table["to_id"].map(mapping)
            if indices.isna().any():
                raise ValueError("Unknown event endpoint")
            role_map = pd.Series(indices.to_numpy(), index=table["from_id"])
            frame[name] = frame.index.map(role_map).fillna(0).astype(np.int32)
            if len(table):
                clocks = frame.reindex(table["from_id"])
                if (clocks["event_seq"].to_numpy() != table["event_seq"].to_numpy()).any() or (
                    clocks["event_ts_ms"].to_numpy() != table["event_ts_ms"].to_numpy()
                ).any():
                    raise ValueError("Participation clocks differ from payment clocks")
        events.append(frame.reset_index())
    payments = pd.concat(events, ignore_index=True).sort_values("event_seq").reset_index(drop=True)
    # An empty rail export otherwise promotes concatenated numeric columns to
    # object dtype. Single-rail datasets are valid inputs.
    payments["amount"] = pd.to_numeric(payments["amount"], errors="raise").astype(np.float64)
    payments["event_seq"] = payments["event_seq"].astype(np.int64)
    payments["event_ts_ms"] = payments["event_ts_ms"].astype(np.int64)
    if payments["event_seq"].duplicated().any() or payments["event_id"].duplicated().any():
        raise ValueError("Duplicate payment identity or sequence")
    if (np.diff(payments["event_ts_ms"].to_numpy(np.int64)) < 0).any():
        raise ValueError("Payment timestamps decrease in sequence order")
    if (payments["sender"] == 0).any() or (
        (payments["recipient"] == 0) & (payments["recipient_token"] == 0)
    ).any():
        raise ValueError("Payment lacks sender or recipient")
    if not np.isfinite(payments["amount"].to_numpy()).all() or (payments["amount"] < 0).any():
        raise ValueError("Invalid payment amount")
    if payments["currency"].nunique() > 1:
        raise ValueError("Convert multi-currency amounts with cutoff-available FX before staging")
    payments["counterparty"] = np.where(
        payments["recipient"] > 0, payments["recipient"], payments["recipient_token"]
    ).astype(np.int32)
    # Integer subtraction: converting epoch milliseconds to float first loses precision.
    previous = (
        payments.groupby(["sender", "counterparty", "rail"], sort=False)["event_ts_ms"]
        .shift(fill_value=0)
        .to_numpy(np.int64)
    )
    payments["gap_present"] = previous != 0
    payments["pair_gap_ms"] = np.where(
        previous != 0, payments["event_ts_ms"].to_numpy(np.int64) - previous, 0
    )
    payments = payments.drop(columns=["currency", "payment_rail"])
    payments.to_parquet(work / "events.parquet", index=False)
    associations = []
    for relation in ASSOCIATIONS:
        frame = read_dataset(
            manifest,
            relation,
            ["from_id", "to_id", "valid_from_seq", "valid_to_seq", "confidence", "source_system"],
        )
        frame["src"] = frame["from_id"].map(mapping)
        frame["dst"] = frame["to_id"].map(mapping)
        if frame[["src", "dst"]].isna().any().any():
            raise ValueError("Unknown association endpoint")
        frame["relation"] = relation
        associations.append(frame[["src", "dst", "relation", "valid_from_seq", "valid_to_seq"]])
    associations_frame = pd.concat(associations, ignore_index=True)
    associations_frame.to_parquet(work / "associations.parquet", index=False)
    report = {
        "source_manifest_sha256": manifest_hash,
        "source_database": manifest["source_database"],
        "loader_verification_sha256": digest(verification_path),
        "graphname": manifest["graphname"],
        "nodes": len(nodes),
        "events": len(payments),
        "associations": len(associations_frame),
        "min_ts_ms": int(payments["event_ts_ms"].min()),
        "max_ts_ms": int(payments["event_ts_ms"].max()),
        "source_vertex_counts": {
            name: manifest["datasets"][name]["rows"] for name in (*ENTITY_COLUMNS, *EVENT_COLUMNS)
        },
        "feature_columns": list(payments.columns),
        "supervision_in_features": False,
        "known_account_labels": label_count,
    }
    (work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    work.rename(output)
    return report
