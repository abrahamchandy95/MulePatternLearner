"""Explicit opt-in importer for synthetic laundering-intermediary supervision.

This target is NOT confirmed mule-account ground truth. The source has no
fraud_mule_in/fraud_mule_forward events. This importer never edits TigerGraph.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np
import pandas as pd

from mule_pattern_learner.temporal.staging import digest

# Restrict roles: victims funding a chain and its terminal cash-out destinations
# are not automatically intermediaries. Invoice and structuring roles are omitted.
ROLES = {
    "fraud_layering_in": ("dst_acct",),
    "fraud_layering_hop": ("src_acct", "dst_acct"),
    "fraud_layering_out": ("src_acct",),
    "fraud_scatter_gather_split": ("dst_acct",),
    "fraud_scatter_gather_merge": ("src_acct",),
    "fraud_cycle": ("src_acct", "dst_acct"),
}


def account_id(raw: str) -> str:
    return "a_" + hashlib.blake2b(("a:" + raw).encode(), digest_size=16).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-laundering-proxy", action="store_true", required=True)
    parser.add_argument("--database", default="dbname=phantomledger")
    parser.add_argument("--stage", type=Path, default=Path("artifacts/temporal/stage"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/temporal/labels/account_labels.parquet")
    )
    parser.add_argument("--label-delay-days", type=int, default=7)
    args = parser.parse_args()
    if args.label_delay_days < 0:
        raise ValueError("Label delay cannot be negative")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Use a new output path for a different label policy")
    # Fixed query, source roles only; raw account IDs stay in process memory.
    sql = """COPY (
      SELECT row_seq,src_acct,dst_acct,ring_id,fraud_type,
             (extract(epoch FROM ts AT TIME ZONE 'UTC')*1000)::bigint AS event_ts_ms
      FROM public.transactions
      WHERE is_fraud=1 AND ring_id>=0 AND fraud_type IN (
        'fraud_layering_in','fraud_layering_hop','fraud_layering_out',
        'fraud_scatter_gather_split','fraud_scatter_gather_merge','fraud_cycle')
      ORDER BY row_seq
    ) TO STDOUT WITH CSV HEADER"""
    completed = subprocess.run(
        ["psql", args.database, "-X", "-v", "ON_ERROR_STOP=1", "-c", sql],
        env={**os.environ, "PGOPTIONS": "-c default_transaction_read_only=on"},
        check=True,
        capture_output=True,
        text=True,
    )
    audit = pd.read_csv(io.StringIO(completed.stdout))
    nodes = pd.read_parquet(args.stage / "nodes.parquet")
    events = pd.read_parquet(
        args.stage / "events.parquet",
        columns=["event_id", "event_ts_ms", "event_seq", "sender", "recipient"],
    )
    lookup = events.set_index("event_id")
    identities = nodes.set_index("id")["node_index"].to_dict()
    records = []
    for row in audit.to_dict("records"):
        event = cast("pd.Series[Any]", lookup.loc["T" + str(row["row_seq"])])
        if int(event["event_ts_ms"]) != int(row["event_ts_ms"]):
            raise ValueError("Audit event clock does not match the staged graph")
        for role in ROLES[str(row["fraud_type"])]:
            opaque = account_id(str(row[role]))
            node_index = identities.get(opaque)
            if node_index is None or int(
                event["sender" if role == "src_acct" else "recipient"]
            ) != int(node_index):
                raise ValueError("Audit participant does not match the staged graph")
            records.append(
                {
                    "account_id": opaque,
                    "ring_id": int(row["ring_id"]),
                    "effective_ts_ms": int(row["event_ts_ms"]),
                }
            )
    positive = pd.DataFrame(records)
    if not len(positive):
        raise ValueError("No explicit intermediary roles found")
    if (positive.groupby("account_id")["ring_id"].nunique() > 1).any():
        raise ValueError("Multiple ring memberships require a membership-table adapter")
    positive = positive.groupby("account_id", as_index=False).agg(
        ring_id=("ring_id", "first"), effective_ts_ms=("effective_ts_ms", "min")
    )
    accounts = nodes[nodes["node_type"] == "Account"][["id"]].rename(columns={"id": "account_id"})
    labels = accounts.merge(positive, on="account_id", how="left", validate="one_to_one")
    labels["target"] = labels["effective_ts_ms"].notna().astype(np.int8)
    labels["ring_id"] = labels["ring_id"].fillna(-1).astype(np.int64)
    labels["effective_ts_ms"] = labels["effective_ts_ms"].fillna(0).astype(np.int64)
    labels["available_ts_ms"] = np.where(
        labels["target"] == 1, labels["effective_ts_ms"] + args.label_delay_days * 86_400_000, 0
    )
    metadata = {
        "target_definition": "synthetic_laundering_intermediary",
        "complete_negative_ground_truth": True,
        "negative_definition": "Not an explicit intermediary in the selected simulator roles; not a claim of legitimate real-world behavior",
        "positive_definition": "First observed participation in an explicitly enumerated intermediary role",
        "availability": "Simulated delay; source does not contain investigator adjudication times",
        "label_delay_days": args.label_delay_days,
        "roles": ROLES,
        "stage_sha256": digest(args.stage / "manifest.json"),
        "source_audit_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "accounts": len(labels),
        "positive_accounts": int(labels["target"].sum()),
        "ring_counts": positive.groupby("ring_id").size().to_dict(),
        "warning": "Proxy only; cannot establish mule-account performance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
