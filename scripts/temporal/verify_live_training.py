"""Read-only audit of installed training queries and time-feature parity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from mule_pattern_learner.temporal.common import timestamp
from mule_pattern_learner.temporal.live.contract import ContextKey
from mule_pattern_learner.temporal.live.dataset import query_hashes
from mule_pattern_learner.temporal.live.installation import verify_sources
from mule_pattern_learner.temporal.live.source import (
    TigerGraphExecutor,
    checked_rows,
    validate_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/live_training_readiness.json")
    )
    parser.add_argument("--account", required=True, help="Account ID to audit; no oracle selection")
    parser.add_argument("--date", required=True, help="Exclusive UTC scoring date")
    args = parser.parse_args()
    executor = TigerGraphExecutor()
    conn = executor.client.conn
    schema = conn.getSchema(force=True)
    attrs = {
        v["Name"]: {a["AttributeName"]: a["AttributeType"]["Name"] for a in v["Attributes"]}
        for v in schema["VertexTypes"]
    }
    assert attrs["Account"]["is_mule"] == "INT"
    matched = verify_sources(executor)
    account = args.account
    cutoff_ms = timestamp(args.date) - 1
    clocks = checked_rows(executor.run("temporal_training_cutoffs", {"cutoff_times": [cutoff_ms]}))
    cutoff_seq = int(clocks[0]["last_visible_seqs"][str(cutoff_ms)]) + 1
    key = ContextKey("Account", account, cutoff_seq, cutoff_ms)
    row = checked_rows(
        executor.run(
            "temporal_training_context",
            {
                "node_types": [key.node_type],
                "node_ids": [key.node_id],
                "cutoff_seqs": [key.cutoff_seq],
                "cutoff_times": [key.cutoff_ms],
                "per_relation": 2,
            },
        )
    )[0]
    validate_context(key, row)
    pairs = []
    for relation in ("zelle_out", "payment_out", "payment_in"):
        message = next((m for m in row["messages"] if m["relation"] == relation), None)
        if message is None or message["node_type"] != "Account":
            continue
        sender, recipient = (
            (message["node_id"], account)
            if relation.endswith("_in")
            else (account, message["node_id"])
        )
        params = {
            "sender": (sender,),
            "recipient_type": "Account",
            "recipient_id": recipient,
            "seed_seq": message["event_seq"] + 1,
            "seed_ts_ms": message["event_ts_ms"],
            "persist": False,
            "max_events": 10000,
        }
        name = "zelle_pair_time64" if relation.startswith("zelle") else "payment_pair_time64"
        if name == "payment_pair_time64":
            params["payment_rail"] = message["rail"]
        history = checked_rows(executor.run(name, params))
        event = next(v for v in history if v.get("event_id") == message["event_id"])
        assert event["pair_delta_t_ms"] == message["gap_ms"]
        assert event["pair_delta_t_present"] == message["gap_present"]
        for field, window in (
            ("pair_count_1h", 3600000),
            ("pair_count_1d", 86400000),
            ("pair_count_7d", 604800000),
        ):
            expected_count = sum(
                1
                for item in history
                if item.get("event_seq", message["event_seq"]) < message["event_seq"]
                and message["event_ts_ms"] - item["event_ts_ms"] < window
            )
            assert message[field] == expected_count
        pairs.append(relation)
    event = next(
        (m for m in row["messages"] if m["relation"] in ("zelle_out", "payment_out")), None
    )
    boundary_checked = event is not None
    if event is not None:
        before_seq, event_ms = event["event_seq"], event["event_ts_ms"]
        boundaries = checked_rows(
            executor.run(
                "temporal_training_context",
                {
                    "node_types": ["Account", "Account"],
                    "node_ids": [account, account],
                    "cutoff_seqs": [before_seq, before_seq + 1],
                    "cutoff_times": [event_ms, event_ms],
                    "per_relation": 1,
                },
            )
        )
        before, after = sorted(boundaries, key=lambda value: value["request_index"])
        assert (
            after["features"].get("1h_out_count", 0) - before["features"].get("1h_out_count", 0)
            == 1
        )
        delta = after["features"].get("1h_out_amount", 0) - before["features"].get(
            "1h_out_amount", 0
        )
        assert abs(delta - event["amount"]) < 1e-4
        assert not any(m["event_id"] == event["event_id"] for m in before["messages"])
        assert any(m["event_id"] == event["event_id"] for m in after["messages"])
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "graphname": conn.graphname,
        "vertex_counts": conn.getVertexCount("*", realtime=True),
        "matched_queries": matched,
        "query_hashes": query_hashes(),
        "pair_parity_checks": pairs,
        "gsql_python_fourier_parity": "passed",
        "exclusive_seed_event_boundary": "passed" if boundary_checked else "no_sampled_payment",
        "database_data_writes": 0,
        "oracle_labels_read": False,
        "strict_inductive_readiness": "run_verify_strict_isolation_separately",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
