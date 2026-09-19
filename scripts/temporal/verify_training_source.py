"""Read-only live count and pair-encoding parity checks against a staged export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from mule_pattern_learner.temporal.encoding import BASIS_ID, fourier64
from mule_pattern_learner.temporal.staging import RAILS, digest
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path("artifacts/temporal/stage"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/experiments/training_source_verification.json")
    )
    args = parser.parse_args()
    manifest = json.loads((args.stage / "manifest.json").read_text())
    conn = Client(Settings()).conn
    if conn.graphname != manifest["graphname"]:
        raise ValueError("Connected graph differs from staging manifest")
    counts = conn.getVertexCount("*")
    if counts != manifest["source_vertex_counts"]:
        raise ValueError("Live counts changed; refresh loader verification and staging")
    nodes = pd.read_parquet(args.stage / "nodes.parquet").set_index("node_index")
    events = pd.read_parquet(args.stage / "events.parquet")
    seed_seq = int(events["event_seq"].max()) + 1
    seed_ts = int(events["event_ts_ms"].max()) + 1
    checks = []
    max_error = 0.0
    for rail in (1, 2):
        candidates = events[events["rail"] == rail].groupby(["sender", "counterparty"]).size()
        pairs = candidates[(candidates >= 3) & (candidates <= 20)].head(2)
        for pair, count in pairs.items():
            src, dst = cast(tuple[int, int], pair)
            source = cast("pd.Series[Any]", nodes.loc[src])
            target = cast("pd.Series[Any]", nodes.loc[dst])
            params: dict[str, Any] = {
                "sender": (str(source["id"]),),
                "recipient_type": str(target["node_type"]),
                "recipient_id": str(target["id"]),
                "seed_seq": seed_seq,
                "seed_ts_ms": seed_ts,
                "persist": False,
                "max_events": 1000,
            }
            query = "zelle_pair_time64" if rail == 1 else "payment_pair_time64"
            if rail != 1:
                params["payment_rail"] = RAILS[rail]
            result = conn.runInstalledQuery(query, params, usePost=True, timeout=120000)
            summary = next(row for row in result if "status" in row)
            if summary["status"] != "ok" or summary["basis_id"] != BASIS_ID or summary["persisted"]:
                raise ValueError("Live time query failed")
            actual = {row["event_id"]: row for row in result if "event_id" in row}
            expected = events[
                (events["sender"] == src)
                & (events["counterparty"] == dst)
                & (events["rail"] == rail)
            ]
            if set(actual) != set(expected["event_id"]):
                raise ValueError("Live pair events differ from staged pair events")
            for row in expected.to_dict("records"):
                live = actual[row["event_id"]]
                if int(live["pair_delta_t_ms"]) != int(row["pair_gap_ms"]) or bool(
                    live["pair_delta_t_present"]
                ) != bool(row["gap_present"]):
                    raise ValueError("Live and staged pair gap differ")
                vectors = [(live["age_time_encoding"], seed_ts - int(row["event_ts_ms"]))]
                if row["gap_present"]:
                    vectors.append((live["pair_time_encoding"], int(row["pair_gap_ms"])))
                elif live["pair_time_encoding"]:
                    raise ValueError("First payment unexpectedly has a pair encoding")
                for values, delta in vectors:
                    if len(values) != 64:
                        raise ValueError("Expected 64 coordinates")
                    reference = fourier64(np.array([delta], dtype=np.int64))[0]
                    error = float(np.max(np.abs(np.asarray(values) - reference)))
                    if error > 1e-5:
                        raise ValueError("Python and GSQL Fourier bases differ")
                    max_error = max(max_error, error)
            checks.append({"rail": RAILS[rail], "events": int(count)})
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "graphname": conn.graphname,
        "stage_sha256": digest(args.stage / "manifest.json"),
        "vertex_counts": counts,
        "pair_samples": checks,
        "basis_id": BASIS_ID,
        "maximum_coordinate_error": max_error,
        "writes": 0,
        "scope": "All vertex counts; sampled pair events and encodings. Loader verification covers all source shard counts; this is not a full live attribute checksum.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
