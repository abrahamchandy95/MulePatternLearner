"""Read-only numerical parity of new GSQL features against an independent reference.

Use account IDs chosen without labels. This intentionally audits unscoped raw
history; run verify_strict_isolation.py separately for endpoint isolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from mule_pattern_learner.temporal.common import timestamp
from mule_pattern_learner.temporal.live.contract import (
    ContextKey,
    FeaturePlan,
    FEATURE_GROUPS,
    SamplerPlan,
)
from mule_pattern_learner.temporal.live.history_reference import (
    payment_features,
    stratify,
    visible_history,
)
from mule_pattern_learner.temporal.live.queries import as_interpreted
from mule_pattern_learner.temporal.live.source import (
    TigerGraphExecutor,
    checked_rows,
    validate_context,
)


def audit_history(executor: TigerGraphExecutor, key: ContextKey) -> list[dict[str, Any]]:
    parts = [
        f"""INTERPRET QUERY () FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {{
    TYPEDEF TUPLE<STRING event_id, UINT event_seq, UINT event_ts_ms, STRING relation,
                  STRING node_type, STRING node_id, STRING rail, STRING currency,
                  DOUBLE amount, BOOL amount_present> Event;
    ListAccum<Event> @@history;
    SumAccum<INT> @@degree;
    VERTEX seed_vertex = to_vertex({json.dumps(key.node_id)}, "Account");
    Root = {{seed_vertex}};
    Guard = SELECT a FROM Root:a ACCUM @@degree += a.outdegree("Account_Sent_Zelle_Transfer")
      + a.outdegree("Account_Received_Zelle_Transfer") + a.outdegree("Account_Initiated_Transaction")
      + a.outdegree("Account_Received_Transaction");
    IF @@degree > 8192 THEN PRINT "audit_degree_exceeded" AS status; RETURN; END;
    """
    ]
    for prefix, vertex, pid, rail, stem, edges in (
        (
            "Transfer",
            "Zelle_Transfer",
            "transfer_id",
            '"zelle"',
            "zelle",
            ("Account_Sent_Zelle_Transfer", "Account_Received_Zelle_Transfer"),
        ),
        (
            "Transaction",
            "Payment_Transaction",
            "transaction_id",
            "t.payment_rail",
            "payment",
            ("Account_Initiated_Transaction", "Account_Received_Transaction"),
        ),
    ):
        for direction, edge in zip(("out", "in"), edges):
            role = "To" if direction == "out" else "From"
            parts.append(f"""Events_{stem}_{direction} = SELECT t FROM Root:a -({edge}>:e)- {vertex}:t
              WHERE t.event_seq < {key.cutoff_seq} AND t.event_ts_ms <= {key.cutoff_ms};""")
            for typ, attr in (("Account", "id"), ("Token", "token_id")):
                fallback = (
                    f' AND t.outdegree("{prefix}_{role}_Account") == 0' if typ == "Token" else ""
                )
                parts.append(f'''Peers_{stem}_{direction}_{typ} = SELECT t
                  FROM Events_{stem}_{direction}:t -({prefix}_{role}_{typ}>:e)- {typ}:p
                  WHERE e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms{fallback}
                  ACCUM @@history += Event(t.{pid}, t.event_seq, t.event_ts_ms, "{stem}_{direction}",
                    "{typ}", p.{attr}, {rail}, t.currency, t.amount, t.amount_present);''')
    parts.append('PRINT "ok" AS status, @@history AS history; }')
    rows = executor.client.conn.runInterpretedQuery("\n".join(parts))
    return checked_rows(rows)[0]["history"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", action="append", required=True)
    parser.add_argument("--date", default="2025-01-01")
    parser.add_argument("--query-name", default="temporal_training_context")
    parser.add_argument(
        "--interpreted",
        action="store_true",
        help="Validate the repository query without installation",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/temporal/feature_parity.json")
    )
    args = parser.parse_args()
    executor = TigerGraphExecutor()
    ms = timestamp(args.date) - 1
    clocks = checked_rows(executor.run("temporal_training_cutoffs", {"cutoff_times": [ms]}))
    seq = int(clocks[0]["last_visible_seqs"][str(ms)]) + 1
    groups = tuple(
        g
        for g in FEATURE_GROUPS
        if g not in ("rolling_windows", "amount_ratios", "pair_window_counts", "device_ip_context")
    )
    plan = FeaturePlan(groups, "split")
    sampler = SamplerPlan("stratified", 4, 3, 2)
    reports = []
    for account in args.account:
        key = ContextKey("Account", account, seq, ms)
        history = audit_history(executor, key)
        expected, decay = payment_features(history, key)
        params: dict[str, Any] = {
            "node_types": ["Account"],
            "node_ids": [account],
            "cutoff_seqs": [seq],
            "cutoff_times": [ms],
            **sampler.query_params(),
            # Print the Fourier vectors so they are compared with numpy fourier64.
            "emit_encodings": True,
            **plan.query_flags(),
        }
        # The temporary query can be checked before the optional device group is installed.
        if args.query_name.endswith("_v4_check"):
            params.pop("include_device_ip_context", None)
        started = time.perf_counter()
        if args.interpreted:
            # The repository text runs under INTERPRET with only its header swapped.
            query = as_interpreted(Path("gsql/temporal/training_context.gsql").read_text())
            row = checked_rows(executor.client.conn.runInterpretedQuery(query, params))[0]
        else:
            row = checked_rows(executor.run(args.query_name, params))[0]
        validate_context(key, row, plan, sampler, require_encodings=True)
        actual = [m for m in row["messages"] if m["event_id"]]
        selected = stratify(visible_history(history, key), sampler)

        def identity(m: dict[str, Any]) -> tuple[str, str, str]:
            return m["relation"], m["event_id"], m["stratum"]

        assert {identity(m) for m in actual} == {identity(m) for m in selected}, "Sampler parity"
        for m in actual:
            for name, value in expected[m["relation"] + ":" + m["event_id"]].items():
                assert np.isclose(m[name], value, rtol=1e-5, atol=1e-6), (name, m[name], value)
        for name, value in decay.items():
            assert np.isclose(row["features"][name], value, rtol=1e-5, atol=1e-6), (
                name,
                row["features"][name],
                value,
            )
        reports.append(
            {
                "history_events": len(history),
                "sampled_events": len(actual),
                "query_seconds": time.perf_counter() - started,
                "status": "passed",
            }
        )
    report = {
        "accounts": reports,
        "hidden_truth_read": False,
        "data_writes": 0,
        "checks": [
            "pair_count",
            "pair_first_age",
            "pair_gap",
            "flow_delay_and_censoring",
            "decayed_activity",
            "sampler_strata",
            "fourier64",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
