"""Check the installed label reveal against its independent Python mirror.

Read-only. Fetches every input the reveal uses (mules, scope partitions, draw keys,
fraud-labelled Zelle inflows and mule-to-mule events), recomputes the whole plan with
reveal_model.plan, runs temporal_reveal_mule_labels with apply = FALSE, and compares
discovery channel, availability clock and revealed set mule by mule. The dry run
passes force = TRUE only to skip the job's already-revealed check, so the check also
works on a graph whose labels were revealed; with apply = FALSE nothing is written.

  python scripts/temporal/verify_label_reveal.py [--salt 42] [--budget 20]

Exit code 0 means the two implementations agree.
"""

from __future__ import annotations

import argparse
import json
import sys

from mule_pattern_learner.temporal.live.config_schema import run_config
from mule_pattern_learner.temporal.live.executor import TigerGraphExecutor, merged_rows
from mule_pattern_learner.temporal.live.labels import REVEAL_QUERY, reveal_parameters
from mule_pattern_learner.temporal.live.reveal_model import (
    INPUTS_QUERY,
    available_ms,
    counts_by_split,
    plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--salt", type=int)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    config = run_config()
    params = reveal_parameters(config, apply=False)
    params["force"] = True
    if args.salt is not None:
        params["salt"] = args.salt
    if args.budget is not None:
        params["budget"] = args.budget
    executor = TigerGraphExecutor()
    inputs = executor.client.conn.runInterpretedQuery(
        INPUTS_QUERY, {"scope_id": params["scope_id"]}
    )
    expected = plan(inputs, params)
    result = merged_rows(executor.run(REVEAL_QUERY, params, timeout_s=3600.0, attempts=1))
    if result.get("status") != "dry_run":
        print(json.dumps(result))
        return 1
    actual = {r["account_id"]: r for r in result["revealed_mules"]}
    failures = []
    if set(actual) != expected["revealed"]:
        failures.append(
            f"revealed sets differ: only GSQL {sorted(set(actual) - expected['revealed'])[:5]}, "
            f"only Python {sorted(expected['revealed'] - set(actual))[:5]}"
        )
    data_end = int(result["data_end_ts_ms"])
    for account, row in actual.items():
        m = expected["mules"].get(account)
        if m is None:
            continue
        known = available_ms(m, data_end)
        if row["channel"] != m["channel"] or row["known_ts_ms"] != known:
            failures.append(
                f"{account}: GSQL {row['channel']} {row['known_ts_ms']}, "
                f"Python {m['channel']} {known}"
            )
    eligible = counts_by_split(expected, "eligible")
    # The job's map has no entry for a split without eligible mules.
    gsql_eligible = {part: int(result["eligible"].get(str(part), 0)) for part in eligible}
    if eligible != gsql_eligible:
        failures.append(f"eligible counts differ: GSQL {gsql_eligible}, Python {eligible}")
    print(
        json.dumps(
            {"eligible": eligible, "revealed": len(actual), "failures": failures[:10]}, indent=1
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
