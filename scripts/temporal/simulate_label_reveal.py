"""Simulate the label reveal over many salts to show how its outcome varies.

Read-only. Fetches the reveal's inputs once (the query verify_label_reveal.py uses),
then runs reveal_model.plan, the Python mirror of temporal_reveal_mule_labels with the
job's own hash and defaults, once per salt. Prints, per split, the mules and the
median and 5th to 95th percentile of the mules discovered before the split's cutoff
(eligible) and of those revealed. It never calls the reveal job and writes nothing.

  python scripts/temporal/simulate_label_reveal.py [--runs 1000] [--first-salt 0]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mule_pattern_learner.temporal.live.config_schema import run_config
from mule_pattern_learner.temporal.live.contract import PHASE_SPLIT
from mule_pattern_learner.temporal.live.executor import TigerGraphExecutor
from mule_pattern_learner.temporal.live.labels import reveal_parameters
from mule_pattern_learner.temporal.live.reveal_model import INPUTS_QUERY, counts_by_split, plan


def spread(values: list[int]) -> dict[str, float]:
    p5, median, p95 = np.percentile(values, [5, 50, 95])
    return {"median": float(median), "p5": float(p5), "p95": float(p95)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=1000, help="number of salts")
    parser.add_argument("--first-salt", type=int, default=0)
    parser.add_argument("--budget", type=int, help="default: the run's reveal_per_split")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    params = reveal_parameters(run_config(), apply=False)
    if args.budget is not None:
        params["budget"] = args.budget
    executor = TigerGraphExecutor()
    inputs = executor.client.conn.runInterpretedQuery(
        INPUTS_QUERY, {"scope_id": params["scope_id"]}
    )
    salts = range(args.first_salt, args.first_salt + args.runs)
    plans = [plan(inputs, {**params, "salt": salt}) for salt in salts]
    eligible = [counts_by_split(result, "eligible") for result in plans]
    revealed = [counts_by_split(result, "revealed") for result in plans]
    mules = plans[0]["mules"].values()
    print(
        json.dumps(
            {
                "scope_id": params["scope_id"],
                "budget": params["budget"],
                "salts": [salts.start, salts.stop - 1],
                "splits": {
                    name: {
                        "mules": sum(1 for m in mules if m["part"] == part),
                        "eligible": spread([counts[part] for counts in eligible]),
                        "revealed": spread([counts[part] for counts in revealed]),
                    }
                    for part, name in PHASE_SPLIT.items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
