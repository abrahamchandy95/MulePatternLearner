"""Predeclared model/label-budget comparisons on one immutable prepared cohort.

Model seeds vary, but the prepared seed reservoir must not: every run pins
cohort_seed to the base configuration's value, so all runs match the dataset's
preparation settings. The base is the built-in run with the optional --config
overrides and the dataset identity of --dataset.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from mule_pattern_learner.temporal.live.checkpoint import ModelCheckpoint
from mule_pattern_learner.temporal.live.cohort import cohort_seed
from mule_pattern_learner.temporal.live.config_schema import run_config, validate_config
from mule_pattern_learner.temporal.live.dataset import read_manifest
from mule_pattern_learner.temporal.live.pipeline import prepared_config
from mule_pattern_learner.temporal.live.training import TRAINING_PROTOCOL, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional overrides of the built-in run")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("temporal", "no_fourier", "tabular"),
        default=["temporal", "no_fourier", "tabular"],
    )
    parser.add_argument("--class-priors", nargs="+", type=float)
    args = parser.parse_args()
    manifest = read_manifest(args.dataset)
    base = prepared_config(run_config(args.config), manifest)
    reservoir = cohort_seed(base)
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for variant, seed, prior in product(
        args.variants, args.seeds, args.class_priors or [base["class_prior"]]
    ):
        config = validate_config(
            {
                **base,
                "cohort_seed": reservoir,
                "variant": variant,
                "seed": seed,
                "class_prior": prior,
            }
        )
        output = args.output / f"{variant}_seed{seed}_prior{prior:g}"
        if (output / "metrics.json").exists():
            if json.loads((output / "config.json").read_text()) != config:
                raise ValueError(f"Existing run has a different configuration: {output}")
            checkpoint = ModelCheckpoint.load(output / "model.pt")
            if checkpoint.training_protocol != TRAINING_PROTOCOL:
                raise ValueError(f"Existing run uses an older training protocol: {output}")
            checkpoint.check_dataset(
                args.dataset, f"Existing run belongs to a different dataset: {output}"
            )
            result = json.loads((output / "metrics.json").read_text())
        else:
            try:
                result = train(config, args.dataset, output)
            except ValueError as error:
                if "Training needs revealed positives" not in str(error):
                    raise
                result = {
                    "status": "skipped",
                    "reason": str(error),
                    "variant": variant,
                    "seed": seed,
                }
        results.append({"run": str(output), **result})
        (args.output / "matrix.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"runs": len(results), "output": str(args.output / "matrix.json")}, indent=2))


if __name__ == "__main__":
    main()
