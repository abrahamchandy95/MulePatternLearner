"""Run a prespecified comparison; each run selects its checkpoint on validation."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config

from mule_pattern_learner.temporal.training import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["tabular", "no_fourier", "temporal"],
        default=["tabular", "no_fourier", "temporal"],
    )
    args = parser.parse_args()
    base = load_config(args.config)
    fractions = (
        args.fractions
        if base["task"] == "mule_pu" and base.get("mask_source", "resample") == "resample"
        else [1.0]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "base_config": base,
        "seeds": args.seeds,
        "fractions": fractions,
        "variants": args.variants,
        "selection": "Each run: validation AP for checkpoint, validation F1 for threshold; test evaluated after selection",
    }
    plan_path = args.output / "experiment_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing experiment plan differs; use a new output directory")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for variant, seed, fraction in product(args.variants, args.seeds, fractions):
        name = f"{variant}_seed{seed}_reveal{fraction:g}"
        path = args.output / name
        config = {**base, "variant": variant, "seed": seed, "reveal_fraction": fraction}
        if (path / "metrics.json").exists():
            result = json.loads((path / "metrics.json").read_text())
            if result["config"] != config:
                raise ValueError("Stored result configuration mismatch")
        else:
            result = run(config, path)
        results.append(result)
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
