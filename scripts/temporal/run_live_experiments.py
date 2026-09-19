"""Predeclared model/label-budget comparisons on one immutable prepared cohort."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config

import torch

from mule_pattern_learner.temporal.common import digest

from mule_pattern_learner.temporal.live.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    base = load_config(args.config)
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for variant, seed, prior in product(
        args.variants, args.seeds, args.class_priors or [base["class_prior"]]
    ):
        config = {
            **base,
            "variant": variant,
            "seed": seed,
            "class_prior": prior,
        }
        output = args.output / f"{variant}_seed{seed}_prior{prior:g}"
        if (output / "metrics.json").exists():
            if json.loads((output / "config.json").read_text()) != config:
                raise ValueError(f"Existing run has a different configuration: {output}")
            checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
            if checkpoint.get("training_protocol") != "scoped_observed_label_nnpu_v4":
                raise ValueError(f"Existing run uses an older training protocol: {output}")
            if checkpoint["dataset_manifest_sha256"] != digest(args.dataset / "manifest.json"):
                raise ValueError(f"Existing run belongs to a different dataset: {output}")
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
