"""Command line entry point for staging, snapshots and temporal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config

from .snapshots import build_snapshot
from .staging import stage
from .training import run, timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("--manifest", type=Path, required=True)
    staging.add_argument("--output", type=Path, default=Path("artifacts/temporal/stage"))
    snapshots = commands.add_parser("snapshots")
    snapshots.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--variant", choices=["temporal", "no_fourier", "tabular"])
    train.add_argument("--seed", type=int)
    train.add_argument("--reveal-fraction", type=float)
    train.add_argument("--class-prior", type=float)
    train.add_argument("--labels", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage(args.manifest, args.output)
    else:
        config = load_config(args.config)
        if args.command == "snapshots":
            dates = sorted({date for values in config["dates"].values() for date in values})
            result = {
                date: build_snapshot(
                    Path(config["stage"]),
                    Path(config["snapshots"]) / date,
                    timestamp(date),
                    history_days=int(config.get("history_days", 90)),
                    slots=int(config.get("slots", 32)),
                    per_relation=int(config.get("per_relation", 4)),
                )
                for date in dates
            }
        else:
            for field in ("variant", "seed", "reveal_fraction", "class_prior", "labels"):
                value = getattr(args, field)
                if value is not None:
                    config[field] = str(value) if isinstance(value, Path) else value
            result = run(config, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
