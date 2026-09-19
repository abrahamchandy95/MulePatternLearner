"""Train from TigerGraph with observed labels, bounded contexts and device selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mule_pattern_learner.configuration import load_config

from .installation import install
from .source import TigerGraphExecutor
from .inference import score
from .predictor import score_new_accounts, read_account_ids
from .evaluation import ParquetEvaluationTruth, evaluate_predictions
from .pipeline import DEFAULT_CONFIG, DEFAULT_MODEL, dataset_path, prepare_live, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install")
    prep = commands.add_parser("prepare", help="Optionally stage the data ahead of training")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--output", type=Path)
    training = commands.add_parser("train", help="Prepare as needed, train with nnPU and save")
    training.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Model .pt path")
    advanced = training.add_argument_group("optional experiment overrides")
    advanced.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    advanced.add_argument("--dataset", type=Path, help="Reuse an existing prepared dataset")
    scoring = commands.add_parser("score")
    scoring.add_argument("--checkpoint", type=Path, required=True)
    scoring.add_argument("--dataset", type=Path, required=True)
    scoring.add_argument("--date", required=True)
    scoring.add_argument("--split", choices=("train", "validation", "test"), default="test")
    scoring.add_argument("--output", type=Path, required=True)
    new = commands.add_parser(
        "score-new", help="Score arbitrary accounts without a training dataset"
    )
    new.add_argument("--checkpoint", type=Path, required=True)
    new.add_argument(
        "--accounts", type=Path, required=True, help="Text file: one Account ID per line"
    )
    new.add_argument("--date", required=True)
    new.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate saved predictions against external truth"
    )
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--truth", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "install":
        result = install(TigerGraphExecutor())
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = evaluate_predictions(
            args.predictions, args.checkpoint, ParquetEvaluationTruth(args.truth)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "score-new":
        result = score_new_accounts(
            args.checkpoint, read_account_ids(args.accounts), args.date, args.output
        )
    elif args.command == "score":
        result = score(args.checkpoint, args.dataset, args.date, args.split, args.output)
    elif args.command == "prepare":
        config = load_config(args.config)
        result = prepare_live(config, args.output or dataset_path(config))
    else:
        result = run(args.output, config_path=args.config, dataset=args.dataset)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
