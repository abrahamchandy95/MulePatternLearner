"""Train from TigerGraph with observed labels, bounded contexts and device selection."""

from __future__ import annotations

import os

# Deterministic cuBLAS GEMMs need a fixed workspace, read once when CUDA initializes,
# so it is set before anything imports torch. An explicit user value wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from mule_pattern_learner.configuration import load_config  # noqa: E402

from .evaluation import (  # noqa: E402
    ParquetEvaluationTruth,
    evaluate_final_population,
    evaluate_predictions,
)
from .inference import score  # noqa: E402
from .installation import install  # noqa: E402
from .pipeline import DEFAULT_CONFIG, DEFAULT_MODEL, dataset_path, prepare_live  # noqa: E402
from .predictor import read_account_ids, score_new_accounts  # noqa: E402
from .source import TigerGraphExecutor  # noqa: E402
from .training import output_paths, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    installing = commands.add_parser(
        "install",
        help="Create and install the training queries that differ from the repository",
    )
    installing.add_argument(
        "--include-optional",
        action="store_true",
        help="Also install the pair_time64 parity queries (training never calls them)",
    )
    installing.add_argument(
        "--force",
        action="store_true",
        help="Reinstall every query, even those already installed with the repository text",
    )
    prep = commands.add_parser("prepare", help="Optionally stage the data ahead of training")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--output", type=Path)
    prep.add_argument(
        "--create-scope",
        action="store_true",
        help="Allow writing a new training scope to TigerGraph (sets create_scope = true)",
    )
    training = commands.add_parser("train", help="Prepare as needed, train with nnPU and save")
    training.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Model .pt path")
    training.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from its checkpoint_last.pt (or start it)",
    )
    advanced = training.add_argument_group("optional experiment overrides")
    advanced.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    advanced.add_argument("--dataset", type=Path, help="Reuse an existing prepared dataset")
    advanced.add_argument(
        "--create-scope",
        action="store_true",
        help="Allow preparation to write a new training scope to TigerGraph",
    )
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
    new.add_argument(
        "--output", type=Path, required=True, help="Parquet path; rejected IDs go beside it"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate saved predictions against external truth"
    )
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--truth", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    final = commands.add_parser(
        "evaluate-final",
        help="Frozen-model audit: all test positives and weighted sampled negatives",
    )
    final.add_argument("--checkpoint", type=Path, required=True)
    final.add_argument("--truth", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument(
        "--dataset", type=Path, help="Prepared dataset (default: recorded in the checkpoint)"
    )
    return parser


def validated_config(path: Path) -> dict[str, Any]:
    """Load a TOML/JSON training configuration and check it against the schema."""
    from .config_schema import validate_config

    return validate_config(load_config(path))


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare when no dataset is given, then train (or resume) and save the model."""
    config = validated_config(args.config)
    checkpoint, run_dir = output_paths(args.output)
    if not args.resume and (checkpoint.exists() or run_dir.exists()):
        raise FileExistsError(f"Experiment already exists: {args.output}; pass --resume")
    dataset = args.dataset
    if dataset is None:
        dataset = dataset_path(config)
        prepare_live({**config, "create_scope": True} if args.create_scope else config, dataset)
    return train(config, dataset, args.output, resume=args.resume)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "install":
        result = install(
            TigerGraphExecutor(), include_optional=args.include_optional, force=args.force
        )
    elif args.command == "evaluate-final":
        result = evaluate_final_population(
            args.checkpoint, ParquetEvaluationTruth(args.truth), args.output, dataset=args.dataset
        )
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
        config = validated_config(args.config)
        if args.create_scope:
            config["create_scope"] = True
        result = prepare_live(config, args.output or dataset_path(config))
    else:
        result = train_command(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
