"""Train from TigerGraph with graph-revealed labels, bounded contexts and CUDA by default.

`mule-temporal train` needs nothing but the TigerGraph credentials in .env: settings
are built in (config_schema.DEFAULT_RUN), and the first run installs the queries,
creates the scope and reveals the known mules. Every flag is optional.
"""

from __future__ import annotations

import os

# Deterministic cuBLAS GEMMs need a fixed workspace, read once when CUDA initializes,
# so it is set before anything imports torch. An explicit user value wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from .checkpoint import ModelCheckpoint  # noqa: E402
from .config_schema import run_config  # noqa: E402
from .evaluation import (  # noqa: E402
    GraphEvaluationTruth,
    ParquetEvaluationTruth,
    evaluate_final_population,
    evaluate_predictions,
)
from .executor import TigerGraphExecutor  # noqa: E402
from .inference import score  # noqa: E402
from .installation import install  # noqa: E402
from .pipeline import DEFAULT_MODEL, dataset_path, prepare_live, run  # noqa: E402
from .predictor import read_account_ids, score_new_accounts  # noqa: E402

CONFIG_HELP = "Optional TOML/JSON file whose keys override the built-in settings"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    installing = commands.add_parser(
        "install",
        help="Install the training queries that differ from the repository (train does this)",
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
    prep.add_argument("--config", type=Path, help=CONFIG_HELP)
    prep.add_argument(
        "--output", type=Path, default=DEFAULT_MODEL, help="Model .pt path whose run it prepares"
    )
    training = commands.add_parser(
        "train", help="Prepare as needed, train with nnPU and save (resumes an interrupted run)"
    )
    training.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Model .pt path")
    advanced = training.add_argument_group("optional experiment overrides")
    advanced.add_argument("--config", type=Path, help=CONFIG_HELP)
    advanced.add_argument("--dataset", type=Path, help="Reuse an existing prepared dataset")
    scoring = commands.add_parser("score")
    scoring.add_argument("--checkpoint", type=Path, required=True)
    scoring.add_argument("--dataset", type=Path, help="Default: the checkpoint's prepared data")
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
        "evaluate", help="Evaluate saved predictions against the graph's oracle truth"
    )
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--truth", type=Path, help="Truth parquet (default: the graph)")
    evaluation.add_argument("--output", type=Path, required=True)
    final = commands.add_parser(
        "evaluate-final",
        help="Frozen-model audit: all test positives and weighted sampled negatives",
    )
    final.add_argument("--checkpoint", type=Path, required=True)
    final.add_argument("--truth", type=Path, help="Truth parquet (default: the graph)")
    final.add_argument("--output", type=Path, required=True)
    final.add_argument(
        "--dataset", type=Path, help="Prepared dataset (default: recorded in the checkpoint)"
    )
    return parser


def truth_source(path: Path | None) -> GraphEvaluationTruth | ParquetEvaluationTruth:
    """Oracle truth for evaluation only: a supplied parquet, else the graph's labels."""
    return ParquetEvaluationTruth(path) if path is not None else GraphEvaluationTruth()


def checkpoint_dataset(checkpoint: ModelCheckpoint) -> Path:
    """The prepared dataset a checkpoint was trained on."""
    if checkpoint.dataset is None:
        raise ValueError(f"{checkpoint.path} records no prepared dataset; pass --dataset")
    return checkpoint.dataset


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare when no dataset is given, then train (or resume) and save the model."""
    # An interrupted run continues from its checkpoint; a finished one is an error.
    return run(args.output, config_path=args.config, dataset=args.dataset, resume=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "install":
        result = install(
            TigerGraphExecutor(), include_optional=args.include_optional, force=args.force
        )
    elif args.command == "evaluate-final":
        result = evaluate_final_population(
            args.checkpoint, truth_source(args.truth), args.output, dataset=args.dataset
        )
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = evaluate_predictions(args.predictions, args.checkpoint, truth_source(args.truth))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "score-new":
        result = score_new_accounts(
            args.checkpoint, read_account_ids(args.accounts), args.date, args.output
        )
    elif args.command == "score":
        checkpoint = ModelCheckpoint.load(args.checkpoint)
        dataset = args.dataset or checkpoint_dataset(checkpoint)
        result = score(checkpoint, dataset, args.date, args.split, args.output)
    elif args.command == "prepare":
        config = run_config(args.config)
        result = prepare_live(config, dataset_path(config, args.output))
    else:
        result = train_command(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
