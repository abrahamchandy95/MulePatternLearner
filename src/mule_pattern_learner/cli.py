"""Train from TigerGraph with graph-revealed labels, bounded contexts and CUDA by default.

`mule-temporal train` needs nothing but the TigerGraph credentials in .env: settings
are built in (config.DEFAULT_CONFIG), the run goes to results/baseline/seed-42/, and
the first run installs the queries, creates the scope and reveals the known mules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG
from .data.manifest import read_manifest
from .inference.saved_model import SavedModel
from .inference.score_accounts import read_account_ids, score
from .paths import DatasetPaths, RunPaths
from .pipeline.connect import connect, open_context_source
from .pipeline.evaluate import evaluate, final_audit
from .pipeline.prepare import prepare_dataset
from .pipeline.score import score_new
from .pipeline.train import BASELINE_RUN, train_run
from .runtime.device import reserve_deterministic_cublas
from .tigergraph.installer import install


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
    commands.add_parser(
        "prepare", help="Optionally stage the built-in run's dataset in data/ ahead of training"
    )
    commands.add_parser(
        "train",
        help="Prepare as needed, then train with nnPU into results/baseline/seed-42/ "
        "(resumes an interrupted run)",
    )
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
        help="Frozen-model audit: all test positives and weighted sampled negatives, "
        "written to the run's audit/",
    )
    final.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=BASELINE_RUN.root,
        help="Run directory (default: results/baseline/seed-42)",
    )
    final.add_argument("--truth", type=Path, help="Truth parquet (default: the graph)")
    final.add_argument(
        "--dataset", type=Path, help="Prepared dataset (default: recorded in the model)"
    )
    return parser


def model_dataset(model: SavedModel) -> DatasetPaths:
    """The prepared dataset a model was trained on."""
    dataset = model.dataset()
    if dataset is None:
        raise ValueError(f"{model.path} records no prepared dataset; pass --dataset")
    return dataset


def train_command() -> dict[str, Any]:
    """Prepare the dataset as needed, then train (or resume) the built-in run."""
    # An interrupted run continues from its resume.pt; a finished one is an error.
    return train_run(resume=True)


def main() -> None:
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    args = build_parser().parse_args()
    if args.command == "install":
        # The built-in run's retry budgets.
        result = install(
            connect(DEFAULT_CONFIG.transport),
            include_optional=args.include_optional,
            force=args.force,
        )
    elif args.command == "evaluate-final":
        dataset = None if args.dataset is None else DatasetPaths(args.dataset)
        result = final_audit(RunPaths(args.run), args.truth, dataset=dataset)
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = evaluate(args.predictions, args.checkpoint, args.truth)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "score-new":
        result = score_new(args.checkpoint, read_account_ids(args.accounts), args.date, args.output)
    elif args.command == "score":
        model = SavedModel.load(args.checkpoint)
        dataset = model_dataset(model) if args.dataset is None else DatasetPaths(args.dataset)
        result = score(
            model,
            dataset,
            args.date,
            args.split,
            args.output,
            open_contexts=open_context_source,
        )
    elif args.command == "prepare":
        result = read_manifest(prepare_dataset(DEFAULT_CONFIG))
    else:
        result = train_command()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
