"""mule: train, evaluate, score and report the built-in mule detection run on TigerGraph.

The commands need nothing but the TigerGraph connection in .env and take no options:
the settings are built in (config.DEFAULT_CONFIG), and RUN defaults to the built-in
run's directory, results/baseline/seed-42. Each command prints one JSON result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .paths import REPOSITORY_ROOT, RunPaths
from .pipeline.check import check
from .pipeline.evaluate import evaluate_run
from .pipeline.prepare import install_queries
from .pipeline.score import score_accounts
from .pipeline.train import BASELINE_RUN, train_run
from .reporting.report import report_run
from .runtime.device import reserve_deterministic_cublas


def run_directory(value: str) -> RunPaths:
    return RunPaths(Path(value))


def build_parser() -> argparse.ArgumentParser:
    baseline = BASELINE_RUN.root.relative_to(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(prog="mule", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser(
        "install",
        help="Add the scope vertex type if it is missing, install the training queries whose "
        "text differs, drop the retired queries still installed (train does this too) and "
        "list installed queries no file defines",
    )
    commands.add_parser(
        "train",
        help=f"Prepare the dataset as needed, then train the built-in run into {baseline}/ "
        "with its training figures; an interrupted run resumes, and a complete one is "
        "reported from its metrics.json",
    )
    evaluating = commands.add_parser(
        "evaluate",
        help="Ground-truth audits of a run's model on validation (for decisions) and test "
        "(for reporting), written to the run's audit/ with the audit figures",
    )
    evaluating.add_argument(
        "run",
        nargs="?",
        type=run_directory,
        default=BASELINE_RUN,
        metavar="RUN",
        help=f"run directory (default: {baseline})",
    )
    scoring = commands.add_parser(
        "score", help="Score the accounts listed in a file with the built-in run's model"
    )
    scoring.add_argument("accounts", type=Path, metavar="ACCOUNTS", help="one account id per line")
    scoring.add_argument(
        "date", nargs="?", metavar="DATE", help="ISO date (default: the model's test cutoff)"
    )
    reporting = commands.add_parser(
        "report",
        help="Redraw a run's figures and report.md from the files it saved, offline",
    )
    reporting.add_argument(
        "run",
        nargs="?",
        type=run_directory,
        default=BASELINE_RUN,
        metavar="RUN",
        help=f"run directory (default: {baseline})",
    )
    commands.add_parser(
        "check",
        help="Read-only readiness: the graph, its queries, the cuGraph probe, then one "
        "batch's tensor digests and the first training loss",
    )
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """The result of the command args name."""
    match args.command:
        case "install":
            return install_queries()
        case "train":
            # An interrupted run continues from its resume.pt; a complete one is reported.
            return train_run(resume=True)
        case "evaluate":
            return evaluate_run(args.run)
        case "score":
            return score_accounts(BASELINE_RUN, args.accounts, args.date)
        case "report":
            return report_run(args.run)
        case "check":
            return check()
        case other:
            raise ValueError(f"Unknown command {other!r}")


def main() -> None:
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    result = run_command(build_parser().parse_args())
    # One line, like the event lines before it.
    print(json.dumps(result, allow_nan=False))
    if result.get("status") == "not_ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
