"""mule: train, evaluate, score and report the built-in mule detection run on TigerGraph.

The commands need nothing but the TigerGraph connection in .env and take no options:
the settings are built in (config.DEFAULT_CONFIG), and RUN defaults to the built-in
run's directory, results/baseline/seed-42. Each command prints one JSON result, and
exits 1 when the graph is not ready (check) or the study is incomplete (diagnose).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .diagnostics.study import ANALYSES, INCOMPLETE
from .paths import REPOSITORY_ROOT, RunPaths
from .pipeline.check import check
from .pipeline.diagnose import diagnose_built_in
from .pipeline.evaluate import evaluate_run
from .pipeline.prepare import install_queries
from .pipeline.score import score_accounts
from .pipeline.train import BASELINE_RUN, train_run
from .reporting.report import report_directory
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
        "text differs (train does this too), drop the retired queries still installed (only "
        "install does) and list installed queries no file defines",
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
        help="Redraw the figures and report.md of a run, a control-experiment suite "
        "(results/experiments/<suite>) or a diagnostic study (results/diagnostics/<dataset "
        "id>), from the files it saved, offline",
    )
    reporting.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=BASELINE_RUN.root,
        metavar="RUN",
        help=f"run, suite or study directory (default: {baseline})",
    )
    diagnosing = commands.add_parser(
        "diagnose",
        help="Run the diagnostic study of the built-in run's dataset against the ground "
        "truth, beside the built-in run, into results/diagnostics/<dataset id>/ with its "
        "figures; the only command that installs the analytics queries (where their text "
        "differs)",
    )
    diagnosing.add_argument(
        "analysis",
        nargs="?",
        choices=ANALYSES,
        metavar="ANALYSIS",
        help=f"one of {', '.join(ANALYSES)} (default: all of them, in this order)",
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
            return report_directory(args.directory)
        case "check":
            return check()
        case "diagnose":
            return diagnose_built_in(args.analysis)
        case other:
            raise ValueError(f"Unknown command {other!r}")


def main() -> None:
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    result = run_command(build_parser().parse_args())
    # One line, like the event lines before it.
    print(json.dumps(result, allow_nan=False))
    if result.get("status") in ("not_ready", INCOMPLETE):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
