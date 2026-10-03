"""mule: train, evaluate, score and report the built-in mule detection run on TigerGraph.

The commands need nothing but the TigerGraph connection in .env and take no options:
the settings are built in (config.DEFAULT_CONFIG), and RUN defaults to the built-in
run's directory, results/baseline/seed-42. Each command shows its progress and then a
short summary; the full records are in the files it writes (events.jsonl, history.csv,
epochs.csv, metrics.json, audit/), and what it did before a run, a dataset or a study
recorded its events is in results/events.jsonl. It exits 1 when the graph is not ready
(check), the study is incomplete (diagnose) or TigerGraph's failures outlast the retries.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .diagnostics.study import ANALYSES, INCOMPLETE
from .metrics import REVIEW_BUDGETS, budget_name
from .paths import REPOSITORY_ROOT, RESULTS_DIR, RunPaths, check_report, command_events
from .pipeline.check import check
from .pipeline.diagnose import diagnose_built_in
from .pipeline.evaluate import evaluate_run
from .pipeline.prepare import install_queries
from .pipeline.score import score_accounts
from .pipeline.train import BASELINE_RUN, train_run
from .reporting.report import report_directory
from .runtime.console import (
    count,
    duration,
    end_progress,
    estimate,
    number,
    plural,
    show,
    shown_path,
    table,
)
from .runtime.device import reserve_deterministic_cublas
from .runtime.progress import recording
from .tigergraph.executor import TigerGraphUnavailableError, TransientQueryError


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
        "batch's digest and the first training loss; the full report goes to "
        "results/check.json",
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


def summary(args: argparse.Namespace, result: Mapping[str, Any]) -> str:
    """What the command args name shows once it is done: a few lines of its result."""
    match args.command:
        case "install":
            return install_summary(result)
        case "train":
            return train_summary(result, BASELINE_RUN)
        case "evaluate":
            return evaluate_summary(result, args.run)
        case "score":
            return score_summary(result)
        case "report":
            return report_summary(result)
        case "check":
            return check_summary(result, check_report())
        case "diagnose":
            return diagnose_summary(result)
        case other:
            raise ValueError(f"Unknown command {other!r}")


def train_summary(metrics: Mapping[str, Any], run: RunPaths) -> str:
    """A trained run: its time, best epoch and proxy metrics, then where its files are."""
    rows = [["", "known mules", "AP", "ROC AUC", "recall at 1%"]]
    for split, values in metrics["observed_label_proxy"].items():
        rows.append(
            [
                split,
                count(values.get("positives", 0)),
                number(values.get("average_precision")),
                number(values.get("roc_auc")),
                number(values.get("recall_at_1pct")),
            ]
        )
    return "\n".join(
        [
            f"Trained in {duration(float(metrics['elapsed_seconds']))}; model.pt holds the "
            f"weights of the best epoch, {metrics['best_epoch']}.",
            "Proxy metrics, on the revealed labels at the selected epoch:",
            *table(rows),
            "These count unlabelled accounts as negatives; `mule evaluate` gives the "
            "ground-truth audit.",
            f"Run directory: {shown_path(run.root)}/",
            "  model.pt, metrics.json, history.csv, epochs.csv, events.jsonl, predictions/, "
            "plots/, report.md",
        ]
    )


def evaluate_summary(reports: Mapping[str, Mapping[str, Any]], run: RunPaths) -> str:
    """A run's audits: each split's ranking metrics with their intervals, side by side."""
    audits = list(reports.values())
    level = audits[0]["constants"]["interval"] if audits else 0.9
    rows = [["", *(f"{split} ({report['purpose']})" for split, report in reports.items())]]
    rows.append(["mules", *(count(report["metrics"]["sample_positives"]) for report in audits)])
    metrics = [("AP", "average_precision"), ("ROC AUC", "roc_auc")]
    for kind in ("recall", "precision"):
        metrics += [
            (f"{kind} at {fraction:.0%}", f"{kind}_at_{budget_name(fraction)}")
            for fraction in REVIEW_BUDGETS
        ]
    for label, key in metrics:
        cells = [estimate(r["metrics"].get(key), r["intervals"].get(key)) for r in audits]
        rows.append([label, *cells])
    audit = shown_path(run.audit_report("validation").parent)
    return "\n".join(
        [
            f"Ground-truth audit of {shown_path(run.root)}, with {level:.0%} intervals:",
            *table(rows),
            "Decide on validation; the test audit is for reporting only.",
            f"Files: {audit}/ (reports and scored samples), plots/ and report.md",
        ]
    )


def report_summary(result: Mapping[str, Any]) -> str:
    """The figures `mule report` drew, and the report.md it wrote."""
    figures = [Path(figure) for figure in result["figures"]]
    report = shown_path(result["report"])
    if not figures:
        return f"Wrote {report}; there was no figure to draw"
    plots = shown_path(figures[0].parent)
    return f"Drew {plural(len(figures), 'figure')} into {plots}/ and wrote {report}"


def score_summary(result: Mapping[str, Any]) -> str:
    """Where the scores went, and what TigerGraph rejected."""
    lines = [
        f"Scored {plural(int(result['accounts']), 'account')} into {shown_path(result['output'])}"
    ]
    rejected = int(result["rejected"])
    if rejected:
        statuses = ", ".join(
            f"{status} {count(n)}" for status, n in result["rejected_roots_by_status"].items()
        )
        lines.append(
            f"TigerGraph rejected {plural(rejected, 'account')} ({statuses}), listed in "
            f"{shown_path(result['rejected_output'])}"
        )
    children = int(result.get("rejected_children", 0))
    if children:
        lines.append(
            f"{plural(children, 'child context')} TigerGraph rejected were left out of the "
            "scored accounts' neighbourhoods"
        )
    return "\n".join(lines)


def check_summary(report: Mapping[str, Any], saved: Path) -> str:
    """mule check's checklist ([x] ready, [ ] not ready, [-] for information), then what to run.

    ``saved`` is where the full report is, results/check.json.
    """
    queries = report["queries"]
    total = len(queries["up_to_date"]) + len(queries["stale"])
    problems: list[str] = report["problems"]
    lines = [f"Readiness of graph {report['graph']}, which mule check only reads:"]

    def item(ready: bool | None, text: str) -> None:
        mark = "-" if ready is None else "x" if ready else " "
        lines.append(f"  [{mark}] {text}")

    item(report["scope_schema"] == "present", f"scope vertex type {report['scope_schema']}")
    stale = queries["stale"]
    if stale:
        named = "; ".join(f"{name} ({', '.join(issues)})" for name, issues in stale.items())
        item(False, f"{len(stale)} of {total} training queries are stale: {named}")
    else:
        item(True, f"{total} training queries installed with the repository's text")
    if queries["retired"]:
        retired = ", ".join(queries["retired"])
        item(None, f"retired queries still installed, which `mule install` drops: {retired}")
    item(*_cugraph(report["cugraph"], problems))
    dataset = report.get("dataset")
    if dataset:
        item(True, f"dataset {str(dataset)[:12]} prepared")
    else:
        item(False, "no prepared dataset of the built-in run; `mule train` prepares one")
    item(*_first_step(report.get("first_step")))
    if not problems:
        lines.append("Ready to train.")
    else:
        commands = [
            c for c in ("mule install", "mule train") if any(f"`{c}`" in p for p in problems)
        ]
        todo = ", then ".join(f"`{c}`" for c in commands)
        lines.append(f"Not ready: run {todo}." if todo else "Not ready: see the items not ticked.")
    lines.append(f"The full report, with every tensor's digest, is in {shown_path(saved)}")
    return "\n".join(lines)


def _cugraph(probe: Mapping[str, Any], problems: list[str]) -> tuple[bool | None, str]:
    """The checklist item of the cuGraph probe; a failure counts only when it is required."""
    required = any(problem.startswith("sampler.backend is cugraph") for problem in problems)
    match probe["status"]:
        case "passed":
            return True, f"cuGraph probe passed on {probe['device']}"
        case "no_cuda":
            return None, "no CUDA device: training samples with the torch sampler"
        case "missing":
            text = "cuGraph is not installed: training samples with the torch sampler"
        case _:
            failed = f"cuGraph probe failed on {probe['device']} ({probe['reason']})"
            text = f"{failed}: training samples with the torch sampler"
    return (False if required else None), text


def _first_step(step: Mapping[str, Any] | None) -> tuple[bool, str]:
    """The checklist item of the first training batch and step, with the batch's digest."""
    if step is None:
        return False, "first training batch: built once everything above is ready"
    if step["status"] != "passed":
        return False, f"first training batch on {step['device']}: TigerGraph rejected every root"
    return True, (
        f"first training batch on {step['device']}: {count(step['accepted_roots'])} of "
        f"{count(step['roots'])} roots, {count(step['context_requests'])} context requests, "
        f"{duration(float(step['batch_seconds']))}; one step: loss {step['loss']:.6f}, "
        f"objective {step['objective']:.6f}\n      batch digest {step['batch_digest'][:12]}"
    )


def install_summary(result: Mapping[str, Any]) -> str:
    """What `mule install` leaves: the queries in place, and what it did not touch."""
    lines = ["Added the scope vertex type"] if "scope_schema" in result else []
    verified, installed = result["verified"], result["installed"]
    line = f"{plural(len(verified), 'training query', 'training queries')} installed with the "
    line += f"repository's text, {len(installed)} of them now"
    dropped = result.get("dropped", [])
    if dropped:
        line += f"; {plural(len(dropped), 'retired query', 'retired queries')} dropped"
    lines.append(line)
    if result.get("not_defined"):
        named = ", ".join(result["not_defined"])
        lines.append(f"Left in place, since no repository file defines them: {named}")
    return "\n".join(lines)


def diagnose_summary(result: Mapping[str, Any]) -> str:
    """Whether the study is complete, and where its files are."""
    skipped: Mapping[str, str] = result.get("skipped", {})
    if skipped:
        head = (
            f"Study incomplete: {plural(len(skipped), 'analysis', 'analyses')} skipped "
            f"({', '.join(skipped)}); the reasons are above and in study.json"
        )
    else:
        head = (
            f"Study complete, of dataset {str(result['dataset_id'])[:12]} and run {result['run']}"
        )
    return "\n".join(
        [
            head,
            f"Study directory: {shown_path(result['directory'])}/",
            "  study.json, features.parquet, <analysis>.csv, events.jsonl, plots/, report.md",
        ]
    )


def stopped(command: str, error: TransientQueryError) -> str:
    """What a command says on stderr when TigerGraph's failures outlast its retries."""
    if isinstance(error, TigerGraphUnavailableError):
        return (
            f"{command} stopped: TigerGraph stayed unavailable. {error}. Run it again once "
            "TigerGraph answers; an interrupted run resumes where it stopped."
        )
    return f"{command} stopped: a TigerGraph request kept failing. {error}."


def main() -> None:
    # Before any CUDA work: deterministic cuBLAS GEMMs need a fixed workspace.
    reserve_deterministic_cublas()
    args = build_parser().parse_args()
    # What the command emits before a run, a dataset or a study records its events, and
    # what it emits outside them, is kept there: an install, connecting, a dataset found.
    events = command_events(RESULTS_DIR)
    events.parent.mkdir(parents=True, exist_ok=True)
    try:
        with recording(events):
            result = run_command(args)
    except TransientQueryError as error:
        # The retries were shown as they happened; the failure that ended them goes to
        # stderr, without the traceback of a bug, and the command exits 1.
        raise SystemExit(stopped(f"mule {args.command}", error)) from None
    finally:
        # A line rewritten in place ends before anything else is written.
        end_progress()
    show(summary(args, result))
    if result.get("status") in ("not_ready", INCOMPLETE):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
