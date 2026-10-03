"""The console: a short line for each event a person needs to follow, and nothing more.

emit (runtime.progress) appends every event's full record to the recorded events.jsonl
and prints the line event_text gives it, chosen by the event's name (LINES): a line for
what a person follows (retries, installs, preparation, the start of training, each
epoch, audits, analyses, warnings) and nothing for the rest (running totals, batch
counts, the records of scoring), whose records are in the files. The progress of
training steps and of a query install (PROGRESS), and how far scoring has come
(show_scoring: validation and test in training, an audit's sample, `mule score`), is
rewritten in place on a terminal, and not printed at all when stdout is a file or a
pipe, so a log holds no progress lines. Other text for the console goes through show,
which first clears a line being rewritten in place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import sys
import threading
from typing import Any

from ..contract.graph_schema import SPLIT_PHASE, SPLITS

_LOCK = threading.RLock()


class _Screen:
    """The width of the line stdout shows being rewritten in place; 0 when there is none."""

    progress = 0


_SCREEN = _Screen()


def is_terminal() -> bool:
    """Whether stdout is a terminal, as it is at the moment of asking."""
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _cleared(terminal: bool) -> str:
    """What clears the line being rewritten in place, if a terminal shows one."""
    width, _SCREEN.progress = _SCREEN.progress, 0
    return "\r" + " " * width + "\r" if width and terminal else ""


def show(text: str) -> None:
    """Print text, one line or several, after clearing a line being rewritten in place."""
    with _LOCK:
        stream = sys.stdout
        stream.write(_cleared(is_terminal()) + text + "\n")
        stream.flush()


def show_progress(text: str) -> None:
    """Rewrite the progress line in place on a terminal; print nothing to a file or pipe."""
    with _LOCK:
        if not is_terminal():
            return
        stream = sys.stdout
        # Spaces cover what is left of a longer line before it.
        stream.write("\r" + text.ljust(_SCREEN.progress))
        stream.flush()
        _SCREEN.progress = len(text)


def show_scoring(what: str, done: int, total: int | None = None) -> None:
    """Rewrite how far scoring has come in place on a terminal: scoring validation 640/2,011.

    ``what`` names what is scored and ``done`` how many of its ``total`` accounts (the
    count alone without one). Nothing is printed to a file or a pipe, and nothing is
    recorded: the records of scoring are in the files.
    """
    of = "" if total is None else f"/{total:,}"
    show_progress(f"scoring {what} {done:,}{of}")


def end_progress() -> None:
    """End a line being rewritten in place, so whatever follows starts a line of its own."""
    with _LOCK:
        if _SCREEN.progress and is_terminal():
            sys.stdout.write("\n")
            sys.stdout.flush()
        _SCREEN.progress = 0


# How the lines write numbers, counts, durations and paths.


def number(value: Any, digits: int = 3) -> str:
    """A metric to ``digits`` decimals; n/a when it is missing or not a number."""
    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if value != value else f"{value:.{digits}f}"


def count(value: Any) -> str:
    """A count with thousands separators."""
    return f"{int(value):,}"


def plural(value: int, word: str, words: str | None = None) -> str:
    """A count and its noun: 1 query, 12 queries."""
    return f"{value:,} {word if value == 1 else words or word + 's'}"


def duration(seconds: float) -> str:
    """Seconds as a person reads them: 41 s, 3.1 min, 2.4 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def shown_path(path: str | Path) -> str:
    """A path relative to the working directory when it lies under it, else as it is."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def sentence(text: str) -> str:
    """Text ending in one full stop: its own trailing dots give way, but an ellipsis stays."""
    text = text.rstrip()
    return text if text.endswith("...") else text.rstrip(".") + "."


def brief(text: str, length: int = 100) -> str:
    """Text on one line, cut to about ``length`` characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= length else flat[: length - 3].rstrip() + "..."


def by_split(counts: Mapping[str, Any]) -> str:
    """Counts of train, validation and test: 20 / 11 / 20."""
    return " / ".join(count(counts.get(split, 0)) for split in SPLITS)


def estimate(value: Any, interval: Sequence[Any] | None) -> str:
    """A metric with its interval, if it has one: 0.312 [0.251, 0.371]."""
    if not interval:
        return number(value)
    low, high = interval
    return f"{number(value)} [{number(low)}, {number(high)}]"


def table(rows: Sequence[Sequence[str]], indent: int = 2) -> list[str]:
    """Rows as aligned lines: the first column to the left, the others to the right."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        (
            " " * indent
            + "  ".join(
                cell.ljust(width) if i == 0 else cell.rjust(width)
                for i, (cell, width) in enumerate(zip(row, widths, strict=True))
            )
        ).rstrip()
        for row in rows
    ]


# The line of each event, by its name.

Line = Callable[[Mapping[str, Any]], str | None]


def _nothing(record: Mapping[str, Any]) -> None:
    return None


def _operation(operation: str) -> str:
    """An operation as a line names it: "fetch_training_context (512 keys)" is
    "fetch_training_context, 512 keys", so the reason is the line's only parenthesis."""
    name, _, detail = operation.partition(" (")
    return f"{name}, {detail.removesuffix(')')}" if detail else name


def _retry(record: Mapping[str, Any]) -> str:
    reason = record.get("reason") or "no answer"
    attempt, wait = record["attempt"], duration(record["retry_in_s"])
    match record["failure"]:
        case "availability":
            what = "TigerGraph is not answering yet"
        case "server_timeout":
            what = f"TigerGraph timed out on {_operation(record['operation'])}"
        case _:
            what = f"TigerGraph failed {_operation(record['operation'])}"
    return f"{what} ({reason}): attempt {attempt}, retrying in {wait}"


def _context_split(record: Mapping[str, Any]) -> str:
    return (
        f"TigerGraph timed out on a request of {plural(int(record['keys']), 'key')} at hop "
        f"{record['hop']}; requesting its halves on their own"
    )


def _install(record: Mapping[str, Any]) -> str | None:
    stale = record["stale"]
    if not stale:
        return None
    queries = plural(len(stale), "query", "queries")
    if record["up_to_date"]:
        return f"Installing {queries} on TigerGraph..."
    # Every query, as on a fresh graph: compiling them all took about 50 minutes on the
    # reference graph, most of it the context query (tigergraph.installer).
    return f"Installing all {queries} on TigerGraph (about 50 minutes)..."


def _install_unanswered(record: Mapping[str, Any]) -> str:
    return (
        f"TigerGraph has not answered the install request ({record['error']}); waiting "
        "until the installed queries are enabled"
    )


def _install_wait(record: Mapping[str, Any]) -> str:
    waiting = len(record["awaiting"]) if "awaiting" in record else int(record["installing"])
    queries = plural(waiting, "query", "queries")
    return f"Waiting for {queries} to compile: {duration(record['elapsed_s'])} so far"


def _installed(record: Mapping[str, Any]) -> str:
    queries = plural(len(record["installed"]), "query", "queries")
    return f"Installed {queries} in {duration(record['seconds'])}"


def _drop_retired(record: Mapping[str, Any]) -> str | None:
    dropped = record["dropped"]
    if not dropped:
        return None
    retired = plural(len(dropped), "retired query", "retired queries")
    return f"Dropped {retired}: {', '.join(dropped)}"


def _scope(record: Mapping[str, Any]) -> str | None:
    # A scope found in place, or the end of its creation, adds nothing to read.
    if record.get("creating"):
        return f"Creating scope {record['scope']} on TigerGraph..."
    return None


def _reveal(record: Mapping[str, Any]) -> str:
    if record["labels"] == "already revealed":
        return f"Known mules already revealed on TigerGraph: {count(record['revealed_labels'])}"
    revealed = record.get("revealed") or {}
    counts = {split: int(revealed.get(str(phase), 0)) for split, phase in SPLIT_PHASE.items()}
    line = (
        f"Revealed {plural(sum(counts.values()), 'known mule')}: {by_split(counts)} in train "
        "/ validation / test"
    )
    short = record.get("shortfall_discovered_by_cutoff")
    if short:
        line += f"; fewer than the budget were discovered by the cutoff of {', '.join(short)}"
    return line


def _dataset(record: Mapping[str, Any]) -> str:
    dataset, known = str(record["dataset_id"])[:12], record.get("known_mules")
    if not known:
        return f"Dataset {dataset} is ready"
    return f"Dataset {dataset}: {by_split(known)} known mules in train / validation / test"


def _already_complete(record: Mapping[str, Any]) -> str:
    return f"{shown_path(record['run'])} is complete; its metrics.json is summarised below"


def _already_audited(record: Mapping[str, Any]) -> str:
    run, reports = Path(record["run"]), [Path(report) for report in record["reports"]]
    splits = " and ".join(report.stem for report in reports)
    files = " and ".join(report.relative_to(run).as_posix() for report in reports)
    summarised = "is summarised" if len(reports) == 1 else "are summarised"
    return f"{shown_path(run)} is already audited on {splits}; its {files} {summarised} below"


def _sampler(backend: Any) -> str:
    return "cuGraph" if backend == "cugraph" else str(backend)


def _plan(record: Mapping[str, Any]) -> str:
    # The steps each epoch's schedule takes: training.steps_per_epoch (the record's
    # steps_per_epoch) limits those of each train cutoff, so an epoch may take more.
    patience = int(record["patience"])
    stop = (
        f"early stop after {plural(patience, 'epoch')} without gain"
        if patience
        else "no early stop"
    )
    per_epoch = f"{plural(int(record['steps']), 'step')} per epoch"
    return f"{per_epoch}, at most {plural(int(record['epochs']), 'epoch')}, {stop}"


def _start(record: Mapping[str, Any]) -> str:
    on = f"on {record['device']} ({_sampler(record['sampler_backend'])} sampler)"
    run = shown_path(record["run"])
    if record["event"] == "start":
        return f"Training {on} into {run}: {_plan(record)}"
    epoch, step = int(record["epoch"]), int(record["step"])
    if record.get("stopped") or epoch >= int(record["epochs"]):
        where = "after its last epoch"
    else:
        where = f"at epoch {epoch + 1}" + (f", step {step}" if step else "")
    return f"Resuming {run} {where}, {on}: {_plan(record)}"


def _train(record: Mapping[str, Any]) -> str:
    return (
        f"epoch {record['epoch']}  step {record['step']}/{record['steps']}  "
        f"loss {number(record['loss'])}  {record['seconds_per_step']:.1f} s/step"
    )


def _epoch(record: Mapping[str, Any]) -> str:
    epoch = int(record["epoch"])
    took = duration(float(record["epoch_seconds"])) if "epoch_seconds" in record else ""
    line = (
        f"epoch {epoch:>2}  loss {number(record['loss'])}  "
        f"validation AP {number(record['validation_ap']):<5}  "
        f"ROC AUC {number(record['validation_roc_auc']):<5}  {took:>7}"
    )
    if record.get("selected"):
        line += "  best so far"
    if record.get("stopped") and "best_epoch" in record:
        line += f"\nearly stop: no gain for {plural(epoch - int(record['best_epoch']), 'epoch')}"
    return line.rstrip()


def _sampler_backend(record: Mapping[str, Any]) -> str:
    return (
        f"Note: the run sampled with the {record['saved']} backend and resumes with "
        f"{record['resumed']}, so the remaining steps sample a different stream"
    )


def _host_settings(record: Mapping[str, Any]) -> str:
    saved, resumed = record["saved"], record["resumed"]
    changes = ", ".join(f"{name} {resumed[name]} (was {saved[name]})" for name in resumed)
    return (
        f"Note: this segment runs with {changes}, so the remaining steps may not reproduce "
        "an uninterrupted run"
    )


def _warning(record: Mapping[str, Any]) -> str:
    return f"Warning: {record['message']}"


def _audit(record: Mapping[str, Any]) -> str:
    accounts = plural(int(record["accounts"]), "account")
    return (
        f"Audited {record['split']} at {record['date']}: {accounts} scored, "
        f"{count(record['rejected_accounts'])} rejected"
    )


def _feature_table(record: Mapping[str, Any]) -> str:
    return (
        f"Feature table, {record['split']} at {record['date']}: "
        f"{plural(int(record['accounts']), 'account')}, {plural(int(record['mules']), 'mule')}, "
        f"{count(record['rejected'])} rejected"
    )


def _diagnose(record: Mapping[str, Any]) -> str:
    head = f"{record['analysis']:<16} {record['status']:<8}"
    if record["status"] == "skipped":
        return f"{head} {record['reason']}"
    rows = int(record["rows"])
    # The noun keeps the width of "rows", so the times stay in one column.
    noun = "row" if rows == 1 else "rows"
    return f"{head} {count(rows):>9} {noun:<4}  {duration(float(record['seconds']))}"


def _suite(record: Mapping[str, Any]) -> str:
    runs: Mapping[str, Mapping[str, str]] = record["runs"]
    seeds = sorted({seed for actions in runs.values() for seed in actions}, key=int)
    total = sum(len(actions) for actions in runs.values())
    pending = sum(action != "keep" for actions in runs.values() for action in actions.values())
    bound = record.get("bound_hours")
    timing = f", at most {bound} hours" if bound is not None and pending else ""
    lines = [
        f"Suite {record['suite']} on dataset {str(record['dataset_id'])[:12]}: "
        f"{plural(total, 'run')}, {pending} to train{timing}"
    ]
    width = max(len("variant"), *(len(variant) for variant in runs)) + 2
    lines.append("  " + "variant".ljust(width) + "".join(f"seed {s}".ljust(10) for s in seeds))
    for variant, actions in runs.items():
        cells = "".join(actions.get(seed, "").ljust(10) for seed in seeds)
        lines.append("  " + variant.ljust(width) + cells)
    return "\n".join(line.rstrip() for line in lines)


def _who(record: Mapping[str, Any]) -> str:
    return f"{record['variant']} seed {record['seed']}"


def _run_archived(record: Mapping[str, Any]) -> str:
    return (
        f"Moved {shown_path(record['run'])} aside to {shown_path(record['archive'])} "
        f"(differs: {', '.join(record['differs'])})"
    )


def _run_finished(record: Mapping[str, Any]) -> str:
    if record["step"] == "train":
        seconds = record.get("seconds")
        took = "" if seconds is None else f", {duration(float(seconds))}"
        return (
            f"{_who(record)} trained: best epoch {record['best_epoch']}, validation proxy AP "
            f"{number(record['validation_proxy_ap'])}{took}"
        )
    return (
        f"{_who(record)} audited: validation AP {number(record['validation_ap'])}, test AP "
        f"{number(record['test_ap'])}"
    )


def _run_failed(record: Mapping[str, Any]) -> str:
    return f"{_who(record)} failed in {record['step']}: {brief(record['error'])}"


def _suite_stopped(record: Mapping[str, Any]) -> str:
    return (
        f"TigerGraph stayed unavailable, so the suite stops at {_who(record)} "
        f"({record['step']}): {brief(record['error'])}"
    )


# The events whose line is rewritten in place on a terminal and left out of a log.
PROGRESS = frozenset({"train", "install_wait"})
# The line of every event the package emits, by its name; _nothing shows none, since the
# record is in the files. tests/runtime/test_console.py checks that each event is here.
LINES: dict[str, Line] = {
    "retry": _retry,
    "context_split": _context_split,
    "install": _install,
    "gsql": _nothing,
    "install_unanswered": _install_unanswered,
    "install_wait": _install_wait,
    "installed": _installed,
    "drop_retired": _drop_retired,
    "scope": _scope,
    "reveal": _reveal,
    "hubs": _nothing,
    "dataset": _dataset,
    "already_complete": _already_complete,
    "already_audited": _already_audited,
    "start": _start,
    "resume": _start,
    "train": _train,
    "score": _nothing,
    "epoch": _epoch,
    "complete": _nothing,
    "sampler_backend": _sampler_backend,
    "host_settings": _host_settings,
    "warning": _warning,
    "audit": _audit,
    "feature_table": _feature_table,
    "diagnose": _diagnose,
    "suite": _suite,
    "run_archived": _run_archived,
    "run_finished": _run_finished,
    "run_failed": _run_failed,
    "suite_stopped": _suite_stopped,
    # What stops a command is on stderr: its one line, or a bug's traceback.
    "command_stopped": _nothing,
    "command_failed": _nothing,
}


def event_text(record: Mapping[str, Any]) -> str | None:
    """The console text of an event record, or None when it prints nothing.

    An event LINES does not name prints nothing, and so does a record that lacks a
    field its line reads: a console line never stops a command, and the record is in
    events.jsonl whatever its line.
    """
    line = LINES.get(str(record.get("event")))
    if line is None:
        return None
    try:
        return line(record)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def show_event(record: Mapping[str, Any]) -> None:
    """Print an event's console text: in place for PROGRESS, as lines for the others."""
    text = event_text(record)
    if text is None:
        return
    if record["event"] in PROGRESS:
        show_progress(text)
    else:
        show(text)
