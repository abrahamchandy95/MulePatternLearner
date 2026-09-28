"""A run's figures and report.md, drawn from the files the run saved; nothing else is read.

This module alone reads files and saves figures: the plot functions of training,
ranking and scores draw on the Axes they are given. Every figure is a
matplotlib.figure.Figure saved through the Agg canvas, so pyplot is never imported, as
a PNG at style.DPI under plots/<topic>_<figure>.png (paths.RunPaths.figure).

`mule train` writes the training figures once model.pt and every other file of the
run are saved (pipeline.train), `mule evaluate` the audit figures once the audits are
(pipeline.evaluate), and `mule report` redraws all of them from the files, offline
(report_run). Each then rewrites report.md, the run's tables with links to the figures
it has. A figure that fails to draw does not stop the others or report.md, and leaves
no older drawing under its name; the error is raised after them, naming every figure
that failed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from ..artifacts import (
    atomic_write,
    read_audit_scores,
    read_epochs,
    read_history,
    read_json,
    read_predictions,
)
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import REVIEW_BUDGETS, budget_name
from ..paths import RunPaths
from .ranking import SplitScores, plot_capture, plot_precision_recall, plot_roc
from .scores import plot_revealed_vs_hidden, plot_score_distribution, plot_threshold_metrics
from .style import DPI, PANEL, RC, STACKED, estimate, number
from .training import (
    plot_context_counts,
    plot_corrections,
    plot_objective,
    plot_run_health,
    plot_throughput,
    plot_validation_ranking,
)

# The figures of each kind, and what report.md calls them.
TRAINING_FIGURES = {
    "training_objective": "Training loss and nnPU objective",
    "training_corrections": "Steps whose non-negative correction fired",
    "validation_ranking": "Proxy validation ranking per epoch",
    "training_throughput": "Training throughput and contexts",
    "proxy_precision_recall": "Proxy precision and recall on observed labels",
    "run_health": "Rejections, sampler totals and database calls",
}
AUDIT_FIGURES = {
    "audit_precision_recall": "Audit precision and recall",
    "audit_roc": "Audit ROC",
    "audit_capture": "Audit capture of mules by review budget",
    "audit_threshold": "Test audit metrics at each threshold",
    "audit_score_distribution": "Test audit score densities",
    "audit_revealed_hidden": "Test audit ranks of revealed and hidden mules",
}
# A figure's drawing: it adds its Axes to the Figure and draws on them.
Drawing = Callable[[Figure], object]


def one(draw: Callable[[Axes], Axes]) -> Drawing:
    """The drawing of a one-panel figure."""
    return lambda figure: draw(figure.add_subplot())


def stacked(top: Callable[[Axes], Axes], bottom: Callable[[Axes], Axes]) -> Drawing:
    """The drawing of two panels stacked on one x axis."""

    def draw(figure: Figure) -> None:
        upper, lower = figure.subplots(2, 1, sharex=True)
        top(upper)
        bottom(lower)
        upper.set_xlabel("")

    return draw


def save_figure(path: Path, drawing: Drawing, size: tuple[float, float] = PANEL) -> None:
    """Draw a figure in the reporting style and save it at path as a PNG, atomically."""
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=size, layout="constrained")
        FigureCanvasAgg(figure)
        drawing(figure)
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(path) as pending:
            figure.savefig(pending, dpi=DPI, format="png")


@dataclass(frozen=True)
class TrainingFiles:
    """What a complete run saved about its training, as the training figures read it."""

    history: pd.DataFrame
    epochs: pd.DataFrame
    metrics: dict[str, Any]
    predictions: dict[str, SplitScores]


def training_files(run: RunPaths) -> TrainingFiles:
    """history.csv, epochs.csv, metrics.json and the proxy predictions of a complete run."""
    metrics = read_json(run.metrics)
    predictions = {}
    for split in HELD_OUT_SPLITS:
        frame = read_predictions(run.predictions(split))
        predictions[split] = SplitScores(
            y=frame.observed_label.to_numpy(np.int64),
            score=frame.score.to_numpy(np.float64),
            weight=np.ones(len(frame)),
            metrics=metrics["observed_label_proxy"][split],
        )
    return TrainingFiles(read_history(run.history), read_epochs(run.epochs), metrics, predictions)


def audited_splits(run: RunPaths) -> list[str]:
    """The splits the run has audited: a split's report is written last."""
    return [split for split in HELD_OUT_SPLITS if run.audit_report(split).exists()]


def audit_files(run: RunPaths) -> dict[str, SplitScores]:
    """The scored audit sample of each audited split, with its report's metrics."""
    audits = {}
    for split in audited_splits(run):
        report = read_json(run.audit_report(split))
        frame = read_audit_scores(run.audit_scores(split))
        audits[split] = SplitScores(
            y=frame.is_mule.to_numpy(np.int64),
            score=frame.score.to_numpy(np.float64),
            weight=1 / frame.inclusion_probability.to_numpy(np.float64),
            metrics=report["metrics"],
            intervals=report["intervals"],
            revealed=frame.revealed.to_numpy(bool),
        )
    return audits


def training_drawings(files: TrainingFiles) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each training figure."""
    history, metrics = files.history, files.metrics
    prevalence = metrics["validation_proxy"].get("prevalence")
    title = "Proxy: revealed mules against unlabelled accounts (not the ground truth)"
    return {
        "training_objective": (one(lambda ax: plot_objective(ax, history)), PANEL),
        "training_corrections": (one(lambda ax: plot_corrections(ax, history)), PANEL),
        "validation_ranking": (
            one(lambda ax: plot_validation_ranking(ax, files.epochs, prevalence)),
            PANEL,
        ),
        "training_throughput": (
            stacked(
                lambda ax: plot_throughput(ax, history),
                lambda ax: plot_context_counts(ax, history),
            ),
            STACKED,
        ),
        "proxy_precision_recall": (
            one(lambda ax: plot_precision_recall(ax, files.predictions, title=title)),
            PANEL,
        ),
        "run_health": (one(lambda ax: plot_run_health(ax, metrics)), PANEL),
    }


def audit_drawings(
    audits: Mapping[str, SplitScores],
) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each audit figure the audited splits allow.

    The precision-recall, ROC and capture figures draw every audited split; the
    threshold, density and revealed-and-hidden figures need the test split's audit.
    """
    drawings: dict[str, tuple[Drawing, tuple[float, float]]] = {}
    if not audits:
        return drawings
    weighted = "Ground-truth audit, weighted to each split's population"
    drawings["audit_precision_recall"] = (
        one(lambda ax: plot_precision_recall(ax, audits, title=weighted)),
        PANEL,
    )
    drawings["audit_roc"] = (one(lambda ax: plot_roc(ax, audits, title=weighted)), PANEL)
    drawings["audit_capture"] = (one(lambda ax: plot_capture(ax, audits, title=weighted)), PANEL)
    test = audits.get("test")
    if test is not None:
        threshold = float(test.metrics["threshold"])
        drawings["audit_threshold"] = (
            one(lambda ax: plot_threshold_metrics(ax, test, threshold)),
            PANEL,
        )
        drawings["audit_score_distribution"] = (
            one(lambda ax: plot_score_distribution(ax, test, threshold)),
            PANEL,
        )
        drawings["audit_revealed_hidden"] = (
            one(lambda ax: plot_revealed_vs_hidden(ax, test)),
            PANEL,
        )
    return drawings


def draw(
    run: RunPaths, drawings: Mapping[str, tuple[Drawing, tuple[float, float]]]
) -> dict[str, Any]:
    """Save every figure, then report.md; raise after both if any figure failed.

    A figure that fails loses its older PNG, so report.md links only figures drawn from
    the files as they are. Returns report.md's path and the figures drawn.
    """
    drawn: list[str] = []
    failed: dict[str, Exception] = {}
    for name, (drawing, size) in drawings.items():
        try:
            save_figure(run.figure(name), drawing, size)
        except Exception as error:  # the other figures and report.md are still written
            failed[name] = error
            # No older drawing stays under its name for report.md to link as current.
            run.figure(name).unlink(missing_ok=True)
        else:
            drawn.append(name)
    report = write_report(run)
    if failed:
        errors = "; ".join(f"{name}: {error!r}" for name, error in failed.items())
        raise RuntimeError(
            f"Figures of {run.root} failed ({errors}); every other file is written, and "
            "`mule report` draws them again"
        ) from next(iter(failed.values()))
    return {"report": str(report), "figures": [str(run.figure(name)) for name in drawn]}


def write_training_report(run: RunPaths) -> dict[str, Any]:
    """The training figures of a complete run, then report.md (what `mule train` adds)."""
    return draw(run, training_drawings(training_files(run)))


def write_audit_report(run: RunPaths) -> dict[str, Any]:
    """The audit figures of the audited splits, then report.md (what `mule evaluate` adds)."""
    return draw(run, audit_drawings(audit_files(run)))


def report_run(run: RunPaths) -> dict[str, Any]:
    """Redraw every figure the run's files allow and report.md: `mule report`, offline.

    The training figures need a complete run (metrics.json), the audit figures an audit.
    """
    complete, audits = run.metrics.exists(), audit_files(run)
    if not complete and not audits:
        raise ValueError(f"{run.root} holds neither a complete run nor an audit to report")
    drawings = training_drawings(training_files(run)) if complete else {}
    return draw(run, {**drawings, **audit_drawings(audits)})


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def figure_links(run: RunPaths, figures: Mapping[str, str]) -> list[str]:
    """Links to the figures the run has, relative to report.md."""
    lines: list[str] = []
    for name, caption in figures.items():
        path = run.figure(name)
        if path.exists():
            lines += [f"![{caption}]({path.relative_to(run.root).as_posix()})", ""]
    return lines


def _budget_rows(
    metrics: Mapping[str, Mapping[str, Any]], intervals: Mapping[str, Mapping[str, Any]]
) -> list[list[str]]:
    rows = []
    for fraction in REVIEW_BUDGETS:
        name = budget_name(fraction)
        for label, key in (("Recall", f"recall_at_{name}"), ("Precision", f"precision_at_{name}")):
            rows.append(
                [f"{label} in the top {fraction:.0%}"]
                + [estimate(metrics[s].get(key), intervals[s].get(key)) for s in metrics]
            )
    return rows


def _threshold_row(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    cells = [
        " / ".join(number(values.get(key)) for key in ("precision", "recall", "f1"))
        for values in metrics.values()
    ]
    threshold = next(iter(metrics.values())).get("threshold")
    return [f"At the selected threshold {threshold:.6g}: precision / recall / F1", *cells]


def audit_section(run: RunPaths) -> list[str]:
    """report.md's audit tables and figures, for the audited splits."""
    reports = {split: read_json(run.audit_report(split)) for split in audited_splits(run)}
    metrics = {split: report["metrics"] for split, report in reports.items()}
    intervals = {split: report["intervals"] for split, report in reports.items()}
    header = ["", *(f"{split} ({report['purpose']})" for split, report in reports.items())]
    rows = [
        ["Cutoff", *(report["date"] for report in reports.values())],
        ["Population accounts", *(number(r["population_accounts"]) for r in reports.values())],
        [
            "Audit sample: mules / accounts",
            *(
                f"{number(m['sample_positives'])} / {number(m['sample_accounts'])}"
                for m in metrics.values()
            ),
        ],
        [
            "Revealed / hidden mules",
            *(
                f"{number(r['revealed_positives'])} / {number(r['hidden_positives'])}"
                for r in reports.values()
            ),
        ],
        ["Weighted prevalence", *(number(m["weighted_prevalence"]) for m in metrics.values())],
        *(
            [label, *(estimate(metrics[s].get(key), intervals[s].get(key)) for s in reports)]
            for label, key in (("Average precision", "average_precision"), ("ROC AUC", "roc_auc"))
        ),
        *_budget_rows(metrics, intervals),
        _threshold_row(metrics),
        ["Rejected accounts", *(number(r["rejected_accounts"]) for r in reports.values())],
    ]
    level = next(iter(reports.values()))["constants"]["interval"]
    return [
        "## Ground-truth audit",
        "",
        "Decisions use the validation audit; the test audit is for reporting. Each audit "
        "scores every mule of its split and a uniform sample of the other accounts, "
        "weighted to the split's whole population. In parentheses: the ring-clustered "
        f"{level:.0%} bootstrap interval.",
        "",
        *table(header, rows),
        "",
        *figure_links(run, AUDIT_FIGURES),
    ]


def training_section(run: RunPaths) -> list[str]:
    """report.md's training tables and figures, for a complete run."""
    metrics = read_json(run.metrics)
    epochs = read_epochs(run.epochs)
    proxy = {split: metrics["observed_label_proxy"][split] for split in HELD_OUT_SPLITS}
    contexts = metrics["contexts"]
    rejected = metrics["rejected_roots"]
    run_rows = [
        ["Epochs run (selected)", f"{len(epochs)} ({metrics['best_epoch']})"],
        ["Parameters", number(metrics["parameter_count"])],
        ["Hours", f"{metrics['elapsed_seconds'] / 3600:.2f}"],
        ["Device, sampler backend", f"{metrics['device']}, {metrics['sampler_backend']}"],
        ["Objective", f"{metrics['objective']} (positive weight {metrics['positive_weight']})"],
        ["Database calls", number(metrics["database_calls_during_training"])],
        [
            "Contexts requested / distinct / from memory",
            " / ".join(number(contexts[key]) for key in ("requested", "distinct", "memory_hits")),
        ],
        [
            "Rejected roots: " + " / ".join(rejected),
            " / ".join(number(counts["rejected"]) for counts in rejected.values()),
        ],
    ]
    empty: dict[str, Mapping[str, Any]] = {split: {} for split in HELD_OUT_SPLITS}
    proxy_rows = [
        [
            "Observed positives / accounts",
            *(f"{number(m['positives'])} / {number(m['n'])}" for m in proxy.values()),
        ],
        ["Prevalence", *(number(m.get("prevalence")) for m in proxy.values())],
        ["Average precision", *(number(m.get("average_precision")) for m in proxy.values())],
        ["ROC AUC", *(number(m.get("roc_auc")) for m in proxy.values())],
        *_budget_rows(proxy, empty),
        _threshold_row(proxy),
    ]
    return [
        "## Training",
        "",
        *table(["Run", ""], run_rows),
        "",
        "The proxy scores each split's revealed mules against a sample of unlabelled "
        "accounts, which count as negatives: it is what training selects on, not the "
        "ground truth.",
        "",
        *table(["Proxy", *HELD_OUT_SPLITS], proxy_rows),
        "",
        *figure_links(run, TRAINING_FIGURES),
    ]


def report_text(run: RunPaths) -> str:
    """report.md: the run's provenance, then its audit and training tables with figures."""
    lines = [f"# Run {run.root.parent.name}/{run.root.name}", ""]
    if run.config.exists():
        provenance = read_json(run.config)["provenance"]
        commit = str(provenance.get("git_commit") or "unknown")[:12]
        dirty = " with uncommitted changes" if provenance.get("git_dirty") else ""
        lines += [
            f"Dataset `{str(provenance.get('dataset_id'))[:12]}`, commit `{commit}`{dirty}, "
            f"started {provenance.get('started')}.",
            "",
        ]
    if audited_splits(run):
        lines += audit_section(run)
    if run.metrics.exists():
        lines += training_section(run)
    else:
        lines += ["The run is not complete: it has no metrics.json yet.", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(run: RunPaths) -> Path:
    """Replace the run's report.md with report_text."""
    with atomic_write(run.report) as pending:
        pending.write_text(report_text(run))
    return run.report
