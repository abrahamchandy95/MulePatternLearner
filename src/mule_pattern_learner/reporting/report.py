"""A run's figures and report.md, drawn from the files the run saved; nothing else is read.

This module alone reads files and saves figures: the plot functions of training,
ranking and scores draw on the Axes they are given. Every figure is a
matplotlib.figure.Figure saved through the Agg canvas, so pyplot is never imported, as
a PNG at style.DPI under plots/<topic>_<figure>.png (paths.RunPaths.figure).

`mule train` writes the training figures once model.pt and every other file of the
run are saved (pipeline.train), `mule evaluate` the audit figures once the audits are
(pipeline.evaluate), and `mule report` redraws all of them from the files, offline
(report_run). Each then rewrites report.md, the run's tables with links to the figures
it has. A control-experiment suite (write_suite_report) and a diagnostic study
(write_diagnostics_report, from the long tables of `mule diagnose`) have figures and a
report.md of their own, drawn the same way. A figure that fails to draw does not stop
the others or report.md, and leaves no older drawing under its name; the error is raised
after them, naming every figure that failed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import Any, Protocol

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..artifacts import (
    DELTA_METRIC,
    DIAGNOSTIC_TABLES,
    PROXY_METRIC,
    atomic_write,
    read_audit_scores,
    read_comparison,
    read_diagnostic_table,
    read_epochs,
    read_feature_table,
    read_history,
    read_json,
    read_predictions,
    read_run_provenance,
    read_summary,
)
from ..contract.graph_schema import HELD_OUT_SPLITS
from ..metrics import INTERVAL, REVIEW_BUDGETS, budget_name
from ..paths import BASELINE_VARIANT, DiagnosticsPaths, RunPaths, SuitePaths
from .comparison import (
    MeanCurve,
    VariantSeeds,
    mean_capture,
    plot_budget_recall,
    plot_capture_overlay,
    plot_comparison,
    plot_paired_delta,
    plot_proxy_vs_audit,
    plot_validation_overlay,
)
from .diagnostics import (
    baseline_rows,
    plot_ap_concentration,
    plot_baselines,
    plot_drift,
    plot_label_curve,
    plot_nnpu_simulation,
    plot_proxy_validity,
    plot_reveal_spread,
    plot_ring_coverage,
    plot_univariate,
    strongest_features,
    strongest_shifts,
)
from .ranking import SplitScores, plot_capture, plot_precision_recall, plot_roc, share_label
from .scores import plot_revealed_vs_hidden, plot_score_distribution, plot_threshold_metrics
from .style import BASELINE, DPI, MUTED, PANEL, RC, SPLIT_COLOURS, STACKED, estimate, number
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


class Reported(Protocol):
    """A directory with figures and a report.md: a run's (RunPaths) or a suite's (SuitePaths)."""

    @property
    def root(self) -> Path: ...
    def figure(self, name: str) -> Path: ...


def draw(
    home: Reported,
    drawings: Mapping[str, tuple[Drawing, tuple[float, float]]],
    write: Callable[[], Path],
) -> dict[str, Any]:
    """Save every figure, then report.md (``write``); raise after both if any figure failed.

    A figure that fails loses its older PNG, so report.md links only figures drawn from
    the files as they are. Returns report.md's path and the figures drawn.
    """
    drawn: list[str] = []
    failed: dict[str, Exception] = {}
    for name, (drawing, size) in drawings.items():
        try:
            save_figure(home.figure(name), drawing, size)
        except Exception as error:  # the other figures and report.md are still written
            failed[name] = error
            # No older drawing stays under its name for report.md to link as current.
            home.figure(name).unlink(missing_ok=True)
        else:
            drawn.append(name)
    report = write()
    if failed:
        errors = "; ".join(f"{name}: {error!r}" for name, error in failed.items())
        raise RuntimeError(
            f"Figures of {home.root} failed ({errors}); every other file is written, and "
            "`mule report` draws them again"
        ) from next(iter(failed.values()))
    return {"report": str(report), "figures": [str(home.figure(name)) for name in drawn]}


def write_training_report(run: RunPaths) -> dict[str, Any]:
    """The training figures of a complete run, then report.md (what `mule train` adds)."""
    return draw(run, training_drawings(training_files(run)), lambda: write_report(run))


def write_audit_report(run: RunPaths) -> dict[str, Any]:
    """The audit figures of the audited splits, then report.md (what `mule evaluate` adds)."""
    return draw(run, audit_drawings(audit_files(run)), lambda: write_report(run))


def report_run(run: RunPaths) -> dict[str, Any]:
    """Redraw every figure the run's files allow and report.md: `mule report`, offline.

    The training figures need a complete run (metrics.json), the audit figures an audit.
    """
    complete, audits = run.metrics.exists(), audit_files(run)
    if not complete and not audits:
        raise ValueError(f"{run.root} holds neither a complete run nor an audit to report")
    drawings = training_drawings(training_files(run)) if complete else {}
    return draw(run, {**drawings, **audit_drawings(audits)}, lambda: write_report(run))


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """The lines of a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def figure_links(home: Reported, figures: Mapping[str, str]) -> list[str]:
    """Links to the figures a run or suite has, relative to its report.md."""
    lines: list[str] = []
    for name, caption in figures.items():
        path = home.figure(name)
        if path.exists():
            lines += [f"![{caption}]({path.relative_to(home.root).as_posix()})", ""]
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
            "Contexts requested / distinct / from memory / from disk",
            " / ".join(
                number(contexts.get(key))
                for key in ("requested", "distinct", "memory_hits", "disk_hits")
            ),
        ],
        ["Disk cache hit rate", number(contexts.get("disk_hit_rate"))],
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


# The figures of a control-experiment suite, and what its report.md calls them.
SUITE_FIGURES = {
    "comparison_ap": "Audit AP per variant on validation and test",
    "comparison_delta": "Validation audit AP against the baseline, paired",
    "comparison_budget": "Validation audit recall at the review budgets",
    "comparison_capture": "Seed-mean validation audit capture against the baseline",
    "comparison_validation": "Seed-mean proxy validation AP per epoch against the baseline",
    "comparison_proxy_vs_audit": "Selected proxy AP against the validation audit AP, per run",
}
# The small multiples of a suite: panels per row.
PANELS_PER_ROW = 3


@dataclass(frozen=True)
class SuiteFiles:
    """What a suite's tables and its complete runs saved, as its figures read them.

    ``comparison`` is ranked by the seed-mean validation audit AP, best first, with the
    variants the audit could not rank last in the suite's order. ``captures`` and
    ``epochs`` hold each complete run's validation audit sample and epochs.csv, by
    variant and seed.
    """

    summary: pd.DataFrame
    comparison: pd.DataFrame
    captures: dict[str, dict[int, SplitScores]]
    epochs: dict[str, dict[int, pd.DataFrame]]
    prevalence: float | None


def complete_runs(summary: pd.DataFrame) -> list[tuple[str, int]]:
    """The (variant, seed) of every complete run of a suite, in summary.csv's order."""
    complete = summary[summary.status == "complete"]
    return list(dict.fromkeys(zip(complete.variant, complete.seed.astype(int), strict=True)))


def suite_files(suite: SuitePaths) -> SuiteFiles:
    """summary.csv, comparison.csv, and the validation audits and epochs of complete runs."""
    summary = read_summary(suite.summary)
    comparison = read_comparison(suite.comparison)
    comparison = comparison.sort_values("validation_ap", ascending=False, na_position="last")
    captures: dict[str, dict[int, SplitScores]] = {}
    epochs: dict[str, dict[int, pd.DataFrame]] = {}
    prevalence = None
    for variant, seed in complete_runs(summary):
        run = suite.run(variant, seed)
        report = read_json(run.audit_report("validation"))
        frame = read_audit_scores(run.audit_scores("validation"))
        captures.setdefault(variant, {})[seed] = SplitScores(
            y=frame.is_mule.to_numpy(np.int64),
            score=frame.score.to_numpy(np.float64),
            weight=1 / frame.inclusion_probability.to_numpy(np.float64),
            metrics=report["metrics"],
        )
        epochs.setdefault(variant, {})[seed] = read_epochs(run.epochs)
        if prevalence is None:
            prevalence = read_json(run.metrics)["validation_proxy"].get("prevalence")
    return SuiteFiles(summary, comparison.reset_index(drop=True), captures, epochs, prevalence)


def _optional(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def variant_seeds(
    files: SuiteFiles, split: str, metric: str, column: str, *, interval: str | None = None
) -> list[VariantSeeds]:
    """Each ranked variant's per-seed values of a summary metric and its comparison column.

    ``interval`` names the comparison columns of the interval without their _low and
    _high endings.
    """
    summary = files.summary
    chosen = summary[(summary.split == split) & (summary.metric == metric)]
    chosen = chosen[chosen.status == "complete"]
    rows = []
    for record in records(files.comparison):
        name = str(record["variant"])
        mine = chosen[chosen.variant == name]
        bounds = None
        if interval is not None:
            low, high = record[f"{interval}_low"], record[f"{interval}_high"]
            bounds = None if pd.isna(low) or pd.isna(high) else (float(low), float(high))
        rows.append(
            VariantSeeds(
                name,
                dict(zip(mine.seed.astype(int), mine.value.astype(float), strict=True)),
                _optional(record[column]),
                bounds,
                str(record["consistent"]) == "True",
            )
        )
    return rows


def panels(
    draws: list[Callable[[Axes], Axes]],
    legend: tuple[list[Line2D], list[str]],
    labels: tuple[str, str],
) -> Drawing:
    """Small multiples: one panel per drawing, sharing axes, with one legend below them."""

    def drawing(figure: Figure) -> None:
        rows = math.ceil(len(draws) / PANELS_PER_ROW)
        columns = min(len(draws), PANELS_PER_ROW)
        axes = figure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
        flat = list(axes.flat)
        for ax, draw_panel in zip(flat, draws, strict=False):
            draw_panel(ax)
        for index, ax in enumerate(flat[len(draws) :], start=len(draws)):
            ax.set_visible(False)
            # The panel above an empty place keeps its x tick labels.
            flat[index - columns].xaxis.set_tick_params(labelbottom=True)
        figure.supxlabel(labels[0], fontsize=9)
        figure.supylabel(labels[1], fontsize=9)
        # Above the panels, clear of the x label below them.
        figure.legend(*legend, loc="outside upper center", ncols=len(legend[0]))

    return drawing


def panels_size(count: int) -> tuple[float, float]:
    """Small multiples of count panels: at least a panel's width, taller with more rows."""
    rows = math.ceil(count / PANELS_PER_ROW)
    return (max(PANEL[0], 2.5 * min(count, PANELS_PER_ROW) + 0.8), 2.2 * rows + 1.0)


def rows_size(count: int, per_row: float = 0.36, width: float = 7.0) -> tuple[float, float]:
    """A figure of one row per variant: taller with more variants."""
    return (width, 2.0 + per_row * count)


def overlay_legend(split: str, reference: str) -> tuple[list[Line2D], list[str]]:
    handles = [
        Line2D([], [], color=BASELINE, linewidth=1.4),
        Line2D([], [], color=SPLIT_COLOURS[split], linewidth=1.8),
        Line2D([], [], color=MUTED, linestyle="--", linewidth=0.9),
    ]
    return handles, ["baseline, mean over seeds", "the panel's variant", reference]


def suite_drawings(files: SuiteFiles) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each suite figure its complete runs allow."""
    drawings: dict[str, tuple[Drawing, tuple[float, float]]] = {}
    if not files.captures:
        return drawings
    count = len(files.comparison)
    splits = {
        split: variant_seeds(
            files, split, "average_precision", f"{split}_ap", interval=f"{split}_ap"
        )
        for split in HELD_OUT_SPLITS
    }

    def both(figure: Figure) -> None:
        axes = figure.subplots(1, len(splits), sharey=True, squeeze=False)[0]
        for ax, (split, rows) in zip(axes, splits.items(), strict=True):
            baseline = next((row.mean for row in rows if row.name == BASELINE_VARIANT), None)
            plot_comparison(ax, rows, split=split, baseline=baseline)

    drawings["comparison_ap"] = (both, rows_size(count, width=9.0))
    deltas = variant_seeds(
        files, "validation", DELTA_METRIC, "validation_ap_delta", interval="validation_ap_delta"
    )
    deltas = sorted(
        (row for row in deltas if row.mean is not None),
        key=lambda row: -(row.mean if row.mean is not None else 0.0),
    )
    if deltas:
        drawings["comparison_delta"] = (
            one(lambda ax: plot_paired_delta(ax, deltas)),
            rows_size(len(deltas)),
        )
    budgets = {
        fraction: variant_seeds(
            files,
            "validation",
            f"recall_at_{budget_name(fraction)}",
            f"validation_recall_at_{budget_name(fraction)}",
        )
        for fraction in REVIEW_BUDGETS
    }
    drawings["comparison_budget"] = (
        one(lambda ax: plot_budget_recall(ax, budgets)),
        rows_size(count, per_row=0.55),
    )
    # Every panel holds the baseline beside its variant, so the baseline has no panel of
    # its own unless it is alone.
    audited = [name for name in files.comparison.variant if name in files.captures]
    order = [name for name in audited if name != BASELINE_VARIANT] or audited
    prevalences = [
        scores.prevalence for runs in files.captures.values() for scores in runs.values()
    ]
    grid = np.geomspace(min([1e-4, *(p / 2 for p in prevalences)]), 1.0, 200)
    captures = {
        name: mean_capture(name, list(files.captures[name].values()), grid) for name in audited
    }
    drawings["comparison_capture"] = (
        panels(
            [
                partial(
                    plot_capture_overlay,
                    variant=captures[name],
                    baseline=captures.get(BASELINE_VARIANT),
                )
                for name in order
            ],
            overlay_legend("validation", "random ranking"),
            ("Top share of accounts reviewed, by score (log scale)", "Share of mules found"),
        ),
        panels_size(len(order)),
    )
    curves = {name: mean_epochs(name, files.epochs[name]) for name in audited}
    last = max(int(curve.x.max()) for curve in curves.values())
    drawings["comparison_validation"] = (
        panels(
            [
                partial(
                    plot_validation_overlay,
                    variant=curves[name],
                    baseline=curves.get(BASELINE_VARIANT),
                    prevalence=files.prevalence,
                    last_epoch=last,
                )
                for name in order
            ],
            overlay_legend("validation", "prevalence: a random ranking's AP"),
            ("Epoch", "Proxy validation AP"),
        ),
        panels_size(len(order)),
    )
    points = proxy_points(files.summary)
    if len(points):
        drawings["comparison_proxy_vs_audit"] = (
            one(
                lambda ax: plot_proxy_vs_audit(
                    ax,
                    points.variant.tolist(),
                    points[PROXY_METRIC].to_numpy(np.float64),
                    points.average_precision.to_numpy(np.float64),
                )
            ),
            PANEL,
        )
    return drawings


def mean_epochs(name: str, epochs: Mapping[int, pd.DataFrame]) -> MeanCurve:
    """A variant's proxy validation AP per epoch, averaged over its seeds.

    The curve ends at the last epoch every seed trained: early stopping ends seeds at
    different epochs, and a mean over the seeds that went on would read as the same one.
    """
    reached = min(int(frame.epoch.max()) for frame in epochs.values())
    joined = pd.concat(
        [frame.loc[frame.epoch <= reached, ["epoch", "validation_ap"]] for frame in epochs.values()]
    )
    mean = joined.groupby("epoch").validation_ap.mean()
    return MeanCurve(name, mean.index.to_numpy(np.float64), mean.to_numpy(np.float64), len(epochs))


def proxy_points(summary: pd.DataFrame) -> pd.DataFrame:
    """Each complete run's selected proxy AP and validation audit AP (variant, seed, both)."""
    chosen = summary[
        (summary.status == "complete")
        & (summary.split == "validation")
        & summary.metric.isin([PROXY_METRIC, "average_precision"])
    ]
    wide = chosen.pivot_table(index=["variant", "seed"], columns="metric", values="value")
    wanted = [PROXY_METRIC, "average_precision"]
    if not set(wanted) <= set(wide.columns):
        return pd.DataFrame(columns=["variant", "seed", *wanted])
    return wide.dropna(subset=wanted).reset_index()


def write_suite_report(suite: SuitePaths) -> dict[str, Any]:
    """Draw a suite's figures from its tables and runs, then its report.md.

    What the runner writes after comparing a suite (experiments.runner), and what
    `mule report` redraws offline from a suite directory.
    """
    files = suite_files(suite)
    return draw(suite, suite_drawings(files), lambda: write_suite_text(suite, files))


def _interval_text(value: Any, low: Any, high: Any) -> str:
    interval = None if pd.isna(low) or pd.isna(high) else [float(low), float(high)]
    return estimate(_optional(value), interval)


def suite_text(suite: SuitePaths, files: SuiteFiles) -> str:
    """A suite's report.md: its runs, the validation ranking, the test audit, the figures."""
    summary, comparison = files.summary, files.comparison
    runs = summary.drop_duplicates(["variant", "seed"])
    statuses = runs.status.value_counts()
    counted = ", ".join(f"{number(int(n))} {status}" for status, n in statuses.items())
    seeds = ", ".join(str(seed) for seed in sorted(runs.seed.astype(int).unique()))
    lines = [
        f"# Suite {suite.root.name}",
        "",
        f"{len(comparison)} variants with the seeds {seeds}: {len(runs)} runs, {counted}.",
    ]
    provenance = [read_run_provenance(suite.run(v, s).config) for v, s in complete_runs(summary)]
    if provenance:
        datasets = sorted({str(p.get("dataset_id"))[:12] for p in provenance})
        commits = sorted({str(p.get("git_commit") or "unknown")[:12] for p in provenance})
        dirty = sum(bool(p.get("git_dirty")) for p in provenance)
        lines[-1] += f" Dataset `{'`, `'.join(datasets)}`, commit `{'`, `'.join(commits)}`" + (
            f"; {dirty} of the complete runs had uncommitted changes." if dirty else "."
        )
    unpaired = comparison.unpaired_accounts.dropna()
    # The variants compared with the baseline.
    variants = max(len(comparison) - 1, 0)
    lines += [
        "",
        "Decisions use the validation audit; the test audit is for reporting, not selection. "
        "The pool groups (pool_activity and pool_internal_inflows) were designed after "
        "reading test-split mules and the data generator's mule typology, so the test "
        "audit is optimistic for every variant that keeps them.",
        "",
        "## Validation audit, for decisions",
        "",
        f"Variants ranked by their mean validation audit AP over seeds. In parentheses: the "
        f"{INTERVAL:.0%} interval of the mean over paired bootstrap replicates, which draw one "
        "resample of the accounts every audit scored and apply it to every run. It covers "
        "the audit sample's uncertainty for these seeds, not the spread between seeds (the "
        "standard deviation beside it). The delta is the variant's mean AP minus the "
        "baseline's over the seeds both completed, on those same accounts; a consistent "
        "delta has one sign in every seed and an interval that excludes zero. "
        f"With {variants} variant{'' if variants == 1 else 's'} at {INTERVAL:.0%}, about "
        f"{variants * (1 - INTERVAL):.1f} would exclude zero by chance, so a single "
        "consistent delta is exploratory until repeated with more seeds.",
        "",
    ]
    if len(unpaired) and unpaired.iloc[0] > 0:
        lines += [
            f"{number(int(unpaired.iloc[0]))} validation accounts that some run's audit "
            "rejected are left out of the pairing.",
            "",
        ]
    budgets = [budget_name(f) for f in REVIEW_BUDGETS]
    header = [
        "Rank",
        "Variant",
        "Seeds",
        "AP",
        "Standard deviation",
        "Delta from the baseline",
        "Consistent",
        "ROC AUC",
        "Recall in the top " + " / ".join(share_label(f) for f in REVIEW_BUDGETS),
    ]
    rows = []
    for rank, row in enumerate(records(comparison), start=1):
        delta = (
            _interval_text(
                row["validation_ap_delta"],
                row["validation_ap_delta_low"],
                row["validation_ap_delta_high"],
            )
            if not pd.isna(row["validation_ap_delta"])
            else ""
        )
        rows.append(
            [
                str(rank),
                f"**{row['variant']}**" if row["variant"] == BASELINE_VARIANT else row["variant"],
                row["seeds"] or "none",
                _interval_text(
                    row["validation_ap"], row["validation_ap_low"], row["validation_ap_high"]
                ),
                number(_optional(row["validation_ap_spread"])),
                delta,
                {True: "yes", False: "no"}.get(row["consistent"], "") if delta else "",
                number(_optional(row["validation_roc_auc"])),
                " / ".join(number(_optional(row[f"validation_recall_at_{b}"])) for b in budgets),
            ]
        )
    lines += [*table(header, rows), ""]
    test_rows = [
        [
            row["variant"],
            _interval_text(row["test_ap"], row["test_ap_low"], row["test_ap_high"]),
            number(_optional(row["test_ap_spread"])),
            number(_optional(row["test_roc_auc"])),
            " / ".join(number(_optional(row[f"test_recall_at_{b}"])) for b in budgets),
        ]
        for row in records(comparison)
    ]
    lines += [
        "## Test audit, for reporting, not selection",
        "",
        "In the validation ranking's order; never rank or choose variants by these.",
        "",
        *table(["Variant", "AP", "Standard deviation", "ROC AUC", header[-1]], test_rows),
        "",
        "## Variants and runs",
        "",
    ]
    variant_rows = [
        [
            row["variant"],
            row["question"],
            row["changes"],
            "" if pd.isna(row["best_epoch"]) else f"{row['best_epoch']:.1f}",
            number(None if pd.isna(row["parameter_count"]) else int(row["parameter_count"])),
            "" if pd.isna(row["training_hours"]) else f"{row['training_hours']:.2f}",
            row["differs"],
        ]
        for row in records(comparison)
    ]
    lines += [
        *table(
            [
                "Variant",
                "Question",
                "Changes",
                "Mean best epoch",
                "Parameters",
                "Mean hours",
                "Runs that differ from the others",
            ],
            variant_rows,
        ),
        "",
    ]
    others = runs[runs.status != "complete"]
    if len(others):
        listed = ", ".join(
            f"{v} seed {int(s)} ({status})"
            for v, s, status in zip(others.variant, others.seed, others.status, strict=True)
        )
        lines += [f"Not compared: {listed}.", ""]
    lines += figure_links(suite, SUITE_FIGURES)
    return "\n".join(lines).rstrip("\n") + "\n"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A table's rows as dicts by column name."""
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def write_suite_text(suite: SuitePaths, files: SuiteFiles) -> Path:
    """Replace the suite's report.md with suite_text."""
    with atomic_write(suite.report) as pending:
        pending.write_text(suite_text(suite, files))
    return suite.report


# The figures of a diagnostic study, and what its report.md calls them.
DIAGNOSTICS_FIGURES = {
    "baselines": "Each baseline's audit AP beside the run's",
    "label_curve": "Audit AP against the oracle-labelled train mules a learner was fitted on",
    "univariate_auc": "Each feature's ROC AUC alone",
    "drift": "The non-mules' feature drift from the train cutoff",
    "ap_concentration": "How few mules make the audit AP",
    "ring_coverage": "Rings with a member in the review budgets",
    "proxy_validity": "The proxy predictions against the ground truth",
    "reveal_spread": "The label reveal replayed over salts",
    "nnpu_simulation": "The nnPU positive weight, simulated",
}
# The figures each analysis' table draws.
ANALYSIS_FIGURES = {
    "baselines": ("baselines",),
    "learning_curve": ("label_curve",),
    "univariate": ("univariate_auc",),
    "drift": ("drift",),
    "subgroups": ("ap_concentration", "ring_coverage"),
    "proxy_validity": ("proxy_validity",),
    "reveal_spread": ("reveal_spread",),
    "nnpu_simulation": ("nnpu_simulation",),
}
# The rows report.md lists of the longest tables.
REPORTED_ROWS = 10


@dataclass(frozen=True)
class StudyFiles:
    """What a diagnostic study saved: study.json, the feature table and its analyses' tables.

    ``tables`` holds the analyses whose table exists, by DIAGNOSTIC_TABLES key.
    """

    record: dict[str, Any]
    features: pd.DataFrame | None
    tables: dict[str, pd.DataFrame]


def study_files(study: DiagnosticsPaths) -> StudyFiles:
    """study.json, features.parquet and every analysis' table the study holds."""
    record = read_json(study.study) if study.study.exists() else {}
    features = read_feature_table(study.features) if study.features.exists() else None
    tables = {
        name: read_diagnostic_table(study.table(name), name)
        for name in DIAGNOSTIC_TABLES
        if study.table(name).exists()
    }
    return StudyFiles(record, features, tables)


def paired(
    draws: tuple[Callable[[Axes], Axes], Callable[[Axes], Axes]], *, stacked: bool
) -> Drawing:
    """The drawing of two panels, validation's then test's: side by side sharing the y
    axis, or ``stacked`` one above the other, each with its own axes and legend."""

    def drawing(figure: Figure) -> None:
        first, second = figure.subplots(2, 1) if stacked else figure.subplots(1, 2, sharey=True)
        draws[0](first)
        draws[1](second)

    return drawing


def study_drawings(files: StudyFiles) -> dict[str, tuple[Drawing, tuple[float, float]]]:
    """The drawing and size of each figure the study's tables allow."""
    tables = files.tables
    drawings: dict[str, tuple[Drawing, tuple[float, float]]] = {}
    if "baselines" in tables:
        table = tables["baselines"]
        drawings["baselines"] = (
            paired(
                (
                    partial(plot_baselines, table=table, split="validation"),
                    partial(plot_baselines, table=table, split="test", labels=False),
                ),
                stacked=False,
            ),
            rows_size(len(baseline_rows(table)), per_row=0.3, width=12.0),
        )
    if "learning_curve" in tables:
        table = tables["learning_curve"]
        drawings["label_curve"] = (
            paired(
                (
                    partial(plot_label_curve, table=table, split="validation"),
                    partial(plot_label_curve, table=table, split="test"),
                ),
                stacked=True,
            ),
            (7.0, 10.0),
        )
    if "univariate" in tables:
        table = tables["univariate"]
        drawings["univariate_auc"] = (
            one(partial(plot_univariate, table=table)),
            rows_size(len(strongest_features(table)), per_row=0.3, width=8.0),
        )
    if "drift" in tables:
        table = tables["drift"]
        drawings["drift"] = (
            one(partial(plot_drift, table=table)),
            rows_size(len(strongest_shifts(table)), per_row=0.3, width=8.0),
        )
    if "subgroups" in tables:
        table = tables["subgroups"]
        drawings["ap_concentration"] = (one(partial(plot_ap_concentration, table=table)), PANEL)
        if (_metric(table, "rings").value > 0).any():
            drawings["ring_coverage"] = (one(partial(plot_ring_coverage, table=table)), PANEL)
    if "proxy_validity" in tables:
        table = tables["proxy_validity"]
        drawings["proxy_validity"] = (one(partial(plot_proxy_validity, table=table)), PANEL)
    if "reveal_spread" in tables:
        reveal = files.record.get("reveal", {})
        drawings["reveal_spread"] = (
            one(
                partial(
                    plot_reveal_spread,
                    table=tables["reveal_spread"],
                    budget=reveal.get("budget"),
                    salt=reveal.get("salt"),
                )
            ),
            PANEL,
        )
    if "nnpu_simulation" in tables:
        table = tables["nnpu_simulation"]
        drawings["nnpu_simulation"] = (one(partial(plot_nnpu_simulation, table=table)), PANEL)
    return drawings


def write_diagnostics_report(study: DiagnosticsPaths) -> dict[str, Any]:
    """Draw a diagnostic study's figures from its tables, then its report.md.

    What `mule diagnose` writes after its analyses (diagnostics.study), and what
    `mule report` redraws offline from a diagnostics directory.
    """
    files = study_files(study)
    return draw(study, study_drawings(files), lambda: write_study_text(study, files))


def _metric(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    return table[table.metric == metric]


def _value(frame: pd.DataFrame, **keys: Any) -> float | None:
    """The one value of a long table's rows with these keys, None if there is none."""
    chosen = frame
    for column, value in keys.items():
        chosen = chosen[chosen[column] == value]
    return None if chosen.empty else float(chosen.value.iloc[0])


def features_section(features: pd.DataFrame) -> list[str]:
    """report.md's lines on the feature table: each split's sample."""
    rows = []
    for split, part in features.groupby("split", sort=False):
        mules = part.is_mule.eq(1)
        weight = part.weight.to_numpy(np.float64)
        prevalence = float(weight[mules.to_numpy()].sum() / weight.sum())
        rows.append(
            [
                str(split),
                str(part.date.iloc[0]),
                number(len(part)),
                number(int(mules.sum())),
                number(int((mules & part.revealed.astype(bool)).sum())),
                number(int(part.rejected.astype(bool).sum())),
                number(prevalence),
            ]
        )
    header = ["Split", "Cutoff", "Accounts", "Mules", "Revealed", "Rejected", "Prevalence"]
    return [
        "## Feature table",
        "",
        "Each split's sample is its audit sample: every mule and a uniform sample of the "
        "other accounts, each weighted to the split's population. Prevalence is weighted.",
        "",
        *table(header, rows),
        "",
    ]


def baselines_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the baselines: AP and ROC AUC on validation and test."""
    rows = []
    keys = [*baseline_rows(frame), ("chance (a random ranking)", "chance", "", "")]
    for label, baseline, features, model in keys:
        mine = frame[
            (frame.baseline == baseline) & (frame.features == features) & (frame.model == model)
        ]
        cells = [label]
        for split in HELD_OUT_SPLITS:
            for metric in ("average_precision", "roc_auc"):
                found = mine[(mine.split == split) & (mine.metric == metric)]
                if found.empty:
                    cells.append("")
                    continue
                row = found.iloc[0]
                bounds = None if pd.isna(row.low) else [float(row.low), float(row.high)]
                cells.append(estimate(float(row.value), bounds))
        rows.append(cells)
    header = ["Ranking", "Validation AP", "Validation ROC AUC", "Test AP", "Test ROC AUC"]
    return [
        "## Baselines",
        "",
        "How well does a table of the account's own activity rank mules, with no neighbour, "
        "association or pool input (the question of the retired no_graph control)? The PU "
        "baselines are fitted at the train cutoff on the revealed train mules, against every "
        "other sampled train account weighted to the population; `account` reads only the "
        "account's own history, which no model reads, `model` the root's own model inputs, "
        "`messages` its candidate pool, and `all` everything. In parentheses: the "
        f"ring-clustered {INTERVAL:.0%} interval.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"baselines": DIAGNOSTICS_FIGURES["baselines"]}),
    ]


def curve_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the label-count curve: the mean AP and ROC AUC of each count."""
    rows = []
    learners = [m for m in ("lr", "hgb", "model") if (frame.model == m).any()]
    for model in learners:
        mine = frame[frame.model == model]
        for labels, mules in sorted({*zip(mine["labels"], mine.mules.astype(int), strict=True)}):
            part = mine[(mine["labels"] == labels) & (mine.mules == mules)]
            cells = [model, str(labels), number(mules), number(int(part.repeat.nunique()))]
            for split in HELD_OUT_SPLITS:
                for metric in ("average_precision", "roc_auc"):
                    values = part[(part.split == split) & (part.metric == metric)].value
                    cells.append(number(float(values.mean())) if len(values) else "")
            rows.append(cells)
    header = ["Learner", "Labels", "Train mules", "Draws", "Validation AP", "Validation ROC AUC"]
    header += ["Test AP", "Test ROC AUC"]
    return [
        "## Label-count curve",
        "",
        "A learner fitted at the train cutoff on k train mules with oracle labels, drawn at "
        "random, against every sampled train non-mule (the mean of the draws), beside the "
        "same learner on the revealed train mules alone and the run's audit.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"label_curve": DIAGNOSTICS_FIGURES["label_curve"]}),
    ]


def univariate_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on each feature alone: the strongest on validation."""
    auc = _metric(frame, "roc_auc").pivot_table(index="feature", columns="split", values="value")
    ap = _metric(frame, "average_precision")
    rows = []
    for name in strongest_features(frame, REPORTED_ROWS):
        cells = [f"`{name}`"]
        cells += [
            number(_optional(auc.loc[name].get(split))) for split in ("train", *HELD_OUT_SPLITS)
        ]
        cells.append(number(_value(ap, feature=name, split="validation")))
        rows.append(cells)
    header = ["Feature", "Train ROC AUC", "Validation ROC AUC", "Test ROC AUC", "Validation AP"]
    return [
        "## Each feature alone",
        "",
        f"The {REPORTED_ROWS} features whose validation ROC AUC is farthest from 0.5. The AP "
        "ranks by the value in the direction the train split gives it.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"univariate_auc": DIAGNOSTICS_FIGURES["univariate_auc"]}),
    ]


def drift_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on drift: the largest shifts, and what the shift costs."""
    shifts = frame[frame.feature != ""]
    rows = []
    for name in strongest_shifts(frame, REPORTED_ROWS):
        mine = shifts[shifts.feature == name]
        rows.append(
            [
                f"`{name}`",
                number(_value(mine, split="validation", metric="smd")),
                number(_value(mine, split="test", metric="smd")),
                number(_value(mine, split="test", metric="shift_auc")),
                number(_value(mine, split="test", metric="above_train_q90")),
            ]
        )
    cost = frame[frame.feature == ""]
    cost_rows = []
    for (model, setup), part in cost.groupby(["model", "setup"], sort=False):
        cells = [str(model), str(setup)]
        for split in HELD_OUT_SPLITS:
            for metric in ("average_precision", "roc_auc"):
                cells.append(number(_value(part, split=split, metric=metric)))
        cost_rows.append(cells)
    return [
        "## Drift",
        "",
        "Each split is read at its own cutoff, so its accounts have seen more history. The "
        "standardised mean difference of a held-out split's non-mules from train's, the ROC "
        "AUC of telling the two apart by the value (0.5 means no shift) and the share above "
        "train's 90th percentile (10% without shift):",
        "",
        *table(
            ["Feature", "Validation SMD", "Test SMD", "Test shift AUC", "Test above train q90"],
            rows,
        ),
        "",
        "What the shift costs a learner fitted with every train mule labelled: at the train "
        "cutoff on raw values, on each split's own percentiles, and fitted inside each "
        "held-out split by cross-validation, which no shift touches:",
        "",
        *table(
            ["Learner", "Setup", "Validation AP", "Validation ROC AUC", "Test AP", "Test ROC AUC"],
            cost_rows,
        ),
        "",
        *figure_links(home, {"drift": DIAGNOSTICS_FIGURES["drift"]}),
    ]


def subgroups_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the run's revealed and hidden mules and its rings."""
    budgets = [budget_name(f) for f in REVIEW_BUDGETS]
    rows = []
    for split in HELD_OUT_SPLITS:
        mine = frame[frame.split == split]
        for subset in ("revealed", "hidden"):
            part = mine[mine.subset == subset]
            if part.empty:
                continue
            rows.append(
                [
                    split,
                    subset,
                    number(_int(_value(part, metric="mules"))),
                    number(_value(part, metric="roc_auc")),
                    " / ".join(number(_int(_value(part, metric=f"in_top_{b}"))) for b in budgets),
                    number(_int(_value(part, metric="median_population_rank"))),
                ]
            )
    ring_rows = []
    for split in HELD_OUT_SPLITS:
        rings = frame[(frame.split == split) & (frame.subset == "rings")]
        if rings.empty:
            continue
        ring_rows.append(
            [
                split,
                number(_int(_value(rings, metric="rings"))),
                " / ".join(number(_value(rings, metric=f"coverage_at_{b}")) for b in budgets),
            ]
        )
    shares = " / ".join(share_label(f) for f in REVIEW_BUDGETS)
    lines = [
        "## Revealed and hidden mules, AP concentration and rings",
        "",
        "From the run's audit samples. A ranking that finds the revealed mules and not the "
        "hidden ones measures the reveal, not mule detection. The population rank is the "
        "estimated number of non-mules scoring at least as high.",
        "",
        *table(
            [
                "Split",
                "Mules",
                "Number",
                "ROC AUC against the non-mules",
                f"In the top {shares}",
                "Median population rank",
            ],
            rows,
        ),
        "",
    ]
    if ring_rows:
        lines += [
            *table(["Split", "Rings", f"Rings with a member in the top {shares}"], ring_rows),
            "",
        ]
    figures = {name: DIAGNOSTICS_FIGURES[name] for name in ANALYSIS_FIGURES["subgroups"]}
    return [*lines, *figure_links(home, figures)]


def proxy_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the proxy predictions against the ground truth."""
    rows = []
    for split in HELD_OUT_SPLITS:
        for subset in ("all", "hidden", "revealed"):
            part = frame[(frame.split == split) & (frame.subset == subset)]
            if part.empty:
                continue
            rows.append(
                [
                    split,
                    subset,
                    number(_int(_value(part, metric="n"))),
                    number(_int(_value(part, metric="positives"))),
                    number(_value(part, metric="average_precision")),
                    number(_value(part, metric="roc_auc")),
                ]
            )
    return [
        "## Proxy validity",
        "",
        "The run's proxy predictions (its observed positives and a sample of unlabelled "
        "accounts) scored against the ground truth, unweighted: all of them, the hidden "
        "mules against the non-mules, and the revealed mules against them.",
        "",
        *table(["Split", "Subset", "Accounts", "Mules", "AP", "ROC AUC"], rows),
        "",
        *figure_links(home, {"proxy_validity": DIAGNOSTICS_FIGURES["proxy_validity"]}),
    ]


def reveal_section(home: Reported, frame: pd.DataFrame, record: Mapping[str, Any]) -> list[str]:
    """report.md's lines on the reveal over salts: each split's spread."""
    reveal = record.get("reveal", {})
    salt = reveal.get("salt")
    rows = []
    for split in ("train", *HELD_OUT_SPLITS):
        mine = frame[frame.split == split]
        if mine.empty:
            continue
        cells = [split, number(_int(_value(mine, metric="mules")))]
        for metric in ("eligible", "revealed"):
            values = mine[mine.metric == metric].value.to_numpy(np.float64)
            low, middle, high = (f"{q:g}" for q in np.percentile(values, [5, 50, 95]))
            configured = _value(mine, metric=metric, salt=salt) if salt is not None else None
            cells += [f"{middle} ({low} to {high})", number(_int(configured))]
        rows.append(cells)
    header = ["Split", "Mules", "Discovered by the cutoff", f"Salt {salt}", "Revealed"]
    header.append(f"Salt {salt}")
    return [
        "## The label reveal over salts",
        "",
        f"The reveal's mirror replayed for {frame.salt.nunique():,} salts: the median over "
        "salts and, in parentheses, its 5th to 95th percentile, beside the configured salt's "
        f"outcome. The budget is {reveal.get('budget', 'n/a')} mules per split.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"reveal_spread": DIAGNOSTICS_FIGURES["reveal_spread"]}),
    ]


def nnpu_section(home: Reported, frame: pd.DataFrame) -> list[str]:
    """report.md's lines on the nnPU simulation: each weight's mean over seeds."""
    wide = frame.pivot_table(index=["positive_weight", "seed"], columns="metric", values="value")
    rows = []
    for weight in sorted({float(w) for w in wide.index.get_level_values("positive_weight")}):
        part = wide.xs(weight, level="positive_weight")
        rows.append(
            [
                f"{weight:g}",
                number(len(part)),
                number(float(part.test_roc_auc.mean())),
                number(float(part.test_average_precision.mean())),
                number(int(part.collapsed.sum())),
                number(float(part.labelled_positive_mean_score.mean())),
            ]
        )
    header = ["Positive weight", "Seeds", "Test ROC AUC", "Test AP", "Collapsed"]
    header.append("Labelled positives' mean score")
    return [
        "## The nnPU positive weight, simulated",
        "",
        "A synthetic problem of the dataset's proportions trained with the repository's "
        "loss (offline; the means over seeds). A run collapsed when its labelled positives "
        "score below 0.05 on average.",
        "",
        *table(header, rows),
        "",
        *figure_links(home, {"nnpu_simulation": DIAGNOSTICS_FIGURES["nnpu_simulation"]}),
    ]


def _int(value: float | None) -> int | None:
    """A count a long table holds as a float, rounded to the account."""
    return None if value is None else round(value)


def study_text(study: DiagnosticsPaths, files: StudyFiles) -> str:
    """A study's report.md: what ran, then each analysis' tables with its figures."""
    record, tables = files.record, files.tables
    dataset = str(record.get("dataset_id", study.root.name))
    lines = [f"# Diagnostics of dataset `{dataset[:12]}`", ""]
    run = record.get("run")
    if run is not None:
        compared = record.get("run_compared")
        lines += [
            f"Compared with the run {run}."
            if compared
            else f"No run is compared: {run} was not trained on this dataset, or is missing.",
            "",
        ]
    lines += [
        "Ground truth chooses the samples and labels the rows, for analysis only; nothing "
        "here feeds a model. Decisions use the validation split; test is for reporting, not "
        "selection. The pool groups (pool_activity and pool_internal_inflows) were designed "
        "after reading test-split mules and the data generator's mule typology, so test "
        "results that depend on them are optimistic.",
        "",
    ]
    outcomes = record.get("analyses", {})
    if outcomes:
        rows = [
            [
                name,
                str(outcome.get("status")),
                str(outcome.get("finished", "")),
                str(outcome.get("reason", "")),
            ]
            for name, outcome in outcomes.items()
        ]
        lines += [*table(["Analysis", "Last outcome", "Finished (UTC)", "Reason"], rows), ""]
    if files.features is not None:
        lines += features_section(files.features)
    sections: list[tuple[str, Callable[[pd.DataFrame], list[str]]]] = [
        ("baselines", partial(baselines_section, study)),
        ("learning_curve", partial(curve_section, study)),
        ("univariate", partial(univariate_section, study)),
        ("drift", partial(drift_section, study)),
        ("subgroups", partial(subgroups_section, study)),
        ("proxy_validity", partial(proxy_section, study)),
        ("reveal_spread", lambda frame: reveal_section(study, frame, record)),
        ("nnpu_simulation", partial(nnpu_section, study)),
    ]
    for name, section in sections:
        if name in tables:
            lines += section(tables[name])
    return "\n".join(lines).rstrip("\n") + "\n"


def write_study_text(study: DiagnosticsPaths, files: StudyFiles) -> Path:
    """Replace the study's report.md with study_text."""
    with atomic_write(study.report) as pending:
        pending.write_text(study_text(study, files))
    return study.report


def report_directory(directory: Path) -> dict[str, Any]:
    """`mule report`: redraw a suite's or a study's report if directory holds one, else the run's."""
    suite = SuitePaths(directory)
    if suite.summary.exists():
        return write_suite_report(suite)
    study = DiagnosticsPaths(directory)
    if study.study.exists():
        return write_diagnostics_report(study)
    return report_run(RunPaths(directory))
