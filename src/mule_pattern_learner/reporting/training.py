"""The training figures, drawn from a run's history.csv, epochs.csv and metrics.json.

Each plot function draws on the Axes it is given and returns it; none reads or saves a
file (reporting.run_report does). ``history`` and ``epochs`` are the frames
artifacts.read_history and read_epochs return, and ``metrics`` is metrics.json. The x
axis of the history figures is the training position in epochs: the interval that
ended at step 50 of epoch 2's 100 sits at 1.5, so the whole numbers are the epoch
boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from .style import (
    AXIS,
    INK,
    MEASURES,
    MUTED,
    SECONDARY_INK,
    SURFACE,
    legend_below,
    measure,
    number,
)


def training_position(history: pd.DataFrame) -> NDArray[np.float64]:
    """Where each log interval ended, in epochs (the x axis of the history figures)."""
    return (history.epoch - 1 + history.step / history.steps).to_numpy(np.float64)


def interval_steps(history: pd.DataFrame) -> NDArray[np.int64]:
    """The steps of each log interval: its step less the previous interval's in its epoch."""
    previous = history.groupby("epoch").step.shift(fill_value=0)
    return (history.step - previous).to_numpy(np.int64)


def _epoch_axis(ax: Axes, history: pd.DataFrame) -> None:
    """Epochs on the x axis: whole-number ticks and a hairline at each boundary."""
    end = float(history.epoch.max())
    for boundary in range(1, int(end)):
        ax.axvline(boundary, color=AXIS, linewidth=0.8, zorder=0)
    ax.set_xlim(0, end)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Epoch")


def plot_objective(ax: Axes, history: pd.DataFrame) -> Axes:
    """Loss and unclamped nnPU objective per log interval, with their mean over an epoch.

    The faint lines are the intervals; the solid ones their rolling mean over as many
    intervals as an epoch logs. The loss is the clamped risk training minimised; the two
    differ where the non-negative correction fires (plot_corrections).
    """
    x = training_position(history)
    window = max(int(history.groupby("epoch").size().max()), 1)
    measures = (("loss", "loss (clamped)"), ("objective", "nnPU objective (unclamped)"))
    for index, (column, label) in enumerate(measures):
        values = history[column].astype(float)
        style = measure(index)
        ax.plot(x, values, color=style["color"], linewidth=0.8, alpha=0.35)
        mean = values.rolling(window, min_periods=1).mean()
        ax.plot(x, mean, label=f"{label}, mean over an epoch", **style)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, linewidth=0.8, alpha=0.6))
    labels.append("each log interval")
    ax.legend(handles, labels, loc="upper right")
    _epoch_axis(ax, history)
    ax.set_ylabel("Mean per training step")
    ax.set_title("Training loss and nnPU objective")
    return ax


def plot_corrections(ax: Axes, history: pd.DataFrame) -> Axes:
    """The share of each log interval's steps whose non-negative correction fired.

    One bar per interval, as wide as the interval; the dashed line is the whole run's share.
    """
    end = training_position(history)
    steps = interval_steps(history)
    width = steps / history.steps.to_numpy(np.float64)
    share = history.corrected_steps.to_numpy(np.float64) / np.maximum(steps, 1)
    ax.bar(
        end - width,
        share,
        width=width,
        align="edge",
        color=MEASURES[0],
        edgecolor=SURFACE,
        linewidth=0.6,
    )
    overall = float(history.corrected_steps.sum()) / max(int(steps.sum()), 1)
    ax.axhline(
        overall,
        color=INK,
        linestyle="--",
        linewidth=1.0,
        label=f"whole run: {overall:.1%} of {number(int(steps.sum()))} steps",
    )
    ax.legend(loc="upper left")
    _epoch_axis(ax, history)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Steps corrected in the log interval")
    ax.set_title("Steps whose non-negative correction fired")
    return ax


def plot_validation_ranking(ax: Axes, epochs: pd.DataFrame, prevalence: float | None) -> Axes:
    """Proxy AP, ROC AUC and nnPU risk of validation per epoch, the selected epoch, chance.

    ``prevalence`` is the share of observed positives among validation's scored rows,
    the AP of a random ranking; None leaves that line out. The nnPU risk, lower being
    better, is drawn where epochs.csv has it (an epochs.csv written before it has not).
    The title names the weights validation scored (the moving average or the raw
    weights).
    """
    x = epochs.epoch.to_numpy(np.int64)
    ax.plot(x, epochs.validation_ap, marker="o", label="average precision", **measure(0))
    ax.plot(x, epochs.validation_roc_auc, marker="s", label="ROC AUC", **measure(1))
    risk = epochs.get("validation_pu_risk")
    top = 1.02
    if risk is not None and risk.notna().any():
        ax.plot(x, risk, marker="^", label="nnPU risk (lower is better)", **measure(2))
        top = max(top, float(risk.max()) * 1.02)
    if prevalence is not None:
        ax.axhline(
            prevalence,
            color=MUTED,
            linestyle="--",
            linewidth=1.0,
            label=f"AP of a random ranking ({number(prevalence)})",
        )
    selected = epochs[epochs.selected.astype(bool)]
    if len(selected):
        epoch, ap = int(selected.epoch.iloc[0]), float(selected.validation_ap.iloc[0])
        ax.axvline(epoch, color=INK, linestyle=":", linewidth=1.0, zorder=0)
        ax.plot(
            epoch,
            ap,
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor=INK,
            markeredgewidth=1.4,
            linestyle="none",
            label=f"selected: epoch {epoch}, AP {number(ap)}",
        )
    stopped = epochs[epochs.stopped.astype(bool)]
    if len(stopped):
        ax.plot(
            [], [], linestyle="none", label=f"early stopping after epoch {stopped.epoch.iloc[0]}"
        )
    ax.set_xlim(0.5, float(x.max()) + 0.5 if len(x) else 1.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, top)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Proxy metric on observed labels")
    # Under the axes: the curves cross every part of them.
    legend_below(ax, *ax.get_legend_handles_labels())
    weights = " and ".join(sorted(set(epochs.weights.astype(str))))
    ax.set_title(f"Proxy validation ranking per epoch ({weights} weights)")
    return ax


def plot_throughput(ax: Axes, history: pd.DataFrame) -> Axes:
    """Seconds per training step and the part of it spent waiting for a batch."""
    x = training_position(history)
    ax.plot(x, history.seconds_per_step, label="seconds per step", **measure(0))
    ax.plot(x, history.batch_wait_seconds, label="waiting for the next batch", **measure(1))
    _epoch_axis(ax, history)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Seconds per step")
    ax.legend(loc="center right")
    ax.set_title("Training throughput")
    return ax


def _count(value: float, position: object) -> str:
    """A tick of counts, with thousands separators."""
    return number(int(value))


def plot_context_counts(ax: Axes, history: pd.DataFrame) -> Axes:
    """The run's contexts requested, distinct, and served from memory or disk, over training."""
    x = training_position(history)
    counts = (
        ("contexts_requested", "requested"),
        ("contexts_distinct", "distinct"),
        ("memory_hits", "served from memory"),
        ("disk_hits", "read from the disk cache"),
    )
    for index, (column, label) in enumerate(counts):
        ax.plot(x, history[column], label=label, **measure(index))
    _epoch_axis(ax, history)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(_count))
    ax.set_ylabel("Contexts, run total")
    ax.legend(loc="upper left")
    ax.set_title("Contexts requested, distinct, and served from memory or disk")
    return ax


def health_rows(metrics: Mapping[str, Any]) -> list[tuple[str, str, int, str]]:
    """The counts of plot_run_health: (group, name, count, note), in drawing order."""
    rows: list[tuple[str, str, int, str]] = []
    for split, counts in metrics["rejected_roots"].items():
        rows.append(
            ("Rejected roots", split, counts["rejected"], f"of {number(counts['requested'])}")
        )
    statuses = metrics["rejections"] or {"none": 0}
    rows.extend(("Rejected context rows", status, count, "") for status, count in statuses.items())
    backend = f"Sampler totals ({metrics['sampler_backend']})"
    for name, count in metrics["sampler_totals"].items():
        rows.append((backend, name.replace("_", " "), count, ""))
    rows.append(("Database", "REST calls", metrics["database_calls_during_training"], ""))
    return rows


def plot_run_health(ax: Axes, metrics: Mapping[str, Any]) -> Axes:
    """Rejections by split and status, sampler totals and database calls of a run.

    One bar per count on a log scale, grouped under bold headings, each labelled with its
    value; a zero has no bar.
    """
    rows = health_rows(metrics)
    labels: list[str] = []
    positions: list[int] = []
    heading_rows: list[int] = []
    row = 0
    group = None
    largest = max([count for _, _, count, _ in rows] + [1])
    for name, label, count, note in rows:
        if name != group:
            group = name
            labels.append(name)
            heading_rows.append(row)
            positions.append(row)
            row += 1
        labels.append(label)
        positions.append(row)
        if count > 0:
            ax.barh(row, count - 0.8, left=0.8, height=0.7, color=MEASURES[0])
        text = f"{number(count)} {note}".strip()
        ax.annotate(
            text,
            (max(count, 0.8), row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
            fontsize=8,
        )
        row += 1
    ax.set_xscale("log")
    ax.set_xlim(0.8, largest * 12)
    ax.set_ylim(row - 0.5, -0.5)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0)
    for index, tick in enumerate(ax.get_yticklabels()):
        if positions[index] in heading_rows:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Count over the run (log scale)")
    ax.set_title("Run health: rejections, sampler totals and database calls")
    return ax
