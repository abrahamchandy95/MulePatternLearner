"""The figures of a control-experiment suite: its variants against the baseline.

Each plot function draws on the Axes it is given and returns it; none reads or saves a
file (reporting.report does, from the suite's summary.csv and comparison.csv and its
runs' files). A variant's numbers are VariantSeeds: one value per seed, the seed mean
and, where the suite has one, the interval of the paired bootstrap. The rows come in
the order they are to be drawn, top to bottom, and the baseline is always ink. The
panels of a split take its colour, so a figure of the validation audit is drawn in the
validation colour throughout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from numpy.typing import NDArray

from ..metrics import INTERVAL, REVIEW_BUDGETS, capture_curve
from ..paths import BASELINE_VARIANT
from .ranking import SplitScores, share_label, top_share_axis
from .style import BASELINE, MEASURES, MUTED, SPLIT_COLOURS, SURFACE, legend_below, number

# What each audit is for, as the figures' titles say it.
PURPOSES = {"validation": "for decisions", "test": "for reporting, not selection"}
# The markers of the review budgets, one per budget beside its colour.
BUDGET_MARKERS = ("o", "s", "D")


@dataclass(frozen=True)
class VariantSeeds:
    """One variant's value of a metric: per seed, the mean over seeds and its interval.

    ``consistent`` marks a paired delta whose seeds all agree in sign with an interval
    that excludes zero (experiments.tables.Delta).
    """

    name: str
    seeds: Mapping[int, float] = field(default_factory=dict[int, float])
    mean: float | None = None
    interval: tuple[float, float] | None = None
    consistent: bool = False


def _colour(name: str, split: str) -> str:
    return BASELINE if name == BASELINE_VARIANT else SPLIT_COLOURS[split]


def _rows(ax: Axes, rows: Sequence[VariantSeeds]) -> NDArray[np.float64]:
    """Variant names down the y axis, the first row on top; returns each row's y."""
    y = np.arange(len(rows), 0, -1, dtype=np.float64) - 1
    ax.set_yticks(y, [row.name for row in rows])
    for label in ax.get_yticklabels():
        if label.get_text() == BASELINE_VARIANT:
            label.set_fontweight("bold")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    return y


def _seeds(
    ax: Axes, y: float, row: VariantSeeds, colour: str, offset: float = 0.0, spread: float = 0.12
) -> None:
    """A row's per-seed values as small hollow markers, spread a little so none hide."""
    values = list(row.seeds.values())
    shifts = np.linspace(-spread, spread, len(values)) if len(values) > 1 else np.zeros(1)
    for value, shift in zip(values, shifts, strict=False):
        ax.plot(
            value,
            y + offset + shift,
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=0.9,
            linestyle="none",
            alpha=0.8,
        )


def _estimate(ax: Axes, y: float, row: VariantSeeds, colour: str, *, filled: bool = True) -> None:
    """A row's seed mean as a large marker on its interval's line."""
    if row.interval is not None:
        ax.plot(row.interval, [y, y], color=colour, linewidth=2.2, solid_capstyle="round")
    if row.mean is not None:
        ax.plot(
            row.mean,
            y,
            marker="o",
            markersize=8,
            markerfacecolor=colour if filled else SURFACE,
            markeredgecolor=colour if not filled else SURFACE,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=3,
        )


def _dot_legend(split: str, *, interval: str) -> tuple[list[Line2D], list[str]]:
    colour = SPLIT_COLOURS[split]
    handles = [
        Line2D(
            [], [], marker="o", markersize=4, markerfacecolor="none", color=colour, linestyle="none"
        ),
        Line2D(
            [],
            [],
            marker="o",
            markersize=8,
            color=colour,
            markeredgecolor=SURFACE,
            linestyle="none",
        ),
        Line2D([], [], color=colour, linewidth=2.2),
    ]
    return handles, ["one seed", "mean over seeds", interval]


def plot_comparison(
    ax: Axes, rows: Sequence[VariantSeeds], *, split: str, baseline: float | None
) -> Axes:
    """Each variant's audit AP on one split: its seeds, their mean and the mean's interval.

    ``baseline`` is the baseline's seed-mean AP, drawn as a dashed ink line through the
    panel. The interval is the paired bootstrap's for the seed mean: it covers the audit
    sample's uncertainty, not the spread between seeds, which the seeds show.
    """
    y = _rows(ax, rows)
    for position, row in zip(y, rows, strict=True):
        colour = _colour(row.name, split)
        _seeds(ax, position, row, colour)
        _estimate(ax, position, row, colour)
    handles, labels = _dot_legend(split, interval=f"{INTERVAL:.0%} interval of the mean")
    if baseline is not None:
        ax.axvline(baseline, color=BASELINE, linestyle="--", linewidth=1.0, zorder=1)
        handles.append(Line2D([], [], color=BASELINE, linestyle="--", linewidth=1.0))
        labels.append(f"baseline mean {number(baseline)}")
    ax.set_xlabel(f"{split.capitalize()} audit average precision")
    legend_below(ax, handles, labels)
    ax.set_title(f"{split.capitalize()} audit, {PURPOSES[split]}")
    return ax


def plot_paired_delta(ax: Axes, rows: Sequence[VariantSeeds]) -> Axes:
    """Each variant's validation audit AP minus the baseline's, paired on shared accounts.

    Per seed, the difference from the baseline of the same seed; the large marker is the
    difference of the seed means, on its paired interval. A filled marker is consistent:
    every seed's difference has its sign and the interval excludes zero.
    """
    y = _rows(ax, rows)
    colour = SPLIT_COLOURS["validation"]
    for position, row in zip(y, rows, strict=True):
        _seeds(ax, position, row, colour)
        _estimate(ax, position, row, colour, filled=row.consistent)
    ax.axvline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
    handles, labels = _dot_legend("validation", interval=f"paired {INTERVAL:.0%} interval")
    handles.insert(
        2,
        Line2D(
            [],
            [],
            marker="o",
            markersize=8,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.6,
            linestyle="none",
        ),
    )
    labels[1] = "difference of the means, consistent"
    labels.insert(2, "not consistent")
    handles.append(Line2D([], [], color=BASELINE, linewidth=1.0))
    labels.append("no difference from the baseline")
    ax.set_xlabel("Validation audit AP minus the baseline's (same accounts, same seed)")
    legend_below(ax, handles, labels)
    ax.set_title("Validation audit AP against the baseline, paired")
    return ax


def plot_budget_recall(ax: Axes, rows: Mapping[float, Sequence[VariantSeeds]]) -> Axes:
    """Each variant's validation audit recall in the top 1, 5 and 10% of accounts.

    ``rows`` holds, for each review budget, the variants in drawing order. Each budget
    has its colour and marker; the small markers are the seeds, the large ones their mean.
    """
    budgets = [fraction for fraction in REVIEW_BUDGETS if fraction in rows]
    names = rows[budgets[0]]
    y = _rows(ax, names)
    offsets = np.linspace(0.22, -0.22, len(budgets)) if len(budgets) > 1 else np.zeros(1)
    handles, labels = [], []
    for index, (fraction, offset) in enumerate(zip(budgets, offsets, strict=True)):
        colour, marker = MEASURES[index], BUDGET_MARKERS[index]
        for position, row in zip(y, rows[fraction], strict=True):
            _seeds(ax, position, row, colour, offset, spread=0.06)
            if row.mean is not None:
                ax.plot(
                    row.mean,
                    position + offset,
                    marker=marker,
                    markersize=7,
                    color=colour,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    linestyle="none",
                    zorder=3,
                )
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                markersize=7,
                color=colour,
                markeredgecolor=SURFACE,
                linestyle="none",
            )
        )
        labels.append(f"top {share_label(fraction)}: mean over seeds")
    handles.append(
        Line2D(
            [], [], marker="o", markersize=4, markerfacecolor="none", color=MUTED, linestyle="none"
        )
    )
    labels.append("one seed")
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of the validation split's mules found (recall)")
    legend_below(ax, handles, labels)
    ax.set_title("Validation audit recall at the review budgets")
    return ax


@dataclass(frozen=True)
class MeanCurve:
    """A variant's curve averaged over its seeds: y at each x, and the seeds averaged."""

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    seeds: int


def mean_capture(name: str, runs: Sequence[SplitScores], grid: NDArray[np.float64]) -> MeanCurve:
    """The seed mean of the share of mules found at each top share of accounts in grid.

    Each run's capture curve (metrics.capture_curve) is linear between the ends of its
    blocks of tied scores, as the audit's review budgets are.
    """
    found = []
    for scores in runs:
        reviewed, hits = capture_curve(scores.y, scores.score, scores.weight)
        found.append(np.interp(grid, reviewed / reviewed[-1], hits / max(hits[-1], 1e-12)))
    return MeanCurve(name, grid, np.mean(found, axis=0), len(runs))


def plot_capture_overlay(
    ax: Axes, variant: MeanCurve, baseline: MeanCurve | None, *, split: str = "validation"
) -> Axes:
    """One variant's seed-mean capture curve against the baseline's, on the log top-share axis.

    A suite draws one panel per variant, so every panel holds two lines: the baseline in
    ink and the variant in the split's colour. The dashed grey line is a random ranking.
    """
    ax.plot(variant.x, variant.x, color=MUTED, linestyle="--", linewidth=0.9)
    if baseline is not None and variant.name != BASELINE_VARIANT:
        ax.plot(baseline.x, baseline.y, color=BASELINE, linewidth=1.4)
    ax.plot(variant.x, variant.y, color=_colour(variant.name, split), linewidth=1.8)
    top_share_axis(ax, float(variant.x[0]), budgets=False)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"{variant.name} ({variant.seeds} seeds)", fontsize=9)
    return ax


def plot_validation_overlay(
    ax: Axes,
    variant: MeanCurve,
    baseline: MeanCurve | None,
    *,
    prevalence: float | None,
    last_epoch: int,
) -> Axes:
    """One variant's seed-mean proxy validation AP per epoch against the baseline's.

    An epoch's mean is over the seeds that trained it (early stopping ends seeds at
    different epochs). ``prevalence`` is the AP of a random ranking, drawn dashed grey,
    and ``last_epoch`` the last epoch any run of the suite trained, so the panels share
    their epochs.
    """
    if prevalence is not None:
        ax.axhline(prevalence, color=MUTED, linestyle="--", linewidth=0.9)
    if baseline is not None and variant.name != BASELINE_VARIANT:
        ax.plot(baseline.x, baseline.y, color=BASELINE, linewidth=1.4, marker="o", markersize=2.5)
    colour = _colour(variant.name, "validation")
    ax.plot(variant.x, variant.y, color=colour, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlim(0.5, last_epoch + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.set_ylim(bottom=0)
    ax.set_title(f"{variant.name} ({variant.seeds} seeds)", fontsize=9)
    return ax


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    """Spearman's rank correlation of two arrays (ties share their mean rank)."""
    if len(x) < 3:
        return None

    def ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
        order = np.argsort(values, kind="stable")
        ranked = np.empty(len(values))
        ranked[order] = np.arange(len(values), dtype=np.float64)
        for value in np.unique(values):
            tied = values == value
            ranked[tied] = ranked[tied].mean()
        return ranked

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def plot_proxy_vs_audit(
    ax: Axes, names: Sequence[str], proxy: NDArray[np.float64], audit: NDArray[np.float64]
) -> Axes:
    """Each run's selected proxy AP against its validation audit AP: is the proxy informative?

    One point per run (``names`` holds each run's variant): the proxy AP on validation's
    observed labels at the epoch training selected, and the validation audit AP of the
    same model. The legend gives Spearman's rank correlation over the runs; a proxy that
    ranks the runs as the audit does is one to select on.
    """
    colour = SPLIT_COLOURS["validation"]
    others = np.array([name != BASELINE_VARIANT for name in names], dtype=bool)
    ax.plot(
        proxy[others],
        audit[others],
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor=colour,
        markeredgewidth=1.2,
        linestyle="none",
        label="a variant's run",
    )
    ax.plot(
        proxy[~others],
        audit[~others],
        marker="o",
        markersize=7,
        color=BASELINE,
        markeredgecolor=SURFACE,
        linestyle="none",
        label="a baseline run",
    )
    correlation = rank_correlation(proxy, audit)
    shown = "n/a" if correlation is None else f"{correlation:.2f}"
    ax.plot(
        [], [], linestyle="none", label=f"Spearman rank correlation {shown} over {len(names)} runs"
    )
    ax.set_xlabel("Selected proxy AP on validation's observed labels")
    ax.set_ylabel("Validation audit AP")
    legend_below(ax, *ax.get_legend_handles_labels())
    ax.set_title("Proxy selection against the ground-truth audit, per run")
    return ax
