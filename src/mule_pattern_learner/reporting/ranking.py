"""The ranking figures of the proxy predictions and the audits: precision-recall, ROC, capture.

Each plot function draws on the Axes it is given and returns it; none reads or saves a
file (reporting.report does). The curves come from metrics.py, one point per block of
tied scores, so they agree with the recorded AP, ROC AUC and review budgets, which the
labels print from the report itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, PercentFormatter
import numpy as np
from numpy.typing import NDArray

from ..metrics import (
    INTERVAL,
    REVIEW_BUDGETS,
    budget_name,
    capture_curve,
    precision_recall_curve,
    roc_curve,
)
from .style import MUTED, SECONDARY_INK, SPLIT_COLOURS, SURFACE, estimate, number


@dataclass(frozen=True)
class SplitScores:
    """A split's scored accounts and what its report recorded about them.

    Each account stands for ``weight`` accounts of the split's population: 1 for the
    proxy's observed labels (predictions/<split>.parquet), 1 / its inclusion probability
    for an audit sample. ``metrics`` are the split's recorded metrics (metrics.json's
    observed_label_proxy, or the audit report's), ``intervals`` the audit's intervals and
    ``revealed`` whether the graph revealed each account's label before the cutoff.
    """

    y: NDArray[np.int64]
    score: NDArray[np.float64]
    weight: NDArray[np.float64]
    metrics: Mapping[str, Any]
    intervals: Mapping[str, list[float] | None] = field(default_factory=dict[str, Any])
    revealed: NDArray[np.bool_] | None = None

    @property
    def prevalence(self) -> float:
        """The weighted share of positives: the precision of a random ranking."""
        return float(self.weight[self.y == 1].sum() / self.weight.sum())

    def summary(self, metric: str) -> str:
        """A recorded metric with its interval, if the report has one."""
        return estimate(self.metrics.get(metric), self.intervals.get(metric))


def _interval_title(splits: Mapping[str, SplitScores], metric: str) -> str | None:
    """The legend title that says what the intervals are, when a split has one."""
    if any(scores.intervals.get(metric) for scores in splits.values()):
        return f"ring-clustered {INTERVAL:.0%} intervals"
    return None


def plot_precision_recall(ax: Axes, splits: Mapping[str, SplitScores], *, title: str) -> Axes:
    """Precision against recall per split, with each split's chance line and AP.

    Each step is the precision at a block of tied scores that holds positives, over the
    recall that block adds, so the area under a split's steps is its AP; each split's
    dashed line is its prevalence, the precision of a random ranking. A split without
    positives has no curve.
    """
    for split, scores in splits.items():
        colour = SPLIT_COLOURS[split]
        if scores.y.any():
            recall, precision, _ = precision_recall_curve(scores.y, scores.score, scores.weight)
            # Blocks of negatives add no recall and no area.
            gains = np.flatnonzero(np.diff(recall, prepend=0.0) > 0)
            ax.plot(
                np.concatenate(([0.0], recall[gains])),
                np.concatenate((precision[gains[:1]], precision[gains])),
                drawstyle="steps-pre",
                color=colour,
                label=f"{split}: AP {scores.summary('average_precision')}",
            )
        ax.axhline(
            scores.prevalence,
            color=colour,
            linestyle="--",
            linewidth=0.9,
            label=f"{split} chance: prevalence {number(scores.prevalence)}",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", title=_interval_title(splits, "average_precision"))
    ax.set_title(title)
    return ax


def plot_roc(ax: Axes, splits: Mapping[str, SplitScores], *, title: str) -> Axes:
    """True against false positive rate per split, with each split's ROC AUC.

    Tied scores draw one diagonal segment; the dashed diagonal is a random ranking.
    """
    for split, scores in splits.items():
        if len(np.unique(scores.y)) != 2:
            continue
        fpr, tpr, _ = roc_curve(scores.y, scores.score, scores.weight)
        ax.plot(
            fpr,
            tpr,
            color=SPLIT_COLOURS[split],
            label=f"{split}: ROC AUC {scores.summary('roc_auc')}",
        )
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.0, label="random ranking")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right", title=_interval_title(splits, "roc_auc"))
    ax.set_title(title)
    return ax


def share_label(share: float) -> str:
    """A share of accounts as a percentage to two significant digits: 0.0001 is 0.01%."""
    return f"{float(f'{share * 100:.2g}'):g}%"


def top_share_axis(ax: Axes, start: float) -> None:
    """A log x axis of the top share of accounts, from start to all of them.

    It is labelled at the powers of ten and at the review budgets.
    """
    ax.set_xscale("log")
    ax.set_xlim(start, 1.0)
    decades = 10.0 ** np.arange(np.ceil(np.log10(start)), 1)
    ticks = sorted({*decades.tolist(), *REVIEW_BUDGETS})
    ax.set_xticks(ticks, [share_label(tick) for tick in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_capture(ax: Axes, splits: Mapping[str, SplitScores], *, title: str) -> Axes:
    """The share of mules found against the share of accounts reviewed, highest scores first.

    The x axis is logarithmic, so the review budgets of 1, 5 and 10% and the accounts
    above them stay apart. The dashed line is a random ranking and the dotted one a
    perfect ranking; each review budget's marker is labelled with the split's recorded
    recall and precision there.
    """
    shares = [scores.prevalence for scores in splits.values() if scores.y.any()]
    start = min([1e-4, *(share / 2 for share in shares)])
    grid = np.geomspace(start, 1.0, 200)
    for position, (split, scores) in enumerate(splits.items()):
        if not scores.y.any():
            continue
        colour = SPLIT_COLOURS[split]
        reviewed, found = capture_curve(scores.y, scores.score, scores.weight)
        # Linear between block ends: a budget inside a block of tied scores takes the same
        # share of each of its accounts.
        ax.plot(reviewed[1:] / reviewed[-1], found[1:] / found[-1], color=colour, label=split)
        perfect = np.minimum(grid / scores.prevalence, 1.0)
        ax.plot(grid, perfect, color=MUTED, linestyle=":", linewidth=1.0)
        # The first split's labels sit above and left of its markers, the second's below
        # and right, clear of the curves that rise through them.
        above = position % 2 == 0
        for fraction in REVIEW_BUDGETS:
            name = budget_name(fraction)
            recall = float(scores.metrics[f"recall_at_{name}"])
            precision = float(scores.metrics[f"precision_at_{name}"])
            ax.plot(
                fraction,
                recall,
                marker="o",
                markersize=7,
                color=colour,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                linestyle="none",
                zorder=3,
            )
            ax.annotate(
                f"{number(recall)} / {number(precision)}",
                (fraction, recall),
                xytext=(-8, 6) if above else (8, -6),
                textcoords="offset points",
                ha="right" if above else "left",
                va="bottom" if above else "top",
                color=SECONDARY_INK,
                fontsize=7.5,
                bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
                zorder=4,
            )
    ax.plot(grid, grid, color=MUTED, linestyle="--", linewidth=1.0, label="random ranking")
    ax.plot([], [], color=MUTED, linestyle=":", linewidth=1.0, label="perfect ranking")
    ax.plot(
        [], [], marker="o", color=MUTED, linestyle="none", label="review budget: recall / precision"
    )
    top_share_axis(ax, start)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Top share of accounts reviewed, by score (log scale)")
    ax.set_ylabel("Share of mules found (recall)")
    # Where the curves leave room, which depends on the run.
    ax.legend(loc="best")
    ax.set_title(title)
    return ax
