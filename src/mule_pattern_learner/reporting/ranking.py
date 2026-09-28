"""The ranking figures of the proxy predictions and the audits: precision-recall, ROC, capture.

Each plot function draws on the Axes it is given and returns it; none reads or saves a
file (reporting.report does). The curves come from metrics.py, one point per block of
tied scores, so they agree with the recorded AP, ROC AUC and review budgets, which the
labels print from the report itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from matplotlib.axes import Axes
from matplotlib.text import Annotation
from matplotlib.ticker import NullFormatter, PercentFormatter
from matplotlib.transforms import Bbox
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
from .style import MUTED, SPLIT_COLOURS, SURFACE, estimate, number


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


# A review budget's marker: its size and edge width in points.
MARKER_SIZE = 7.0
MARKER_EDGE = 1.2
# A place for a review budget's label: its offset from the marker in points, and its
# alignment there.
Place = tuple[tuple[float, float], Literal["left", "right"], Literal["top", "center", "bottom"]]
# The places a label tries, in order. The label of the highest recall at a budget tries
# above and left of its marker first, the others below and right: there they stay clear
# of the curves, which rise through the markers, and of the other split's marker.
HIGHER: tuple[Place, ...] = (
    ((-8.0, 6.0), "right", "bottom"),
    ((-8.0, 0.0), "right", "center"),
    ((8.0, 6.0), "left", "bottom"),
    ((8.0, 0.0), "left", "center"),
)
LOWER: tuple[Place, ...] = (
    ((8.0, -6.0), "left", "top"),
    ((8.0, 0.0), "left", "center"),
    ((-8.0, -6.0), "right", "top"),
    ((-8.0, 0.0), "right", "center"),
)


def _put(label: Annotation, place: Place) -> Bbox:
    """Move a label to a place by its marker, and return its box in pixels."""
    offset, horizontal, vertical = place
    label.xyann = offset
    label.set_horizontalalignment(horizontal)
    label.set_verticalalignment(vertical)
    return label.get_window_extent()


def _shift(label: Annotation, pixels: float) -> None:
    """Move a label up by pixels (down when negative); it is offset in points."""
    figure = label.get_figure(root=True)
    assert figure is not None
    x, y = label.xyann
    label.xyann = (x, y + pixels * 72 / figure.dpi)


def _clear(
    label: Annotation, placed: list[Bbox], direction: int, span: tuple[float, float]
) -> None:
    """Move a label up (direction 1) or down (-1) until it overlaps none of the boxes placed.

    ``span`` is the bottom and top of the axes in pixels. The label first moves inside
    them, beside its marker if there is no room above or below it, and it turns back
    once if moving on would take it out of them. It then joins the boxes placed.
    """
    bottom, top = span
    box = label.get_window_extent()
    _shift(label, max(bottom - box.y0, 0.0) + min(top - box.y1, 0.0))
    turned = False
    for _ in range(2 * len(placed)):
        box = label.get_window_extent()
        hit = next((other for other in placed if box.overlaps(other)), None)
        if hit is None:
            break
        beyond = (
            hit.y0 - box.height - 1 < bottom if direction < 0 else hit.y1 + box.height + 1 > top
        )
        if beyond and not turned:
            direction, turned = -direction, True
        _shift(label, hit.y1 - box.y0 + 1 if direction > 0 else hit.y0 - box.y1 - 1)
    placed.append(label.get_window_extent())


def _distance(box: Bbox, point: NDArray[np.float64]) -> float:
    """The distance in pixels from a box to a point, 0 when the box holds it."""
    x, y = point
    return float(np.hypot(max(box.x0 - x, 0.0, x - box.x1), max(box.y0 - y, 0.0, y - box.y1)))


def _place(
    label: Annotation,
    places: tuple[Place, ...],
    marker: NDArray[np.float64],
    markers: NDArray[np.float64],
    placed: list[Bbox],
    span: tuple[float, float],
) -> None:
    """Put a label at the first of its places that suits it, and add it to the boxes placed.

    A place suits the label when it stays inside the axes (``span``, their bottom and top
    in pixels), covers none of the boxes placed (the markers and the labels before it)
    and sits nearer its own marker than any marker elsewhere. When none does, the label
    takes its first place and moves clear from there.
    """
    others = [point for point in markers if not np.allclose(point, marker)]
    for place in places:
        box = _put(label, place)
        own = _distance(box, marker)
        if (
            span[0] <= box.y0
            and box.y1 <= span[1]
            and not any(box.overlaps(other) for other in placed)
            and all(_distance(box, point) > own for point in others)
        ):
            placed.append(box)
            return
    _put(label, places[0])
    _clear(label, placed, 1 if places[0][0][1] > 0 else -1, span)


def plot_capture(ax: Axes, splits: Mapping[str, SplitScores], *, title: str) -> Axes:
    """The share of mules found against the share of accounts reviewed, highest scores first.

    The x axis is logarithmic, so the review budgets of 1, 5 and 10% and the accounts
    above them stay apart. The dashed grey line is a random ranking, and each split's
    dotted line its perfect ranking. Each review budget's marker is labelled, in its
    split's colour, with the split's recorded recall and precision there. At each budget
    the split with the higher recall has its label above and left of its marker and the
    other below and right, so neither label sits by the other split's marker. A label
    that would leave the axes, cover a marker or another label, or sit nearer another
    marker than its own tries the other sides of its marker, and moves up or down clear
    if none suits.
    """
    drawn = {split: scores for split, scores in splits.items() if scores.y.any()}
    start = min([1e-4, *(scores.prevalence / 2 for scores in drawn.values())])
    grid = np.geomspace(start, 1.0, 200)
    # Each split's recorded recall and precision at the review budgets.
    recorded: dict[str, list[tuple[float, float]]] = {}
    for split, scores in drawn.items():
        colour = SPLIT_COLOURS[split]
        reviewed, found = capture_curve(scores.y, scores.score, scores.weight)
        # Linear between block ends: a budget inside a block of tied scores takes the same
        # share of each of its accounts.
        ax.plot(reviewed[1:] / reviewed[-1], found[1:] / found[-1], color=colour, label=split)
        # A perfect ranking finds every mule once the split's prevalence is reviewed.
        perfect = np.minimum(grid / scores.prevalence, 1.0)
        ax.plot(
            grid,
            perfect,
            color=colour,
            linestyle=":",
            linewidth=1.0,
            label=f"{split} perfect ranking",
        )
        recorded[split] = [
            (
                float(scores.metrics[f"recall_at_{budget_name(fraction)}"]),
                float(scores.metrics[f"precision_at_{budget_name(fraction)}"]),
            )
            for fraction in REVIEW_BUDGETS
        ]
        ax.plot(
            REVIEW_BUDGETS,
            [recall for recall, _ in recorded[split]],
            marker="o",
            markersize=MARKER_SIZE,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=MARKER_EDGE,
            linestyle="none",
            zorder=3,
        )
    ax.plot(grid, grid, color=MUTED, linestyle="--", linewidth=1.0, label="random ranking")
    ax.plot(
        [], [], marker="o", color=MUTED, linestyle="none", label="review budget: recall / precision"
    )
    top_share_axis(ax, start)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Top share of accounts reviewed, by score (log scale)")
    ax.set_ylabel("Share of mules found (recall)")
    # Below the axes, clear of the curves and their labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    ax.set_title(title)
    higher: list[Annotation] = []
    lower: list[Annotation] = []
    for index, fraction in enumerate(REVIEW_BUDGETS):
        # Highest recall first; a tie keeps the splits' order.
        ranked = sorted(recorded, key=lambda split: -recorded[split][index][0])
        for rank, split in enumerate(ranked):
            recall, precision = recorded[split][index]
            label = ax.annotate(
                f"{number(recall)} / {number(precision)}",
                (fraction, recall),
                xytext=(0.0, 0.0),
                textcoords="offset points",
                color=SPLIT_COLOURS[split],
                fontsize=7.5,
                bbox={
                    "boxstyle": "square,pad=0.1",
                    "facecolor": SURFACE,
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
                zorder=4,
            )
            # It is kept inside the axes, so the figure's layout need not make room for it.
            label.set_in_layout(False)
            (lower if rank else higher).append(label)
    # Place the labels where the figure's layout puts the axes.
    figure = ax.get_figure(root=True)
    assert figure is not None
    figure.draw_without_rendering()
    points = [
        (fraction, recall)
        for values in recorded.values()
        for fraction, (recall, _) in zip(REVIEW_BUDGETS, values, strict=True)
    ]
    markers = ax.transData.transform(np.array(points, dtype=float).reshape(-1, 2))
    radius = (MARKER_SIZE + MARKER_EDGE) / 2 * figure.dpi / 72
    placed = [Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius) for x, y in markers]
    span = (ax.bbox.y0, ax.bbox.y1)
    # The highest recall's labels reach left and are placed from left to right, the
    # others' reach right and are placed from right to left, so each finds the neighbour
    # it reaches towards already placed.
    for label in higher:
        _place(label, HIGHER, ax.transData.transform(label.xy), markers, placed, span)
    for label in lower[::-1]:
        _place(label, LOWER, ax.transData.transform(label.xy), markers, placed, span)
    return ax
