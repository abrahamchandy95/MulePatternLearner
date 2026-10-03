"""The fixed look of every figure: colours, sizes, resolution and matplotlib settings.

A colour means the same thing in every figure. Mules, non-mules and the baseline run
never change colour, and neither do the audited splits; the measures a figure compares
(a loss and an objective, precision, recall and F1, the context counts) take MEASURES in
order, each with its own line style too, since some of those colours sit close for
colour-blind readers. The
hues are the categorical steps of a palette checked for colour-blind separation; lines
and bars of the lighter ones always carry a legend or a label. A reference line of the
whole figure (a random ranking, a threshold, a run's average) is grey or ink, never a
data colour; one that belongs to a split or a group (its chance or perfect ranking, its
median) takes that one's colour, dashed or dotted and thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.transforms import offset_copy
from matplotlib.typing import RcKeyType

# PNG resolution of every figure.
DPI = 150
# One panel, and two panels stacked on a shared x axis (inches).
PANEL = (7.0, 4.2)
STACKED = (7.0, 6.4)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MULE = "#eb6834"
NON_MULE = "#2a78d6"
# The built-in run in the comparisons of control experiments: ink, so it stands out
# against the variants.
BASELINE = INK
SPLIT_COLOURS = {"validation": "#4a3aa7", "test": "#008300"}
# Hidden mules are the mules the model never saw labelled; revealed ones recede.
HIDDEN = MULE
REVEALED = MUTED
# Aqua, red, yellow and magenta: the palette's steps that no fixed colour above takes.
MEASURES = ("#1baf7a", "#e34948", "#eda100", "#e87ba4")
MEASURE_LINES = ("-", "--", ":", "-.")

# matplotlib settings for matplotlib.rc_context, around a figure's creation and drawing.
RC: dict[RcKeyType, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelsize": 9,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
}


# How far below the axes a legend under them starts, clear of one line of tick labels and
# the x label (points); each further line of the tick labels adds a line's height.
LEGEND_DROP = 34.0
LINE_SPACING = 1.2


def legend_below(
    ax: Axes, handles: Sequence[Artist], labels: Sequence[str], *, ncols: int = 2
) -> None:
    """A legend under the axes, clear of their tick labels and x label, off the data."""
    figure = ax.get_figure(root=True)
    assert figure is not None
    ticks = ax.get_xticklabels()
    lines = max((tick.get_text().count("\n") + 1 for tick in ticks), default=1)
    size = max((tick.get_fontproperties().get_size_in_points() for tick in ticks), default=0.0)
    height = size * LINE_SPACING
    drop = LEGEND_DROP + (lines - 1) * height
    below = offset_copy(ax.transAxes, fig=figure, y=-drop, units="points")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=below,
        ncols=ncols,
    )


def measure(index: int) -> dict[str, Any]:
    """The colour and line style of a figure's index-th measure."""
    return {"color": MEASURES[index], "linestyle": MEASURE_LINES[index]}


def number(value: Any) -> str:
    """A value as the figures and report.md print it.

    Counts get thousands separators and other numbers three decimals, or two
    significant digits below 0.01 (a prevalence of 0.00084); None is n/a.
    """
    if value is None:
        return "n/a"
    if isinstance(value, bool | str):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.2g}"
    return f"{value:.3f}"


def estimate(value: Any, interval: list[float] | None) -> str:
    """A metric with its interval, if it has one: 0.134 (0.081 to 0.212)."""
    if interval is None:
        return number(value)
    low, high = interval
    return f"{number(value)} ({number(low)} to {number(high)})"
