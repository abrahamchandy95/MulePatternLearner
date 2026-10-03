"""What every report shares: a figure's drawing and saving, and report.md's tables.

Every figure is a matplotlib.figure.Figure drawn in the reporting style (style.RC) and
saved through the Agg canvas, so pyplot is never imported, as a PNG at style.DPI
(save_figure). draw saves a report's figures, then its report.md: a figure that fails to
draw does not stop the others or report.md, and leaves no older drawing under its name;
the error is raised after them, naming every figure that failed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd

from ..artifacts import atomic_write
from .style import DPI, PANEL, RC

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


class Reported(Protocol):
    """A directory with figures and a report.md: a run's, a suite's or a study's paths."""

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


def rows_size(count: int, per_row: float = 0.36, width: float = 7.0) -> tuple[float, float]:
    """A figure of one row per item (a variant, a baseline): taller with more rows."""
    return (width, 2.0 + per_row * count)


def value_or_none(value: Any) -> float | None:
    """A table's value as a float, None where it is missing (NaN)."""
    return None if pd.isna(value) else float(value)
