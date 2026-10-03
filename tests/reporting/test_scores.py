"""The score figures of an audit: log10 odds, weighted densities and ranks from the top."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
import numpy as np
import pytest

from mule_pattern_learner.metrics import ranking_metrics, threshold_metrics
from mule_pattern_learner.reporting.ranking import SplitScores
from mule_pattern_learner.reporting.scores import (
    log10_odds,
    plot_revealed_vs_hidden,
    plot_score_distribution,
    plot_threshold_metrics,
    top_shares,
)
from mule_pattern_learner.reporting.style import HIDDEN, MULE, NON_MULE, REVEALED

# As in tests/test_metrics.py: a mule of weight 1 at 0.9, a non-mule of weight 2 and a
# mule of weight 4 tied at 0.5, and a non-mule of weight 8 at 0.1.
Y = np.array([1, 0, 1, 0])
SCORE = np.array([0.9, 0.5, 0.5, 0.1])
WEIGHT = np.array([1.0, 2.0, 4.0, 8.0])


def scores(threshold: float = 0.5, revealed: np.ndarray | None = None) -> SplitScores:
    metrics = {
        **ranking_metrics(Y, SCORE, WEIGHT),
        **threshold_metrics(Y, SCORE, WEIGHT, threshold),
    }
    return SplitScores(Y, SCORE, WEIGHT, metrics, revealed=revealed)


def axes() -> Axes:
    return Figure().add_subplot()


def xs(line: Line2D) -> list[float]:
    return np.asarray(line.get_xdata(), dtype=float).tolist()


def ys(line: Line2D) -> list[float]:
    return np.asarray(line.get_ydata(), dtype=float).tolist()


def bin_mass(mass: np.ndarray, edges: np.ndarray, score: float) -> float:
    """The mass of the histogram bin that holds a score's log10 odds (the last bin is closed)."""
    index = int(np.searchsorted(edges, log10_odds(score), side="right")) - 1
    return float(mass[min(index, len(mass) - 1)])


def test_log10_odds_are_finite_at_scores_of_zero_and_one() -> None:
    assert log10_odds(np.array([0.5, 0.99, 0.01])) == pytest.approx(
        [0, np.log10(99), -np.log10(99)]
    )
    assert log10_odds(np.array([0.0, 1.0])).tolist() == [-16.0, 16.0]


def test_ranks_from_the_top_count_half_of_the_tied_accounts() -> None:
    # Of 15 accounts: none above 0.9 (half its own 1), 1 above the tied 0.5 (plus half of
    # their 6), 7 above 0.1 (plus half of its 8).
    assert top_shares(scores()) == pytest.approx([0.5 / 15, 4 / 15, 4 / 15, 11 / 15])


def test_the_threshold_figure_steps_with_the_blocks_and_marks_the_threshold() -> None:
    ax = axes()
    plot_threshold_metrics(ax, scores(), 0.5)
    precision, recall, f1, threshold = ax.get_lines()
    # Lowest threshold first: 0.1 flags everything, 0.5 seven accounts, 0.9 one.
    assert xs(precision) == pytest.approx(log10_odds(np.array([0.1, 0.5, 0.9])))
    assert ys(precision) == pytest.approx([5 / 15, 5 / 7, 1])
    assert ys(recall) == pytest.approx([1, 1, 0.2])
    assert ys(f1)[1] == pytest.approx(2 * (5 / 7) / (5 / 7 + 1))
    assert xs(threshold) == [0, 0]
    legend = ax.get_legend()
    assert legend is not None
    # The threshold in the axis' units, log10 odds, and as the score it is.
    assert legend.get_texts()[3].get_text() == (
        "selected threshold: log10 odds 0.00 (score 0.5),\nprecision 0.714, recall 1.000"
    )


def test_the_densities_are_weighted_and_normalised_per_class() -> None:
    ax = axes()
    plot_score_distribution(ax, scores(), 0.5)
    outlines = [p for p in ax.patches if isinstance(p, StepPatch) and not p.get_fill()]
    assert [to_hex(p.get_edgecolor()) for p in outlines] == [NON_MULE, MULE]
    # Each class's density integrates to one, and its weights split it: of the non-mules,
    # the one of weight 8 at 0.1 holds 8 of 10; of the mules, the one of weight 4 at 0.5
    # holds 4 of 5.
    for patch, (heavy, light) in zip(outlines, ((0.1, 0.5), (0.5, 0.9)), strict=True):
        density, edges, _ = patch.get_data()
        assert density is not None and edges is not None
        mass = density * np.diff(edges)
        assert float(mass.sum()) == pytest.approx(1)
        assert (bin_mass(mass, edges, heavy), bin_mass(mass, edges, light)) == pytest.approx(
            (0.8, 0.2)
        )
    legend = ax.get_legend()
    assert legend is not None
    texts = [text.get_text() for text in legend.get_texts()]
    assert texts == [
        "non-mules: 2 sampled, standing for 10",
        "mules: 2 sampled, standing for 5",
        "selected threshold: log10 odds 0.00 (score 0.5)",
    ]


def test_revealed_and_hidden_mules_are_plotted_at_their_ranks() -> None:
    with pytest.raises(ValueError, match="revealed flags"):
        plot_revealed_vs_hidden(axes(), scores())
    ax = axes()
    plot_revealed_vs_hidden(ax, scores(revealed=np.array([True, False, False, False])))
    points = [line for line in ax.get_lines() if line.get_marker() == "o"]
    hidden, revealed = points
    assert hidden.get_color() == HIDDEN and revealed.get_color() == REVEALED
    assert xs(hidden) == pytest.approx([4 / 15])
    assert xs(revealed) == pytest.approx([0.5 / 15])
    assert ax.get_xscale() == "log"
