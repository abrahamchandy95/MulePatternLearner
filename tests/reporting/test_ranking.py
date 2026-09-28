"""The ranking figures agree with the recorded metrics they label."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mule_pattern_learner.metrics import capture_at_budgets, ranking_metrics
from mule_pattern_learner.reporting.ranking import (
    SplitScores,
    plot_capture,
    plot_precision_recall,
    plot_roc,
    share_label,
)
from mule_pattern_learner.reporting.style import SPLIT_COLOURS, number

# Four accounts: a mule of weight 1 at 0.9, a non-mule of weight 2 and a mule of weight 4
# tied at 0.5, and a non-mule of weight 8 at 0.1 (as in tests/test_metrics.py).
Y = np.array([1, 0, 1, 0])
SCORE = np.array([0.9, 0.5, 0.5, 0.1])
WEIGHT = np.array([1.0, 2.0, 4.0, 8.0])


def split_scores(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> SplitScores:
    metrics = ranking_metrics(y, score, weight)
    return SplitScores(y, score, weight, metrics, {"average_precision": [0.1, 0.9]})


def axes() -> Axes:
    return Figure().add_subplot()


def xs(line: Line2D) -> list[float]:
    return np.asarray(line.get_xdata(), dtype=float).tolist()


def ys(line: Line2D) -> list[float]:
    return np.asarray(line.get_ydata(), dtype=float).tolist()


def test_the_area_under_the_drawn_precision_steps_is_the_recorded_ap() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.1).astype(np.int64)
    score = np.round(rng.random(300) * 0.6 + 0.4 * y, 2)
    weight = np.where(y == 1, 1.0, 20.0)
    for scores in (split_scores(Y, SCORE, WEIGHT), split_scores(y, score, weight)):
        ax = axes()
        plot_precision_recall(ax, {"test": scores}, title="t")
        curve = ax.get_lines()[0]
        recall, precision = np.array(xs(curve)), np.array(ys(curve))
        area = float(np.sum(np.diff(recall) * precision[1:]))
        expected = average_precision_score(scores.y, scores.score, sample_weight=scores.weight)
        assert area == pytest.approx(expected)
        assert curve.get_drawstyle() == "steps-pre"
        assert curve.get_color() == SPLIT_COLOURS["test"]
    # The hand case: recall 1/5 at precision 1, then all 5 mules at 5/7.
    ax = axes()
    plot_precision_recall(ax, {"validation": split_scores(Y, SCORE, WEIGHT)}, title="t")
    curve, chance = ax.get_lines()
    assert xs(curve) == pytest.approx([0, 0.2, 1])
    assert ys(curve) == pytest.approx([1, 1, 5 / 7])
    assert ys(chance) == pytest.approx([5 / 15, 5 / 15])
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "ring-clustered 90% intervals"
    ap = average_precision_score(Y, SCORE, sample_weight=WEIGHT)
    assert legend.get_texts()[0].get_text() == f"validation: AP {ap:.3f} (0.100 to 0.900)"


def test_the_roc_figure_ends_at_the_corners_and_labels_the_auc() -> None:
    ax = axes()
    plot_roc(ax, {"test": split_scores(Y, SCORE, WEIGHT)}, title="t")
    curve = ax.get_lines()[0]
    # (0, 0), then 0.9 flags 1 of 5 mules, 0.5 all of them and 2 of 10 others, then all.
    assert xs(curve) == pytest.approx([0, 0, 0.2, 1])
    assert ys(curve) == pytest.approx([0, 0.2, 1, 1])
    auc = roc_auc_score(Y, SCORE, sample_weight=WEIGHT)
    assert np.trapezoid(ys(curve), xs(curve)) == pytest.approx(auc)


def test_the_capture_figure_marks_the_recorded_budgets_on_its_curve() -> None:
    rng = np.random.default_rng(5)
    y = (rng.random(400) < 0.05).astype(np.int64)
    score = rng.random(400) * 0.5 + 0.5 * y
    weight = np.where(y == 1, 1.0, 12.0)
    scores = split_scores(y, score, weight)
    ax = axes()
    plot_capture(ax, {"test": scores}, title="t")
    curve = ax.get_lines()[0]
    budgets = capture_at_budgets(y, score, weight)
    for fraction, name in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")):
        # On the curve, which interpolates between blocks as the budgets do.
        drawn = np.interp(fraction, xs(curve), ys(curve))
        assert drawn == pytest.approx(budgets[f"recall_at_{name}"])
    labels = [text.get_text() for text in ax.texts]
    assert labels == [
        f"{number(budgets[f'recall_at_{n}'])} / {number(budgets[f'precision_at_{n}'])}"
        for n in ("1pct", "5pct", "10pct")
    ]
    assert ax.get_xscale() == "log"
    ticks = [label.get_text() for label in ax.get_xticklabels()]
    assert ticks == ["0.01%", "0.1%", "1%", "5%", "10%", "100%"]


def test_the_capture_labels_move_clear_of_each_other() -> None:
    # Two splits with the same curve, flat from 5% to 10%: every label starts crowded.
    y = np.r_[np.ones(10, np.int64), np.zeros(990, np.int64)]
    score = np.r_[np.linspace(0.9, 0.99, 10), np.linspace(0.0, 0.5, 990)]
    scores = split_scores(y, score, np.ones(1000))
    ax = axes()
    plot_capture(ax, {"validation": scores, "test": scores}, title="t")
    boxes = [text.get_window_extent() for text in ax.texts]
    assert len(boxes) == 6
    assert not any(box.overlaps(other) for i, box in enumerate(boxes) for other in boxes[i + 1 :])


def test_shares_print_as_percentages_to_two_digits() -> None:
    assert [share_label(s) for s in (1e-5, 0.0001, 0.0123, 0.05, 1.0)] == [
        "0.001%",
        "0.01%",
        "1.2%",
        "5%",
        "100%",
    ]
