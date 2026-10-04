"""The training figures draw what history.csv, epochs.csv and metrics.json record."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.reporting.style import MEASURES
from mule_pattern_learner.reporting.training import (
    health_rows,
    interval_steps,
    plot_context_counts,
    plot_corrections,
    plot_objective,
    plot_run_health,
    plot_throughput,
    plot_validation_ranking,
    training_position,
)

# Two epochs of 4 steps, logged after steps 2 and 4 of the first and after step 4 of
# the second: 2, 2 and 4 steps, of which 1, 0 and 2 were corrected.
HISTORY = pd.DataFrame(
    {
        "epoch": [1, 1, 2],
        "step": [2, 4, 4],
        "steps": [4, 4, 4],
        "loss": [0.8, 0.6, 0.5],
        "objective": [0.7, 0.6, 0.4],
        "corrected_steps": [1, 0, 2],
        "seconds_per_step": [3.0, 2.9, 3.1],
        "batch_wait_seconds": [0.4, 0.3, 0.5],
        "contexts_requested": [100, 200, 400],
        "contexts_distinct": [90, 170, 300],
        "memory_hits": [5, 20, 60],
        "disk_hits": [0, 30, 100],
    }
)
EPOCHS = pd.DataFrame(
    {
        "epoch": [1, 2, 3],
        "validation_ap": [0.2, 0.5, 0.4],
        "validation_roc_auc": [0.8, 0.9, 0.85],
        "weights": ["averaged"] * 3,
        "selected": [False, True, False],
        "stopped": [False, False, True],
    }
)


def axes() -> Axes:
    return Figure().add_subplot()


def xs(line: Line2D) -> list[float]:
    return np.asarray(line.get_xdata(), dtype=float).tolist()


def ys(line: Line2D) -> list[float]:
    return np.asarray(line.get_ydata(), dtype=float).tolist()


def test_intervals_sit_at_their_position_in_epochs_with_their_steps() -> None:
    assert training_position(HISTORY).tolist() == [0.5, 1.0, 2.0]
    assert interval_steps(HISTORY).tolist() == [2, 2, 4]


def test_the_objective_figure_draws_each_interval_and_its_epoch_mean() -> None:
    ax = axes()
    assert plot_objective(ax, HISTORY) is ax
    lines = [line for line in ax.get_lines() if len(xs(line)) == 3]
    faint_loss, mean_loss, faint_objective, mean_objective = lines
    assert ys(faint_loss) == [0.8, 0.6, 0.5]
    # The rolling mean over as many intervals as an epoch logs (two).
    assert ys(mean_loss) == pytest.approx([0.8, 0.7, 0.55])
    assert ys(mean_objective) == pytest.approx([0.7, 0.65, 0.5])
    assert xs(faint_objective) == [0.5, 1.0, 2.0]
    assert mean_loss.get_color() == MEASURES[0] and mean_objective.get_color() == MEASURES[1]
    # The epoch boundary at 1 is a hairline.
    assert any(xs(line) == [1, 1] for line in ax.get_lines())


def test_the_corrections_figure_shares_each_interval_steps() -> None:
    ax = axes()
    plot_corrections(ax, HISTORY)
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert [bar.get_height() for bar in bars] == pytest.approx([0.5, 0.0, 0.5])
    # Each bar spans its interval: steps 0 to 2 of epoch 1 are 0 to 0.5.
    assert [bar.get_x() for bar in bars] == pytest.approx([0.0, 0.5, 1.0])
    assert [bar.get_width() for bar in bars] == pytest.approx([0.5, 0.5, 1.0])
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == "whole run: 37.5% of 8 steps"


def test_the_validation_figure_marks_the_selected_epoch_and_chance() -> None:
    ax = axes()
    plot_validation_ranking(ax, EPOCHS, 0.0055)
    legend = ax.get_legend()
    assert legend is not None
    texts = [text.get_text() for text in legend.get_texts()]
    assert texts == [
        "average precision",
        "ROC AUC",
        "AP of a random ranking (0.0055)",
        "selected: epoch 2, AP 0.500",
        "early stopping after epoch 3",
    ]
    assert ax.get_title() == "Proxy validation ranking per epoch (averaged weights)"
    # The nnPU risk where epochs.csv has it, the axis tall enough for it.
    ax = axes()
    plot_validation_ranking(ax, EPOCHS.assign(validation_pu_risk=[1.4, 0.9, 1.0]), None)
    legend = ax.get_legend()
    assert legend is not None
    assert "nnPU risk (lower is better)" in [text.get_text() for text in legend.get_texts()]
    assert ax.get_ylim()[1] == pytest.approx(1.4 * 1.02)


def test_the_throughput_panels_draw_seconds_and_context_totals() -> None:
    top, bottom = axes(), axes()
    plot_throughput(top, HISTORY)
    plot_context_counts(bottom, HISTORY)
    seconds = [line for line in top.get_lines() if len(xs(line)) == 3]
    assert [ys(line) for line in seconds] == [[3.0, 2.9, 3.1], [0.4, 0.3, 0.5]]
    counts = [line for line in bottom.get_lines() if len(xs(line)) == 3]
    assert [ys(line) for line in counts] == [
        [100, 200, 400],
        [90, 170, 300],
        [5, 20, 60],
        [0, 30, 100],
    ]


METRICS: dict[str, Any] = {
    "rejected_roots": {
        "train": {"requested": 96, "rejected": 2},
        "validation": {"requested": 24, "rejected": 0},
    },
    "rejections": {},
    "sampler_backend": "torch",
    "sampler_totals": {"roots": 169, "stub_children": 0},
    "database_calls_during_training": 37,
}


def test_the_health_figure_lists_every_count_under_its_heading() -> None:
    assert health_rows(METRICS) == [
        ("Rejected roots", "train", 2, "of 96"),
        ("Rejected roots", "validation", 0, "of 24"),
        ("Rejected context rows", "none", 0, ""),
        ("Sampler totals (torch)", "roots", 169, ""),
        ("Sampler totals (torch)", "stub children", 0, ""),
        ("Database", "REST calls", 37, ""),
    ]
    ax = axes()
    plot_run_health(ax, METRICS)
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == [
        "Rejected roots",
        "train",
        "validation",
        "Rejected context rows",
        "none",
        "Sampler totals (torch)",
        "roots",
        "stub children",
        "Database",
        "REST calls",
    ]
    # A bar for each count above zero, and every count printed.
    assert len(ax.patches) == 3
    printed = [text.get_text() for text in ax.texts]
    assert printed == ["2 of 96", "0 of 24", "0", "169", "0", "37"]
    assert np.isclose(ax.get_xlim()[0], 0.8)
