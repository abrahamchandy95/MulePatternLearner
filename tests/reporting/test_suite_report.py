"""A suite's report: its seed-mean curves and its proxy reliability (figures: test_comparison)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.artifacts import HIDDEN_METRIC, PROXY_METRIC
from mule_pattern_learner.config import SELECTION_RULES
from mule_pattern_learner.reporting.suite_report import (
    RULE_RUNS,
    mean_epochs,
    proxy_points,
    proxy_reliability,
)


def test_a_seed_mean_curve_ends_at_the_last_epoch_every_seed_trained() -> None:
    def epochs(count: int, ap: float) -> pd.DataFrame:
        return pd.DataFrame({"epoch": range(1, count + 1), "validation_ap": [ap] * count})

    # Early stopping ended seed 43 after three epochs: the mean of seed 42 alone after it
    # would read as a mean over both.
    curve = mean_epochs("variant", {42: epochs(5, 0.2), 43: epochs(3, 0.4)})
    assert curve.x.tolist() == [1, 2, 3] and curve.y == pytest.approx([0.3, 0.3, 0.3])
    assert curve.seeds == 2


def test_the_proxy_reliability_of_a_small_case_worked_by_hand() -> None:
    points = pd.DataFrame(
        {
            "variant": ["baseline", "baseline", "a", "a", "b", "b"],
            "seed": [1, 2, 1, 2, 1, 2],
            PROXY_METRIC: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            "average_precision": [0.6, 0.3, 0.5, 0.2, 0.1, 0.0],
            HIDDEN_METRIC: [0.3, 0.2, 0.25, np.nan, 0.05, 0.02],
        }
    )
    rules = {(v, s): "validation_ap" for v in ("baseline", "a") for s in (1, 2)}
    rules |= {("b", 1): "none", ("b", 2): "none"}
    found = proxy_reliability(points, rules)
    # Over the six runs the ranks differ only where the second and third swap: Spearman
    # is 1 - 6 * 2 / (6 * 35). Over the four runs that select on the AP, 1 - 12 / 60.
    # The two runs of "none" are too few. The variants' seed means rank alike; on the
    # hidden mules five runs have a value, two of them swapped: 1 - 12 / 120.
    assert [(c.label, c.n) for c in found] == [
        ("runs", 6),
        ("runs selected on the proxy AP", 4),
        ("runs keeping the last epoch", 2),
        ("variants, by their seed means", 3),
        ("runs, audit on the hidden mules", 5),
    ]
    assert found[2].value is None
    values = [c.value for c in found if c.value is not None]
    assert values == pytest.approx([1 - 12 / 210, 0.8, 1.0, 0.9])
    assert found[0].text() == "runs: Spearman 0.94 (n = 6)"
    assert found[2].text() == "runs keeping the last epoch: Spearman n/a (n = 2)"
    # Every rule's runs are named in words.
    assert list(RULE_RUNS) == list(SELECTION_RULES)


def test_proxy_points_take_the_complete_runs_and_leave_the_ensembles_out() -> None:
    def row(variant: str, seed: float, metric: str, value: float, status: str) -> dict[str, object]:
        return {
            "variant": variant,
            "seed": seed,
            "split": "validation",
            "metric": metric,
            "value": value,
            "status": status,
            "commit": "",
        }

    summary = pd.DataFrame(
        [
            row("baseline", 42, PROXY_METRIC, 0.5, "complete"),
            row("baseline", 42, "average_precision", 0.3, "complete"),
            row("baseline", 42, HIDDEN_METRIC, 0.2, "complete"),
            row("a", 42, PROXY_METRIC, 0.4, "complete"),
            row("a", 42, "average_precision", 0.1, "complete"),
            row("a", 43, PROXY_METRIC, 0.4, "failed"),
            row("baseline", np.nan, "average_precision", 0.6, "ensemble"),
        ]
    )
    points = proxy_points(summary)
    assert points.variant.tolist() == ["baseline", "a"]
    assert points[PROXY_METRIC].tolist() == [0.5, 0.4]
    assert points.average_precision.tolist() == [0.3, 0.1]
    assert points[HIDDEN_METRIC].iloc[0] == 0.2 and np.isnan(points[HIDDEN_METRIC].iloc[1])
