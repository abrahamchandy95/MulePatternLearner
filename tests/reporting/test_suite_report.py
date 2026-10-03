"""A suite's report: the seed-mean curves its figures draw (its figures: test_comparison.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from mule_pattern_learner.reporting.suite_report import mean_epochs


def test_a_seed_mean_curve_ends_at_the_last_epoch_every_seed_trained() -> None:
    def epochs(count: int, ap: float) -> pd.DataFrame:
        return pd.DataFrame({"epoch": range(1, count + 1), "validation_ap": [ap] * count})

    # Early stopping ended seed 43 after three epochs: the mean of seed 42 alone after it
    # would read as a mean over both.
    curve = mean_epochs("variant", {42: epochs(5, 0.2), 43: epochs(3, 0.4)})
    assert curve.x.tolist() == [1, 2, 3] and curve.y == pytest.approx([0.3, 0.3, 0.3])
    assert curve.seeds == 2
