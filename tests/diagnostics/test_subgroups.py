"""Revealed and hidden mules, the head of AP and the rings, from a run's audit samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from mule_pattern_learner.artifacts import DIAGNOSTIC_TABLES
from mule_pattern_learner.diagnostics.subgroups import (
    ap_contributions,
    population_rank,
    share_above,
    subgroups,
)
from mule_pattern_learner.testing.builders import audit_frame


def test_the_parts_of_ap_add_up_to_the_weighted_ap_with_ties() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.8, 0.5, 0.2, 0.1])
    weight = np.array([1.0, 3.0, 1.0, 10.0, 1.0, 20.0])
    parts = ap_contributions(y, score, weight)
    # 0.9: 1 of 1; the tie at 0.8: 2 of 5; 0.2: 3 of 16; each mule a third of the mules.
    assert parts.tolist() == pytest.approx([1 / 3, 0.4 / 3, (3 / 16) / 3])
    assert parts.sum() == pytest.approx(average_precision_score(y, score, sample_weight=weight))
    # Shares of the population at or above each account, a tie counting whole.
    assert share_above(score, weight).tolist() == pytest.approx(
        [1 / 36, 5 / 36, 5 / 36, 15 / 36, 16 / 36, 1.0]
    )
    assert population_rank(score, weight, y).tolist() == [0.0, 3.0, 13.0]


def test_each_audited_split_has_its_subsets_rings_and_ap_concentration() -> None:
    rng = np.random.default_rng(0)
    audits = {split: audit_frame(rng, split) for split in ("validation", "test")}
    table = subgroups(audits)
    assert tuple(table.columns) == DIAGNOSTIC_TABLES["subgroups"]
    for split, frame in audits.items():
        rows = table[table.split == split]
        mules = frame[frame.is_mule == 1]
        counts = rows[rows.metric == "mules"].set_index("subset").value
        assert counts.to_dict() == {
            "revealed": mules.revealed.sum(),
            "hidden": (~mules.revealed).sum(),
        }
        head = rows[rows.metric == "cumulative_average_precision"]
        assert head["rank"].tolist() == list(range(1, len(mules) + 1))
        assert head.value.is_monotonic_increasing
        weight = 1 / frame.inclusion_probability
        ap = average_precision_score(frame.is_mule, frame.score, sample_weight=weight)
        assert head.value.iloc[-1] == pytest.approx(ap)
        (recorded,) = rows[rows.metric == "average_precision"].value
        assert recorded == pytest.approx(ap)
        rings = rows[rows.subset == "rings"].set_index("metric").value
        assert rings["rings"] == mules.ring_id[mules.ring_id >= 0].nunique()
        coverage = [rings[f"coverage_at_{b}"] for b in ("1pct", "5pct", "10pct")]
        assert coverage == sorted(coverage) and 0 <= coverage[0] and coverage[-1] <= 1
        top = rows[rows.metric.str.startswith("in_top_")].groupby("metric").value.sum()
        assert top["in_top_1pct"] <= top["in_top_5pct"] <= top["in_top_10pct"] <= len(mules)


def test_a_ring_is_covered_when_one_member_is_in_the_top() -> None:
    frame = pd.DataFrame(
        {
            "is_mule": [1, 1, 1, 1, 0, 0],
            "score": [0.99, 0.2, 0.1, 0.05, 0.5, 0.01],
            "inclusion_probability": [1.0, 1.0, 1.0, 1.0, 0.01, 0.01],
            "revealed": [True, False, False, False, False, False],
            "ring_id": [7, 7, 8, -1, -1, -1],
        }
    )
    rows = subgroups({"test": frame})
    rings = rows[rows.subset == "rings"].set_index("metric").value
    # Ring 7 has the top account; ring 8's member sits below the heavy non-mule.
    assert rings["rings"] == 2 and rings["coverage_at_1pct"] == 0.5
    assert rings["coverage_at_10pct"] == 0.5
