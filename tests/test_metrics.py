"""Pure metrics: curves, review budgets, thresholded metrics and bootstrap intervals.

The small cases are worked by hand in their comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.metrics import (
    bootstrap_intervals,
    capture_curve,
    paired_replicates,
    percentile_interval,
    precision_recall_curve,
    proxy_metrics,
    ranking_metrics,
    resamples,
    ring_clusters,
    roc_curve,
    weighted_metrics,
)

# Four accounts: a mule of weight 1 at 0.9, a non-mule of weight 2 and a mule of weight 4
# tied at 0.5, and a non-mule of weight 8 at 0.1. The population is 15, with 5 mules.
Y = np.array([1, 0, 1, 0])
SCORE = np.array([0.9, 0.5, 0.5, 0.1])
WEIGHT = np.array([1.0, 2.0, 4.0, 8.0])
TOP_KEYS = [f"{kind}_at_{pct}pct" for pct in (1, 5, 10) for kind in ("precision", "recall")]


def test_the_curves_have_one_point_per_block_of_tied_scores() -> None:
    # At or above 0.9: 1 account, 1 mule; at or above 0.5: 7 and 5; at 0.1: 15 and 5.
    reviewed, found = capture_curve(Y, SCORE, WEIGHT)
    assert reviewed.tolist() == [0.0, 1.0, 7.0, 15.0]
    assert found.tolist() == [0.0, 1.0, 5.0, 5.0]
    recall, precision, thresholds = precision_recall_curve(Y, SCORE, WEIGHT)
    assert recall.tolist() == pytest.approx([1 / 5, 1, 1])
    assert precision.tolist() == pytest.approx([1, 5 / 7, 5 / 15])
    assert thresholds.tolist() == [0.9, 0.5, 0.1]
    # False positives above each block: 0, 2 and 10 of the 10 non-mule accounts.
    fpr, tpr, cuts = roc_curve(Y, SCORE, WEIGHT)
    assert fpr.tolist() == pytest.approx([0, 0, 0.2, 1])
    assert tpr.tolist() == pytest.approx([0, 0.2, 1, 1])
    assert cuts.tolist() == [np.inf, 0.9, 0.5, 0.1]
    # AP is precision times the gain in recall: 0.2 * 1 + 0.8 * 5/7 = 27/35. The ROC area
    # is 0.2 * (0.2 + 1) / 2 + 0.8 * 1 = 0.92: of the 50 weighted mule and non-mule
    # pairs, 46 rank the mule higher once the tie of weight 8 counts half.
    metrics = ranking_metrics(Y, SCORE, WEIGHT)
    assert metrics["average_precision"] == pytest.approx(27 / 35)
    assert float(np.sum(np.diff(np.r_[0, recall]) * precision)) == pytest.approx(27 / 35)
    assert metrics["roc_auc"] == pytest.approx(0.92)
    assert float(np.trapezoid(tpr, fpr)) == pytest.approx(0.92)
    # Budgets of 0.15, 0.75 and 1.5 accounts. The first two lie inside the first account,
    # a mule; the third takes it and half an account of the tied block, whose 6 accounts
    # hold 4 mules: 1 + 0.5 * 4/6 = 4/3 mules.
    assert {k: metrics[k] for k in TOP_KEYS} == pytest.approx(
        {
            "precision_at_1pct": 1,
            "recall_at_1pct": 0.15 / 5,
            "precision_at_5pct": 1,
            "recall_at_5pct": 0.75 / 5,
            "precision_at_10pct": (4 / 3) / 1.5,
            "recall_at_10pct": (4 / 3) / 5,
        }
    )


def test_the_curves_without_positives_stay_at_zero() -> None:
    recall, precision, _ = precision_recall_curve(np.zeros(4), SCORE, WEIGHT)
    assert recall.tolist() == [0, 0, 0] and precision.tolist() == [0, 0, 0]
    fpr, tpr, _ = roc_curve(np.zeros(4), SCORE, WEIGHT)
    assert tpr.tolist() == [0, 0, 0, 0] and fpr.tolist() == pytest.approx([0, 1 / 15, 7 / 15, 1])


def test_unit_weights_give_the_unweighted_top_fraction_metrics() -> None:
    rng = np.random.default_rng(3)
    # 1, 5 and 10% of 200 accounts are whole numbers: 2, 10 and 20.
    y = (rng.random(200) < 0.1).astype(np.int64)
    score = rng.random(200)
    weighted = weighted_metrics(y, score, np.ones(200), 0.5)
    unweighted = proxy_metrics(y, score, 0.5)
    for key in ("precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct"):
        assert weighted[key] == pytest.approx(unweighted[key])
    for key in ("average_precision", "roc_auc", "precision", "recall", "f1"):
        assert weighted[key] == pytest.approx(unweighted[key])
    # The thresholded metrics are one formula: unit weights give the counts' ratios.
    predicted = score >= 0.5
    tp = int((predicted & (y == 1)).sum())
    assert unweighted["precision"] == tp / predicted.sum() and unweighted["recall"] == tp / y.sum()
    top = np.argsort(-score)[:20]
    assert weighted["precision_at_10pct"] == pytest.approx(y[top].sum() / 20)
    assert weighted["recall_at_10pct"] == pytest.approx(y[top].sum() / y.sum())


def test_the_proxy_budgets_share_tied_scores_whatever_the_row_order() -> None:
    # 20 accounts, 4 mules; the top two tie, one of them a mule. The budgets are 0.2, 1
    # and 2 accounts: the first two lie inside the tied block and take half a mule per
    # account, the third holds the block.
    y = np.array([1, 0, 1, 1, 1] + [0] * 15)
    score = np.array([0.9, 0.9, 0.5, 0.4, 0.3] + [0.1] * 15)
    expected = {
        "precision_at_1pct": 0.1 / 0.2,
        "recall_at_1pct": 0.1 / 4,
        "precision_at_5pct": 0.5 / 1,
        "recall_at_5pct": 0.5 / 4,
        "precision_at_10pct": 1 / 2,
        "recall_at_10pct": 1 / 4,
    }
    for order in (np.arange(20), np.r_[1, 0, 2:20]):
        metrics = proxy_metrics(y[order], score[order], 0.5)
        assert {k: metrics[k] for k in expected} == pytest.approx(expected)


def audit_frame() -> pd.DataFrame:
    """50 population accounts: 4 mules, all sampled, and 46 non-mules behind 5 sampled ones.

    Ranked by score, the population accounts and mules found so far are: 01 (1, 1), then
    the tied block of 03 and 02 (4, 2), 04 (5, 3), 05 (9, 3), 06 (10, 4), 07 (26, 4),
    08 (42, 4), 09 (50, 4). The block holds 3 population accounts, one of them a mule.
    """
    rows = [
        ("01", 1, 1.0, 0.99),
        ("03", 1, 1.0, 0.97),
        ("02", 0, 0.5, 0.97),
        ("04", 1, 1.0, 0.90),
        ("05", 0, 0.25, 0.80),
        ("06", 1, 1.0, 0.70),
        ("07", 0, 0.0625, 0.50),
        ("08", 0, 0.0625, 0.30),
        ("09", 0, 0.125, 0.20),
    ]
    return pd.DataFrame(rows, columns=["account_id", "is_mule", "inclusion_probability", "score"])


# Top 1% is 0.5 accounts: half of 01. Top 5% is 2.5: 01 and half of the tied block, so
# half of its one mule. Top 10% is 5: 01 to 04.
EXPECTED = {
    "precision_at_1pct": 0.5 / 0.5,
    "recall_at_1pct": 0.5 / 4,
    "precision_at_5pct": 1.5 / 2.5,
    "recall_at_5pct": 1.5 / 4,
    "precision_at_10pct": 3 / 5,
    "recall_at_10pct": 3 / 4,
}


def sample_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """weighted_metrics of an audit sample, each account weighted by 1 / its probability."""
    weight = 1 / frame.inclusion_probability.to_numpy(float)
    return weighted_metrics(frame.is_mule.to_numpy(int), frame.score.to_numpy(float), weight, 0.5)


def test_capture_at_budgets_count_population_accounts() -> None:
    frame = audit_frame()
    metrics = sample_metrics(frame)
    assert metrics["estimated_population"] == 50 and metrics["weighted_prevalence"] == 0.08
    assert {k: metrics[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)
    # Neither row order nor account IDs break the tie: either order of 03 and 02 would
    # find 2 or 1 mules in the top 5%, and the block counts their average.
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    renamed = frame.assign(account_id=frame.account_id.replace({"02": "03", "03": "02"}))
    for variant in (shuffled, renamed):
        assert {k: sample_metrics(variant)[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)
    # Untied, the order decides: 03 first holds its mule inside the top 5%.
    untied = frame.assign(score=frame.score.where(frame.account_id != "03", 0.98))
    assert sample_metrics(untied)["recall_at_5pct"] == pytest.approx(2 / 4)


def test_capture_at_budgets_rank_scores_beyond_float32_precision() -> None:
    frame = audit_frame()
    # The same ranking squeezed within 1e-9 of 1, where float32 rounds every score to 1.
    near_one = frame.assign(score=1 - 1e-9 * (1 - frame.score))
    assert (near_one.score.astype(np.float32) == 1).all()
    assert {k: sample_metrics(near_one)[k] for k in TOP_KEYS} == pytest.approx(EXPECTED)


def test_capture_at_budgets_without_positives_are_zero() -> None:
    metrics = sample_metrics(audit_frame().assign(is_mule=0))
    assert all(metrics[k] == 0 for k in TOP_KEYS) and metrics["average_precision"] is None


def test_positives_are_resampled_by_ring_and_negatives_within_their_class() -> None:
    # Rows 0 and 1 are one ring, row 2 a mule without a ring, rows 3 and 4 non-mules.
    y = np.array([1, 1, 1, 0, 0])
    rings = np.array([7, 7, -1, -1, -1])
    assert [c.tolist() for c in ring_clusters(y, rings)] == [[0, 1], [2]]
    assert [c.tolist() for c in ring_clusters(y, None)] == [[0], [1], [2]]
    drawn = list(resamples(y, rings, replicates=200, seed=3))
    for rows in drawn:
        counts = np.bincount(rows, minlength=5)
        # Two clusters drawn with replacement: the ring's rows come together, twice at most.
        assert counts[0] == counts[1] and counts[0] + counts[2] == 2
        assert counts[3] + counts[4] == 2
    # Every combination of two clusters occurs, and the seed fixes the draws.
    assert {int(np.bincount(rows, minlength=5)[0]) for rows in drawn} == {0, 1, 2}
    again = resamples(y, rings, replicates=200, seed=3)
    assert all(a.tolist() == b.tolist() for a, b in zip(drawn, again, strict=True))
    # Stratified: always three positive rows.
    for rows in resamples(y, None, replicates=50, seed=3):
        assert int(y[rows].sum()) == 3 and len(rows) == 5


def test_percentile_intervals_leave_out_undefined_replicates() -> None:
    # The 5th and 95th percentiles of 0 to 100.
    assert percentile_interval(np.arange(101.0)) == pytest.approx([5, 95])
    assert percentile_interval(np.r_[np.arange(101.0), np.nan]) == pytest.approx([5, 95])
    assert percentile_interval(np.arange(101.0), 0.5) == pytest.approx([25, 75])
    assert percentile_interval(np.full(3, np.nan)) is None


def test_bootstrap_intervals_of_a_ranking_every_replicate_agrees_on() -> None:
    # Two mules above 98 non-mules: every replicate keeps 2 mules on top of 100 accounts.
    # The budgets of 1, 5 and 10 accounts find 1, 2 and 2 mules.
    y = np.r_[1, 1, np.zeros(98, int)]
    score = np.r_[0.9, 0.8, np.linspace(0.5, 0.1, 98)]
    intervals = bootstrap_intervals(y, score, np.ones(100), replicates=50)
    expected = {
        "average_precision": 1,
        "roc_auc": 1,
        "precision_at_1pct": 1,
        "recall_at_1pct": 0.5,
        "precision_at_5pct": 0.4,
        "recall_at_5pct": 1,
        "precision_at_10pct": 0.2,
        "recall_at_10pct": 1,
    }
    assert set(intervals) == set(expected)
    for name, value in expected.items():
        assert intervals[name] == pytest.approx([value, value]), name
    # Without mules AP and ROC AUC have no interval; the budgets find nothing.
    empty = bootstrap_intervals(np.zeros(100, int), score, np.ones(100), replicates=20)
    assert empty["average_precision"] is None and empty["roc_auc"] is None
    assert empty["recall_at_5pct"] == [0.0, 0.0]


def test_ring_clustered_intervals_are_wider_when_rings_rank_together() -> None:
    # Five rings of four mules: rings 0 and 1 rank above every non-mule, rings 2 to 4
    # below them, so the top 10% (20 accounts) holds exactly the 8 mules of rings 0 and 1.
    rings = np.r_[np.repeat(np.arange(5), 4), -np.ones(180, int)]
    y = (rings >= 0).astype(int)
    score = np.r_[np.full(8, 0.9), np.full(12, 0.1), np.full(180, 0.5)]
    weight = np.ones(200)
    assert ranking_metrics(y, score, weight)["recall_at_10pct"] == pytest.approx(0.4)
    clustered = bootstrap_intervals(y, score, weight, rings, replicates=400)
    stratified = bootstrap_intervals(y, score, weight, replicates=400)
    low, high = clustered["recall_at_10pct"] or [0.0, 0.0]
    s_low, s_high = stratified["recall_at_10pct"] or [0.0, 0.0]
    assert high - low > s_high - s_low > 0


def test_paired_replicates_apply_one_resample_to_every_run() -> None:
    rng = np.random.default_rng(5)
    rings = np.r_[np.repeat(np.arange(6), 3), -np.ones(60, int)]
    y = (rings >= 0).astype(int)
    weight = np.r_[np.ones(18), np.full(60, 10.0)]
    first = rng.random(78) + 0.4 * y
    second = rng.random(78) + 0.2 * y

    def ap(yy: np.ndarray, score: np.ndarray, ww: np.ndarray) -> float | None:
        return ranking_metrics(yy, score, ww)["average_precision"]

    values = paired_replicates(y, weight, [first, first, second], ap, rings, replicates=30)
    assert values.shape == (30, 3)
    # A run paired with itself differs by exactly zero in every replicate.
    assert (values[:, 0] == values[:, 1]).all()
    # Each column is the statistic of that run on the shared resamples.
    shared = [ap(y[r], second[r], weight[r]) for r in resamples(y, rings, replicates=30)]
    assert values[:, 2].tolist() == shared
