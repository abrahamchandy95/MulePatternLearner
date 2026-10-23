"""Pure metrics: curves, review budgets, thresholded metrics and the Monday rankings.

The small cases are worked by hand in their comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner import metrics
from mule_pattern_learner.metrics import (
    capture_curve,
    hidden_first,
    hidden_name,
    precision_recall_curve,
    proxy_metrics,
    ranking_metrics,
    roc_curve,
    weighted_metrics,
    weighted_quantiles,
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


def test_weighted_quantiles_interpolate_between_the_middles_of_the_weights() -> None:
    values, weight = np.array([3.0, 1.0, 2.0]), np.array([1.0, 1.0, 2.0])
    # Sorted 1, 2, 3 with weights 1, 2, 1: middles at 1/8, 1/2 and 7/8 of the weight.
    found = weighted_quantiles(values, weight, [0.0, 0.125, 0.5, 0.6875, 1.0])
    assert found.tolist() == [1.0, 1.0, 2.0, 2.5, 3.0]
    unit = weighted_quantiles(np.arange(5.0), np.ones(5), [0.5])
    assert unit.tolist() == [2.0]


def test_the_hidden_mules_are_measured_first_without_the_revealed_ones() -> None:
    # The first mule (weight 1, at 0.9) is revealed: without it the mule of weight 4,
    # tied at 0.5 with a non-mule of weight 2, ranks first, ahead of the non-mule of
    # weight 8. Its AP is 4 / 6; of the 10 weighted pairs it beats 8 and ties 2: 0.9.
    revealed = np.array([True, False, False, False])
    found = hidden_first(Y, SCORE, WEIGHT, revealed)
    assert list(found)[:2] == [hidden_name("average_precision"), hidden_name("roc_auc")]
    assert found["hidden_average_precision"] == pytest.approx(4 / 6)
    assert found["hidden_roc_auc"] == pytest.approx(0.9)
    # Every mule's follow, as ranking_metrics gives them.
    every = ranking_metrics(Y, SCORE, WEIGHT)
    assert {name: found[name] for name in every} == every
    # A sample's description and ranking metrics are weighted_metrics' without a threshold.
    sample = metrics.sample_metrics(Y, SCORE, WEIGHT)
    weighted = weighted_metrics(Y, SCORE, WEIGHT, 0.5)
    assert sample == {name: weighted[name] for name in sample}
    assert (sample["estimated_population"], sample["weighted_prevalence"]) == (15.0, 1 / 3)
    assert set(weighted) - set(sample) == {"threshold", "precision", "recall", "f1"}


# The Monday rankings of the test world. Monday 1: a mule of weight 1 at 0.9, a non-mule of
# weight 2 and a mule of weight 4 tied at 0.5, a non-mule of weight 8 at 0.1 (15 in all).
# Monday 2: a non-mule of weight 1 at 3.0 and a mule of weight 1 at 2.0. A score of 5.0 on
# Monday 2 would top Monday 1, but scores are compared within a Monday only.
MONDAY = np.array([1, 1, 1, 1, 2, 2])
LOGIT = np.array([0.9, 0.5, 0.5, 0.1, 3.0, 2.0])
MULE = np.array([True, False, True, False, False, True])
SAMPLED = np.array([1.0, 2.0, 4.0, 8.0, 1.0, 1.0])
BUDGETS_TESTED = (0.1, 0.5)


def test_each_mondays_budgets_find_shares_of_its_blocks() -> None:
    ranking = metrics.MondayRanking(MONDAY, LOGIT)
    shares = ranking.shares(SAMPLED, BUDGETS_TESTED)
    # Monday 1's 10% is 1.5 of 15: the first account whole, then half an account of the tied
    # block of 6, a twelfth of each of its accounts. Its 50% is 7.5: the first account and the
    # block whole, then 0.5 of the 8 below, a sixteenth. Monday 2's 10% is 0.2 of 2, a fifth
    # of its first account, and its 50% that account whole.
    np.testing.assert_allclose(
        shares, [[1, 1], [1 / 12, 1], [1 / 12, 1], [0, 1 / 16], [0.2, 1], [0, 0]]
    )
    found, positives, sizes = ranking.budget_found(shares, MULE, SAMPLED, BUDGETS_TESTED)
    # Monday 1 finds 1 + 4/12 of its mules' 5 at 10% and all 5 at 50%; Monday 2 none of 1.
    np.testing.assert_allclose(found, [[4 / 3, 5], [0, 0]])
    np.testing.assert_allclose(sizes, [[1.5, 7.5], [0.2, 1]])
    assert positives.tolist() == [5, 1]
    assert ranking.mondays.tolist() == [1, 2]
    # Each row's place: the weight above its block and half its block, over its Monday's.
    assert ranking.rank_fraction(SAMPLED).tolist() == pytest.approx(
        [0.5 / 15, 4 / 15, 4 / 15, 11 / 15, 0.25, 0.75]
    )


def test_rows_outside_a_ranking_are_read_against_it() -> None:
    ranking = metrics.MondayRanking(MONDAY, LOGIT)
    shares, fractions = ranking.against(
        SAMPLED, np.array([1, 1, 1, 2]), np.array([0.5, 0.7, 0.05, 5.0]), BUDGETS_TESTED
    )
    # A score tied with Monday 1's block takes its share; one between the first account and
    # the block lies wholly inside 10%; one under everything is never flagged; on Monday 2 a
    # score of 5.0 tops its Monday.
    np.testing.assert_allclose(shares, [[1 / 12, 1], [1, 1], [0, 0], [1, 1]])
    assert fractions.tolist() == pytest.approx([4 / 15, 1 / 15, 1, 0])
    with pytest.raises(ValueError, match="No ranked row"):
        ranking.against(SAMPLED, np.array([3]), np.array([1.0]), BUDGETS_TESTED)


def test_a_flag_on_any_monday_and_the_first_flag() -> None:
    # Two rows of half a chance each are flagged at least once with chance 3/4; a whole share
    # makes a flag certain.
    found = metrics.ever_flagged(
        np.array([0.5, 0.5, 1.0, 0.0]), np.array([0, 0, 1, 1], dtype=np.intp), 2
    )
    assert found.tolist() == pytest.approx([0.75, 1.0])
    # First flagged 7 days after onset with chance 1/2, 14 days after with 1/2 x 1/2: given a
    # flag, (7 x 0.5 + 14 x 0.25) / 0.75 days.
    ever, days = metrics.first_flag(np.array([14.0, 7.0, 21.0]), np.array([0.5, 0.5, 0.0]))
    assert ever == pytest.approx(0.75) and days == pytest.approx(7 / 0.75)
    ever, days = metrics.first_flag(np.array([7.0]), np.array([0.0]))
    assert ever == 0 and np.isnan(days)
