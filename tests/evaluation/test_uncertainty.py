"""The ring bootstrap and the ranking statistics it resamples: whole clusters drawn within their
class, the same draws for the same seed, and weighted ROC AUC and average precision with ties
equal to scikit-learn's."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mule_pattern_learner.evaluation.uncertainty import (
    Clusters,
    Ranked,
    cluster_keys,
    interval,
    intervals,
    paired_interval,
    percentiles,
    replicate_matrix,
    replicate_weights,
    spread,
)


def test_cluster_keys_are_rings_ringless_mules_and_accounts() -> None:
    keys = cluster_keys(["a", "b", "c", "d"], np.array([1, 1, 1, 0]), np.array([7, 7, -1, -1]))
    assert keys == ["ring:7", "ring:7", "mule:c", "account:d"]


def test_a_cluster_of_both_classes_is_refused() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        Clusters.of(["ring:7", "ring:7"], np.array([True, False]))


def test_a_replicate_draws_whole_clusters_within_their_class() -> None:
    # Three positive clusters and four negative ones: every replicate draws three of the
    # first and four of the second.
    keys = [
        "ring:1",
        "ring:1",
        "mule:x",
        "ring:2",
        "account:a",
        "account:b",
        "account:c",
        "account:d",
    ]
    positive = np.array([True] * 4 + [False] * 4)
    clusters = Clusters.of(keys, positive)
    chunks = list(replicate_weights(clusters, 70, seed=3))
    assert [len(chunk) for chunk in chunks] == [32, 32, 6]
    draws = np.concatenate(chunks)
    assert (draws[:, clusters.positive].sum(axis=1) == 3).all()
    assert (draws[:, ~clusters.positive].sum(axis=1) == 4).all()
    # Both rows of ring 1 are always drawn together.
    weights = draws[:, clusters.row]
    assert (weights[:, 0] == weights[:, 1]).all()
    again = np.concatenate(list(replicate_weights(clusters, 70, seed=3)))
    assert (again == draws).all()


def test_ranked_statistics_equal_scikit_learns_with_weights_and_ties() -> None:
    rng = np.random.default_rng(5)
    score = rng.integers(0, 12, 400).astype(np.float64)
    positive = rng.random(400) < 0.2
    weight = rng.random(400) * 3
    ranked = Ranked(score, positive)
    assert ranked.roc_auc(weight) == pytest.approx(
        roc_auc_score(positive, score, sample_weight=weight)
    )
    assert ranked.average_precision(weight) == pytest.approx(
        average_precision_score(positive, score, sample_weight=weight)
    )


def test_a_hand_worked_ranking() -> None:
    # A mule of weight 1 at 0.9, a non-mule of weight 2 and a mule of weight 4 tied at 0.5, a
    # non-mule of weight 8 at 0.1: of the 50 weighted pairs 46 rank the mule higher once the
    # tie of weight 8 counts half, so the ROC AUC is 0.92; AP is 1 x 0.2 + 5/7 x 0.8 = 27/35.
    ranked = Ranked(np.array([0.9, 0.5, 0.5, 0.1]), np.array([True, False, True, False]))
    weight = np.array([1.0, 2.0, 4.0, 8.0])
    assert ranked.roc_auc(weight) == pytest.approx(0.92)
    assert ranked.average_precision(weight) == pytest.approx(27 / 35)
    assert np.isnan(Ranked(np.zeros(2), np.zeros(2, dtype=bool)).roc_auc(np.ones(2)))


def test_intervals_cover_the_statistic_and_pair_runs() -> None:
    rng = np.random.default_rng(1)
    n = 600
    positive = rng.random(n) < 0.25
    keys = [f"ring:{i % 40}" if p else f"account:{i}" for i, p in enumerate(positive)]
    clusters = Clusters.of(keys, positive)
    good = Ranked(positive + rng.normal(0, 0.8, n), positive)
    worse = Ranked(positive + rng.normal(0, 2.0, n), positive)
    weight = np.ones(n)
    found = interval(good.roc_auc, clusters, weight, replicates=200, level=0.9, seed=0)
    assert found is not None and found[0] < good.roc_auc(weight) < found[1]
    assert intervals([good.roc_auc], clusters, weight, replicates=200, level=0.9, seed=0) == [found]
    difference = paired_interval(
        good.roc_auc, worse.roc_auc, clusters, weight, replicates=200, level=0.9, seed=0
    )
    assert difference is not None and difference[0] > 0
    # A run against itself differs by nothing on every replicate.
    assert paired_interval(
        good.roc_auc, good.roc_auc, clusters, weight, replicates=50, level=0.9, seed=0
    ) == (0.0, 0.0)


def test_a_replicate_matrix_pairs_its_statistics() -> None:
    positive = np.array([True, True, False, False, False])
    clusters = Clusters.of(["ring:1", "mule:b", "account:c", "account:d", "account:e"], positive)
    weight = np.ones(5)

    def totals(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([weights[positive].sum(), weights[~positive].sum()])

    def doubled(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * totals(weights)

    found = replicate_matrix([totals, doubled], clusters, weight, replicates=40, seed=2)
    assert found.shape == (40, 2, 2)
    # Two positive clusters of one row each, three negative: every replicate draws as many.
    assert (found[:, 0, 0] == 2).all() and (found[:, 0, 1] == 3).all()
    assert (found[:, 1] == 2 * found[:, 0]).all()


def test_percentiles_and_spread() -> None:
    values = np.stack([np.arange(101, dtype=np.float64), np.full(101, np.nan)], axis=1)
    low, high = percentiles(values, 0.9)
    assert low[0] == pytest.approx(5) and high[0] == pytest.approx(95)
    assert np.isnan(low[1]) and np.isnan(high[1])
    assert spread([1.0, 2.0, 3.0, float("nan")]) == {
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert spread([4.0])["std"] == 0 and np.isnan(spread([])["mean"])
