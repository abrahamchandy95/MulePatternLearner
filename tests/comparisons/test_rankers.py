"""The two XGBoost comparisons (docs/design.md, "Comparisons"): XGBoost's defaults seeded by the
model seed, the round count chosen by the selection score, the same model from the same seed,
and the ranker's file under its lock. Every fit and prediction runs in a child process
(comparisons.process), so none of these tests imports xgboost."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.comparisons import rankers
from mule_pattern_learner.comparisons.features import (
    GRAPH_COLUMNS,
    OWN_COLUMNS,
    Vectors,
    vectors_of,
)
from mule_pattern_learner.comparisons.rankers import (
    GRAPH,
    OWN,
    fit,
    load,
    lock,
    save,
    scores,
    train,
)
from mule_pattern_learner.comparisons.rounds import MAX_ROUNDS, ROUND_STEP
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.training.selection import selection_score

SUBGRAPH = DEFAULT_CONFIG.subgraph
TRAINING = DEFAULT_CONFIG.training
M1, M2 = 1_000, 2_000
EMPTY_LABELS = np.zeros(0, dtype=np.int64)
EMPTY_WEIGHTS = np.zeros(0, dtype=np.float64)


def _planted(n: int, seed: int, *, labelled: bool = True) -> tuple[Vectors, pd.DataFrame]:
    """n accounts on each of two Mondays with random vectors, positive when their column 3
    (in_amount_7d) is above 1.5 and noise does not hide it; the vectors, carrying their labels
    and weights of 1 unless labelled is false, and the selection keys of the same accounts."""
    rng = np.random.default_rng(seed)
    keys = [(t_ms, f"a{index:03d}") for t_ms in (M1, M2) for index in range(n)]
    graph = rng.normal(size=(2 * n, len(GRAPH_COLUMNS))).astype(np.float32)
    labels = (graph[:, 3] + rng.normal(size=2 * n) * 0.3 > 1.5).astype(np.int64)
    frame = pd.DataFrame(
        {
            "t_ms": np.array([t for t, _ in keys], dtype=np.int64),
            "account_id": [account for _, account in keys],
            "positive": labels.astype(bool),
            "ranked": True,
            "weight": 1.0,
        }
    )
    own = graph[:, : len(OWN_COLUMNS)].copy()
    if not labelled:
        return Vectors(tuple(keys), own, graph, EMPTY_LABELS, EMPTY_WEIGHTS), frame
    return Vectors(tuple(keys), own, graph, labels, np.ones(2 * n)), frame


def test_a_constant_feature_changes_nothing() -> None:
    vectors, _ = _planted(100, 2)
    labels, weights = vectors.labels, vectors.weights
    columns = OWN_COLUMNS[:10]
    plain = fit(vectors.own[:, :10], labels, weights, 0, columns)
    padded = np.concatenate([vectors.own[:, :10], np.full((200, 1), 7.0, np.float32)], axis=1)
    widened = fit(padded, labels, weights, 0, (*columns, "constant"))
    np.testing.assert_array_equal(
        scores(plain, vectors.own[:, :10], columns, 100),
        scores(widened, padded, (*columns, "constant"), 100),
    )


def test_the_rankers_learn_a_planted_cue_and_choose_their_rounds() -> None:
    """Each ranker comes within 10% of the best score a ranking of the selection world can get,
    every positive first: about 15 positives a Monday, of which the top 1% holds 2."""
    training, _ = _planted(300, 3)
    selection, keys = _planted(200, 4, labelled=False)
    perfect = selection_score(keys, keys["positive"].to_numpy(dtype=np.float64))
    assert 0.3 < perfect < 0.5
    for name in (OWN, GRAPH):
        trees = train(name, training, selection, keys, 0)
        assert trees.name == name and trees.seed == 0
        assert trees.rounds % ROUND_STEP == 0 and 10 <= trees.rounds <= MAX_ROUNDS
        assert len(trees.history) == 100
        best = max(score for _, score in trees.history)
        assert trees.score == best
        assert trees.rounds == min(count for count, score in trees.history if score == best)
        assert trees.score > 0.9 * perfect
        np.testing.assert_array_equal(
            trees.scores(selection),
            scores(
                trees.booster, rankers.matrix(name, selection), rankers.COLUMNS[name], trees.rounds
            ),
        )


def test_the_labels_follow_their_rows_whatever_order_the_examples_come_in() -> None:
    """The batches hold their examples shuffled, so rows and labels are cut in one order: the
    shuffled vectors, each row with its own label, learn the planted cue as well as the ordered
    ones, and the same rows with the labels left in the old order learn nothing."""
    training, _ = _planted(300, 3)
    selection, keys = _planted(200, 4, labelled=False)
    perfect = selection_score(keys, keys["positive"].to_numpy(dtype=np.float64))
    order = np.random.default_rng(0).permutation(len(training.keys))
    shuffled = Vectors(
        tuple(training.keys[row] for row in order),
        training.own[order],
        training.graph[order],
        training.labels[order],
        training.weights[order],
    )
    assert list(shuffled.keys) != list(training.keys)
    for name in (OWN, GRAPH):
        assert train(name, shuffled, selection, keys, 0).score > 0.9 * perfect
    misaligned = replace(shuffled, labels=training.labels, weights=training.weights)
    assert train(OWN, misaligned, selection, keys, 0).score < 0.5 * perfect


def test_the_same_seed_gives_the_same_model() -> None:
    training, _ = _planted(150, 5)
    selection, keys = _planted(100, 6, labelled=False)
    first = train(GRAPH, training, selection, keys, 1)
    second = train(GRAPH, training, selection, keys, 1)
    assert first.booster == second.booster
    assert (first.rounds, first.score, first.history) == (
        second.rounds,
        second.score,
        second.history,
    )
    np.testing.assert_array_equal(first.scores(selection), second.scores(selection))


def test_selection_vectors_must_be_those_of_the_keys() -> None:
    training, _ = _planted(50, 7)
    selection, keys = _planted(50, 8, labelled=False)
    with pytest.raises(ValueError, match="not those of the selection keys"):
        train(OWN, training, selection, keys.iloc[::-1], 0)
    with pytest.raises(ValueError, match="No XGBoost ranker"):
        train("xgboost_other", training, selection, keys, 0)
    # Vectors of scoring batches carry no labels to train on.
    with pytest.raises(ValueError, match="carry no labels and weights"):
        train(OWN, selection, selection, keys, 0)


def test_a_rankers_file_reads_back_under_its_lock(tmp_path: Path) -> None:
    training, _ = _planted(100, 9)
    selection, keys = _planted(100, 10, labelled=False)
    trees = train(OWN, training, selection, keys, 2)
    path = tmp_path / "xgboost_own_seed2.pt"
    save(trees, path, lock(OWN, SUBGRAPH, TRAINING))
    back = load(path, lock(OWN, SUBGRAPH, TRAINING))
    assert (back.name, back.seed, back.rounds, back.score, back.history) == (
        OWN,
        2,
        trees.rounds,
        trees.score,
        trees.history,
    )
    assert back.found.accounts == trees.found.accounts
    np.testing.assert_array_equal(back.scores(selection), trees.scores(selection))
    with pytest.raises(ValueError, match=r"settings\.devices_ips \(the model's 8, now 4\)"):
        load(path, lock(OWN, replace(SUBGRAPH, devices_ips=4), TRAINING))
    with pytest.raises(ValueError, match="xgboost_graph.columns"):
        load(path, lock(GRAPH, SUBGRAPH, TRAINING))
    assert lock(OWN, SUBGRAPH, TRAINING)["xgboost_own.columns"] == list(OWN_COLUMNS)


def test_both_rankers_learn_the_batch_worlds_mules(world: Any) -> None:
    """Trained on the batch world's planted mules, each ranker ranks them high on the selection
    world of T: at best (2.24 / 6 + 6 / 6) / 2, about 0.687."""
    training = vectors_of(world.training)
    selection = vectors_of(world.selection.batches)
    for name in (OWN, GRAPH):
        trees = train(name, training, selection, world.selection.keys, 0)
        assert trees.score >= 0.5, name
        assert set(trees.found.accounts) == set(world.mules)
