"""The two XGBoost comparisons (docs/design.md, "Comparisons"): each label's weights rescaled to
equal totals, XGBoost's defaults seeded by the model seed, the round count chosen by the
selection score with a larger count only when strictly better, the same model from the same
seed, and the ranker's file under its lock."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest
import xgboost

from mule_pattern_learner.comparisons import rankers
from mule_pattern_learner.comparisons.features import (
    GRAPH_COLUMNS,
    OWN_COLUMNS,
    Vectors,
    vectors_of,
)
from mule_pattern_learner.comparisons.rankers import (
    GRAPH,
    MAX_ROUNDS,
    OWN,
    ROUND_STEP,
    choose,
    equal_totals,
    fit,
    load,
    lock,
    save,
    scores,
    train,
)
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


def test_each_labels_weights_total_half_the_examples() -> None:
    labels = np.array([1, 0, 0, 0])
    np.testing.assert_allclose(
        equal_totals(labels, np.array([2.0, 1.0, 1.0, 2.0])), [2, 0.5, 0.5, 1]
    )
    with pytest.raises(ValueError, match="no label 1"):
        equal_totals(np.zeros(3, dtype=np.int64), np.ones(3))


def test_xgboost_keeps_its_defaults_and_takes_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, Any]] = []
    real = xgboost.XGBClassifier

    def recording(**keywords: Any) -> xgboost.XGBClassifier:
        made.append(keywords)
        return real(**keywords)

    monkeypatch.setattr(rankers.xgboost, "XGBClassifier", recording)
    vectors, _ = _planted(50, 0)
    booster = fit(vectors.own, vectors.labels, vectors.weights, 3, OWN_COLUMNS)
    assert made == [{"n_estimators": MAX_ROUNDS, "random_state": 3}]
    assert booster.num_boosted_rounds() == MAX_ROUNDS == 1_000 and ROUND_STEP == 10
    assert booster.feature_names == list(OWN_COLUMNS)


def test_a_score_is_the_sum_of_the_trees_up_to_its_round(tmp_path: Path) -> None:
    """XGBoost's margin up to a round, less its base score, which every account shares."""
    vectors, _ = _planted(100, 1)
    booster = fit(vectors.own, vectors.labels, vectors.weights, 0, OWN_COLUMNS)
    frame = pd.DataFrame(vectors.own, columns=list(OWN_COLUMNS))
    margin = booster.predict(xgboost.DMatrix(frame), output_margin=True, iteration_range=(0, 30))
    found = scores(booster, vectors.own, OWN_COLUMNS, 30)
    assert found.dtype == np.float64
    offset = margin.astype(np.float64) - found
    np.testing.assert_allclose(offset, offset[0], atol=1e-5)
    with pytest.raises(ValueError, match="multiple of 10"):
        scores(booster, vectors.own, OWN_COLUMNS, 25)


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


def test_the_round_count_is_the_first_with_the_best_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scores of the counts tried rise to their best at 40 rounds and tie it at 60: 40 is
    kept, since a larger count must be strictly better."""
    keys = pd.DataFrame(
        {
            "t_ms": np.full(100, M1, dtype=np.int64),
            "account_id": [f"a{index:03d}" for index in range(100)],
            "positive": [index == 0 for index in range(100)],
            "ranked": True,
            "weight": 1.0,
        }
    )
    # The positive's place at each count: 50th, 10th, 3rd, 1st, 2nd, 1st, then last.
    places = [50, 10, 3, 1, 2, 1] + [100] * 94

    def planted(*args: Any) -> Iterator[tuple[int, NDArray[np.float64]]]:
        for step, place in enumerate(places):
            total = -np.arange(100, dtype=np.float64)
            total[0] = -(place - 1) + 0.5
            yield (step + 1) * ROUND_STEP, total

    monkeypatch.setattr(rankers, "margins", planted)
    rounds, found, history = choose(xgboost.Booster(), np.zeros((100, 1)), ("x",), keys)
    assert rounds == 40 and found.score == pytest.approx(1.0)
    assert [count for count, _ in history] == list(range(10, 1_001, 10))
    assert history[5] == (60, pytest.approx(1.0))


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
    assert bytes(first.booster.save_raw("ubj")) == bytes(second.booster.save_raw("ubj"))
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


def test_xgboost_refuses_more_than_one_openmp_thread_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On macOS XGBoost crashes beside torch's OpenMP runtime with more than one thread, so it
    refuses to run there unless OMP_NUM_THREADS is 1; elsewhere it runs."""
    vectors, _ = _planted(50, 11)
    monkeypatch.setattr(rankers.sys, "platform", "darwin")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=1"):
        fit(vectors.own, vectors.labels, vectors.weights, 0, OWN_COLUMNS)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    rankers.require_one_openmp_thread()
    monkeypatch.setattr(rankers.sys, "platform", "linux")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    rankers.require_one_openmp_thread()
