"""The rankings of the test world and what one run found, on hand-made rankings (docs/design.md,
"Rankings" and "Metrics"), and the ring link of an early-warning Monday on the reference world.

Two Mondays, T and T + 1 week, each ranking eight non-mules n1 to n8 at logits 0.1 to 0.8, each
standing for 25 accounts, and the true mules active that Monday at weight 1, 202 in all: the
top 1% is 2.02 and the top 5% 10.1.

- m1, of ring 5 and identified, active on both Mondays (onset the day before T): at 0.95 it tops
  T; at 0.25 on T + 1 week, 151 of weight rank above it.
- m2, of ring 5, onset 4 days after T: T is its early-warning Monday, at 0.9, under m1 alone;
  active on T + 1 week, where at 0.99 it tops the Monday.
- m3, of no ring, active at T (its last payment the day before T), where at 0.05 it is last;
  after its activity on T + 1 week, at 0.85, under m2 alone.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.comparisons import rankers
from mule_pattern_learner.comparisons.features import vectors_of
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.evaluation.answer_key import (
    ACTIVE,
    AFTER,
    EARLY_WARNING,
    NON_MULE,
    AnswerKey,
)
from mule_pattern_learner.evaluation.rankings import (
    HEADLINE,
    Ranking,
    evaluation_frame,
    ring_links,
    run_metrics,
    score_gnn,
    score_trees,
    scored_keys,
)
from mule_pattern_learner.model.hetero import HeteroModel
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.subgraph.layouts import LAYOUT
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth
from mule_pattern_learner.training import contract
from mule_pattern_learner.training.batches import ScoringBatches
from mule_pattern_learner.training.selection import Found
from mule_pattern_learner.training.trainer import gnn_lock, logits

D, W = DAY_MS, 7 * DAY_MS
T1 = T + W
KEY = AnswerKey(
    (
        AccountTruth("m1", 1, 5, T - D, T + 3 * D, 0, "deposit", True),
        AccountTruth("m2", 1, 5, T + 4 * D, T + 5 * D, 0, "deposit", True),
        AccountTruth("m3", 1, -1, T - 9 * D, T - D, 0, "deposit", True),
        *(AccountTruth(f"n{i}", 0, -1, 0, 0, 0, "deposit", True) for i in range(1, 9)),
    )
)
LOGITS = {
    T: {"m1": 0.95, "m2": 0.9, "m3": 0.05},
    T1: {"m1": 0.25, "m2": 0.99, "m3": 0.85},
}


def _frame() -> pd.DataFrame:
    rows = [
        (
            t_ms,
            account,
            LOGITS[t_ms].get(account, int(account[1:]) / 10 if account[0] == "n" else 0),
        )
        for t_ms in (T, T1)
        for account in [*LOGITS[t_ms], *(f"n{i}" for i in range(1, 9))]
    ]
    scores = pd.DataFrame(rows, columns=["t_ms", "account_id", "logit"])
    keys = scores.loc[:, ["t_ms", "account_id"]].assign(
        weight=[25.0 if account[0] == "n" else 1.0 for account in scores["account_id"]]
    )
    return evaluation_frame(scores, keys, KEY, frozenset({"m1"}))


def test_the_evaluation_frame_holds_each_keys_status_and_truth() -> None:
    frame = _frame()
    status = dict(
        zip(zip(frame["t_ms"], frame["account_id"], strict=True), frame["status"], strict=True)
    )
    assert status[(T, "m1")] == ACTIVE and status[(T1, "m1")] == ACTIVE
    assert status[(T, "m2")] == EARLY_WARNING and status[(T1, "m2")] == ACTIVE
    assert status[(T, "m3")] == ACTIVE and status[(T1, "m3")] == AFTER
    assert status[(T, "n4")] == NON_MULE
    assert frame.loc[frame["account_id"] == "m1", "identified"].all()
    assert not frame.loc[frame["account_id"] == "m2", "identified"].any()


def test_an_evaluation_frame_needs_a_logit_and_a_truth_for_every_key() -> None:
    scores = pd.DataFrame({"t_ms": [T], "account_id": ["n1"], "logit": [0.1]})
    keys = pd.DataFrame({"t_ms": [T, T], "account_id": ["n1", "n2"], "weight": [1.0, 1.0]})
    with pytest.raises(ValueError, match="not those of the test keys"):
        evaluation_frame(scores, keys, KEY, frozenset())
    scores = pd.DataFrame({"t_ms": [T], "account_id": ["ghost"], "logit": [0.1]})
    with pytest.raises(ValueError, match="no truth"):
        evaluation_frame(scores, keys.iloc[:1].assign(account_id="ghost"), KEY, frozenset())


def test_an_evaluation_frame_refuses_a_truth_that_shares_a_column_with_its_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = AnswerKey.frame

    def with_weight(key: AnswerKey) -> pd.DataFrame:
        return frame(key).assign(weight=0.0)

    monkeypatch.setattr(AnswerKey, "frame", with_weight)
    with pytest.raises(ValueError, match=r"share \['weight'\] with the truth"):
        _frame()


def test_the_ranking_leaves_out_mules_outside_their_activity() -> None:
    frame = _frame()
    ranking = Ranking.of(frame, (frame["status"] == ACTIVE).to_numpy())
    # m2 at T and m3 at T + 1 week are not ranked: ten rows each Monday.
    assert len(ranking.frame) == 20 and int(ranking.positive.sum()) == 4
    values = dict(zip(HEADLINE, ranking.headline().tolist(), strict=True))
    # m1 at T and m2 at T + 1 week top their Mondays, inside even the 2.02 of the top 1%;
    # m3 at T and m1 at T + 1 week are under the top 5%: 2 of 4 found at each budget.
    assert values["recall_at_1pct"] == 0.5 and values["recall_at_5pct"] == 0.5
    assert values["precision_at_1pct"] == pytest.approx(2 / (2 * 2.02))
    assert values["precision_at_5pct"] == pytest.approx(2 / (2 * 10.1))
    table = ranking.per_monday()
    assert table["recall_at_1pct"].tolist() == [0.5, 0.5] and table["positives"].tolist() == [2, 2]


def test_what_one_run_found() -> None:
    found = run_metrics(_frame(), links=frozenset())
    mules = found.mules.set_index("account_id")
    # m1 is flagged at T, one day after its onset; m2 on T + 1 week, 3 days after its onset;
    # m3 never while active.
    assert mules.loc["m1", "ever_at_5pct"] == 1 and mules.loc["m1", "first_flag_days_at_5pct"] == 1
    assert mules.loc["m2", "first_flag_days_at_1pct"] == 3
    assert mules.loc["m3", "ever_at_5pct"] == 0
    # m1 ranks 0.5 of 202 at T and 151.5 at T + 1 week: its median is their mean.
    assert mules.loc["m1", "median_rank"] == pytest.approx((0.5 + 151.5) / 2 / 202)
    # m2 at 0.9 on its early-warning Monday is under m1 alone, inside both budgets; m3 after
    # its activity at 0.85 is under m2 alone.
    assert (
        mules.loc["m2", "early_warning_at_1pct"] == 1
        and mules.loc["m2", "early_warning_monday"] == T
    )
    assert pd.isna(mules.at["m1", "early_warning_at_1pct"])
    assert mules.loc["m3", "after_at_5pct"] == 1 and mules.loc["m1", "after_at_5pct"] == 0
    values = found.values
    assert values["ever_flagged_at_1pct"] == pytest.approx(2 / 3)
    # m3 is the one mule after its activity, on one Monday, inside both budgets.
    assert values["early_warning_at_5pct"] == 1 and values["after_activity_at_1pct"] == 1
    assert values["after_activity_account_mondays"] == 1
    assert np.isnan(values["early_warning_ring_link_at_1pct"])
    assert values["first_flag_mean_days_at_1pct"] == 2 and values["mules_scored"] == 3
    # The non-mules stand for 25 accounts each, so the keys are weighted: no distinct count.
    assert np.isnan(values["accounts_alerted_at_1pct"])
    assert np.isnan(values["accounts_alerted_at_5pct"])


def test_distinct_accounts_alerted_are_counted_when_every_key_is_scored_whole() -> None:
    # m1 tops both Mondays, each of 9 ranked account-Mondays at weight 1: its budgets are 0.09
    # and 0.45 of an account, so it is flagged on a Monday with chance 0.09 at 1% and 0.45 at 5%,
    # and on at least one of the two with 1 - 0.91 ** 2 and 1 - 0.55 ** 2. No non-mule is
    # within either budget.
    rows = [
        (t_ms, account, 0.95 if account == "m1" else int(account[1:]) / 10)
        for t_ms in (T, T1)
        for account in ("m1", *(f"n{i}" for i in range(1, 9)))
    ]
    scores = pd.DataFrame(rows, columns=["t_ms", "account_id", "logit"])
    keys = scores.loc[:, ["t_ms", "account_id"]].assign(weight=1.0)
    values = run_metrics(evaluation_frame(scores, keys, KEY, frozenset()), frozenset()).values
    assert values["accounts_alerted_at_1pct"] == pytest.approx(1 - 0.91**2)
    assert values["accounts_alerted_at_5pct"] == pytest.approx(1 - 0.55**2)
    # Under the fallback list the non-mules are a sample, each of one in ten and standing for
    # ten, and an account is in the list on a few of its Mondays only: a flag on any of them
    # would be counted over those few, so the count is not reported.
    sampled = keys.assign(
        weight=[1.0 if account == "m1" else 10.0 for account in keys["account_id"]]
    )
    found = run_metrics(evaluation_frame(scores, sampled, KEY, frozenset()), frozenset()).values
    assert np.isnan(found["accounts_alerted_at_1pct"])
    assert np.isnan(found["accounts_alerted_at_5pct"])


def test_a_ring_link_visible_on_the_early_warning_monday(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    # a_late's early-warning Monday is T + 1 week; its customer has been enrolled on d_hidden
    # since 3 days after T, as have ring 11's three active mules for weeks.
    cache = Cache(tmp_path / "cache")
    ring = ["a_hr1", "a_hr2", "a_hr3", "a_late", "a_o00"]
    fetch(Spy(graph), cache, {T1: ring}, parity=parity)
    key = AnswerKey(
        tuple(
            AccountTruth(
                account,
                *(1, 11, T - 10 * D, T + 9 * D)
                if account != "a_late"
                else (1, 11, T + 10 * D, T + 12 * D),
                0,
                "deposit",
                True,
            )
            if account != "a_o00"
            else AccountTruth(account, 0, -1, 0, 0, 0, "deposit", True)
            for account in ring
        )
    )
    scores = pd.DataFrame({"t_ms": [T1] * 5, "account_id": ring, "logit": [0.0] * 5})
    frame = evaluation_frame(
        scores, scores.loc[:, ["t_ms", "account_id"]].assign(weight=1.0), key, frozenset()
    )
    assert scored_keys(frame, (EARLY_WARNING,)) == [(T1, "a_late")]
    assert ring_links(cache, frame, scored_keys(frame, (EARLY_WARNING,))) == {(T1, "a_late")}
    # Without its ring-mates on the Monday, nothing shares its device.
    alone = frame[frame["account_id"].isin(["a_late", "a_o00"])]
    assert ring_links(cache, alone, [(T1, "a_late")]) == frozenset()


def test_model_files_score_the_test_keys(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    accounts = [row.account_id for row in list_population(graph, T)]
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: accounts}, parity=parity)
    keys = pd.DataFrame({"t_ms": [T] * len(accounts), "account_id": accounts})
    batches = ScoringBatches(cache, "w", keys, 16)
    config = DEFAULT_CONFIG
    lock = gnn_lock(None, config.model, config.subgraph, config.training)
    torch.manual_seed(0)
    model = HeteroModel(LAYOUT, config.model)
    path = tmp_path / "gnn_seed0.pt"
    contract.save(
        path,
        lock,
        {
            "ranker": "gnn",
            "seed": 0,
            "prior": [1.0, 0.01],
            "epoch": 1,
            "score": 0.5,
            "model_config": asdict(config.model),
            "state": model.state_dict(),
            "found": {"accounts": [], "weights": torch.zeros(0), "shares": torch.zeros(0, 2)},
            "history": [],
            "priors": [],
        },
    )
    found = score_gnn(path, lock, batches)
    _, expected = logits(model, batches, None, torch.device("cpu"))
    assert found["account_id"].tolist() == sorted(accounts)
    assert np.array_equal(found["logit"].to_numpy(), expected)
    # An XGBoost ranker's file scores its margin, summed up to its chosen rounds.
    vectors = vectors_of(batches)
    labels = np.array([account.startswith("a_hr") for _, account in vectors.keys], dtype=np.int64)
    booster = rankers.fit(vectors.own, labels, np.ones(len(labels)), 0, rankers.OWN_COLUMNS)
    nothing = Found((), np.zeros(0), np.zeros((0, 2)))
    trees = rankers.Trees(rankers.OWN, 0, booster, 20, 0.5, nothing, ())
    trees_lock = rankers.lock(rankers.OWN, config.subgraph, config.training)
    rankers.save(trees, tmp_path / "xgboost_own_seed0.pt", trees_lock)
    scored = score_trees(tmp_path / "xgboost_own_seed0.pt", trees_lock, batches)
    assert np.array_equal(scored["logit"].to_numpy(), trees.scores(vectors))
    assert scored["account_id"].tolist() == [account for _, account in vectors.keys]
