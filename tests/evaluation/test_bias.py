"""Identification bias (docs/design.md, "Identification bias"): identified against hidden mules
by victim origin from their own B2 rows, recall of hidden mules by it, how well the features
separate the two, and the correlation of the selection score with recall of every mule over
the kept epochs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import read_answer_key
from mule_pattern_learner.evaluation.bias import (
    NO_NEW_PAYER,
    hidden_recall_by_origin,
    identification_separation,
    origin_bin,
    origin_table,
    rescore_epochs,
    selection_correlation,
    victim_origin,
)
from mule_pattern_learner.evaluation.feature_audit import COLUMNS
from mule_pattern_learner.evaluation.rankings import evaluation_frame
from mule_pattern_learner.model.hetero import HeteroModel
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.subgraph.layouts import LAYOUT
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import T
from mule_pattern_learner.training import contract
from mule_pattern_learner.training.batches import ScoringBatches
from mule_pattern_learner.training.trainer import gnn_lock

MULES_AT_T = ["a_collector", "a_drop", "a_hr1", "a_hr2", "a_hr3", "a_mule", "a_mule2", "a_mule3"]
ORDINARY = [f"a_o{i:02d}" for i in range(12)]


@pytest.fixture
def world(graph: ReferenceGraph, parity: Path, tmp_path: Path) -> tuple[Cache, pd.DataFrame]:
    """The cache of T's mules and ordinary accounts, and their frame, every logit 0."""
    accounts = [*MULES_AT_T, *ORDINARY]
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: accounts}, parity=parity)
    scores = pd.DataFrame({"t_ms": [T] * len(accounts), "account_id": accounts, "logit": 0.0})
    keys = scores.loc[:, ["t_ms", "account_id"]].assign(weight=1.0)
    identified = frozenset({"a_mule", "a_drop"})
    return cache, evaluation_frame(scores, keys, read_answer_key(Spy(graph)), identified, (T,))


def test_the_bins_of_the_internal_share() -> None:
    assert origin_bin(float("nan")) == NO_NEW_PAYER
    assert [origin_bin(share) for share in (0.0, 0.25, 0.6, 0.75, 1.0)] == [
        "[0, 0.25)",
        "[0.25, 0.5)",
        "[0.5, 0.75)",
        "[0.75, 1]",
        "[0.75, 1]",
    ]


def test_victim_origin_from_each_mules_own_rows(world: tuple[Cache, pd.DataFrame]) -> None:
    cache, frame = world
    origin = victim_origin(cache, frame)
    rows = origin.set_index("account_id").to_dict("index")
    # a_mule's new payers over 30 days: a_vic1 of this bank, x_vic2 and x_vic3 of others (its
    # payroll first paid 75 days before T is no new payer). a_drop's one new payer is
    # another bank's.
    assert (rows["a_mule"]["new_internal_payers"], rows["a_mule"]["new_external_payers"]) == (1, 2)
    assert rows["a_mule"]["internal_share"] == pytest.approx(1 / 3)
    assert rows["a_drop"]["internal_share"] == 0 and rows["a_drop"]["identified"]
    # a_hr1 received nothing: no share. a_hr2 was paid by a_hr1, of this bank.
    assert np.isnan(rows["a_hr1"]["internal_share"])
    assert rows["a_hr2"]["internal_share"] == 1
    table = origin_table(origin).set_index("class").to_dict("index")
    assert table["identified"]["mules"] == 2
    assert table["identified"]["mean_internal_share"] == pytest.approx(1 / 6)


def test_hidden_recall_by_victim_origin() -> None:
    origin = pd.DataFrame(
        {
            "account_id": ["h1", "h2", "h3", "i1"],
            "identified": [False, False, False, True],
            "internal_share": [0.1, 0.2, np.nan, 0.9],
        }
    )
    mules = pd.DataFrame(
        {
            "account_id": ["h1", "h2", "h3", "i1"],
            "identified": [False, False, False, True],
            "ever_at_1pct": [1.0, 0.0, 0.5, 1.0],
            "ever_at_5pct": [1.0, 1.0, 0.5, 1.0],
        }
    )
    table = hidden_recall_by_origin(origin, mules).set_index("bin").to_dict("index")
    assert table["[0, 0.25)"]["mules"] == 2 and table["[0, 0.25)"]["ever_at_1pct"] == 0.5
    assert table[NO_NEW_PAYER]["ever_at_5pct"] == 0.5 and "[0.75, 1]" not in table


def test_the_separation_of_identified_from_hidden_mules(
    world: tuple[Cache, pd.DataFrame],
) -> None:
    cache, frame = world
    table = identification_separation(cache, frame, "w")
    assert table["column"].tolist() == list(COLUMNS)
    found = table["roc_auc"].to_numpy(dtype=np.float64)
    assert np.isfinite(found).all() and ((found >= 0) & (found <= 1)).all()


def test_the_correlation_of_the_selection_score_with_recall() -> None:
    epochs = pd.DataFrame(
        {
            "prior": [1.0, 1.0, 1.0, 3.0],
            "epoch": [1, 2, 3, 1],
            "selection_score": [0.1, 0.2, 0.3, 0.2],
            "recall_at_1pct": [0.1, 0.3, 0.5, 0.2],
            "recall_at_5pct": [0.3, 0.5, 0.7, 0.4],
        }
    )
    table = selection_correlation(epochs).set_index("prior").to_dict("index")
    assert table[1.0]["correlation"] == pytest.approx(1.0) and table[1.0]["epochs"] == 3
    assert np.isnan(table[3.0]["correlation"])


def test_kept_epochs_are_rescored_on_the_selection_world(
    world: tuple[Cache, pd.DataFrame], graph: ReferenceGraph, tmp_path: Path
) -> None:
    cache, frame = world
    config = DEFAULT_CONFIG
    lock = gnn_lock(None, config.model, config.subgraph, config.training)
    torch.manual_seed(0)
    model = HeteroModel(LAYOUT, config.model)
    path = tmp_path / "epochs" / "gnn_seed0_prior1_epoch2.pt"
    contract.save(
        path,
        lock,
        {
            "ranker": "gnn",
            "seed": 0,
            "prior": [1.0, 0.01],
            "epoch": 2,
            "score": 0.25,
            "model_config": asdict(config.model),
            "state": model.state_dict(),
            "found": {},
        },
    )
    keys = frame.loc[:, ["t_ms", "account_id", "weight"]]
    batches = ScoringBatches(cache, "w", keys, 8)
    table = rescore_epochs(
        [path], lock, batches, keys, read_answer_key(Spy(graph)), frozenset({"a_mule"}), (T,)
    )
    row = table.iloc[0]
    assert (row["prior"], row["epoch"], row["selection_score"]) == (1.0, 2, 0.25)
    assert 0 <= row["recall_at_1pct"] <= row["recall_at_5pct"] <= 1
