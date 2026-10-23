"""Scoring with a saved model (docs/design.md, "Training and selection"): mule score takes the
seed with the best selection score, refuses a model of another lock, and ranks raw logits in
float64 within each Monday."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.model.hetero import HeteroModel
from mule_pattern_learner.subgraph.layouts import LAYOUT
from mule_pattern_learner.training.batches import ScoringBatches
from mule_pattern_learner.training.priors import Prior
from mule_pattern_learner.training.scoring import best_seed, score
from mule_pattern_learner.training.selection import Found
from mule_pattern_learner.training.trainer import (
    GNN,
    Trained,
    gnn_lock,
    model_file,
    save_trained,
)

MODEL = DEFAULT_CONFIG.model
SUBGRAPH = DEFAULT_CONFIG.subgraph
TRAINING = DEFAULT_CONFIG.training
LOCK = gnn_lock(None, MODEL, SUBGRAPH, TRAINING)


def _trained(seed: int, score_value: float) -> Trained:
    torch.manual_seed(seed)
    model = HeteroModel(LAYOUT, MODEL).eval()
    found = Found(("a",), np.ones(1), np.full((1, 2), score_value))
    return Trained(GNN, seed, MODEL, model, Prior(1.0, 0.01), 1, score_value, found, ())


def _files(folder: Path, scores: dict[int, float]) -> list[Path]:
    paths: list[Path] = []
    for seed, value in scores.items():
        path = model_file(folder, GNN, seed)
        save_trained(_trained(seed, value), path, LOCK)
        paths.append(path)
    return paths


def test_the_best_selection_score_chooses_the_seed(tmp_path: Path) -> None:
    paths = _files(tmp_path, {0: 0.3, 1: 0.5, 2: 0.4, 3: 0.5, 4: 0.1})
    chosen = best_seed(paths, LOCK)
    assert (chosen.seed, chosen.score) == (1, 0.5)
    assert best_seed(paths[2:], LOCK).seed == 3


def test_a_model_of_other_settings_is_refused_with_its_fields(tmp_path: Path) -> None:
    paths = _files(tmp_path, {0: 0.3})
    other = gnn_lock(None, MODEL, replace(SUBGRAPH, community=32), TRAINING)
    with pytest.raises(ValueError, match=r"settings\.community \(the model's 64, now 32\)"):
        best_seed(paths, other)
    with pytest.raises(ValueError, match="at least one model file"):
        best_seed([], LOCK)


def test_scores_are_float64_logits_ranked_within_each_monday(world: Any, tmp_path: Path) -> None:
    saved = best_seed(_files(tmp_path, {0: 0.3}), LOCK)
    keys = world.examples()[["t_ms", "account_id"]]
    keys = keys[keys["account_id"].isin(["a_mule", "a_r1", "a_h1", "a_shop"])]
    batches = ScoringBatches(world.cache, world.identity, keys, 3)
    found = score(saved, batches)
    assert list(found.columns) == ["t_ms", "account_id", "logit", "rank"]
    assert found["logit"].dtype == np.float64 and found["rank"].dtype == np.int64
    assert len(found) == 8
    for _, day in found.groupby("t_ms"):
        expected = day["logit"].rank(method="min", ascending=False).astype(int)
        assert day["rank"].tolist() == expected.tolist()
        assert sorted(day["rank"]) == [1, 2, 3, 4]
    # Logits a float32 sigmoid would tie still rank apart: one batch per Monday, whose four
    # logits are 40 and steps of 1e-5 above it.
    model = saved.model

    class Shifted(torch.nn.Module):
        def forward(self, data: Any) -> torch.Tensor:
            return model(data) * 0 + 40 + torch.arange(len(data.scored_index)) * 1e-5

    shifted = replace(saved, model=cast(HeteroModel, Shifted()))
    found = score(shifted, ScoringBatches(world.cache, world.identity, keys, 4))
    assert torch.sigmoid(torch.tensor([40.0, 40.00001])).unique().numel() == 1
    assert found.groupby("t_ms")["rank"].nunique().tolist() == [4, 4]
