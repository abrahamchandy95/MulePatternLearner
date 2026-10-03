"""The one scoring loop: logits with and without embeddings, order and rejected roots."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mule_pattern_learner.contract.clock import cutoff_ms
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.inference.predictor import (
    Predictor,
    accepted_scores,
    score_batch,
)
from mule_pattern_learner.pipeline.connect import context_source
from mule_pattern_learner.testing.builders import neighbourhood, saved_model, unit_config
from mule_pattern_learner.testing.fake_graph import FakeTigerGraph


def test_embeddings_leave_the_logits_unchanged_and_rejected_roots_are_listed(
    tmp_path: Path,
) -> None:
    config = unit_config()
    path = saved_model(tmp_path / "model.pt", config)
    executor = FakeTigerGraph(factory=neighbourhood, statuses={"A0002": "missing_entity"})
    predictor = Predictor(path, context_source(executor, config), device="cpu")
    ms = cutoff_ms("2024-07-01")
    keys = [ContextKey("Account", f"A{i:04}", 103, ms, config.scope.id, 3) for i in range(5)]
    try:
        prepared = predictor.prepare(keys)
        plain = score_batch(predictor.model, prepared, predictor.device)
        embedded = score_batch(predictor.model, prepared, predictor.device, embeddings=True)
        assert plain.logits is not None and embedded.logits is not None
        assert torch.equal(plain.logits, embedded.logits) and plain.embeddings is None
        assert embedded.embeddings is not None and embedded.embeddings.shape[0] == 4
        frames, rejected = predictor.score_keys([keys[:3], keys[3:]])
    finally:
        predictor.contexts.close()
    assert rejected == ["A0002"]
    assert [f.account_id.tolist() for f in frames] == [["A0000", "A0001"], ["A0003", "A0004"]]


def test_scoring_uses_the_saved_runtime_and_restores_the_global_torch_state(
    tmp_path: Path,
) -> None:
    config = unit_config(runtime={"device": "cpu", "threads": 1, "deterministic": True})
    path = saved_model(tmp_path / "model.pt", config)
    contexts = context_source(FakeTigerGraph(factory=neighbourhood), config)
    original, threads = torch.get_num_threads(), 3
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False)
    try:
        predictor = Predictor(path, contexts)
        assert predictor.device == torch.device("cpu")
        # Building the predictor changes nothing; scoring runs with the saved settings.
        assert torch.get_num_threads() == threads
        with predictor.runtime():
            assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == threads
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        contexts.close()
        torch.set_num_threads(original)
        torch.use_deterministic_algorithms(False)


def test_accepted_scores_leave_rejected_roots_empty_and_refuse_non_finite_ones() -> None:
    scores, mask = accepted_scores(
        [torch.tensor([0.0]), torch.tensor([2.0])],
        [np.array([True, False]), np.array([True])],
        "test",
    )
    assert mask.tolist() == [True, False, True] and math.isnan(scores[1])
    assert scores[0] == 0.5 and scores[2] == pytest.approx(1 / (1 + math.exp(-2)))
    with pytest.raises(ValueError, match="for 1 accepted validation root$"):
        accepted_scores([torch.tensor([float("nan"), 0.0])], [np.ones(2, bool)], "validation")
    empty, none = accepted_scores([], [], "test")
    assert len(empty) == len(none) == 0
