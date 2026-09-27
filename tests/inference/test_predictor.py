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
from mule_pattern_learner.testing.builders import checkpoint, example_config, neighbourhood
from mule_pattern_learner.testing.fake_graph import FakeExecutor
from mule_pattern_learner.tigergraph.context_query import TigerGraphContextFetcher


def test_embeddings_leave_the_logits_unchanged_and_rejected_roots_are_listed(
    tmp_path: Path,
) -> None:
    config = example_config()
    path = checkpoint(tmp_path / "model.pt", config)
    executor = FakeExecutor(factory=neighbourhood, statuses={"A0002": "missing_entity"})
    predictor = Predictor(path, fetcher=TigerGraphContextFetcher(executor), device="cpu")
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


def test_accepted_scores_leave_rejected_roots_empty_and_refuse_non_finite_ones() -> None:
    scores, mask = accepted_scores(
        [torch.tensor([0.0]), torch.tensor([2.0])],
        [np.array([True, False]), np.array([True])],
        "test",
    )
    assert mask.tolist() == [True, False, True] and math.isnan(scores[1])
    assert scores[0] == 0.5 and scores[2] == pytest.approx(1 / (1 + math.exp(-2)))
    with pytest.raises(ValueError, match="for 1 accepted validation roots"):
        accepted_scores([torch.tensor([float("nan"), 0.0])], [np.ones(2, bool)], "validation")
    empty, none = accepted_scores([], [], "test")
    assert len(empty) == len(none) == 0
