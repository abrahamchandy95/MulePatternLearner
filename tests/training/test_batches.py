"""The training and scoring batches built from the batch world's cache of conftest.py: built
once, shuffled across Mondays and labels by a fixed seed, every example with its label and
weight; scoring batches in the order of their keys, the same on every pass."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch

from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.training.batches import ScoringBatches, training_batches


def _examples(batches: list[Batch]) -> list[tuple[int, str, int, float]]:
    return [
        (key.t_ms, key.account_id, int(label), float(weight))
        for batch in batches
        for key, label, weight in zip(
            batch.keys, batch.data.label.tolist(), batch.data.weight.tolist(), strict=True
        )
    ]


def test_training_batches_hold_every_example_once_with_its_label_and_weight(
    world: Any, planted: list[Batch]
) -> None:
    frame = world.examples()
    frame.loc[frame["account_id"] == "a_r1", "weight"] = 2.5
    made = training_batches(world.cache, world.identity, frame, 32)
    assert [len(batch.keys) for batch in made] == [32] * 14
    rows = zip(frame["t_ms"], frame["account_id"], frame["label"], frame["weight"], strict=True)
    assert sorted(_examples(made)) == sorted(
        (int(t), str(a), int(label), float(weight)) for t, a, label, weight in rows
    )
    assert {batch.keys[0].world for batch in made} == {world.identity}


def test_training_batches_are_shuffled_once_across_mondays_and_labels(
    world: Any, planted: list[Batch]
) -> None:
    again = training_batches(world.cache, world.identity, world.examples(), 32)
    assert _examples(again) == _examples(planted)
    # The examples come in (Monday, account) order, and the batches mix them.
    mondays = [{key.t_ms for key in batch.keys} for batch in planted]
    assert sum(len(found) == 2 for found in mondays) >= 12
    labels = [set(batch.data.label.tolist()) for batch in planted]
    assert sum(found == {0, 1} for found in labels) >= 4
    # The order does not follow the examples' order.
    assert [key.account_id for key in planted[0].keys] != sorted(
        key.account_id for key in planted[0].keys
    )


def test_training_batches_refuse_bad_examples(world: Any) -> None:
    frame = world.examples().head(40)
    with pytest.raises(ValueError, match="lack the columns"):
        training_batches(world.cache, world.identity, frame.drop(columns="weight"), 8)
    with pytest.raises(ValueError, match="twice"):
        training_batches(world.cache, world.identity, pd.concat([frame, frame.head(1)]), 8)
    bad = frame.copy()
    bad.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label"):
        training_batches(world.cache, world.identity, bad, 8)
    bad = frame.copy()
    bad.loc[0, "weight"] = 0.0
    with pytest.raises(ValueError, match="weight"):
        training_batches(world.cache, world.identity, bad, 8)
    with pytest.raises(ValueError, match="at least one example"):
        training_batches(world.cache, world.identity, frame, 0)
    with pytest.raises(KeyError, match="lacks"):
        unknown = frame.copy()
        unknown.loc[0, "account_id"] = "a_unknown"
        training_batches(world.cache, world.identity, unknown, 8)


def test_scoring_batches_follow_their_keys_on_every_pass(world: Any) -> None:
    keys = world.examples()[["account_id", "t_ms"]].iloc[::-1]
    batches = ScoringBatches(world.cache, world.identity, keys, 100)
    assert len(batches) == 448
    assert list(batches.keys.columns) == ["t_ms", "account_id"]
    first = list(batches)
    second = list(batches)
    assert [len(batch.keys) for batch in first] == [100, 100, 24, 100, 100, 24]
    found = [(key.t_ms, key.account_id) for batch in first for key in batch.keys]
    assert found == sorted(found)
    assert found == list(zip(batches.keys["t_ms"], batches.keys["account_id"], strict=True))
    for left, right in zip(first, second, strict=True):
        assert "label" not in left.data
        assert torch.equal(left.data["account"].x, right.data["account"].x)
    with pytest.raises(ValueError, match="at least one key"):
        ScoringBatches(world.cache, world.identity, keys.head(0), 10)
