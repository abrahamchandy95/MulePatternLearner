"""The training and scoring batches built from the batch world's cache of conftest.py: built
once, shuffled across Mondays and labels by a fixed seed, every example with its label and
weight; scoring batches in the order of their keys, the same on every pass."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.layouts import ACCOUNT, RELATIONS
from mule_pattern_learner.training import batches as batches_module
from mule_pattern_learner.training.batches import ScoringBatches, training_batches

SEED = DEFAULT_CONFIG.training.shuffle_seed


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
    made = training_batches(world.cache, world.identity, frame, 32, seed=SEED)
    assert [len(batch.keys) for batch in made] == [32] * 14
    rows = zip(frame["t_ms"], frame["account_id"], frame["label"], frame["weight"], strict=True)
    assert sorted(_examples(made)) == sorted(
        (int(t), str(a), int(label), float(weight)) for t, a, label, weight in rows
    )
    assert {batch.keys[0].world for batch in made} == {world.identity}


def test_training_batches_are_shuffled_once_across_mondays_and_labels(
    world: Any, planted: list[Batch]
) -> None:
    again = training_batches(world.cache, world.identity, world.examples(), 32, seed=SEED)
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
        training_batches(world.cache, world.identity, frame.drop(columns="weight"), 8, seed=SEED)
    with pytest.raises(ValueError, match="twice"):
        training_batches(
            world.cache, world.identity, pd.concat([frame, frame.head(1)]), 8, seed=SEED
        )
    bad = frame.copy()
    bad.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label"):
        training_batches(world.cache, world.identity, bad, 8, seed=SEED)
    bad = frame.copy()
    bad.loc[0, "weight"] = 0.0
    with pytest.raises(ValueError, match="weight"):
        training_batches(world.cache, world.identity, bad, 8, seed=SEED)
    with pytest.raises(ValueError, match="at least one example"):
        training_batches(world.cache, world.identity, frame, 0, seed=SEED)
    with pytest.raises(KeyError, match="lacks"):
        unknown = frame.copy()
        unknown.loc[0, "account_id"] = "a_unknown"
        training_batches(world.cache, world.identity, unknown, 8, seed=SEED)


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


def test_the_shuffle_seed_is_the_callers_and_there_is_no_other() -> None:
    assert not hasattr(batches_module, "SHUFFLE_SEED")


def test_another_shuffle_seed_orders_the_same_examples_another_way(world: Any) -> None:
    first = training_batches(world.cache, world.identity, world.examples(), 32, seed=SEED)
    other = training_batches(world.cache, world.identity, world.examples(), 32, seed=SEED + 1)
    assert _examples(first) != _examples(other)
    assert sorted(_examples(first)) == sorted(_examples(other))


def _scored_rows(batches: list[Batch]) -> dict[str, Any]:
    """What a pass over scoring batches holds example by example, the batches' own grouping
    left out: the keys, each scored account's row, each community block, and the edges of
    every relation."""
    return {
        "keys": [(key.t_ms, key.account_id) for batch in batches for key in batch.keys],
        "scored": torch.cat([batch.data[ACCOUNT].x[batch.data.scored_index] for batch in batches]),
        "community": torch.cat([batch.data.community for batch in batches]),
        "edges": {
            relation: sum(batch.data[relation].edge_index.shape[1] for batch in batches)
            for relation in RELATIONS
        },
    }


def test_a_pass_over_several_read_groups_equals_one_over_a_single_group(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every real run reads a Monday in groups of READ_GROUP accounts. With 50 a Monday of 224
    accounts takes five reads (four of 50 and one of 24), and the keys, their order, the scored
    rows, the community blocks and the edges are those of one read of all 224."""
    keys = world.examples()[["t_ms", "account_id"]].iloc[::-1]
    one = ScoringBatches(world.cache, world.identity, keys, 64)
    expected = _scored_rows(list(one))
    reads: list[int] = []
    real = batches_module.read_rows

    def counting(cache: Any, examples: Any) -> Any:
        reads.append(len(examples))
        return real(cache, examples)

    monkeypatch.setattr(batches_module, "read_rows", counting)
    assert reads == []
    list(one)
    assert reads == [224, 224]
    reads.clear()
    monkeypatch.setattr(batches_module, "READ_GROUP", 50)
    several = ScoringBatches(world.cache, world.identity, keys, 64)
    batches = list(several)
    assert reads == [50, 50, 50, 50, 24] * 2
    # A group is cut into batches of its own, so the batches are not those of one read.
    assert [len(batch.keys) for batch in batches] == [50, 50, 50, 50, 24] * 2
    found = _scored_rows(batches)
    assert found["keys"] == expected["keys"]
    assert found["keys"] == sorted(found["keys"]) and len(found["keys"]) == 448
    assert torch.equal(found["scored"], expected["scored"])
    assert torch.equal(found["community"], expected["community"])
    assert found["edges"] == expected["edges"]


def test_a_pass_counts_the_examples_it_built_and_the_time_it_took(world: Any) -> None:
    keys = world.examples()[["t_ms", "account_id"]].head(100)
    batches = ScoringBatches(world.cache, world.identity, keys, 32)
    assert (batches.built, batches.seconds) == (0, 0.0)
    list(batches)
    assert batches.built == 100 and batches.seconds > 0
    seconds = batches.seconds
    list(batches)
    assert batches.built == 200 and batches.seconds > seconds
