"""The vectorised batch features equal the scalar reference, and labels never enter them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_pattern_learner.batching.assemble import build_batch, child_key
from mule_pattern_learner.batching.limits import BatchIndex
from mule_pattern_learner.contract.feature_groups import (
    CORE_GROUPS,
    FEATURE_GROUPS,
    POOL_GROUPS,
    FeaturePlan,
)
from mule_pattern_learner.contract.graph_schema import (
    CHANNELS,
    RAILS,
    RELATIONS,
    STRATA,
    ContextKey,
)
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.reference.batch_features import (
    base_features,
    edge_features,
    node_features,
)
from mule_pattern_learner.testing.builders import POOLED, context, roots, slots
from mule_pattern_learner.testing.fake_graph import FakeStore

# Every group but sampler_meta and the pool groups, whose counts batches give the roots
# only while the scalar features give them to every context.
V4_GROUPS = tuple(g for g in FEATURE_GROUPS if g not in ("sampler_meta", *POOL_GROUPS))


def reference_batch(
    store: FakeStore,
    keys: list[ContextKey],
    fanouts: tuple[int, int],
    plan: FeaturePlan,
    sampler: SamplerPlan,
    *,
    mode: str = "eval",
    step_seed: int = 0,
) -> dict[str, np.ndarray]:
    """The previous per-message assembly loop, on the scalar feature functions.

    Neighbours are drawn by the batches' own resampling; the loop checks how the
    vectorised assembly turns them into tensors.
    """
    root_rows = [store.row(k) for k in keys]
    first = slots(keys, root_rows, sampler, fanouts[0], mode=mode, step_seed=step_seed)
    lookup = BatchIndex(
        keys + [child_key(m, k) for k, msgs in zip(keys, first) for m in msgs], capacity=4096
    )
    unique = lookup.keys
    root_set = set(keys)
    contexts = [store.row(k) if k in root_set else store.row(k, 2) for k in unique]
    second = slots(unique, contexts, sampler, fanouts[1], 2, mode=mode, step_seed=step_seed)
    arrays = {
        "root_positions": np.asarray([lookup[k] for k in keys], dtype=np.int64),
        "x": np.stack([node_features(row, plan) for row in contexts]),
        "neighbor_positions": np.zeros((len(keys), fanouts[0]), dtype=np.int64),
        "second_x": np.zeros((len(unique), fanouts[1], len(plan.names("node"))), np.float32),
    }
    for prefix, rows, messages, fanout in (
        ("first_", root_rows, first, fanouts[0]),
        ("second_", contexts, second, fanouts[1]),
    ):
        arrays[prefix + "edge"] = np.zeros((len(rows), fanout, len(plan.edge_names)), np.float32)
        for name in ("relation", "rail", "channel", "stratum"):
            arrays[prefix + name] = np.zeros((len(rows), fanout), dtype=np.int64)
        arrays[prefix + "mask"] = np.zeros((len(rows), fanout), dtype=bool)
        for i, neighbors in enumerate(messages):
            for j, m in enumerate(neighbors):
                arrays[prefix + "edge"][i, j] = edge_features(m, plan)
                arrays[prefix + "relation"][i, j] = RELATIONS.index(m["relation"])
                arrays[prefix + "rail"][i, j] = RAILS.index(m["rail"])
                channel = m.get("channel", "unknown")
                arrays[prefix + "channel"][i, j] = CHANNELS.index(
                    channel if channel in CHANNELS else "other"
                )
                arrays[prefix + "stratum"][i, j] = STRATA.index(m["stratum"])
                arrays[prefix + "mask"][i, j] = True
                if prefix == "first_":
                    arrays["neighbor_positions"][i, j] = lookup[child_key(m, keys[i])]
                else:
                    arrays["second_x"][i, j] = base_features(m, plan)
    return arrays


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize(
    "plan",
    [FeaturePlan(CORE_GROUPS, "tgat"), FeaturePlan(V4_GROUPS, "tgat")],
    ids=["default", "all-but-pools"],
)
def test_vectorised_assembly_matches_the_scalar_features_bit_for_bit(
    mode: str, plan: FeaturePlan
) -> None:
    sampler = POOLED
    store = FakeStore(sampler)
    keys = roots(12) + roots(2)  # duplicate roots, as PU batches draw with replacement
    batch = build_batch(
        store, keys, fanouts=(8, 4), plan=plan, sampler=sampler, mode=mode, step_seed=5
    )
    expected = reference_batch(store, keys, (8, 4), plan, sampler, mode=mode, step_seed=5)
    assert set(batch) == set(expected)
    for name, value in expected.items():
        assert batch[name].dtype == torch.from_numpy(value).dtype, name
        assert torch.equal(batch[name], torch.from_numpy(value)), name
    assert batch["first_mask"].sum() > len(keys) and batch["second_mask"].any()


def test_labels_cannot_enter_node_features() -> None:
    row = context(ContextKey("Account", "root", 100, 1000))
    assert node_features(row).shape == (len(FeaturePlan().node_names),)
    row["features"]["is_mule"] = 1
    with pytest.raises(ValueError, match="Unrecognized"):
        node_features(row)
