"""The slot_sum option: a count-aware sum over the root's hop-1 slots beside attention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from temporal_fakes import FakeExecutor, context, message
import torch
from torch import nn

from mule_pattern_learner.batching.assemble import make_live_batch
from mule_pattern_learner.config import DEFAULT_RUN, run_config, validate_config
from mule_pattern_learner.contract.feature_groups import (
    DEFAULT_GROUPS,
    FeaturePlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.graph_schema import RAILS, RELATIONS, ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.contract.time_basis import BASIS_ID
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.experiments.variants import feature_experiments
from mule_pattern_learner.inference.predictor import TemporalPredictor
from mule_pattern_learner.inference.saved_model import ModelCheckpoint
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.tgat import LiveTGAT

CONFIG = run_config()
PLAN = FeaturePlan.from_config(CONFIG)
SAMPLER = SamplerPlan.from_config(CONFIG)
HIDDEN = CONFIG["hidden"]
FANOUT = CONFIG["fanouts"][0]
SLOT_KEYS = {"slot_sum.0.weight", "slot_sum.0.bias", "slot_sum.2.weight", "slot_sum.2.bias"}
ROOT = ContextKey("Account", "root", 1000, 100_000_000)
PAYMENTS = [
    message(seq, seq * 60_000, ROOT, relation="payment_in", rail="ach", node_id=f"P{seq % 3}")
    for seq in range(10, 16)
]


def slot_batch(
    plan: FeaturePlan, slots: list[list[int]], width: int = FANOUT
) -> dict[str, torch.Tensor]:
    """Roots with identical inputs whose hop-1 slots name children 0, 1, ...

    Contexts are the roots, then the children. Every hop-1 slot carries the same
    message, so two slots on one child are the same token. Other columns are padding,
    pointing at context 0 like make_live_batch's.
    """
    generator = torch.Generator().manual_seed(0)
    roots, children = len(slots), 1 + max(max(s, default=0) for s in slots)
    # One template row for every root, then one row per child.
    rows = torch.tensor([0] * roots + list(range(1, children + 1)))
    second, edges = 2, len(plan.edge_names)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator)

    batch = {
        "root_positions": torch.arange(roots),
        "x": rand(children + 1, len(plan.node_names))[rows],
        "second_x": rand(children + 1, second, len(plan.names("node")))[rows],
        "second_edge": rand(children + 1, second, edges)[rows],
        "second_mask": torch.ones(len(rows), second, dtype=torch.bool),
        "neighbor_positions": torch.zeros(roots, width, dtype=torch.long),
        "first_edge": rand(edges).expand(roots, width, edges).clone(),
        "first_mask": torch.zeros(roots, width, dtype=torch.bool),
    }
    for prefix, shape in (("first_", (roots, width)), ("second_", (len(rows), second))):
        for name in ("relation", "rail", "channel", "stratum"):
            batch[prefix + name] = torch.zeros(shape, dtype=torch.long)
    for i, named in enumerate(slots):
        batch["neighbor_positions"][i, : len(named)] = torch.tensor(named, dtype=torch.long) + roots
        batch["first_mask"][i, : len(named)] = True
    return batch


def seeded(config: dict[str, Any], plan: FeaturePlan = PLAN) -> LiveTGAT:
    torch.manual_seed(0)
    return build_model(config, plan, dropout=0.0).eval()


def slot_mlp(model: LiveTGAT) -> nn.Module:
    assert model.slot_sum is not None
    return model.slot_sum


def payload(config: dict[str, Any], model: LiveTGAT, contract: str, inputs: str) -> dict[str, Any]:
    """A model.pt payload as training saves it, with the fields scoring checks."""
    return {
        "state_dict": model.state_dict(),
        "config": config,
        "contract": contract,
        "basis_id": BASIS_ID,
        "threshold": 0.5,
        "input_fingerprint": inputs,
    }


@pytest.mark.parametrize("groups", [DEFAULT_RUN["feature_groups"], list(DEFAULT_GROUPS)])
def test_output_shapes(groups: list[str]) -> None:
    config = {**CONFIG, "feature_groups": groups}
    plan = FeaturePlan.from_config(config)
    model = seeded(config, plan)
    assert model.first_fanout == FANOUT == 16
    # Attention, slot sum and, with pool_activity, the summary branch.
    parts = 3 if plan.names("summary") else 2
    batch = slot_batch(plan, [[0, 1], [1], []])
    assert model.encode(batch).shape == (3, parts * HIDDEN)
    assert model.head[0].in_features == parts * HIDDEN
    assert model(batch).shape == (3,)


def test_built_in_batches_fit_the_slot_sum_and_train_it() -> None:
    executor = FakeExecutor({ROOT: context(ROOT, PAYMENTS)})
    with StreamingContextSource(executor, plan=extraction_plan(CONFIG), sampler=SAMPLER) as source:
        batch = make_live_batch(
            source, [ROOT], fanouts=(FANOUT, 4), plan=PLAN, sampler=SAMPLER, mode="train"
        )
    model = build_model(CONFIG, PLAN)
    assert batch["first_mask"].shape == (1, model.first_fanout)
    assert 0 < int(batch["first_mask"].sum()) < FANOUT
    output = model(batch)
    assert output.shape == (1,) and torch.isfinite(output).all()
    output.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in slot_mlp(model).parameters())


def test_the_sum_counts_slots_that_a_mean_cannot_tell_apart() -> None:
    model = seeded(CONFIG)
    # Slots {a, b} and {a, a, b, b}: the same mean, twice the count.
    batch = slot_batch(PLAN, [[0, 1], [0, 1, 0, 1]])
    seen: dict[str, torch.Tensor] = {}
    slot_mlp(model).register_forward_hook(lambda _m, _i, output: seen.update(values=output))
    embedding = model.encode(batch)
    values, mask = seen["values"], batch["first_mask"]
    torch.testing.assert_close(values[0][mask[0]].mean(0), values[1][mask[1]].mean(0))
    # The slot part of the embedding sits between the attention output and the summary.
    total = embedding[:, HIDDEN : 2 * HIDDEN]
    assert total[0].abs().sum() > 0
    torch.testing.assert_close(total[1], 2 * total[0])
    # Divided by the configured fan-out, not by the two or four valid slots.
    torch.testing.assert_close(total[0], values[0, :2].sum(0) / FANOUT)
    logits = model(batch)
    assert logits[0] != logits[1]


def test_padding_changes_nothing() -> None:
    model = seeded(CONFIG)
    batch = slot_batch(PLAN, [[0], [0, 1, 2], []], width=8)
    expected = model.encode(batch)
    # A root without slots adds nothing to the head.
    assert not expected[2, HIDDEN : 2 * HIDDEN].any()
    noisy = {name: value.clone() for name, value in batch.items()}
    padding = ~noisy["first_mask"]
    count = int(padding.sum())
    generator = torch.Generator().manual_seed(1)
    noisy["neighbor_positions"][padding] = torch.randint(
        0, len(noisy["x"]), (count,), generator=generator
    )
    noisy["first_edge"][padding] = torch.randn(
        count, noisy["first_edge"].shape[-1], generator=generator
    )
    for name, size in (("relation", len(RELATIONS)), ("rail", len(RAILS))):
        noisy["first_" + name][padding] = torch.randint(0, size, (count,), generator=generator)
    torch.testing.assert_close(model.encode(noisy), expected)
    # More padded columns, up to the configured fan-out.
    wide = dict(batch)
    for name, value in batch.items():
        if name.startswith("first_") or name == "neighbor_positions":
            extra = value.new_zeros((len(value), FANOUT - 8, *value.shape[2:]))
            wide[name] = torch.cat((value, extra), dim=1)
    torch.testing.assert_close(model.encode(wide), expected)


def test_the_slot_sum_is_on_by_default_and_off_builds_the_model_without_it() -> None:
    absent = {k: v for k, v in CONFIG.items() if k != "slot_sum"}
    assert validate_config(absent)["slot_sum"] is True
    no_pools = {**CONFIG, "slot_sum": False, "feature_groups": list(DEFAULT_GROUPS)}
    for config, parameters in ((no_pools, 83_457), ({**CONFIG, "slot_sum": False}, 88_705)):
        plan = FeaturePlan.from_config(config)
        # The constructor call without the option.
        torch.manual_seed(0)
        old = LiveTGAT(HIDDEN, CONFIG["heads"], CONFIG["dropout"], plan=plan).state_dict()
        torch.manual_seed(0)
        model = build_model(config, plan)
        assert model.slot_sum is None
        state = model.state_dict()
        # Same keys, shapes and initial weights.
        assert list(state) == list(old)
        assert all(torch.equal(state[k], old[k]) for k in old)
        assert sum(p.numel() for p in model.parameters()) == parameters
    old = LiveTGAT(HIDDEN, CONFIG["heads"], CONFIG["dropout"], plan=PLAN).state_dict()
    model = build_model(CONFIG, PLAN)
    assert model.state_dict().keys() - old.keys() == SLOT_KEYS
    assert model.state_dict()["head.0.weight"].shape == (HIDDEN, 3 * HIDDEN)
    assert sum(p.numel() for p in model.parameters()) == 101_121


def test_saved_models_with_the_slot_sum_score_like_the_trained_model(tmp_path: Path) -> None:
    new = seeded(CONFIG)
    torch.save(
        payload(CONFIG, new, contract_fingerprint(), PLAN.fingerprint()), tmp_path / "new.pt"
    )
    executor = FakeExecutor({ROOT: context(ROOT, PAYMENTS)})
    with StreamingContextSource(executor, plan=extraction_plan(CONFIG), sampler=SAMPLER) as source:
        predictor = TemporalPredictor(ModelCheckpoint.load(tmp_path / "new.pt"), source, "cpu")
        assert predictor.model.slot_sum is not None
        prepared = predictor.prepare([ROOT])
        frame = predictor.infer(prepared)
        assert prepared.batch is not None and len(frame.embedding[0]) == 3 * HIDDEN
        with torch.no_grad():
            expected = torch.sigmoid(new(prepared.batch))
        assert frame.score.tolist() == pytest.approx(expected.tolist(), rel=1e-6)


def test_nonsense_options_are_rejected() -> None:
    summary = FeaturePlan(("entity_meta", "decayed_activity", "history_support"), "summary")
    with pytest.raises(ValueError, match="no hop-1 slots"):
        LiveTGAT(16, 4, 0, plan=summary, slot_sum=True)
    for flag in (1, "yes", None):
        with pytest.raises(ValueError, match="slot_sum must be true or false"):
            LiveTGAT(16, 4, 0, plan=PLAN, slot_sum=flag)  # type: ignore[arg-type]
    for fanout in (0, 65, 8.0, True):
        with pytest.raises(ValueError, match="fan-out must be an integer"):
            LiveTGAT(16, 4, 0, plan=PLAN, slot_sum=True, first_fanout=fanout)  # type: ignore[arg-type]
    for flag in ("yes", 1, 0.5):
        with pytest.raises(ValueError, match="slot_sum"):
            validate_config({"slot_sum": flag})
    # The tabular variant of the built-in run has no slots: the switch adds nothing.
    tabular = {**CONFIG, "architecture": "summary"}
    model = build_model(tabular, FeaturePlan.from_config(tabular))
    assert model.slot_sum is None and model.head[0].in_features == HIDDEN
    # A batch wider than the configured fan-out would change the divisor's meaning.
    narrow = build_model({**CONFIG, "fanouts": [8, 4]}, PLAN)
    assert narrow.first_fanout == 8
    with pytest.raises(ValueError, match="16 hop-1 slots, more than the model's fan-out 8"):
        narrow.encode(slot_batch(PLAN, [[0]], width=16))


def test_every_ablation_arm_states_its_slot_sum() -> None:
    arms = feature_experiments(CONFIG)
    for name, arm in arms.items():
        model = build_model(arm, FeaturePlan.from_config(arm))
        assert (model.slot_sum is not None) == arm["slot_sum"], name
    # The feature-group arms keep the model they were designed on.
    assert not any(arm["slot_sum"] for name, arm in arms.items() if not name.startswith("built_in"))
    keys = ("feature_groups", "architecture", "slot_sum")
    assert {k: arms["built_in"][k] for k in keys} == {k: CONFIG[k] for k in keys}
    differences = {
        name: {k for k in keys if arm[k] != CONFIG[k]}
        for name, arm in arms.items()
        if name.startswith("built_in_")
    }
    assert differences == {
        "built_in_no_slot_sum": {"slot_sum"},
        "built_in_no_pool": {"feature_groups"},
        "built_in_no_internal": {"feature_groups"},
        "built_in_tabular": {"architecture", "slot_sum"},
    }
    assert arms["built_in_no_pool"]["feature_groups"] == list(DEFAULT_GROUPS)
    assert "pool_internal_inflows" not in arms["built_in_no_internal"]["feature_groups"]
