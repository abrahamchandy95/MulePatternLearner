"""The client-computed pool groups: exact counts, transport, fingerprints and the built-in run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from temporal_fakes import FakeExecutor, association, context, message
import torch

from mule_pattern_learner.batching import assemble, features
from mule_pattern_learner.batching.assemble import child_key, make_live_batch
from mule_pattern_learner.batching.features import node_matrix
from mule_pattern_learner.batching.pool_counts import pool_activity
from mule_pattern_learner.config import DEFAULT_RUN, run_config
from mule_pattern_learner.contract import feature_groups
from mule_pattern_learner.contract.feature_groups import (
    DEFAULT_GROUPS,
    POOL_ACTIVITY_FEATURES,
    POOL_GROUPS,
    POOL_INTERNAL_FEATURES,
    FeaturePlan,
    contract_fingerprint,
    extraction_plan,
)
from mule_pattern_learner.contract.graph_schema import ContextKey
from mule_pattern_learner.contract.sampler_plan import SamplerPlan
from mule_pattern_learner.data.contexts import StreamingContextSource
from mule_pattern_learner.data.manifest import preparation_mismatches, preparation_view
from mule_pattern_learner.inference.saved_model import ModelCheckpoint
from mule_pattern_learner.model.build import build_model
from mule_pattern_learner.model.tgat import LiveTGAT
from mule_pattern_learner.reference import batch_features as batch_reference
from mule_pattern_learner.reference.batch_features import node_features
from mule_pattern_learner.tigergraph.context_query import validate_context

ROOT = ContextKey("Account", "root", 1000, 100_000_000)
CONFIG = run_config()
PLAN = FeaturePlan.from_config(CONFIG)
SAMPLER = SamplerPlan.from_config(CONFIG)
POOL_NAMES = POOL_ACTIVITY_FEATURES + POOL_INTERNAL_FEATURES
WITHOUT_POOLS = {
    **CONFIG,
    "feature_groups": [g for g in CONFIG["feature_groups"] if g not in POOL_GROUPS],
}


def payment(seq: int, relation: str, peer: str, **changes: Any) -> dict[str, Any]:
    """A pooled payment of ROOT; repeat pair (pair_prior_count 1) unless a test says so."""
    rail = "zelle" if relation.startswith("zelle") else "ach"
    fields = {"relation": relation, "rail": rail, "node_id": peer, "event_id": f"E{seq}"}
    return message(seq, seq * 60_000, ROOT, **(fields | changes))


def forwarded(delay: float, ratio: float) -> dict[str, Any]:
    """Flow fields of a payment whose next (or previous) opposite payment was seen."""
    return {
        "flow_present": True,
        "flow_censored": False,
        "flow_delay_seconds": delay,
        "flow_amount_ratio": ratio,
        "flow_ratio_present": True,
    }


# Ten payments and one association, with the expected pool counts.
POOL = [
    # First-time internal payer, 150, forwarded after an hour at 90%: a pass-through.
    payment(10, "zelle_in", "A", pair_prior_count=0, amount=150.0, **forwarded(3600, 0.9)),
    # The same payer again; its next outflow comes after more than a day.
    payment(11, "zelle_in", "A", amount=2000.0, **forwarded(90_000, 0.95)),
    # Both bands and both pass-through bounds are inclusive.
    payment(12, "payment_in", "B", pair_prior_count=0, amount=1000.0, **forwarded(86_400, 1.0)),
    # First-time but external; forwarded at 40%, below the pass-through band.
    payment(13, "payment_in", "C", pair_prior_count=0, peer_external=True, amount=5000.0)
    | forwarded(60, 0.4),
    # First-time internal payer whose amount is missing: counted, but in no amount band.
    payment(14, "payment_in", "D", pair_prior_count=0, amount=0, amount_present=False),
    # A known payer on another relation; an outflow followed but no amount ratio exists.
    payment(15, "payment_in", "A", pair_prior_count=3, amount=500.0)
    | {"flow_present": True, "flow_censored": False, "flow_delay_seconds": 100.0},
    # A Token "A" is a different counterparty from Account "A".
    payment(16, "zelle_out", "A", node_type="Token", pair_prior_count=0),
    payment(17, "zelle_out", "A", node_type="Token"),
    payment(18, "payment_out", "A", pair_prior_count=0),
    # An outflow's flow fields point back to an earlier inflow: never a pass-through.
    payment(19, "payment_out", "F", pair_prior_count=2, **forwarded(100, 0.9)),
    association(ROOT),
]
EXPECTED = {
    "pool_zelle_out_count": 2,
    "pool_zelle_out_unique": 1,
    "pool_zelle_in_count": 2,
    "pool_zelle_in_unique": 1,
    "pool_payment_out_count": 2,
    "pool_payment_out_unique": 2,
    "pool_payment_in_count": 4,
    "pool_payment_in_unique": 4,
    "pool_in_unique": 4,
    "pool_out_unique": 3,
    "pool_first_in": 4,
    "pool_pass_through_1d": 2,
    "pool_first_in_internal": 3,
    "pool_first_in_internal_100": 2,
    "pool_first_in_internal_1000": 1,
}


class Hubs:
    def __init__(self, ids: set[str]) -> None:
        self.ids = ids

    def is_stub(self, node_type: str, node_id: str, cutoff_seq: int, phase: int = 3, /) -> bool:
        return node_id in self.ids


class RawRows:
    """A context source that skips validation, as a misbehaving server would."""

    def __init__(self, rows: dict[tuple[int, ContextKey], dict[str, Any]]) -> None:
        self.rows = rows

    def fetch(self, keys: list[ContextKey], *, hop: int = 1) -> list[dict[str, Any] | None]:
        return [self.rows[hop, key] for key in keys]


def test_pool_activity_counts_the_candidate_pool_exactly() -> None:
    row = context(ROOT, POOL)
    # The hand-built row is a valid response of the built-in extraction.
    validate_context(ROOT, row, extraction_plan(CONFIG), SAMPLER)
    values = pool_activity(row)
    assert tuple(values) == POOL_NAMES
    assert values == EXPECTED
    # Every count gets log1p; both paths agree.
    x = node_matrix([row], PLAN)[0]
    np.testing.assert_allclose(node_features(row, PLAN), x)
    for name, value in EXPECTED.items():
        assert x[PLAN.node_names.index(name)] == pytest.approx(np.log1p(value), rel=1e-6)
    # Computing the groups never writes into the (cached) TigerGraph row.
    assert not set(row["features"]) & set(POOL_NAMES)
    # Each group feeds only its own columns.
    for group, names in zip(POOL_GROUPS, (POOL_ACTIVITY_FEATURES, POOL_INTERNAL_FEATURES)):
        plan = FeaturePlan((*WITHOUT_POOLS["feature_groups"], group), "split")
        assert plan.names("summary") == names
        vector = node_matrix([row], plan)[0]
        expected = [np.log1p(EXPECTED[name]) for name in names]
        np.testing.assert_allclose(vector[-len(names) :], expected, rtol=1e-6)


def test_stubs_and_contexts_without_payments_get_zeros() -> None:
    link = POOL[0]
    stub = assemble._stub_row(child_key(link, ROOT), link)  # pyright: ignore[reportPrivateUsage]
    rows = [stub, context(ROOT), context(ROOT, [association(ROOT)])]
    for row in rows:
        assert pool_activity(row) == dict.fromkeys(POOL_NAMES, 0)
    columns = [PLAN.node_names.index(name) for name in POOL_NAMES]
    x = node_matrix(rows, PLAN)
    assert not x[:, columns].any()
    for row, vector in zip(rows, x, strict=True):
        np.testing.assert_allclose(node_features(row, PLAN), vector)
    missing = {k: v for k, v in link.items() if k != "pair_prior_count"}
    with pytest.raises(ValueError, match="lacks required field 'pair_prior_count'"):
        pool_activity(context(ROOT, [missing]))


def test_built_in_batches_feed_root_pool_counts_to_the_summary_branch() -> None:
    # Payer B's own context, at the cutoff of its payment to the root, has one inflow.
    payer = child_key(POOL[2], ROOT)
    inflow = message(5, 300_000, payer, relation="payment_in", rail="ach", pair_prior_count=0)
    # A second root at the same cutoff with its own pool, and the first root drawn twice.
    other = ContextKey("Account", "other", ROOT.cutoff_seq, ROOT.cutoff_ms)
    rows = {
        ROOT: context(ROOT, POOL),
        other: context(other, [payment(20, "zelle_in", "Q", pair_prior_count=0, amount=300.0)]),
        payer: context(payer, [inflow]),
    }
    roots = [ROOT, other, ROOT]
    executor = FakeExecutor(rows)
    with StreamingContextSource(executor, plan=extraction_plan(CONFIG), sampler=SAMPLER) as source:
        for _ in range(2):  # the second batch is served from the source's cache
            batch = make_live_batch(
                source, roots, fanouts=(16, 4), plan=PLAN, sampler=SAMPLER, hubs=Hubs({"F"})
            )
    x = batch["x"]
    assert x.shape[1] == len(PLAN.node_names) == 9 + len(POOL_NAMES)
    columns = [PLAN.node_names.index(name) for name in POOL_NAMES]
    positions = batch["root_positions"].tolist()
    assert positions == [0, 1, 0]
    for key, position in zip(roots, positions, strict=True):
        expected = node_matrix([rows[key]], PLAN)[0, columns]
        assert expected.any()
        np.testing.assert_allclose(x[position, columns].numpy(), expected)
    # The split model reads pool columns for roots only, so children (the payer and the
    # hub stub among them) keep zeros there, whatever their own pool holds.
    withheld = x[:, PLAN.node_names.index("history_withheld")] == 1
    assert int(withheld.sum()) == 1 and len(x) > 3
    assert not x[2:][:, columns].any()
    assert node_matrix([rows[payer]], PLAN)[0, columns].any()
    # The summary branch reads the root's pool columns and receives gradient.
    model = build_model(CONFIG, PLAN, dropout=0.0)
    assert model.summary is not None
    assert list(model.summary_indices) == columns
    assert [PLAN.node_names[i] for i in model.node_indices] == list(PLAN.names("node"))
    output = model(batch)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.summary.parameters())


def test_pool_groups_feed_models_that_read_them_for_roots_only() -> None:
    # The tabular control of the built-in run (summary architecture) scores roots only.
    tabular = {**CONFIG, "architecture": "summary"}
    plan = FeaturePlan.from_config(tabular)
    assert plan.architecture == "summary" and plan.names("summary") == POOL_NAMES
    executor = FakeExecutor({ROOT: context(ROOT, POOL)})
    with StreamingContextSource(executor, plan=extraction_plan(tabular), sampler=SAMPLER) as source:
        batch = make_live_batch(source, [ROOT, ROOT], plan=plan, sampler=SAMPLER)
    expected = node_matrix([context(ROOT, POOL)], plan)
    np.testing.assert_allclose(batch["x"].numpy(), np.repeat(expected, 2, axis=0))
    assert build_model(tabular, plan)(batch).shape == (2,)
    # node_matrix gives pool counts to the leading rows only when asked.
    rows = [context(ROOT, POOL)] * 3
    columns = [PLAN.node_names.index(name) for name in POOL_NAMES]
    full = node_matrix(rows, PLAN)
    partial = node_matrix(rows, PLAN, pooled=1)
    np.testing.assert_allclose(partial[0], full[0])
    assert full[1:, columns].all() and not partial[1:, columns].any()
    others = [i for i in range(len(PLAN.node_names)) if i not in columns]
    np.testing.assert_allclose(partial[:, others], full[:, others])


def test_tigergraph_cannot_supply_pool_counts() -> None:
    bad = context(ROOT, POOL)
    bad["features"]["pool_first_in"] = 1
    with pytest.raises(ValueError, match="Unknown node feature"):
        validate_context(ROOT, bad, extraction_plan(CONFIG), SAMPLER)
    with pytest.raises(ValueError, match=r"client-only feature \['pool_first_in'\]"):
        make_live_batch(RawRows({(1, ROOT): bad}), [ROOT], plan=PLAN, sampler=SAMPLER)
    link = message(80, 800_000, ROOT, node_id="N")
    child = child_key(link, ROOT)
    bad_child = context(child)
    bad_child["features"]["pool_first_in_internal"] = 1
    rows = {(1, ROOT): context(ROOT, [link]), (2, child): bad_child}
    with pytest.raises(ValueError, match="client-only feature"):
        make_live_batch(RawRows(rows), [ROOT], plan=PLAN, sampler=SAMPLER)


def test_client_groups_leave_the_wire_and_the_preparation_unchanged() -> None:
    assert set(POOL_GROUPS) <= set(DEFAULT_RUN["feature_groups"])
    assert PLAN.architecture == "split" and PLAN.names("summary") == POOL_NAMES
    assert extraction_plan(CONFIG) == extraction_plan(WITHOUT_POOLS)
    for hop in (1, 2):
        flags = PLAN.query_flags(hop)
        assert flags == FeaturePlan.from_config(WITHOUT_POOLS).query_flags(hop)
        assert not {"include_" + group for group in POOL_GROUPS} & set(flags)
    # A dataset prepared without the groups serves a run that adds them.
    view = preparation_view(WITHOUT_POOLS)
    assert preparation_view(CONFIG) == view
    assert preparation_mismatches(CONFIG, {"source": {"preparation": view}}) == []
    # The counts read pair (and flow) fields, so the model must extract those groups.
    with pytest.raises(ValueError, match="dependencies for pool_activity"):
        FeaturePlan(("entity_meta", "message_core", "pair_history", "pool_activity"), "split")
    with pytest.raises(ValueError, match="dependencies for pool_internal_inflows"):
        FeaturePlan(("entity_meta", "message_core", "pool_internal_inflows"), "split")


def test_pool_definitions_are_part_of_the_input_fingerprint_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, plan, without = (
        contract_fingerprint(),
        PLAN.fingerprint(),
        FeaturePlan.from_config(WITHOUT_POOLS).fingerprint(),
    )
    internal = FeaturePlan((*WITHOUT_POOLS["feature_groups"], "pool_internal_inflows"), "split")
    first = internal.fingerprint()
    changes: list[tuple[str, Any]] = [
        ("PASS_THROUGH_RATIO", (0.8, 1.0)),
        ("PASS_THROUGH_SECONDS", 3_600),
        ("FIRST_INFLOW_BANDS", (100, 500)),
        ("POOL_ACTIVITY_VERSION", 2),
    ]
    for name, value in changes:
        with monkeypatch.context() as patch:
            patch.setattr(feature_groups, name, value)
            # Checkpoints and caches of every model keep their contract, and models
            # without a pool group their inputs.
            assert contract_fingerprint() == contract
            assert FeaturePlan.from_config(WITHOUT_POOLS).fingerprint() == without
            # A model trained with the pool groups is refused once their meaning changes.
            assert PLAN.fingerprint() != plan, name
            assert internal.fingerprint() != first, name
    saved = ModelCheckpoint(Path("model.pt"), {"input_fingerprint": plan})
    saved.check_inputs(PLAN)
    monkeypatch.setattr(feature_groups, "PASS_THROUGH_RATIO", (0.8, 1.0))
    with pytest.raises(ValueError, match="pool definitions differ"):
        saved.check_inputs(PLAN)


@pytest.mark.parametrize(
    ("groups", "architecture"),
    [(DEFAULT_GROUPS, "split"), (DEFAULT_GROUPS, "summary")],
)
def test_plans_without_pool_groups_are_unaffected(
    groups: tuple[str, ...], architecture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = FeaturePlan(groups, architecture)
    assert not set(plan.node_names) & set(POOL_NAMES)

    def refuse(row: dict[str, Any]) -> dict[str, float]:
        raise AssertionError("pool counts computed for a plan without a pool group")

    monkeypatch.setattr(features, "pool_activity", refuse)
    monkeypatch.setattr(batch_reference, "pool_activity", refuse)
    row = context(ROOT, POOL)
    np.testing.assert_allclose(node_features(row, plan), node_matrix([row], plan)[0])
    if architecture == "split":
        # The previous built-in profile keeps its width and has no summary branch.
        model = LiveTGAT(64, 4, 0.15, plan=plan)
        assert model.summary is None and len(plan.node_names) == 9
        assert sum(p.numel() for p in model.parameters()) == 83_457
