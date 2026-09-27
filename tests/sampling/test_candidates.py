"""Candidate tables, selection keys and the merge of kept candidates into slots."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random
from typing import Any

import numpy as np
import pytest
import torch

from mule_pattern_learner.batching import assemble
from mule_pattern_learner.batching.assemble import build_batch
from mule_pattern_learner.contract.feature_groups import DEFAULT_GROUPS, FeaturePlan
from mule_pattern_learner.contract.graph_schema import RELATIONS, ContextKey
from mule_pattern_learner.contract.sampler_plan import PoolPlan, SamplerPlan
from mule_pattern_learner.sampling.backend import select_resampled
from mule_pattern_learner.sampling.candidates import CandidateTable, selection_keys, splitmix64
from mule_pattern_learner.testing.builders import (
    PAYMENTS,
    RESAMPLE,
    candidate_table,
    context_rng,
    payment,
    roots,
    synthetic_association,
    synthetic_row,
)
from mule_pattern_learner.testing.fake_graph import FakeStore

MPS = torch.backends.mps.is_available()
POOLED_RESAMPLE = SamplerPlan(
    roots=PoolPlan(recent=8, older=4, distinct=4, associations=2),
    children=PoolPlan(recent=4, older=2, distinct=2, associations=0),
    relation_fanouts=(8, 4),
)


def test_resample_caps_reserve_and_backfill_follow_the_stratified_merge() -> None:
    sampler = replace(RESAMPLE, relation_fanouts=(3, 2), association_fanout=1, association_slots=2)
    counts = {"zelle_out": 9, "zelle_in": 5, "payment_out": 1, "Account_Owned_By_Party": 3}
    counts |= {"Account_Bound_From_Token": 2, "Account_Uses_Device": 1}
    keys, rows = candidate_table(counts)
    for seed in range(30):
        table = CandidateTable.build(keys, rows)
        slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        chosen = [table.messages[j] for j in slots[0] if j >= 0]
        relations = Counter(m["relation"] for m in chosen)
        assert all(relations[r] <= 3 for r in PAYMENTS)
        assert all(relations[r] <= 1 for r in RELATIONS[4:])
        # 7 payments (3+3+1) fill K - reserve = 6 slots, then 2 reserved associations.
        assert [m["relation"] in PAYMENTS for m in chosen] == [True] * 6 + [False] * 2
        assert [m["relation"] for m in chosen[:3]] == ["zelle_out", "zelle_in", "payment_out"]
        assert [m["relation"] for m in chosen[3:6]] == ["zelle_out", "zelle_in", "zelle_out"]
        assert [m["relation"] for m in chosen[6:]] == [
            "Account_Owned_By_Party",
            "Account_Bound_From_Token",
        ]
    # Few payments: associations backfill the free slots after the reserve.
    keys, rows = candidate_table({"zelle_out": 1} | {r: 2 for r in counts if r not in PAYMENTS})
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=1, sampler=sampler, fanout=8, mode="eval")
    chosen = [table.messages[j]["relation"] for j in slots[0] if j >= 0]
    assert chosen == [
        "zelle_out",
        "Account_Owned_By_Party",
        "Account_Bound_From_Token",
        "Account_Uses_Device",
    ]
    # Hop 2 is payments-only with the second relation fan-out.
    keys, rows = candidate_table(counts)
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=2, sampler=sampler, fanout=4, mode="train", step_seed=3)
    chosen = [table.messages[j]["relation"] for j in slots[0] if j >= 0]
    assert chosen == ["zelle_out", "zelle_in", "payment_out", "zelle_out"]
    slots = select_resampled(table, hop=2, sampler=sampler, fanout=64, mode="eval")
    assert Counter(table.messages[j]["relation"] for j in slots[0] if j >= 0) == {
        "zelle_out": 2,
        "zelle_in": 2,
        "payment_out": 1,
    }


def test_reserve_uses_association_slots_and_quarter_fanout_limit() -> None:
    counts = {
        "zelle_out": 20,
        "zelle_in": 20,
        "Account_Owned_By_Party": 4,
        "Account_Uses_Device": 4,
    }
    keys, rows = candidate_table(counts)
    table = CandidateTable.build(keys, rows)
    for slots_, fanout, expected in ((2, 16, 2), (0, 16, 0), (8, 16, 4), (8, 8, 2), (8, 4, 1)):
        sampler = replace(RESAMPLE, relation_fanouts=(16, 4), association_slots=slots_)
        sampler = replace(sampler, association_fanout=4)
        slots = select_resampled(table, hop=1, sampler=sampler, fanout=fanout, mode="eval")
        chosen = [table.messages[j] for j in slots[0] if j >= 0]
        assert len(chosen) == fanout
        assert sum(not m["event_id"] for m in chosen) == min(slots_, 8, fanout // 4)
        assert sum(not m["event_id"] for m in chosen) == expected


def test_candidates_must_be_strictly_before_the_context_cutoff() -> None:
    key = ContextKey("Account", "a", 100, 1000)
    rng = context_rng(key)
    ok = [
        payment(key, "zelle_out", 99, "p", "recent", rng),
        synthetic_association(key, "Account_Uses_Device", "d", rng),
    ]
    table = CandidateTable.build([key], [{"messages": ok}])
    assert table.time_key.tolist() == [198, 199] and table.seed_time.tolist() == [200]
    for seq in (100, 101):
        bad = payment(key, "zelle_in", seq, "p", "recent", rng)
        with pytest.raises(ValueError, match="strictly before"):
            CandidateTable.build([key], [{"messages": [*ok, bad]}])


def test_eval_is_deterministic_and_device_independent() -> None:
    store = FakeStore(RESAMPLE)
    keys = roots(8)
    table = CandidateTable.build(keys, [store.row(k) for k in keys])
    expected = select_resampled(table, hop=1, sampler=RESAMPLE, fanout=8, mode="eval")
    devices = ["cpu"] + (["mps"] if MPS else [])
    for device in devices:
        for step_seed in (0, 99):
            again = select_resampled(
                table,
                hop=1,
                sampler=RESAMPLE,
                fanout=8,
                mode="eval",
                step_seed=step_seed,
                device=device,
            )
            assert np.array_equal(again, expected)
    other = replace(RESAMPLE, evaluation_seed=5)
    assert not np.array_equal(
        select_resampled(table, hop=1, sampler=other, fanout=8, mode="eval"), expected
    )
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    one = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", step_seed=1)
    two = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", step_seed=2)
    for name in one:
        assert torch.equal(one[name], two[name])
    if MPS:
        three = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="eval", device="mps")
        for name in one:
            if name.endswith("edge"):
                torch.testing.assert_close(three[name].cpu(), one[name], atol=1e-5, rtol=0)
            else:
                assert torch.equal(three[name].cpu(), one[name]), name


def _is_prefix(table: CandidateTable, hop_one: np.ndarray, hop_two: np.ndarray) -> bool:
    """Whether a context's hop-2 draw is the first payments of its hop-1 draw."""
    payments = [int(j) for j in hop_one if j >= 0 and table.relation[j] < 4]
    picked = [int(j) for j in hop_two if j >= 0]
    return picked == payments[: len(picked)]


def test_eval_keys_mix_the_hop_and_keep_hop_one_draws() -> None:
    keys = roots(32)
    table = CandidateTable.build(
        keys, [synthetic_row(k, POOLED_RESAMPLE.roots, full=True) for k in keys]
    )
    for seed in (0, 7):
        # Hop 1 keeps the SplitMix64(seed, context, item) keys of the first scheme.
        state = splitmix64(splitmix64(np.uint64(seed)) ^ table.context_hash[table.context])
        first_scheme = torch.from_numpy(
            (splitmix64(state ^ table.item_hash) >> np.uint64(1)).astype(np.int64)
        )
        options: dict[str, Any] = {"mode": "eval", "step_seed": 3, "evaluation_seed": seed}
        assert torch.equal(selection_keys(table, hop=1, **options), first_scheme)
        hop_two = selection_keys(table, hop=2, **options)
        assert bool((hop_two >= 0).all()) and not bool((hop_two == first_scheme).any())
    # A root is selected at both hops (its hop-2 row is its hop-1 row); its eval hop-2
    # draw must not simply repeat its first hop-1 picks, as it does not in training.
    first = select_resampled(table, hop=1, sampler=POOLED_RESAMPLE, fanout=16, mode="eval")
    second = select_resampled(table, hop=2, sampler=POOLED_RESAMPLE, fanout=4, mode="eval")
    prefixes = sum(_is_prefix(table, a, b) for a, b in zip(first, second, strict=True))
    assert prefixes <= len(keys) // 4, prefixes


def test_eval_batches_draw_root_hops_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore(POOLED_RESAMPLE)
    keys = roots(16)
    for key in keys:
        store.rows[1, key] = synthetic_row(key, POOLED_RESAMPLE.roots, encodings=False, full=True)
    draws: dict[int, list[list[dict[str, Any]]]] = {}
    select = assemble._select  # pyright: ignore[reportPrivateUsage]

    def recording(
        keys_: list[ContextKey], rows: list[dict[str, Any]], **kw: Any
    ) -> list[list[dict[str, Any]]]:
        chosen = select(keys_, rows, **kw)
        draws[kw["hop"]] = chosen
        return chosen

    monkeypatch.setattr(assemble, "_select", recording)
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    for mode in ("eval", "train"):
        build_batch(
            store, keys, fanouts=(16, 4), plan=plan, sampler=POOLED_RESAMPLE, mode=mode, step_seed=5
        )
        prefixes = 0
        for root in range(len(keys)):  # roots come first in the hop-2 context order
            payments = [m["event_id"] for m in draws[1][root] if m["event_id"]]
            picked = [m["event_id"] for m in draws[2][root]]
            prefixes += picked == payments[: len(picked)]
        assert prefixes <= len(keys) // 4, (mode, prefixes)


def _ids(table: CandidateTable, slots: np.ndarray) -> list[list[str]]:
    return [
        [table.messages[j]["event_id"] + table.messages[j]["node_id"] for j in row if j >= 0]
        for row in slots
    ]


def test_train_mode_varies_with_step_seed_and_ignores_wire_order() -> None:
    store = FakeStore(RESAMPLE)
    keys = roots(8)
    rows = [store.row(k) for k in keys]
    table = CandidateTable.build(keys, rows)
    draws = {
        seed: select_resampled(
            table, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=seed
        )
        for seed in range(6)
    }
    assert len({d.tobytes() for d in draws.values()}) == 6
    shuffled = []
    for row in rows:
        messages = list(row["messages"])
        random.Random(4).shuffle(messages)
        shuffled.append({**row, "messages": messages})
    reordered = CandidateTable.build(keys, shuffled)
    again = select_resampled(
        reordered, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=3
    )
    assert _ids(reordered, again) == _ids(table, draws[3])
    if MPS:
        mps = select_resampled(
            table, hop=1, sampler=RESAMPLE, fanout=8, mode="train", step_seed=3, device="mps"
        )
        assert np.array_equal(mps, draws[3])
    plan = FeaturePlan(DEFAULT_GROUPS, "tgat")
    a = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=10)
    b = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=10)
    c = build_batch(store, keys, plan=plan, sampler=RESAMPLE, mode="train", step_seed=11)
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert not all(a[n].shape == c[n].shape and torch.equal(a[n], c[n]) for n in a)
