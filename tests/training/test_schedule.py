"""The nnPU schedule: positive draws, marginal coverage, step seeds and evaluation samples."""

from __future__ import annotations

import hashlib

import numpy as np

from mule_pattern_learner.training.schedule import (
    PUSample,
    epoch_schedule,
    evaluation_indices,
    pu_batches,
    step_seed,
)


def test_evaluation_sample_draws_unlabeled_rows_only_from_the_marginal() -> None:
    indices = np.arange(40)
    observed = np.zeros(40, dtype=bool)
    observed[[1, 5, 30]] = True
    marginal = np.ones(40, dtype=bool)
    marginal[[5, 30, 31, 32, 33]] = False  # label-selected pool rows
    chosen = evaluation_indices(indices, observed, limit=10, seed=3, marginal=marginal)
    assert {1, 5, 30} <= set(chosen)
    unlabeled = set(chosen) - {1, 5, 30}
    assert len(unlabeled) == 10 and not unlabeled & {31, 32, 33}
    everything = evaluation_indices(indices, observed, limit=None, seed=3, marginal=marginal)
    assert set(everything) == set(indices) - {31, 32, 33}
    # Without a marginal mask the former behaviour is unchanged.
    assert np.array_equal(evaluation_indices(indices, observed, limit=None, seed=3), indices)


def test_step_seeds_are_stable_distinct_and_63_bit() -> None:
    seeds = {step_seed(7, epoch, step) for epoch in range(3) for step in range(50)}
    assert len(seeds) == 150 and all(0 <= s < 2**63 for s in seeds)
    expected = hashlib.sha256(b"temporal_live_step:7:1:2").digest()
    assert step_seed(7, 1, 2) == int.from_bytes(expected[:8], "big") >> 1


def test_epoch_schedule_equals_lazy_pu_batches_and_leaves_the_same_generator_state() -> None:
    observed = np.zeros(100, dtype=bool)
    observed[:9] = True
    samples = [
        PUSample("a", np.arange(10, 60), observed, np.arange(0, 5)),
        PUSample("b", np.arange(60, 100), observed, np.arange(5, 9)),
    ]
    lazy_rng, eager_rng = np.random.default_rng(11), np.random.default_rng(11)
    lazy = [
        (s.date, p, m)
        for s in samples
        for p, m in pu_batches(s.marginal, s.observed, lazy_rng, 8, positive_indices=s.positives)
    ]
    steps = epoch_schedule(samples, eager_rng, 8, epoch=2, seed=5)
    assert len(steps) == len(lazy)
    for index, (step, (date, positives, marginal)) in enumerate(zip(steps, lazy, strict=True)):
        assert (step.step, step.date, step.seed) == (index, date, step_seed(5, 2, index))
        assert np.array_equal(step.indices, np.r_[positives, marginal])
    assert lazy_rng.bit_generator.state == eager_rng.bit_generator.state


def test_nnpu_draws_only_known_positives_and_covers_the_label_blind_marginal() -> None:
    indices = np.arange(100)
    observed = indices < 20
    draws = list(pu_batches(indices, observed, np.random.default_rng(42), 16))
    assert all(observed[p].all() for p, _ in draws)
    assert sorted(np.concatenate([u for _, u in draws]).tolist()) == indices.tolist()
    assert all(len(p) == 4 for p, _ in draws)
    assert len(list(pu_batches(indices, observed, np.random.default_rng(42), 16, max_steps=2))) == 2
