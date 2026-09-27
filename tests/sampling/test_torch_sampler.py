"""The torch subset sampler draws uniformly without replacement."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mule_pattern_learner.sampling.backend import select_resampled
from mule_pattern_learner.sampling.candidates import CandidateTable
from mule_pattern_learner.testing.builders import RESAMPLE, candidate_table


def test_subset_is_uniform_without_replacement() -> None:
    n, quota, trials = 12, 3, 3000
    keys, rows = candidate_table({"zelle_out": n})
    table = CandidateTable.build(keys, rows)
    sampler = replace(RESAMPLE, relation_fanouts=(quota, 2))
    counts = np.zeros(n)
    first = np.zeros(n)
    for seed in range(trials):
        slots = select_resampled(
            table, hop=1, sampler=sampler, fanout=8, mode="train", step_seed=seed
        )
        chosen = slots[0][slots[0] >= 0]
        assert len(chosen) == quota == len(set(chosen.tolist()))
        counts[chosen] += 1
        first[chosen[0]] += 1
    # Pearson chi-square with n-1 = 11 degrees of freedom; 31.26 is the 0.999 quantile.
    for observed, total in ((counts, trials * quota), (first, trials)):
        expected = total / n
        assert ((observed - expected) ** 2 / expected).sum() < 31.26
    # Evaluation hashes are uniform across contexts too.
    keys, rows = candidate_table({"zelle_out": n}, contexts=trials)
    table = CandidateTable.build(keys, rows)
    slots = select_resampled(table, hop=1, sampler=sampler, fanout=8, mode="eval")
    rank = np.asarray([[int(table.messages[j]["event_id"][2:]) for j in r[:quota]] for r in slots])
    observed = np.bincount((1000 - 1) - rank.ravel(), minlength=n)
    expected = trials * quota / n
    assert ((observed - expected) ** 2 / expected).sum() < 31.26
