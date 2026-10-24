"""The prior grid from the identified active share, and its decision on hand-made rankings
(docs/design.md, "Training and selection")."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, FULL_DESIGN_CONFIG
from mule_pattern_learner.training.priors import FIXED_MULTIPLE, Prior, decide, grid
from mule_pattern_learner.training.selection import Found

# The grid of the full design, which the proof of concept replaces by a fixed prior.
TRAINING = FULL_DESIGN_CONFIG.training
AUDIT = DEFAULT_CONFIG.audit
MULES = tuple(f"m{index:02d}" for index in range(40))


# An identified active share that every multiple of the grid keeps exact: 2 to the -9.
SHARE = 0.001953125


def test_the_grid_is_1_3_and_10_times_the_identified_active_share() -> None:
    assert grid(SHARE, TRAINING) == (
        Prior(1.0, 0.001953125),
        Prior(3.0, 0.005859375),
        Prior(10.0, 0.01953125),
    )


def test_a_fixed_multiple_is_the_grid_alone() -> None:
    fixed = replace(TRAINING, fixed_prior=FIXED_MULTIPLE)
    assert grid(SHARE, fixed) == (Prior(1.0, SHARE),)
    assert FIXED_MULTIPLE == 1.0


def test_the_grid_refuses_a_share_or_a_prior_outside_0_and_1() -> None:
    with pytest.raises(ValueError, match="share must be in"):
        grid(0.0, TRAINING)
    with pytest.raises(ValueError, match="not in"):
        grid(0.2, TRAINING)


def _found(rate: float, seed: int, flipped: tuple[int, ...] = ()) -> Found:
    """A seed's ranking of 40 mules with one positive Monday each, finding each with chance
    rate in both budgets, the mules of the rows flipped found when they were not and missed
    when they were found."""
    hits = np.random.default_rng(seed).random(len(MULES)) < rate
    hits[list(flipped)] = ~hits[list(flipped)]
    shares = np.repeat(hits[:, None].astype(np.float64), 2, axis=1)
    return Found(MULES, np.ones(len(MULES)), shares)


def test_priors_that_score_alike_fix_the_prior() -> None:
    """Three priors whose five seeds find about half the mules, each prior differing from the
    others on one or two mules of one seed: no pair's interval excludes zero."""
    found = {
        1.0: [_found(0.5, seed) for seed in range(5)],
        3.0: [_found(0.5, seed, (0,) if seed == 0 else ()) for seed in range(5)],
        10.0: [_found(0.5, seed, (1, 2) if seed == 3 else ()) for seed in range(5)],
    }
    decision = decide(found, AUDIT, TRAINING)
    assert list(decision.intervals) == [(1.0, 3.0), (1.0, 10.0), (3.0, 10.0)]
    assert decision.fixed
    assert decision.lines()[-1].startswith("No pair of priors differs")


def test_a_prior_that_scores_higher_keeps_the_grid() -> None:
    """The 3x prior finds every mule on every seed, the others about half: its pairs exclude
    zero."""
    found = {
        1.0: [_found(0.5, seed) for seed in range(5)],
        3.0: [_found(1.0, seed) for seed in range(5)],
        10.0: [_found(0.5, 20 + seed) for seed in range(5)],
    }
    decision = decide(found, AUDIT, TRAINING)
    low, high = decision.intervals[(1.0, 3.0)]
    assert high < 0
    low, high = decision.intervals[(3.0, 10.0)]
    assert low > 0
    assert not decision.fixed
    assert decision.lines()[-1] == "Some pair of priors differs: reported runs keep the grid"


def test_a_decision_needs_two_priors_with_a_run_per_seed() -> None:
    with pytest.raises(ValueError, match="at least two priors"):
        decide({1.0: [_found(0.5, 0)]}, AUDIT, TRAINING)
    with pytest.raises(ValueError, match="one selected run per seed"):
        decide({1.0: [_found(0.5, 0)], 3.0: [_found(0.5, 0), _found(0.5, 1)]}, AUDIT, TRAINING)


def test_the_decision_takes_its_level_and_seed_from_config() -> None:
    """The same runs decided at another interval level and another bootstrap seed give other
    intervals: the decision fixes neither."""
    found = {
        1.0: [_found(0.5, seed) for seed in range(5)],
        3.0: [_found(0.7, seed) for seed in range(5)],
    }
    base = decide(found, AUDIT, TRAINING).intervals[(1.0, 3.0)]
    narrow = decide(found, replace(AUDIT, interval=0.5), TRAINING).intervals[(1.0, 3.0)]
    assert base[0] < narrow[0] < narrow[1] < base[1]
    seeded = decide(found, AUDIT, replace(TRAINING, bootstrap_seed=1)).intervals[(1.0, 3.0)]
    assert seeded != base
