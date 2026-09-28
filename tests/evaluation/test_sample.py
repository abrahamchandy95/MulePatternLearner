"""The audit sample keeps rare positives and recovers the population prevalence."""

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.evaluation.audit import audit_metrics
from mule_pattern_learner.evaluation.sample import AUDIT_NEGATIVES, audit_sample


def population_of(split: str, count: int = 10000) -> pd.DataFrame:
    return pd.DataFrame({"account_id": [f"{i:05}" for i in range(count)], "split": split})


def truth_of(population: pd.DataFrame, mules: int = 40) -> pd.DataFrame:
    """The first ``mules`` accounts are mules, in rings of four."""
    count = len(population)
    ring = np.where(np.arange(count) < mules, np.arange(count) // 4, -1)
    return population[["account_id"]].assign(
        is_mule=np.r_[np.ones(mules, int), np.zeros(count - mules, int)],
        ring_id=ring,
        label_source="phantomledger_role",
    )


def test_the_audit_sample_covers_rare_positives_and_recovers_population_prevalence():
    population = population_of("test")
    truth = truth_of(population)
    sample = audit_sample(population, truth, negatives=100, seed=7)
    assert len(sample) == 140 and sample.is_mule.sum() == 40
    assert sample.account_id.is_monotonic_increasing
    # The truth's ring and label source come with every sampled account.
    assert sample[sample.is_mule == 1].ring_id.tolist() == [i // 4 for i in range(40)]
    assert (sample[sample.is_mule == 0].ring_id == -1).all()
    assert set(sample.inclusion_probability) == {1.0, 100 / 9960}
    sample["score"] = sample.is_mule * 0.8 + 0.1
    metrics = audit_metrics(sample, 0.5)
    assert metrics["weighted_prevalence"] == pytest.approx(0.004)
    assert metrics["estimated_population"] == pytest.approx(10000)
    assert metrics["average_precision"] == pytest.approx(1)
    with pytest.raises(ValueError, match="binary"):
        audit_sample(population, truth.iloc[1:], seed=7)
    with pytest.raises(ValueError, match="one split"):
        audit_sample(pd.concat([population, population_of("train", 3)]), truth, seed=7)


def test_the_sample_of_a_split_depends_only_on_its_population_truth_and_seed():
    assert AUDIT_NEGATIVES == 2000
    validation = population_of("validation", 5000)
    truth = truth_of(validation)
    first = audit_sample(validation, truth, seed=42)
    # Every positive and AUDIT_NEGATIVES negatives, whatever order the population has.
    assert (first.is_mule.sum(), len(first)) == (40, 40 + AUDIT_NEGATIVES)
    shuffled = validation.sample(frac=1, random_state=3)
    again = audit_sample(shuffled, truth.sample(frac=1, random_state=4), seed=42)
    pd.testing.assert_frame_equal(first, again)
    # Another seed draws other negatives from the same population.
    other = audit_sample(validation, truth, seed=43)
    assert set(other.account_id) != set(first.account_id)
    assert set(other[other.is_mule == 1].account_id) == set(first[first.is_mule == 1].account_id)
