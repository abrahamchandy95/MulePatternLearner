"""The audit sample keeps rare positives and recovers the population prevalence."""

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.evaluation.audit import audit_metrics
from mule_pattern_learner.evaluation.sample import audit_sample


def test_final_evaluation_covers_rare_positives_and_recovers_population_prevalence():
    population = pd.DataFrame({"account_id": [str(i) for i in range(10000)], "split": "test"})
    truth = population[["account_id"]].assign(is_mule=np.r_[np.ones(40, int), np.zeros(9960, int)])
    sample = audit_sample(population, truth, negative_limit=100, seed=7)
    assert len(sample) == 140 and sample.is_mule.sum() == 40
    sample["score"] = sample.is_mule * 0.8 + 0.1
    metrics = audit_metrics(sample, 0.5)
    assert metrics["weighted_prevalence"] == pytest.approx(0.004)
    assert metrics["estimated_population"] == pytest.approx(10000)
    assert metrics["average_precision"] == pytest.approx(1)
    with pytest.raises(ValueError, match="binary"):
        audit_sample(population, truth.iloc[1:])
    with pytest.raises(ValueError, match="test-only"):
        audit_sample(population.assign(split="train"), truth)
