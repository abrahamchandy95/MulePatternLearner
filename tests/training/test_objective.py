"""The nnPU objective resolves the named positive weights."""

from __future__ import annotations

import pytest

from mule_pattern_learner.training import objective as training_objective


def test_nnpu_objective_resolves_the_named_positive_weights() -> None:
    for weight, resolved, name in (
        ("prior", 0.001, "nnPU"),
        ("balanced", 0.999, "imbalanced_nnPU"),
        (0.5, 0.5, "positive_reweighted_nnPU"),
    ):
        prior, value = training_objective.nnpu_objective(
            {"class_prior": 0.001, "positive_weight": weight}
        )
        assert (prior, value) == (0.001, pytest.approx(resolved))
        assert training_objective.objective_name(prior, value) == name
