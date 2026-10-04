"""The nnPU objective resolves the named positive weights, and its risk on a proxy sample."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mule_pattern_learner.config import LossConfig
from mule_pattern_learner.model.loss import NonNegativePULoss
from mule_pattern_learner.training import objective as training_objective


def test_nnpu_objective_resolves_the_named_positive_weights() -> None:
    for weight, resolved, name in (
        ("prior", 0.001, "nnPU"),
        ("balanced", 0.999, "imbalanced_nnPU"),
        (0.5, 0.5, "positive_reweighted_nnPU"),
    ):
        prior, value = training_objective.nnpu_objective(
            LossConfig(class_prior=0.001, positive_weight=weight)
        )
        assert (prior, value) == (0.001, pytest.approx(resolved))
        assert training_objective.objective_name(prior, value) == name


def test_the_proxy_risk_is_the_non_negative_risk_of_the_loss() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0])
    # Two positives scored 0.9 and 0.6, four unlabeled accounts 0.2, 0.1, 0.1 and 0.0. The
    # positive risk is 0.5 * (0.1 + 0.4) / 2 = 0.125, and the negative risk under the prior
    # 0.2, 0.4 / 4 - 0.2 * 1.5 / 2 = -0.05, is held at zero.
    scores = np.array([0.9, 0.6, 0.2, 0.1, 0.1, 0.0])
    assert training_objective.pu_risk(labels, scores, 0.2, 0.5) == pytest.approx(0.125)
    # Under the prior 0.01 the negative risk stays positive: 0.125 + 0.1 - 0.01 * 0.75.
    assert training_objective.pu_risk(labels, scores, 0.01, 0.5) == pytest.approx(0.2175)
    # The loss on the logits of those scores: its unclamped objective where the negative
    # risk is positive, and its positive risk alone where it is not.
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = torch.tensor(np.log(clipped / (1 - clipped)))
    targets = torch.tensor(labels, dtype=torch.float64)
    for prior in (0.01, 0.2):
        loss = NonNegativePULoss(prior=prior, positive_weight=0.5)
        _, objective = loss(logits, targets)
        positive = 0.5 * float((1 - torch.sigmoid(logits[:2])).mean())
        expected = max(float(objective), positive)
        assert training_objective.pu_risk(labels, clipped, prior, 0.5) == pytest.approx(expected)
    with pytest.raises(ValueError, match="revealed positives and unlabeled"):
        training_objective.pu_risk(np.ones(3), np.ones(3) / 2, 0.2, 0.5)
