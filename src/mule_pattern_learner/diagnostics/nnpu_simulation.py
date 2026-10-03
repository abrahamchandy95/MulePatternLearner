"""The nnPU positive weight on a synthetic problem shaped like the dataset, offline.

The first full run trained with textbook nnPU, whose positive weight is the class prior
(0.001), and every score collapsed to about 1.7e-7 within an epoch; the built-in run
weighs the positives by one minus the prior ("balanced", imbalanced nnPU). This
simulation asks whether the weight alone explains the collapse, on a problem with the
dataset's proportions and the repository's loss (model.loss.NonNegativePULoss), with no
graph and no ground truth read.

The problem (Problem): positives are normal vectors shifted by `shift` along a few of
the features; 20 labelled positives; a label-blind marginal in which positives are at
the true prevalence, drawn like the labelled ones; a validation proxy of 11 labelled
positives against 2,000 negatives; and a truth test at the class prior. A small MLP
trains on batches of 16 positives drawn with replacement and 48 marginal accounts, as
the trainer's batches are, and the epoch with the best validation proxy AP is kept,
with early stopping as the trainer's. For each positive weight and seed the table
records the kept epoch's test AP, ROC AUC and precision in the top 1%, the mean scores
of the labelled positives and of the test negatives, and whether the run collapsed: the
mean score of its labelled positives is below COLLAPSE, so it scores even the positives
it trains on near zero, the constant scorer that costs the textbook objective only the
prior. Its counterpart on the graph is the prior_weight control variant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import nn

from ..artifacts import DIAGNOSTIC_TABLES
from ..metrics import average_precision, roc_auc
from ..model.loss import NonNegativePULoss

COLUMNS = DIAGNOSTIC_TABLES["nnpu_simulation"]
# The class prior the loss assumes (the built-in run's loss.class_prior).
PRIOR = 0.001
# The positive weights compared: the textbook weight (the prior), two between, and the
# balanced weight of the built-in run (one minus the prior).
WEIGHTS = (PRIOR, 0.1, 0.5, 1 - PRIOR)
SEEDS = (1, 2, 3, 4, 5)
# A run whose labelled positives score below this on average has collapsed.
COLLAPSE = 0.05


@dataclass(frozen=True)
class Problem:
    """The synthetic problem and the training schedule of one simulated run."""

    features: int = 16
    shifted: int = 4
    shift: float = 2.0
    positives: int = 20
    marginal: int = 20_000
    true_prevalence: float = 0.00073
    validation_positives: int = 11
    validation_negatives: int = 2_000
    test_positives: int = 300
    test_negatives: int = 300_000
    batch_positives: int = 16
    batch_marginal: int = 48
    steps: int = 100
    epochs: int = 30
    patience: int = 6
    hidden: int = 64


def measured(value: float | None) -> float:
    """A metric of a simulated split, which holds both classes by construction."""
    if value is None:
        raise ValueError("A simulated split holds no positive or no negative")
    return value


def draw(
    rng: np.random.Generator, problem: Problem, positives: int, negatives: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Positive and negative feature vectors: positives shifted along the first features."""
    mean = np.zeros(problem.features)
    mean[: problem.shifted] = problem.shift / np.sqrt(problem.shifted)
    shifted = rng.normal(size=(positives, problem.features)) + mean
    plain = rng.normal(size=(negatives, problem.features))
    return shifted.astype(np.float32), plain.astype(np.float32)


def scores(model: nn.Module, x: NDArray[np.float32]) -> NDArray[np.float64]:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x)).squeeze(-1)).double().numpy()


def simulate(weight: float, seed: int, problem: Problem = Problem()) -> dict[str, float]:
    """One simulated run with a positive weight and seed; its kept epoch's metrics."""
    rng = np.random.default_rng(seed)
    labelled, _ = draw(rng, problem, problem.positives, 0)
    hidden = int(rng.binomial(problem.marginal, problem.true_prevalence))
    hidden_x, negative_x = draw(rng, problem, hidden, problem.marginal - hidden)
    marginal = np.concatenate([hidden_x, negative_x])
    validation_p, validation_u = draw(
        rng, problem, problem.validation_positives, problem.validation_negatives
    )
    validation = np.concatenate([validation_p, validation_u])
    validation_y = np.r_[np.ones(len(validation_p)), np.zeros(len(validation_u))]
    test_p, test_n = draw(rng, problem, problem.test_positives, problem.test_negatives)
    test = np.concatenate([test_p, test_n])
    test_y = np.r_[np.ones(len(test_p)), np.zeros(len(test_n))]
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(problem.features, problem.hidden),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(problem.hidden, problem.hidden),
        nn.GELU(),
        nn.Linear(problem.hidden, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = NonNegativePULoss(prior=PRIOR, positive_weight=weight)
    targets = torch.tensor(
        [1.0] * problem.batch_positives + [0.0] * problem.batch_marginal, dtype=torch.float32
    )
    labelled_t, marginal_t = torch.from_numpy(labelled), torch.from_numpy(marginal)
    best: tuple[float, int, NDArray[np.float64], NDArray[np.float64]] | None = None
    stopped = corrected = 0
    for epoch in range(1, problem.epochs + 1):
        order = rng.permutation(len(marginal))
        model.train()
        for step in range(problem.steps):
            chosen = rng.integers(0, len(labelled), problem.batch_positives)
            start = (step * problem.batch_marginal) % max(len(marginal) - problem.batch_marginal, 1)
            unlabelled = order[start : start + problem.batch_marginal]
            batch = torch.cat([labelled_t[chosen], marginal_t[unlabelled]])
            trained, objective = loss(model(batch).squeeze(-1), targets)
            corrected += int(trained.item() != objective.item())
            optimizer.zero_grad()
            trained.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        found = measured(average_precision(validation_y, scores(model, validation)))
        if best is None or found > best[0]:
            best = (found, epoch, scores(model, test), scores(model, labelled))
        stopped = epoch
        if epoch - best[1] >= problem.patience:
            break
    assert best is not None
    validation_ap, kept, test_scores, labelled_scores = best
    top = np.argsort(-test_scores, kind="stable")[: max(int(0.01 * len(test_y)), 1)]
    return {
        "validation_average_precision": validation_ap,
        "test_average_precision": measured(average_precision(test_y, test_scores)),
        "test_roc_auc": measured(roc_auc(test_y, test_scores)),
        "test_precision_at_1pct": float(test_y[top].mean()),
        "labelled_positive_mean_score": float(labelled_scores.mean()),
        "test_negative_mean_score": float(test_scores[test_y == 0].mean()),
        "collapsed": float(labelled_scores.mean() < COLLAPSE),
        "kept_epoch": float(kept),
        "stopped_epoch": float(stopped),
        "corrected_steps": float(corrected),
    }


def nnpu_simulation(
    weights: Sequence[float] = WEIGHTS,
    seeds: Sequence[int] = SEEDS,
    problem: Problem = Problem(),
) -> pd.DataFrame:
    """The simulation table: every metric of each positive weight and seed."""
    records: list[tuple[Any, ...]] = []
    for weight in weights:
        for seed in seeds:
            for metric, value in simulate(weight, seed, problem).items():
                records.append((float(weight), int(seed), metric, value))
    return pd.DataFrame(records, columns=list(COLUMNS))
