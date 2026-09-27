"""Positive oversampling, label-blind marginal coverage, the nnPU schedule, evaluation samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..contract.fingerprints import hash64
from ..contract.salts import STEP_SALT


def pu_batches(
    indices: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    *,
    max_steps: int | None = None,
    positive_indices: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Oversample known P; visit each training marginal account once per epoch.

    Like the existing seed sampler, positives are sampled with replacement and
    the large pool is shuffled without replacement. Here the second draw is the
    full marginal (including known P), as required by the standard nnPU prior.
    Hidden truth cannot influence either pool. max_steps bounds epoch work.
    """
    if batch_size < 2 or (max_steps is not None and max_steps < 1):
        raise ValueError("nnPU needs batch_size >= 2 and a positive optional step limit")
    positives = indices[observed[indices]] if positive_indices is None else positive_indices
    if not observed[positives].all():
        raise ValueError("Positive pool contains an unobserved label")
    if not len(positives) or not len(indices):
        raise ValueError("nnPU requires observed positives and a training marginal")
    positive_count = max(1, batch_size // 4)
    marginal_count = batch_size - positive_count
    marginal = rng.permutation(indices)
    for step, start in enumerate(range(0, len(marginal), marginal_count)):
        if max_steps is not None and step >= max_steps:
            break
        yield (
            rng.choice(positives, size=positive_count, replace=True),
            marginal[start : start + marginal_count],
        )


def evaluation_indices(
    indices: np.ndarray,
    observed: np.ndarray,
    *,
    limit: int | None,
    seed: int,
    marginal: np.ndarray | None = None,
) -> np.ndarray:
    """Every visible observed positive plus a uniform unlabeled sample; no truth access.

    The unlabeled sample is drawn only from marginal reservoir rows (``marginal`` is a
    boolean mask over all rows). Positive-pool rows outside the reservoir were kept
    because they carry a label, possibly one revealed after this cutoff, so drawing
    them as "unlabeled" would enrich the sample with hidden positives.
    """
    if limit is not None and limit < 1:
        raise ValueError("training.proxy_unlabeled_limit must be positive")
    visible = observed[indices]
    known = indices[visible]
    unlabeled = indices[~visible]
    if marginal is not None:
        unlabeled = unlabeled[marginal[unlabeled]]
    if limit is not None and len(unlabeled) > limit:
        unlabeled = np.random.default_rng(seed).choice(unlabeled, size=limit, replace=False)
    return np.sort(np.r_[known, unlabeled])


def step_seed(seed: int, epoch: int, step: int) -> int:
    """Stable nonnegative 63-bit seed for one training step, identical on every machine."""
    return hash64(STEP_SALT, seed, epoch, step) >> 1


@dataclass(frozen=True)
class PUSample:
    """One training cutoff: its marginal reservoir rows and its visible positives."""

    date: str
    marginal: np.ndarray
    observed: np.ndarray
    positives: np.ndarray


@dataclass(frozen=True)
class EvaluationSample:
    """One validation or test cutoff: its chosen rows and their observed labels."""

    date: str
    indices: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class TrainingStep:
    epoch: int
    step: int
    date: str
    positives: np.ndarray
    marginal: np.ndarray
    seed: int

    @property
    def indices(self) -> np.ndarray:
        """Row indices of the batch; the first len(positives) rows are labeled positive."""
        return np.r_[self.positives, self.marginal]


def epoch_schedule(
    samples: Iterable[PUSample],
    rng: np.random.Generator,
    batch_size: int,
    *,
    epoch: int,
    seed: int,
    max_steps: int | None = None,
) -> list[TrainingStep]:
    """Draw one epoch of PU batches up front, in the order the trainer consumes them.

    The draws equal lazy consumption of pu_batches with the same generator, so the
    schedule depends only on the generator state at the start of the epoch. Every
    step carries a stable seed derived from (seed, epoch, step).
    """
    steps: list[TrainingStep] = []
    for sample in samples:
        for positives, marginal in pu_batches(
            sample.marginal,
            sample.observed,
            rng,
            batch_size,
            max_steps=max_steps,
            positive_indices=sample.positives,
        ):
            index = len(steps)
            steps.append(
                TrainingStep(
                    epoch, index, sample.date, positives, marginal, step_seed(seed, epoch, index)
                )
            )
    return steps
