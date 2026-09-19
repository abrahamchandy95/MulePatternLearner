"""Positive oversampling and label-blind marginal coverage for nnPU."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


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
    indices: np.ndarray, observed: np.ndarray, *, limit: int | None, seed: int
) -> np.ndarray:
    """Fixed observed-P plus a uniform unlabeled evaluation sample; no truth access."""
    if limit is None:
        return indices
    if limit < 1:
        raise ValueError("evaluation_unlabeled_limit must be positive")
    known = indices[observed[indices]]
    unlabeled = indices[~observed[indices]]
    if len(unlabeled) > limit:
        unlabeled = np.random.default_rng(seed).choice(unlabeled, size=limit, replace=False)
    return np.sort(np.r_[known, unlabeled])
