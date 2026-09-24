"""Positive oversampling, label-blind marginal coverage and the batch schedule for nnPU."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import queue
import threading
from types import TracebackType
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")
MAX_PREFETCH = 8


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
        raise ValueError("evaluation_unlabeled_limit must be positive")
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
    value = hashlib.sha256(f"temporal_live_step:{seed}:{epoch}:{step}".encode()).digest()
    return int.from_bytes(value[:8], "big") >> 1


@dataclass(frozen=True)
class PUSample:
    """One training cutoff: its marginal reservoir rows and its visible positives."""

    date: str
    marginal: np.ndarray
    observed: np.ndarray
    positives: np.ndarray


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


_END = object()


class BatchPrefetcher(Generic[T, R]):
    """Build items ahead of the consumer on worker threads and yield results in input order.

    At most ``depth`` items are queued or being built, so host memory stays bounded.
    A build error is raised when the consumer reaches that item. ``depth=0`` builds
    inline on the consumer thread.

    Shutdown depends on how iteration ends. When it finishes normally (or the
    consumer breaks out inside ``with``) ``close()`` joins the workers. When an
    exception or KeyboardInterrupt ends it (in a build or in the consumer), queued
    work is cancelled and the error propagates at once: builds already running,
    typically blocked in a REST call with retries, are not awaited. The workers are
    daemon threads, so they finish or die with the process on their own.
    """

    def __init__(
        self,
        build: Callable[[T], R],
        items: Iterable[T],
        *,
        depth: int = 2,
        name: str = "temporal-batch",
    ) -> None:
        if not 0 <= depth <= MAX_PREFETCH:
            raise ValueError(f"prefetch_batches must be in [0,{MAX_PREFETCH}]")
        self.build, self.depth, self.name = build, depth, name
        self.items = iter(items)
        self.tasks: queue.SimpleQueue[tuple[Future[R], T] | None] = queue.SimpleQueue()
        self.workers: list[threading.Thread] = []
        self.pending: deque[Future[R]] = deque()
        self.stopped = False

    def _work(self) -> None:
        while True:
            task = self.tasks.get()
            if task is None:
                return
            future, item = task
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = self.build(item)
            except BaseException as error:  # delivered to the consumer by result()
                future.set_exception(error)
            else:
                future.set_result(result)

    def _start(self) -> None:
        for index in range(self.depth):
            worker = threading.Thread(target=self._work, name=f"{self.name}_{index}", daemon=True)
            worker.start()
            self.workers.append(worker)

    def _fill(self) -> None:
        while not self.stopped and len(self.pending) < self.depth:
            item = next(self.items, _END)
            if item is _END:
                return
            future: Future[R] = Future()
            self.pending.append(future)
            self.tasks.put((future, item))  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[R]:
        if not self.depth:
            for item in self.items:
                yield self.build(item)
            return
        if not self.workers and not self.stopped:
            self._start()
        try:
            self._fill()
            while self.pending:
                result = self.pending.popleft().result()
                self._fill()
                yield result
        except BaseException:
            # A build error, a consumer error (seen here as GeneratorExit when the
            # generator is discarded) or Ctrl-C: never wait for running builds.
            self.cancel()
            raise
        self.close()

    def cancel(self) -> None:
        """Cancel queued work and stop the workers without waiting for running builds."""
        if self.stopped:
            return
        self.stopped = True
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        for _ in self.workers:
            self.tasks.put(None)

    def close(self, *, wait: bool = True) -> None:
        """Cancel queued work; with ``wait`` also join the workers (their running builds)."""
        self.cancel()
        if wait:
            for worker in self.workers:
                worker.join()

    def __enter__(self) -> BatchPrefetcher[T, R]:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(wait=error is None)
