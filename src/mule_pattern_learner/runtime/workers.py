"""Worker threads that build items ahead of their consumer."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
import queue
import threading
from types import TracebackType
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")
MAX_PREFETCH = 8
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
