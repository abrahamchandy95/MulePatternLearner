"""The one worker pool: daemon threads for context requests and batch prefetch.

DaemonPool runs submitted calls on at most a fixed number of daemon threads. The
streaming context source sends its REST requests through one, and BatchPrefetcher
builds batches ahead of their consumer on one.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
import queue
import threading
from types import TracebackType
from typing import Any, Generic, TypeVar
import weakref

from ..contract.bounds import PREFETCH_BATCHES

T = TypeVar("T")
R = TypeVar("R")
_END = object()

_Work = tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]


def _pool_worker(work: queue.SimpleQueue[_Work | None]) -> None:
    """Run submitted calls until the None sentinel; never holds a reference to the pool."""
    while True:
        item = work.get()
        if item is None:
            return
        future, function, args = item
        if not future.set_running_or_notify_cancel():
            continue
        try:
            result = function(*args)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        del item, future, function, args


def _stop_workers(work: queue.SimpleQueue[_Work | None], threads: list[threading.Thread]) -> None:
    for _ in threads:
        work.put(None)


class DaemonPool:
    """At most `workers` daemon threads running submitted calls (a ThreadPoolExecutor subset).

    ThreadPoolExecutor joins its workers when the interpreter exits, so a request
    abandoned after an error or Ctrl-C (a REST retry chain may last max_outage_s)
    would still hold up the exiting process. These workers are daemon threads:
    `shutdown(wait=False, cancel_futures=True)` cancels queued calls and returns at
    once, and the process may exit while a request is still in flight. Workers
    start on demand and stop when the pool is shut down or garbage collected.
    """

    def __init__(self, workers: int, name: str) -> None:
        self._workers, self._name = workers, name
        self._queue: queue.SimpleQueue[_Work | None] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._release = weakref.finalize(self, _stop_workers, self._queue, self._threads)

    def submit(self, function: Callable[..., T], /, *args: Any) -> Future[T]:
        future: Future[T] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.put((future, function, args))
            if len(self._threads) < self._workers:
                thread = threading.Thread(
                    target=_pool_worker,
                    args=(self._queue,),
                    name=f"{self._name}_{len(self._threads)}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not None:
                            item[0].cancel()
                self._release()
            threads = list(self._threads)
        if wait:
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join()


class BatchPrefetcher(Generic[T, R]):
    """Build items ahead of the consumer on a DaemonPool and yield results in input order.

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
        if not PREFETCH_BATCHES.holds(depth):
            raise ValueError(
                f"prefetch_batches must be in [{PREFETCH_BATCHES.low},{PREFETCH_BATCHES.high}]"
            )
        self.build, self.depth = build, depth
        self.items = iter(items)
        self.pool = DaemonPool(depth, name)
        self.pending: deque[Future[R]] = deque()
        self.stopped = False

    def _fill(self) -> None:
        while not self.stopped and len(self.pending) < self.depth:
            item = next(self.items, _END)
            if item is _END:
                return
            self.pending.append(self.pool.submit(self.build, item))

    def __iter__(self) -> Iterator[R]:
        if not self.depth:
            for item in self.items:
                yield self.build(item)
            return
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
        self.pool.shutdown(wait=False, cancel_futures=True)

    def close(self, *, wait: bool = True) -> None:
        """Cancel queued work; with ``wait`` also join the workers (their running builds)."""
        self.cancel()
        if wait:
            self.pool.shutdown(wait=True)

    def __enter__(self) -> BatchPrefetcher[T, R]:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(wait=error is None)
