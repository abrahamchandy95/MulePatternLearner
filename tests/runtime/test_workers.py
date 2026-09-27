"""The worker pool and the batch prefetcher: order, bounded work, errors and shutdown."""

from __future__ import annotations

import threading
import time

import pytest

from mule_pattern_learner.runtime.workers import BatchPrefetcher, DaemonPool


def _prefetch_threads() -> int:
    return sum(t.name.startswith("temporal-batch") for t in threading.enumerate())


def _wait_for_no_prefetch_threads(timeout: float = 5.0) -> int:
    """Workers abandoned after an error finish on their own; wait for them."""
    deadline = time.monotonic() + timeout
    while _prefetch_threads() and time.monotonic() < deadline:
        time.sleep(0.01)
    return _prefetch_threads()


def test_prefetcher_preserves_order_and_bounds_work_in_flight() -> None:
    lock, active, peak, pulled = threading.Lock(), [0], [0], []

    def items():
        for value in range(30):
            pulled.append(value)
            yield value

    def build(value: int) -> int:
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep((value * 7 % 5) / 1000)
        with lock:
            active[0] -= 1
        return value * value

    results = []
    with BatchPrefetcher(build, items(), depth=3) as batches:
        for value in batches:
            # Lookahead never exceeds the prefetch depth.
            assert len(pulled) <= len(results) + 1 + 3
            results.append(value)
    assert results == [v * v for v in range(30)]
    assert peak[0] <= 3
    assert _prefetch_threads() == 0


def test_prefetcher_raises_at_the_failing_item_and_shuts_down() -> None:
    def build(value: int) -> int:
        if value == 4:
            raise KeyError("boom")
        return value

    seen = []
    with pytest.raises(KeyError, match="boom"):
        with BatchPrefetcher(build, range(20), depth=2) as batches:
            for value in batches:
                seen.append(value)
    assert seen == [0, 1, 2, 3]
    assert _wait_for_no_prefetch_threads() == 0


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_prefetcher_error_does_not_wait_for_running_builds(error: type[BaseException]) -> None:
    release = threading.Event()

    def build(value: int) -> int:
        if value:
            release.wait(30)  # an in-flight REST call with retries
        return value

    started = time.perf_counter()
    try:
        with pytest.raises(error):
            with BatchPrefetcher(build, range(10), depth=3) as batches:
                for _ in batches:
                    raise error("consumer failure")
        assert time.perf_counter() - started < 5
        assert _prefetch_threads() > 0, "running builds are abandoned, not joined"
    finally:
        release.set()
    assert _wait_for_no_prefetch_threads() == 0


def test_prefetcher_build_error_does_not_wait_for_other_running_builds() -> None:
    release = threading.Event()

    def build(value: int) -> int:
        if value == 1:
            raise KeyError("boom")
        if value > 1:
            release.wait(30)
        return value

    started = time.perf_counter()
    try:
        with pytest.raises(KeyError, match="boom"):
            with BatchPrefetcher(build, range(10), depth=2) as batches:
                for _ in batches:
                    pass
        assert time.perf_counter() - started < 5
    finally:
        release.set()
    assert _wait_for_no_prefetch_threads() == 0


def test_prefetcher_cancels_queued_work_on_early_exit_and_runs_inline() -> None:
    built = []

    def build(value: int) -> int:
        built.append(value)
        time.sleep(0.01)
        return value

    with BatchPrefetcher(build, range(100), depth=2) as batches:
        for value in batches:
            if value == 1:
                break
    assert len(built) <= 5 and _prefetch_threads() == 0
    inline = list(BatchPrefetcher(lambda v: (v, threading.current_thread()), range(3), depth=0))
    assert inline == [(v, threading.current_thread()) for v in range(3)]
    with pytest.raises(ValueError):
        BatchPrefetcher(build, range(1), depth=9)


def test_pool_runs_on_bounded_daemon_threads_and_cancels_queued_calls() -> None:
    entered, release = threading.Semaphore(0), threading.Event()

    def call(value: int) -> int:
        entered.release()
        release.wait(30)
        return value

    pool = DaemonPool(2, "test-pool")
    futures = [pool.submit(call, value) for value in range(5)]
    assert entered.acquire(timeout=5) and entered.acquire(timeout=5)
    workers = [t for t in threading.enumerate() if t.name.startswith("test-pool")]
    assert len(workers) == 2 and all(t.daemon for t in workers)
    started = time.perf_counter()
    pool.shutdown(wait=False, cancel_futures=True)
    assert time.perf_counter() - started < 5
    with pytest.raises(RuntimeError, match="after shutdown"):
        pool.submit(print)
    release.set()
    pool.shutdown()
    assert [f.result() for f in futures[:2]] == [0, 1]
    assert all(f.cancelled() for f in futures[2:])
    assert not any(t.is_alive() for t in workers)
