"""The run's history: its events, log intervals and epochs, as they happen.

Progress appends every record to the run's events.jsonl and echoes selected records
to stdout. It also sums the batch statistics and counts the REST calls, rejections and
contexts of every segment of a resumed run. LogInterval keeps one log interval's losses on the
device, so the host reads them once per interval.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..batching.assemble import batch_counts
from ..data.contexts import ContextSource


def plain(stats: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of batch statistics (numpy scalars become Python numbers)."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in stats.items()}


class Progress:
    """Append JSON lines to the run's events.jsonl; echo selected records to stdout."""

    def __init__(self, started: float, store: ContextSource, backend: str) -> None:
        self.started, self.store = started, store
        self.path: Path | None = None
        self.totals: Counter[str] = Counter()
        # The backend resolved once for the run (batch statistics name the same one).
        self.backend = backend
        # Counts of earlier segments of a resumed run; the source counts this one. The
        # source's own set of distinct contexts is restored instead (ContextCounts.seen).
        self.base_calls = 0
        self.base_rejections: Counter[str] = Counter()
        self.base_contexts: Counter[str] = Counter()

    def add(self, stats: dict[str, Any]) -> None:
        self.totals.update(batch_counts(stats))

    def calls(self) -> int:
        """REST calls of every segment of the run."""
        return self.base_calls + self.store.query_calls

    def rejections(self) -> dict[str, int]:
        """Rejected rows served by the source (both hops) in every segment, by status."""
        return dict(self.base_rejections + Counter(self.store.rejections))

    def contexts(self) -> dict[str, int]:
        """Contexts requested, distinct and served from memory in every segment."""
        counts = self.store.counts
        return {
            "requested": self.base_contexts["requested"] + counts.requested,
            "distinct": counts.distinct,
            "cache_hits": self.base_contexts["cache_hits"] + counts.cache_hits,
        }

    def emit(self, record: dict[str, Any], *, echo: bool = True) -> dict[str, Any]:
        contexts = self.contexts()
        record = {
            **record,
            "query_calls": self.calls(),
            "contexts_requested": contexts["requested"],
            "contexts_distinct": contexts["distinct"],
            "cache_hits": contexts["cache_hits"],
            "rejections": self.rejections(),
            "stub_children": int(self.totals["stub_children"]),
            "rejected_children": int(self.totals["rejected_children"]),
            "sampler_backend": self.backend,
            "elapsed_seconds": round(time.perf_counter() - self.started, 3),
        }
        line = json.dumps(record, allow_nan=False)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(line + "\n")
        if echo:
            print(line, flush=True)
        return record


class LogInterval:
    """One log interval of training: its losses on the device, its steps and its timing."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.start(time.perf_counter())

    def start(self, now: float) -> None:
        """Begin a new interval at ``now``."""
        self.loss = torch.zeros((), device=self.device)
        self.objective = torch.zeros((), device=self.device)
        self.corrected = torch.zeros((), device=self.device)
        self.finite = torch.ones((), dtype=torch.bool, device=self.device)
        self.started, self.waited, self.steps = now, 0.0, 0

    def add(self, value: torch.Tensor, objective: torch.Tensor) -> None:
        """Count one step's loss and unclamped risk, without a host read."""
        self.loss += value
        self.objective += objective
        self.corrected += (value != objective).to(self.corrected.dtype)
        self.finite &= torch.isfinite(value)
        self.steps += 1

    def totals(self) -> tuple[float, float, float, bool]:
        """Summed loss and risk, corrected steps, and whether every loss was finite.

        Reading them is the interval's one device-to-host copy.
        """
        loss, risk, corrections, finite = torch.stack(
            (self.loss, self.objective, self.corrected, self.finite.to(self.loss.dtype))
        ).tolist()
        return loss, risk, corrections, bool(finite)

    def record(self, loss: float, risk: float, corrections: float, now: float) -> dict[str, Any]:
        """The interval's means per trained step, corrected steps and timing."""
        trained = max(self.steps, 1)
        return {
            "loss": loss / trained,
            # The unclamped risk and the steps whose nnPU correction fired.
            "objective": risk / trained,
            "corrected_steps": int(corrections),
            "seconds_per_step": (now - self.started) / trained,
            "batch_wait_seconds": self.waited / trained,
        }


def epoch_record(
    epoch: int, loss_sum: torch.Tensor, steps: int, validation: dict[str, Any]
) -> dict[str, Any]:
    """The history entry of a finished epoch (``epoch`` counts from 1)."""
    return {
        "epoch": epoch,
        "loss": float(loss_sum.item() / steps),
        "steps": steps,
        "validation_proxy_ap": validation["average_precision"],
        "validation_proxy_roc_auc": validation["roc_auc"],
    }
