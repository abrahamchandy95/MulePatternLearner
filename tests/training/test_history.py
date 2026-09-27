"""The log interval's device sums and the progress totals."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from mule_pattern_learner.data.contexts import ContextCounts
from mule_pattern_learner.training.history import LogInterval, Progress


def test_log_interval_counts_corrections_and_non_finite_losses() -> None:
    interval = LogInterval(torch.device("cpu"))
    interval.add(torch.tensor(2.0), torch.tensor(2.0))
    interval.add(torch.tensor(1.0), torch.tensor(-0.5))
    loss, risk, corrections, finite = interval.totals()
    assert (loss, risk, corrections, finite) == (3.0, 1.5, 1.0, True)
    record = interval.record(loss, risk, corrections, interval.started + 4.0)
    assert (record["loss"], record["objective"], record["corrected_steps"]) == (1.5, 0.75, 1)
    assert record["seconds_per_step"] == 2.0
    interval.add(torch.tensor(float("nan")), torch.tensor(0.0))
    assert not interval.totals()[3]
    interval.start(0.0)
    assert interval.totals() == (0.0, 0.0, 0.0, True) and interval.steps == 0


def test_progress_sums_integer_statistics_across_segments() -> None:
    counts = ContextCounts(requested=7, cache_hits=2, seen={1, 2, 3})
    rejections = Counter({"missing_entity": 1})
    source = SimpleNamespace(query_calls=3, rejections=rejections, counts=counts)
    progress = Progress(0.0, cast(Any, source), "torch")
    progress.base_calls, progress.base_rejections = 2, Counter({"missing_entity": 4})
    progress.base_contexts = Counter({"requested": 10, "cache_hits": 1})
    progress.add({"roots": np.int64(4), "sampler_backend": "torch", "stub_children": 1})
    progress.add({"roots": 2, "flag": True})
    assert progress.totals == Counter({"roots": 6, "stub_children": 1})
    assert progress.calls() == 5 and progress.rejections() == {"missing_entity": 5}
    # A resumed source holds every segment's distinct contexts itself.
    assert progress.contexts() == {"requested": 17, "distinct": 3, "cache_hits": 3}
