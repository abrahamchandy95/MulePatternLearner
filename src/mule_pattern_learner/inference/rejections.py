"""Rejected roots: the limit that fails a set, the counts that check it, and their report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.contexts import ContextReader

# The counts of a set's rejected roots (rejection_counts), in the order runs report them.
REJECTION_KEYS = ("requested", "rejected", "positive", "unlabeled")


def exceeds_rejection_limit(rejected: int, positives: int, total: int, limit: float) -> bool:
    """Whether rejected roots censor a set: any observed positive, or over limit * total."""
    return bool(positives) or rejected > limit * total


def rejection_counts(labels: np.ndarray, accepted: np.ndarray) -> dict[str, int]:
    """Requested and rejected roots, split into observed positives and unlabeled."""
    lost = ~accepted
    positives = int(labels[lost].astype(bool).sum())
    rejected = int(lost.sum())
    return {
        "requested": len(labels),
        "rejected": rejected,
        "positive": positives,
        "unlabeled": rejected - positives,
    }


def check_split_rejections(
    split: str,
    labels: np.ndarray,
    accepted: np.ndarray,
    limit: float,
    statuses: Mapping[str, int],
) -> None:
    """Fail an evaluation whose rejected roots would censor its metrics.

    ``labels`` are the split's observed labels and ``statuses`` the source's rejections
    by status, which the message reports. Validation must also keep both observed
    classes after its rejections.
    """
    counts = rejection_counts(labels, accepted)
    rejected, positives = counts["rejected"], counts["positive"]
    problem = None
    if exceeds_rejection_limit(rejected, positives, len(labels), limit):
        problem = (
            f"{positives} observed positives among them"
            if positives
            else f"above max_rejected_root_fraction={limit}"
        )
    elif split == "validation" and len(np.unique(labels[accepted])) != 2:
        problem = "validation no longer has both observed classes"
    if problem is not None:
        raise ValueError(
            f"{split}: TigerGraph rejected {rejected} of {len(labels)} roots ({problem}); "
            f"statuses {dict(statuses)}"
        )


class TrainingRejections:
    """Rejected training roots of the whole run and of the current epoch, under a limit."""

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.run: Counter[str] = Counter()
        self.epoch: Counter[str] = Counter()

    def start_epoch(self) -> None:
        self.epoch = Counter()

    def count(
        self,
        epoch: int,
        labels: np.ndarray,
        accepted: np.ndarray,
        requested: int,
        statuses: Callable[[], Mapping[str, int]],
    ) -> None:
        """Count a step's rejected roots; fail past the limit or on an observed positive.

        ``labels`` are the step's observed labels. The limit applies to the whole epoch
        (``requested`` roots, ``epoch`` counting from 0). The count only grows, so
        failing as soon as it is crossed equals failing at the epoch end. ``statuses``
        gives the source's rejections by status for the message.
        """
        self.run["requested"] += len(accepted)
        if accepted.all():
            return
        counts = rejection_counts(labels, accepted)
        del counts["requested"]
        self.epoch.update(counts)
        self.run.update(counts)
        rejected, positives = self.epoch["rejected"], self.epoch["positive"]
        if exceeds_rejection_limit(rejected, positives, requested, self.limit):
            raise ValueError(
                f"Epoch {epoch + 1}: TigerGraph rejected {rejected} of {requested} training "
                f"roots so far ({positives} observed positives; max_rejected_root_fraction="
                f"{self.limit}); statuses {dict(statuses())}"
            )

    def totals(self) -> dict[str, int]:
        """The run's counts (rejection_counts) over every segment."""
        return {key: int(self.run[key]) for key in REJECTION_KEYS}

    def saved(self) -> dict[str, dict[str, int]]:
        # Plain dicts: torch.load(weights_only=True) refuses a Counter.
        return {"run": dict(self.run), "epoch": dict(self.epoch)}

    def restore(self, saved: Mapping[str, Mapping[str, int]]) -> None:
        self.run, self.epoch = Counter(saved["run"]), Counter(saved["epoch"])


@dataclass(frozen=True)
class SourceRejections:
    """A context source's rejection counters at one moment: by status, and by hop."""

    by_status: Counter[str]
    by_hop: dict[int, Counter[str]]

    @classmethod
    def of(cls, contexts: ContextReader) -> SourceRejections:
        """A copy of the source's counters, which its later fetches leave as they are."""
        by_hop = {hop: Counter(counts) for hop, counts in contexts.rejections_by_hop.items()}
        return cls(Counter(contexts.rejections), by_hop)

    def since(self, earlier: SourceRejections) -> SourceRejections:
        """The rejections served after ``earlier``, a copy of the same source's counters."""
        by_hop = {
            hop: counts - earlier.by_hop.get(hop, Counter()) for hop, counts in self.by_hop.items()
        }
        return SourceRejections(self.by_status - earlier.by_status, by_hop)


def rejection_summary(
    contexts: ContextReader,
    rejected_roots: int,
    totals: Counter[str],
    *,
    since: SourceRejections | None = None,
) -> dict[str, Any]:
    """Root and child rejections reported separately.

    ``rejected`` counts the roots that were not scored. ``rejected_children`` counts
    the child contexts masked out of scored batches. ``rejection_events_by_status``
    is the source's raw counter: every rejected row served by a fetch at either hop,
    cache replays included, so it is not a count of accounts. The source's per-hop
    counts (``rejections_by_hop``) give the root and child statuses. The counters are
    the source's whole life unless ``since`` (SourceRejections.of, taken before the set
    was scored) leaves out what it served earlier, so that each set scored on a shared
    source reports its own.
    """
    served = SourceRejections.of(contexts)
    if since is not None:
        served = served.since(since)
    return {
        "rejected": rejected_roots,
        "rejected_roots_by_status": dict(served.by_hop.get(1, {})),
        "rejected_children": int(totals["rejected_children"]),
        "rejected_children_by_status": dict(served.by_hop.get(2, {})),
        "stub_children": int(totals["stub_children"]),
        "rejection_events_by_status": dict(served.by_status),
    }
