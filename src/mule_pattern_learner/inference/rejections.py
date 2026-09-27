"""Rejected roots: the limit that fails a set, and their report."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..data.contexts import ContextSource


def exceeds_rejection_limit(rejected: int, positives: int, total: int, limit: float) -> bool:
    """Whether rejected roots censor a set: any observed positive, or over limit * total."""
    return bool(positives) or rejected > limit * total


def rejection_summary(
    source: ContextSource, rejected_roots: int, totals: Counter[str]
) -> dict[str, Any]:
    """Root and child rejections reported separately.

    ``rejected`` counts the roots that were not scored. ``rejected_children`` counts
    the child contexts masked out of scored batches. ``rejection_events_by_status``
    is the source's raw counter: every rejected row served by a fetch at either hop,
    cache replays included, so it is not a count of accounts. The source's per-hop
    counts (``rejections_by_hop``) give the root and child statuses.
    """
    by_hop = source.rejections_by_hop
    return {
        "rejected": rejected_roots,
        "rejected_roots_by_status": dict(by_hop.get(1, {})),
        "rejected_children": int(totals["rejected_children"]),
        "rejected_children_by_status": dict(by_hop.get(2, {})),
        "stub_children": int(totals["stub_children"]),
        "rejection_events_by_status": dict(source.rejections),
    }
