"""Reveal the known mules in the graph, through the Account labels.

A fresh PhantomLedger load masks every mule, so training would have no positives.
Preparing a dataset calls the reveal query (gsql/queries/label_reveal.gsql), which
simulates when a bank would have discovered each mule (victim reports, network tracing,
monitoring; see docs/explanation/label-reveal.md) and reveals every mule discovered
before its split's cutoff. Training then reads only the revealed positives and their
discovery clocks; ground truth stays in the graph for the oracle audit.
"""

from __future__ import annotations

from typing import Any

from ..config import ScopeConfig, SplitDates
from ..contract.clock import timestamp
from ..contract.server import REVEAL_QUERY
from ..runtime.progress import emit
from .executor import QueryExecutor, merged_rows

# The reveal's statuses for a graph that has known labels: written by the configured
# reveal, or by another one (another salt, a capped reveal, or code that recorded no
# budget).
CURRENT, DIFFERENT = "already_revealed", "revealed_differently"
# How a `reveal` event names the configured reveal's labels revealed again for a split
# the run created.
NEW_SPLIT = "revealed for the new split"


def reveal_parameters(scope: ScopeConfig, dates: SplitDates, *, apply: bool) -> dict[str, Any]:
    """Query parameters: each split's latest cutoff and the salt."""
    return {
        "train_cutoff_ms": max(timestamp(d) for d in dates.train),
        "validation_cutoff_ms": max(timestamp(d) for d in dates.validation),
        "test_cutoff_ms": max(timestamp(d) for d in dates.test),
        "salt": scope.reveal_salt,
        "apply": apply,
    }


def run_reveal(
    executor: QueryExecutor, scope: ScopeConfig, dates: SplitDates, *, force: bool = False
) -> dict[str, Any]:
    """One run of the reveal that writes the labels (unless the graph has them and not force)."""
    params = reveal_parameters(scope, dates, apply=True)
    if force:
        params["force"] = True
    return merged_rows(executor.run(REVEAL_QUERY, params, timeout_s=3600.0, attempts=1))


def verify_labels(executor: QueryExecutor, scope: ScopeConfig, dates: SplitDates) -> None:
    """Refuse a graph whose labels are not those the reveal of these settings writes.

    A dry run of the reveal answers from the label sources alone when the graph has
    known labels. A dataset reads its labels once, and the audits read which mules were
    revealed from the graph, so a run of a dataset whose reveal was replaced would
    measure one set of labels against another.
    """
    params = reveal_parameters(scope, dates, apply=False)
    result = merged_rows(executor.run(REVEAL_QUERY, params, timeout_s=3600.0, attempts=1))
    status = result.get("status")
    if status == CURRENT:
        return
    if status == DIFFERENT:
        raise ValueError(
            f"The graph's labels are not the reveal this dataset read "
            f"({result.get('reveal_record')}): {result.get('other_reveal')} mules have "
            "another reveal's label. Another dataset's preparation revealed again; prepare "
            "this one's labels with its settings, or train a dataset of the current ones"
        )
    raise ValueError(f"The graph has no revealed labels for this dataset: {result}")


def ensure_revealed_labels(
    executor: QueryExecutor, scope: ScopeConfig, dates: SplitDates, *, new_split: bool = False
) -> dict[str, Any]:
    """Make the graph's labels the configured reveal's.

    A graph without known labels is revealed; one whose labels the configured reveal
    wrote is kept; one whose labels another reveal wrote (another salt, a capped reveal,
    or code that recorded no budget) is revealed again, which a `reveal` event records
    with the count of mules whose label differed. Datasets prepared from the former
    labels no longer match the graph, and every run of them is refused
    (tigergraph.provenance). Prints the mules revealed per split, or how many labels the
    graph had revealed.

    After the run created the split (``new_split``), labels the configured reveal wrote
    are revealed again with force, since they followed the partitions of the split
    before; an identical split gets identical labels. Labels another source wrote, on a
    graph with no internal mule, are kept as always.
    """
    result = run_reveal(executor, scope, dates)
    replaced: dict[str, Any] = {}
    status = result.get("status")
    if status == DIFFERENT:
        replaced = {
            "other_reveal": result.get("other_reveal"),
            "reveal_record": result.get("reveal_record"),
        }
        result = run_reveal(executor, scope, dates, force=True)
    renewed = new_split and status == CURRENT and bool(result.get("mules"))
    if renewed:
        result = run_reveal(executor, scope, dates, force=True)
    status = result.get("status")
    if status not in ("ok", CURRENT):
        raise ValueError(f"TigerGraph rejected the label reveal: {result}")
    if status == CURRENT:
        summary = {"labels": "already revealed", "revealed_labels": result.get("revealed_labels")}
    else:
        labels = "revealed again" if replaced else NEW_SPLIT if renewed else "revealed now"
        summary = {"labels": labels, "revealed": result.get("revealed")}
        if replaced:
            summary["replaced"] = replaced
    emit({"event": "reveal", **summary})
    return summary
