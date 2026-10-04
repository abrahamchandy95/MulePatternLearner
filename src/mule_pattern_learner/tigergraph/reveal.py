"""Reveal the known mules in the graph, through the Account label contract.

A fresh PhantomLedger load masks every mule, so training would have no positives.
Preparing a dataset calls the reveal query (gsql/queries/label_reveal.gsql), which
simulates when a bank would have discovered each mule (victim reports, network tracing,
monitoring; see docs/explanation/label-reveal.md) and reveals the mules discovered
before each split's cutoff: every one, or at most `scope.reveal_per_split` per split.
Training then reads only the revealed positives and their discovery clocks; ground
truth stays in the graph for the oracle audit.
"""

from __future__ import annotations

from typing import Any

from ..config import ScopeConfig, SplitDates
from ..contract.bounds import REVEAL_PER_SPLIT
from ..contract.clock import timestamp
from ..contract.graph_schema import SPLIT_PHASE
from ..contract.server import REVEAL_QUERY
from ..runtime.progress import emit
from .executor import QueryExecutor, merged_rows
from .labels import validate_supervision

# The reveal's statuses for a graph that has known labels: written by the configured
# reveal, or by another one (other settings, or code that recorded no budget).
CURRENT, DIFFERENT = "already_revealed", "revealed_differently"


def reveal_budget(scope: ScopeConfig) -> int:
    """The most mules the reveal makes known per split: the cap, or the bound for none."""
    if scope.reveal_per_split is None:
        return REVEAL_PER_SPLIT.high
    return scope.reveal_per_split


def reveal_parameters(scope: ScopeConfig, dates: SplitDates, *, apply: bool) -> dict[str, Any]:
    """Query parameters: the scope, each split's latest cutoff, the budget and the salt."""
    return {
        "scope_id": scope.id,
        "train_cutoff_ms": max(timestamp(d) for d in dates.train),
        "validation_cutoff_ms": max(timestamp(d) for d in dates.validation),
        "test_cutoff_ms": max(timestamp(d) for d in dates.test),
        "budget": reveal_budget(scope),
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
    executor: QueryExecutor, scope: ScopeConfig, dates: SplitDates
) -> dict[str, Any]:
    """Make the graph's labels the configured reveal's, then check the label contract.

    A graph without known labels is revealed; one whose labels the configured reveal
    wrote is kept; one whose labels another reveal wrote (other settings, or code that
    recorded no budget) is revealed again, which a `reveal` event records with the count
    of mules whose label differed. Datasets prepared from the former labels no longer
    match the graph, and every run of them is refused (tigergraph.provenance).
    Without a cap, a split with more discovered mules than the reveal can take is an
    error, never a silent cap. Prints the reveal plan (mules, eligible and revealed per
    split, and channels) or the existing label counts.
    """
    result = run_reveal(executor, scope, dates)
    replaced: dict[str, Any] = {}
    if result.get("status") == DIFFERENT:
        replaced = {
            "other_reveal": result.get("other_reveal"),
            "reveal_record": result.get("reveal_record"),
        }
        result = run_reveal(executor, scope, dates, force=True)
    status = result.get("status")
    if status not in ("ok", CURRENT):
        raise ValueError(f"TigerGraph rejected the label reveal: {result}")
    if status == CURRENT:
        summary = {
            "labels": "already revealed",
            "known_labels": result.get("known_labels"),
            "revealed_labels": result.get("revealed_labels"),
        }
    else:
        summary = {
            "labels": "revealed again" if replaced else "revealed now",
            **{
                key: result.get(key)
                for key in (
                    "version",
                    "budget",
                    "reveal_record",
                    "mules",
                    "eligible",
                    "revealed",
                    "eligible_by_channel",
                    "revealed_by_channel",
                )
            },
        }
        if replaced:
            summary["replaced"] = replaced
        budget = reveal_budget(scope)
        eligible = {
            split: int(result.get("eligible", {}).get(str(part), 0))
            for split, part in SPLIT_PHASE.items()
        }
        if scope.reveal_per_split is None:
            over = {split: count for split, count in eligible.items() if count > budget}
            if over:
                raise ValueError(
                    f"More mules were discovered by the cutoff than the reveal takes ({budget} "
                    f"per split): {over}; the reveal revealed only {budget} of them"
                )
        else:
            short = {split: count for split, count in eligible.items() if count < budget}
            if short:
                # Never filled with mules no discovery channel had found by the cutoff.
                summary["shortfall_discovered_by_cutoff"] = short
    audit = validate_supervision(executor)
    summary["contract"] = {
        key: audit.get(key) for key in ("known_labels", "true_mules", "revealed_positives")
    }
    emit({"event": "reveal", **summary})
    return summary
