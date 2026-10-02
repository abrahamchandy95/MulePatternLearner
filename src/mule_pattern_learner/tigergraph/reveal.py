"""Reveal the known mules in the graph once, through the Account label contract.

A fresh PhantomLedger load masks every mule, so training would have no positives.
The first run calls the reveal query (gsql/queries/label_reveal.gsql), which simulates
when a bank would have discovered each mule (victim reports, network tracing,
monitoring; see docs/explanation/label-reveal.md) and reveals up to
`scope.reveal_per_split` discovered mules per split. Training then reads only the
revealed positives and their discovery clocks; ground truth stays in the graph for the
oracle audit.
"""

from __future__ import annotations

from typing import Any

from ..config import ScopeConfig, SplitDates
from ..contract.clock import timestamp
from ..contract.discovery import REVEAL_DEFAULTS
from ..contract.graph_schema import SPLIT_PHASE
from ..contract.server import REVEAL_QUERY
from ..runtime.progress import emit
from .executor import QueryExecutor, merged_rows
from .labels import validate_supervision


def reveal_parameters(scope: ScopeConfig, dates: SplitDates, *, apply: bool) -> dict[str, Any]:
    """Query parameters: the scope, each split's latest cutoff, the budget and the salt."""
    return {
        "scope_id": scope.id,
        "train_cutoff_ms": max(timestamp(d) for d in dates.train),
        "validation_cutoff_ms": max(timestamp(d) for d in dates.validation),
        "test_cutoff_ms": max(timestamp(d) for d in dates.test),
        "budget": scope.reveal_per_split,
        "salt": scope.reveal_salt,
        "apply": apply,
    }


def ensure_revealed_labels(
    executor: QueryExecutor, scope: ScopeConfig, dates: SplitDates
) -> dict[str, Any]:
    """Reveal known mules on first use; a graph that already has known labels is kept.

    Prints the reveal plan (mules, eligible and revealed per split, and channels) or
    the existing label counts, then checks the label contract.
    """
    result = merged_rows(
        executor.run(
            REVEAL_QUERY, reveal_parameters(scope, dates, apply=True), timeout_s=3600.0, attempts=1
        )
    )
    status = result.get("status")
    if status not in ("ok", "already_revealed"):
        raise ValueError(f"TigerGraph rejected the label reveal: {result}")
    if status == "already_revealed":
        summary = {
            "labels": "already revealed",
            "known_labels": result.get("known_labels"),
            "revealed_labels": result.get("revealed_labels"),
        }
    else:
        summary = {
            "labels": "revealed now",
            **{
                key: result.get(key)
                for key in (
                    "version",
                    "budget",
                    "mules",
                    "eligible",
                    "revealed",
                    "eligible_by_channel",
                    "revealed_by_channel",
                )
            },
        }
        budget = int(result.get("budget", REVEAL_DEFAULTS["budget"]))
        eligible = {
            split: int(result.get("eligible", {}).get(str(part), 0))
            for split, part in SPLIT_PHASE.items()
        }
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
