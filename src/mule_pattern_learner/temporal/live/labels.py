"""Reveal the known mules in the graph once, through the Account label contract.

A fresh PhantomLedger load masks every mule, so training would have no positives.
The first run calls temporal_reveal_mule_labels (gsql/temporal/label_reveal.gsql),
which simulates when a bank would have discovered each mule (victim reports, network
tracing, monitoring; see docs/label_reveal.md) and reveals up to `reveal_per_split`
discovered mules per split. Training then reads only the revealed positives and
their discovery clocks (label_policy = "graph_observed"); ground truth stays in the
graph for the oracle audit.
"""

from __future__ import annotations

import json
from typing import Any

from ..common import timestamp
from .config_schema import model_seed
from .contract import SPLIT_PHASE
from .executor import QueryExecutor, merged_rows

REVEAL_QUERY = "temporal_reveal_mule_labels"
VALIDATE_QUERY = "temporal_validate_account_supervision"
# The defaults temporal_reveal_mule_labels declares (tests compare them with the GSQL).
# reveal_parameters sends the budget and salt; the model parameters stay the query's own.
REVEAL_DEFAULTS: dict[str, float] = {
    "budget": 20,
    "salt": 42,
    "p_report": 0.65,
    "p_action_first": 0.5,
    "p_action_later": 0.7,
    "proactive_per_day": 0.00045,
    "trace_probability": 0.25,
    "propensity_slope": 1.0,
    "propensity_floor": 0.05,
}
# Contract violations temporal_validate_account_supervision counts; all must be zero.
VIOLATIONS = ("invalid_mule", "invalid_pu", "invalid_unknown", "invalid_clocks", "invalid_ring")
# Account CSV/PSV input columns (gsql/schema/temporal_account_loading.gsql): five
# account facts, then the ten supervision fields of the label contract.
ACCOUNT_LOAD_COLUMNS = [
    "id",
    "account_type",
    "is_external",
    "first_seen_seq",
    "first_seen_ts_ms",
    "is_mule",
    "mule_label_known",
    "is_mule_masked",
    "pu_label",
    "mule_label_effective_seq",
    "mule_label_effective_ts_ms",
    "mule_label_available_seq",
    "mule_label_available_ts_ms",
    "mule_ring_id",
    "mule_label_source",
]
# The integer migration appends is_mule in graph storage. CSV/PSV input order
# stays unchanged; the loading job maps named columns to storage positions.
ACCOUNT_STORAGE_COLUMNS = [name for name in ACCOUNT_LOAD_COLUMNS if name != "is_mule"] + ["is_mule"]
_PRIME = 2147483647


def reveal_uniforms(key: int, salt: int, n: int) -> list[float]:
    """The uniforms temporal_reveal_uniforms returns (same modular mixer, exactly)."""
    values = []
    for i in range(1, n + 1):
        x = ((key % _PRIME) + i * 1000003 + (salt % _PRIME) * 7919) % _PRIME
        x = (x * 48271 + 11) % _PRIME
        y = (x * x + 1013904223) % _PRIME
        x = (y * 69621 + x) % _PRIME
        y = (x * x + 12345) % _PRIME
        x = (y * 48271 + x) % _PRIME
        values.append((x + 0.5) / _PRIME)
    return values


def reveal_parameters(config: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    """Query parameters: the scope, each split's latest cutoff, budget and salt.

    An absent or null reveal_per_split takes the query's default budget; an absent
    or null reveal_salt takes the run's seed (model_seed).
    """
    dates = config["dates"]
    budget = config.get("reveal_per_split")
    salt = config.get("reveal_salt")
    return {
        "scope_id": config["scope_id"],
        "train_cutoff_ms": max(timestamp(d) for d in dates["train"]),
        "validation_cutoff_ms": max(timestamp(d) for d in dates["validation"]),
        "test_cutoff_ms": max(timestamp(d) for d in dates["test"]),
        "budget": int(REVEAL_DEFAULTS["budget"] if budget is None else budget),
        "salt": model_seed(config) if salt is None else int(salt),
        "apply": apply,
    }


def validate_supervision(executor: QueryExecutor) -> dict[str, Any]:
    """Label-contract audit counts; raises if any violation counter is nonzero."""
    counts = merged_rows(executor.run(VALIDATE_QUERY, {}, timeout_s=900.0))
    bad = {name: int(counts.get(name, 0)) for name in VIOLATIONS if int(counts.get(name, 0))}
    if bad:
        raise ValueError(f"Account label contract violated after the reveal: {bad}")
    return counts


def ensure_revealed_labels(executor: QueryExecutor, config: dict[str, Any]) -> dict[str, Any]:
    """Reveal known mules on first use; a graph that already has known labels is kept.

    Prints the reveal plan (mules, eligible and revealed per split, and channels) or
    the existing label counts, then checks the label contract.
    """
    if config.get("evaluation_protocol") != "strict_inductive":
        raise ValueError("Revealing labels needs the strict_inductive scope partitions")
    result = merged_rows(
        executor.run(
            REVEAL_QUERY, reveal_parameters(config, apply=True), timeout_s=3600.0, attempts=1
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
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary
