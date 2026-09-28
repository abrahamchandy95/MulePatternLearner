"""Reveal the known mules in the graph once, through the Account label contract.

A fresh PhantomLedger load masks every mule, so training would have no positives.
The first run calls the reveal query (gsql/queries/label_reveal.gsql),
which simulates when a bank would have discovered each mule (victim reports, network
tracing, monitoring; see docs/label_reveal.md) and reveals up to
`scope.reveal_per_split` discovered mules per split. Training then reads only the revealed positives and
their discovery clocks; ground truth stays in the
graph for the oracle audit.
"""

from __future__ import annotations

from typing import Any

from ..config import ScopeConfig, SplitDates
from ..contract.clock import timestamp
from ..contract.discovery import REVEAL_DEFAULTS
from ..contract.graph_schema import SPLIT_PHASE
from ..contract.server import GRAPH_NAME, REVEAL_QUERY
from ..runtime.progress import emit
from .executor import ConnectionExecutor, QueryExecutor, merged_rows
from .labels import validate_supervision

# Read-only interpreted query: every input of the reveal, with the job's traversals, for
# its Python mirror (reference.label_reveal.plan). The first result holds the mules
# (split, draw key, first observation and the "event_seq:label_available_ts_ms"
# fraud-labelled Zelle inflows), the next two the Zelle and payment events between two
# mules.
REVEAL_INPUTS_QUERY = f"""
INTERPRET QUERY (STRING scope_id) FOR GRAPH {GRAPH_NAME} {{
  MaxAccum<INT> @part;
  MinAccum<INT> @key;
  OrAccum @mule;
  ListAccum<STRING> @inflows;
  SetAccum<STRING> @ends;
  Scopes = {{Temporal_Training_Scope.*}};
  Ready = SELECT r FROM Scopes:r WHERE r.scope_id == scope_id;
  M = {{Account.*}};
  M = SELECT a FROM M:a WHERE a.is_mule == 1 AND NOT a.is_external POST-ACCUM a.@mule += TRUE;
  S = SELECT a FROM Ready:r -(Training_Scope_Has_Entity>:e)- Account:a WHERE a.@mule ACCUM a.@part += e.partition;
  K1 = SELECT t FROM M:a -(Account_Initiated_Transaction>:e)- Payment_Transaction:t ACCUM a.@key += t.event_seq;
  K2 = SELECT z FROM M:a -(Account_Sent_Zelle_Transfer>:e)- Zelle_Transfer:z ACCUM a.@key += z.event_seq;
  F = SELECT z FROM M:a -(Account_Received_Zelle_Transfer>:e)- Zelle_Transfer:z
      WHERE z.fraud_label == 1 AND z.label_known
      ACCUM a.@inflows += (to_string(z.event_seq) + ":" + to_string(z.label_available_ts_ms));
  LZ = SELECT z FROM M:a -((Account_Sent_Zelle_Transfer>|Account_Received_Zelle_Transfer>):e)- Zelle_Transfer:z
       ACCUM z.@ends += a.id;
  LP = SELECT t FROM M:a -((Account_Initiated_Transaction>|Account_Received_Transaction>):e)- Payment_Transaction:t
       ACCUM t.@ends += a.id;
  LZ = SELECT z FROM LZ:z WHERE z.@ends.size() > 1;
  LP = SELECT t FROM LP:t WHERE t.@ends.size() > 1;
  PRINT M[M.id, M.first_seen_ts_ms, M.@part, M.@key, M.@inflows];
  PRINT LZ[LZ.event_seq, LZ.event_ts_ms, LZ.@ends] AS zelle_links;
  PRINT LP[LP.event_seq, LP.event_ts_ms, LP.@ends] AS payment_links;
}}
"""


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


class TigerGraphRevealInputReader:
    """The reveal's inputs in a scope, read through the executor with REVEAL_INPUTS_QUERY.

    Read-only: the interpreted query writes nothing, and the job itself never runs.
    diagnostics.reveal_spread replays the job's mirror on these rows over many salts.
    """

    def __init__(self, executor: ConnectionExecutor) -> None:
        self.executor = executor

    def read(self, scope_id: str) -> list[dict[str, Any]]:
        rows = self.executor.call(
            lambda conn: conn.runInterpretedQuery(REVEAL_INPUTS_QUERY, {"scope_id": scope_id}),
            what="reveal inputs",
        )
        if not isinstance(rows, list) or not any("M" in row for row in rows):
            raise ValueError("The reveal's inputs query returned no mules result")
        return rows
