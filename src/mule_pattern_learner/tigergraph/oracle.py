"""Oracle truth read from the graph, for evaluation and diagnostics; training never imports it.

TigerGraphTruthReader reads each account's ground truth for the audits.
TigerGraphRevealInputReader reads what the one-time reveal decides from (the true mules,
their draw keys and their fraud-labelled inflows), so diagnostics can replay the reveal
offline; both read the oracle, so they live here, out of training's reach (the import
contract "Training never reads ground truth").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..contract.graph_schema import TRUTH_COLUMNS
from ..contract.server import GRAPH_NAME, TRUTH_QUERY
from .executor import ConnectionExecutor, QueryExecutor, account_pages

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


@dataclass
class TigerGraphTruthReader:
    """Oracle truth paged from the graph's label contract, for evaluation only.

    contract.server.TRUTH_QUERY is the oracle endpoint; training never calls it. An
    account whose label is not known (mule_label_known false) reports is_mule = -1,
    which the evaluators treat as unknown, never as a negative. The rows have the
    contract.graph_schema.TRUTH_COLUMNS: the ring id (-1 for none) and the label's
    source come with the label.
    """

    executor: QueryExecutor

    def read(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        pages = account_pages(self.executor, TRUTH_QUERY, {}, timeout_s=900.0)
        for page in pages:
            for row in page:
                known = bool(row["mule_label_known"])
                rows.append(
                    {
                        "account_id": str(row["account_id"]),
                        "is_mule": int(row["is_mule"]) if known else -1,
                        "ring_id": int(row["mule_ring_id"]),
                        "label_source": str(row["mule_label_source"]),
                    }
                )
        return pd.DataFrame(rows, columns=list(TRUTH_COLUMNS))


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
