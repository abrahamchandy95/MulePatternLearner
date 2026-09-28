"""Render the context queries from the shared relation, window and feature contracts.

The generated texts are the single source of `gsql/queries/training_context.gsql`,
which training requests, and `gsql/analytics/analytics_context.gsql`, which only
analyses do (re-render both with `scripts/render_queries.py`). Both run unchanged
through INTERPRET (see `as_interpreted`), so parity checks never need an install.

One renderer writes both, from the feature groups each computes. The training query
computes the groups of contract.feature_groups.FEATURE_GROUPS that TigerGraph serves;
the analytics query computes those, the same way, and every group of
contract.analytics_features, which training never reads. A group a query does not
compute leaves no trace in its text: no include flag, no accumulator, no message field.
Every row a query returns prints its contract (contract.server.CONTEXT_CONTRACT and
ANALYTICS_CONTRACT), which names its text: `context_contract` and `analytics_contract`
derive them from the rendered texts, and a test keeps the constants equal to them, so
a changed query cannot ship under an old contract.

Processing is per request, but set-based inside a request: role, peer, device/IP
and prior-pair lookups for the sampled events are single SELECTs over all of them
instead of point SELECTs per event, and each relation is traversed once unless
order-sensitive floating sums (rolling windows, decayed activity) are requested.
Outputs are bit-identical to the reviewed v4 text apart from the documented
changes: `emit_encodings`, per-request statuses (a failed request prints a status
row and the call continues) and `missing_entity` for unknown IDs.
"""

import hashlib
import re

from ..contract.analytics_features import (
    AMOUNT_RATIO_CAP,
    AMOUNT_RATIO_FLOOR,
    AMOUNT_RATIO_WINDOWS,
    ANALYTICS_GROUPS,
    HALF_LIVES,
    IDENTITY_ORDER_RELATIONS,
    PAIR_WINDOWS,
    WINDOWS,
)
from ..contract.bounds import POOL, REQUEST_KEYS
from ..contract.feature_groups import CLIENT_GROUPS, FEATURE_GROUPS
from ..contract.graph_schema import ASSOCIATION_TARGETS, ASSOCIATIONS, NODE_TYPES
from ..contract.server import (
    ANALYTICS_CONTEXT_QUERY,
    ANALYTICS_CONTRACT,
    CONTEXT_CONTRACT,
    CONTEXT_QUERY,
    FOURIER_QUERY,
    GRAPH_NAME,
)
from .gsql_text import normalized

# The groups each query computes, in the order of its include_* parameters (every one
# defaults to TRUE; callers pass them all). The training groups are those
# FeaturePlan.query_flags names: message_core is always computed and the client
# computes its own groups.
TRAINING_GROUPS = tuple(
    name for name in FEATURE_GROUPS if name != "message_core" and name not in CLIENT_GROUPS
)
ANALYTICS_QUERY_GROUPS = (*TRAINING_GROUPS, *ANALYTICS_GROUPS)
PRIMARY_KEYS = {"Token": "token_id", "Address": "address_id"}
# (edge prefix, vertex type, primary key, rail expression, relation stem, out edges, in edges)
EVENT_TYPES = (
    (
        "Transfer",
        "Zelle_Transfer",
        "transfer_id",
        '"zelle"',
        "zelle",
        "Account_Sent_Zelle_Transfer>|Token_Sent_Transfer>",
        "Account_Received_Zelle_Transfer>|Token_Received_Transfer>",
    ),
    (
        "Transaction",
        "Payment_Transaction",
        "transaction_id",
        "t.payment_rail",
        "payment",
        "Account_Initiated_Transaction>|Token_Sent_Transaction>",
        "Account_Received_Transaction>|Token_Received_Transaction>",
    ),
)
SENDER_EDGES = {
    "Transfer": "Account_Sent_Zelle_Transfer",
    "Transaction": "Account_Initiated_Transaction",
}
# What each query's header says it is for.
TRAINING_NOTE = ""
ANALYTICS_NOTE = f"""
   For analysis only; training never calls it. It computes every feature group:
   those of training as {CONTEXT_QUERY} does, and those no model reads."""


def primary_key(node_type: str) -> str:
    return PRIMARY_KEYS.get(node_type, "id")


def as_interpreted(text: str) -> str:
    """Turn a rendered `CREATE OR REPLACE QUERY` into an equivalent INTERPRET QUERY text."""
    body = re.split(r"CREATE OR REPLACE QUERY \w+\(", text, maxsplit=1)[1]
    return "INTERPRET QUERY (" + body


def _indent(text: str, indent: str) -> str:
    return "".join(indent + line if line.strip() else line for line in text.splitlines(True))


def _when(on: bool, text: str) -> str:
    """text where a query computes the group it belongs to, else nothing."""
    return text if on else ""


def scope_entities(vertex_set: str, label: str, indent: str = "    ") -> str:
    return _indent(
        f"""IF scope_id != "" THEN
  Scope_{label} = SELECT v FROM {vertex_set}:v -(Entity_In_Training_Scope>:membership)- Temporal_Training_Scope:r
    WHERE r.scope_id == scope_id AND r.ready AND membership.partition >= 1
      AND membership.partition <= visibility_phase
    ACCUM v.@scope_allowed += TRUE;
END;
""",
        indent,
    )


def scope_events(vertex_set: str, prefix: str, label: str, indent: str = "    ") -> str:
    """Block events with any From/To Account outside the visible partitions.

    Scope and phase are fixed per call, so membership is looked up once per account
    (`@scope_checked`) and the sticky `@scope_allowed`/`@scope_blocked` flags stay valid.
    """
    return _indent(
        f"""IF scope_id != "" THEN
  Endpoints_{label} = SELECT a FROM {vertex_set}:t -(({prefix}_From_Account>|{prefix}_To_Account>):role)- Account:a
    WHERE NOT a.@scope_checked
    POST-ACCUM a.@scope_checked = TRUE;
  Scope_{label} = SELECT v FROM Endpoints_{label}:v -(Entity_In_Training_Scope>:membership)- Temporal_Training_Scope:r
    WHERE r.scope_id == scope_id AND r.ready AND membership.partition >= 1
      AND membership.partition <= visibility_phase
    ACCUM v.@scope_allowed += TRUE;
  Blocked_{label} = SELECT t FROM {vertex_set}:t -(({prefix}_From_Account>|{prefix}_To_Account>):role)- Account:a
    WHERE NOT a.@scope_allowed ACCUM t.@scope_blocked += TRUE;
END;
""",
        indent,
    )


def _header(name: str, note: str, groups: frozenset[str]) -> str:
    # The request and candidate pool checks render the ranges of contract.bounds.
    recent, older, distinct = POOL["recent"], POOL["older"], POOL["distinct"]
    associations, history = POOL["associations"], POOL["max_history"]
    ordered = [group for group in ANALYTICS_QUERY_GROUPS if group in groups]
    flags = ",\n  ".join(f"BOOL include_{group} = TRUE" for group in ordered)
    windows = "pair_window_counts" in groups
    device = "device_ip_context" in groups
    pair_counts = _when(
        windows, " INT pair_count_1h,\n                INT pair_count_1d, INT pair_count_7d,"
    )
    device_fields = _when(
        device,
        ",\n                DOUBLE device_age_seconds, BOOL device_present,"
        " DOUBLE ip_age_seconds, BOOL ip_present",
    )
    device_clocks = _when(device, ",\n    @@device_first_ms, @@ip_first_ms")
    prior_windows = _when(windows, ", @@prior_1h, @@prior_1d, @@prior_7d")
    return f"""USE GRAPH {GRAPH_NAME}

/* Generated by scripts/render_queries.py. Read-only.
   Each request has independent clocks and gets exactly one status row; a failed
   request never aborts the others. Labels and persisted feature caches are
   deliberately absent. Bounded response; adjacency scans are not indexed by
   time, so callers keep hub accounts out of child requests (hub registry).{note} */
CREATE OR REPLACE QUERY {name}(
  LIST<STRING> node_types, LIST<STRING> node_ids,
  LIST<UINT> cutoff_seqs, LIST<UINT> cutoff_times,
  INT per_relation = 2, STRING scope_id = "", INT visibility_phase = 3,
  INT k_old = 0, INT k_div = 0, INT k_assoc = 2, INT max_history = 2048,
  BOOL emit_encodings = FALSE,
  {flags}
) FOR GRAPH {GRAPH_NAME} SYNTAX V2 {{
  TYPEDEF TUPLE<UINT seq, UINT ts, STRING event_id, STRING event_type,
                STRING relation, STRING rail, DOUBLE amount, BOOL amount_present, STRING channel, STRING peer_key> EventRow;
  TYPEDEF TUPLE<STRING node_type, STRING node_id, STRING relation,
                STRING rail, STRING event_id, UINT event_seq, UINT event_ts_ms,
                DOUBLE amount, BOOL amount_present, UINT age_ms,
                UINT gap_ms, BOOL gap_present,{pair_counts} UINT peer_first_ms,
                BOOL peer_external, BOOL peer_deposit, STRING channel, STRING stratum,
                INT pair_prior_count, DOUBLE pair_first_age_seconds, BOOL pair_first_present,
                DOUBLE flow_delay_seconds, BOOL flow_present, BOOL flow_censored,
                DOUBLE flow_observation_seconds, DOUBLE flow_amount_ratio,
                BOOL flow_ratio_present, BOOL flow_same_rail{device_fields}> MessageRow;
  TYPEDEF TUPLE<STRING relation, STRING rail, STRING peer_key> PairKey;
  TYPEDEF TUPLE<UINT seq, STRING node_id, UINT first_ms, BOOL external, BOOL deposit> AssociationRow;
  TYPEDEF TUPLE<STRING event_key, UINT seq, UINT ts, STRING rail> PairItem;
  HeapAccum<EventRow>(max_history, seq DESC, event_id ASC) @@zelle_out, @@zelle_in,
    @@payment_out, @@payment_in;
  HeapAccum<AssociationRow>(8, seq DESC, node_id ASC) @@association;
  HeapAccum<EventRow>(16384, seq ASC, event_id ASC, relation ASC) @@chronology;
{_when("identity_order" in groups, "  HeapAccum<EventRow>(10, seq DESC, event_id ASC) @@last_ten;\n")}\
  MapAccum<PairKey, UINT> @@pair_counts, @@pair_first_times;
  MapAccum<PairKey, MaxAccum<UINT>> @@pair_last_times, @@pair_last_seqs;
  MapAccum<STRING, UINT> @@event_counts, @@event_first_times, @@event_previous_times;
  ListAccum<EventRow> @@events, @@ordered, @@history, @@ascending;
  ListAccum<AssociationRow> @@associations;
  ListAccum<MessageRow> @@messages;
  SetAccum<INT> @@older_ranks;
  MapAccum<STRING, STRING> @@strata;
  SetAccum<STRING> @@selected, @@selected_peers;
  MapAccum<STRING, SumAccum<DOUBLE>> @@features;
{_when("rolling_windows" in groups, "  MapAccum<STRING, SetAccum<STRING>> @@distinct;\n")}\
  MapAccum<STRING, ListAccum<DOUBLE>> @@age_encoding, @@gap_encoding;
  MapAccum<STRING, SumAccum<INT>> @@diagnostics;
  SumAccum<INT> @@visible_count, @@excluded_currency, @@relation_count;
{_when("recency" in groups, "  MaxAccum<UINT> @@out_last, @@in_last;\n")}\
  OrAccum @@invalid;
  ListAccum<STRING> @@request_types, @@request_ids;
  ListAccum<UINT> @@request_seqs, @@request_times;
  SetAccum<STRING> {", ".join("@@requested_" + t for t in NODE_TYPES)};
  MapAccum<STRING, MaxAccum<UINT>> @@root_first_seq, @@root_first_ms;
  MapAccum<STRING, OrAccum<BOOL>> @@root_external, @@root_deposit;
  SetAccum<STRING> @@root_allowed;
  SetAccum<STRING> @@selected_zelle, @@selected_payment;
  MapAccum<STRING, SumAccum<INT>> @@event_senders, @@event_recipients, @@event_tokens;
  MapAccum<STRING, OrAccum<BOOL>> @@event_role_invalid;
  MapAccum<STRING, MaxAccum<STRING>> @@sender_ids, @@recipient_ids, @@token_ids;
  MapAccum<STRING, MaxAccum<UINT>> @@sender_first_ms, @@recipient_first_ms, @@token_first_ms{device_clocks};
  MapAccum<STRING, OrAccum<BOOL>> @@sender_external, @@sender_deposit,
    @@recipient_external, @@recipient_deposit;
  SetAccum<VERTEX<Account>> @@prior_senders_Transfer, @@prior_senders_Transaction;
  MapAccum<STRING, MaxAccum<UINT>> @@prior_seq_bound_Transfer, @@prior_ts_bound_Transfer,
    @@prior_seq_bound_Transaction, @@prior_ts_bound_Transaction;
  MapAccum<STRING, ListAccum<PairItem>> @@pair_items;
  MapAccum<STRING, SumAccum<INT>> @@prior_counts{prior_windows};
  MapAccum<STRING, MaxAccum<UINT>> @@prior_last_ts, @@prior_last_seq;
  MapAccum<STRING, MinAccum<UINT>> @@prior_first_ts;
  MaxAccum<STRING> @peer_key;
  SumAccum<INT> @from_accounts, @to_accounts, @to_tokens;
  OrAccum @role_invalid;
  MaxAccum<INT> @clock_mismatch;
  SetAccum<STRING> @prior_senders;
  OrAccum @scope_allowed, @scope_blocked, @scope_checked;
  PairKey pair_key;
  VERTEX root;
  STRING root_type = "";
  STRING root_id = "";
  STRING root_key = "";
  UINT seed_seq = 0;
  UINT seed_ts_ms = 0;
  UINT state_seq = 0;
  BOOL use_chronology = FALSE;
  BOOL use_prior_scan = FALSE;
  BOOL keep_history = FALSE;
  BOOL request_failed = FALSE;
  INT pass_stamp = 0;
  INT rank = 0;
  INT older_rank = 0;
  INT diverse_taken = 0;
  INT taken = 0;
{_when("identity_order" in groups, "  UINT tenth_seq = 0;\n")}\
  UINT flow_seq = 0;
  UINT flow_ts = 0;
  DOUBLE flow_amount = 0.0;
  BOOL flow_amount_present = FALSE;
  STRING flow_rail = "";
  DOUBLE flow_delay = 0.0;
  DOUBLE flow_ratio = 0.0;
  DOUBLE observation_seconds = 0.0;
  BOOL flow_present = FALSE;
  BOOL flow_censored = FALSE;
  BOOL flow_ratio_present = FALSE;
  STRING sample_key = "";
  STRING event_key = "";
  STRING event_ref = "";
  DOUBLE pair_first_age = 0.0;
  INT pair_prior = 0;
  UINT pair_first = 0;
  UINT previous_ts = 0;
  UINT previous_seq = 0;
{_when(windows, "  INT pair_1h = 0;\n  INT pair_1d = 0;\n  INT pair_7d = 0;\n")}\
  UINT peer_first_ms = 0;
  BOOL peer_external = FALSE;
  BOOL peer_deposit = FALSE;
{_when(device, "  BOOL device_seen = FALSE;\n  BOOL ip_seen = FALSE;\n  DOUBLE device_age = 0.0;\n  DOUBLE ip_age = 0.0;\n")}\
  UINT age_ms = 0;
  UINT gap_ms = 0;
  DOUBLE incoming_amount = 0.0;
  DOUBLE outgoing_amount = 0.0;
{_when("amount_ratios" in groups, "  DOUBLE amount_ratio = 0.0;\n")}\
  STRING recipient_type = "";
  STRING recipient_key = "";
  STRING peer_type = "";
  STRING peer_id = "";

  @@request_ids = node_ids;
  @@request_types = node_types;
  @@request_seqs = cutoff_seqs;
  @@request_times = cutoff_times;
  IF @@request_ids.size() < {REQUEST_KEYS.low} OR @@request_ids.size() > {REQUEST_KEYS.high} OR
     @@request_types.size() != @@request_ids.size() OR @@request_seqs.size() != @@request_ids.size()
     OR @@request_times.size() != @@request_ids.size() OR per_relation < {recent.low} OR per_relation > {recent.high}
     OR k_old < {older.low} OR k_old > {older.high} OR k_div < {distinct.low} OR k_div > {distinct.high} OR k_assoc < {associations.low} OR k_assoc > {associations.high}
     OR max_history < {history.low} OR max_history > {history.high} THEN
    PRINT "invalid_parameters" AS status;
    RETURN;
  END;
  IF visibility_phase < 1 OR visibility_phase > 3 THEN
    PRINT "invalid_visibility_phase" AS status; RETURN;
  END;
  IF scope_id != "" THEN
    ScopeCatalog = {{Temporal_Training_Scope.*}};
    RequestedScope = SELECT r FROM ScopeCatalog:r WHERE r.scope_id == scope_id AND r.ready;
    IF RequestedScope.size() != 1 THEN PRINT "scope_not_ready" AS status; RETURN; END;
  END;
"""


def _root_catalog() -> str:
    """Typed, set-based root lookup: missing IDs become statuses, never runtime errors."""
    lines = ["  FOREACH i IN RANGE[0, @@request_ids.size()-1] DO\n"]
    for node_type in NODE_TYPES:
        lines.append(f"""    IF @@request_types.get(i) == "{node_type}" THEN
      @@requested_{node_type} += @@request_ids.get(i);
    END;
""")
    lines.append("  END;\n")
    for node_type in NODE_TYPES:
        key = f'"{node_type}:" + s.{primary_key(node_type)}'
        account = (
            f""",
      @@root_external += ({key} -> s.is_external),
      @@root_deposit += ({key} -> (s.account_type == "deposit"))"""
            if node_type == "Account"
            else ""
        )
        lines.append(f"""  Requested_{node_type} = to_vertex_set(@@requested_{node_type}, "{node_type}");
  Catalog_{node_type} = SELECT s FROM Requested_{node_type}:s
    ACCUM @@root_first_seq += ({key} -> s.first_seen_seq),
      @@root_first_ms += ({key} -> s.first_seen_ts_ms){account};
""")
        if node_type in ("Account", "Party"):
            lines.append(f"""  IF scope_id != "" THEN
    RootScope_{node_type} = SELECT s FROM Requested_{node_type}:s -(Entity_In_Training_Scope>:membership)- Temporal_Training_Scope:r
      WHERE r.scope_id == scope_id AND membership.partition >= 1
        AND membership.partition <= visibility_phase
      ACCUM s.@scope_allowed += TRUE, @@root_allowed += {key};
  END;
""")
    return "".join(lines)


def _request_setup(groups: frozenset[str]) -> str:
    known = " AND ".join(f'root_type != "{t}"' for t in NODE_TYPES)
    windows, device = "pair_window_counts" in groups, "device_ip_context" in groups
    distinct = _when("rolling_windows" in groups, " @@distinct.clear();")
    last_ten = _when("identity_order" in groups, " @@last_ten.clear();")
    tenth = _when("identity_order" in groups, " tenth_seq = 0;")
    last = _when("recency" in groups, "@@out_last = 0; @@in_last = 0; ")
    prior_windows = _when(windows, " @@prior_1h.clear(); @@prior_1d.clear(); @@prior_7d.clear();")
    if windows:
        paths = """    /* Path A (chronology) gives pair history for Account roots; the prior-pair scan
       (path B) replaces it whenever pair window counts are requested. */
    use_chronology = root_type == "Account" AND (include_time_encoding OR include_pair_history)
      AND NOT include_pair_window_counts;
    use_prior_scan = FALSE;
    IF include_pair_window_counts OR ((include_time_encoding OR include_pair_history) AND root_type != "Account") THEN
      use_prior_scan = TRUE;
    END;
"""
    else:
        paths = """    /* Path A (chronology) gives pair history for Account roots and the prior-pair scan
       (path B) for the others. */
    use_chronology = root_type == "Account" AND (include_time_encoding OR include_pair_history);
    use_prior_scan = root_type != "Account" AND (include_time_encoding OR include_pair_history);
"""
    age = _when(
        "entity_age" in groups,
        """    IF include_entity_age THEN
      @@features += ("age_days" -> (seed_ts_ms - @@root_first_ms.get(root_key)) / 86400000.0);
    END;
""",
    )
    return f"""  FOREACH i IN RANGE[0, @@request_ids.size()-1] DO
    root_type = @@request_types.get(i);
    root_id = @@request_ids.get(i);
    root_key = root_type + ":" + root_id;
    seed_seq = @@request_seqs.get(i);
    seed_ts_ms = @@request_times.get(i);
    IF seed_seq == 0 OR seed_ts_ms == 0 OR ({known}) THEN
      PRINT "invalid_request" AS status, i AS request_index;
      CONTINUE;
    END;
    IF NOT @@root_first_seq.containsKey(root_key) THEN
      PRINT "missing_entity" AS status, i AS request_index;
      CONTINUE;
    END;
    /* A peer may first be observed on the connecting event itself. Its identity
       is available then, while payment/association history remains strictly prior. */
    IF @@root_first_seq.get(root_key) > seed_seq OR @@root_first_ms.get(root_key) > seed_ts_ms
       OR (scope_id != "" AND (root_type == "Account" OR root_type == "Party")
           AND NOT @@root_allowed.contains(root_key)) THEN
      PRINT "invisible_entity" AS status, i AS request_index;
      CONTINUE;
    END;
    state_seq = seed_seq - 1;
    root = to_vertex(root_id, root_type);
    Visible = {{root}};
    @@features.clear();{distinct} @@messages.clear(); @@events.clear();
    @@history.clear(); @@ascending.clear(); @@chronology.clear(); @@pair_counts.clear(); @@pair_first_times.clear(); @@pair_last_times.clear(); @@pair_last_seqs.clear();
    @@event_counts.clear(); @@event_first_times.clear(); @@event_previous_times.clear(); @@strata.clear(); @@diagnostics.clear();{last_ten}
    @@visible_count = 0; @@excluded_currency = 0;{tenth}
    @@age_encoding.clear(); @@gap_encoding.clear();
    {last}@@invalid = FALSE;
    @@zelle_out.clear(); @@zelle_in.clear();
    @@payment_out.clear(); @@payment_in.clear();
    @@selected_zelle.clear(); @@selected_payment.clear();
    @@event_senders.clear(); @@event_recipients.clear(); @@event_tokens.clear(); @@event_role_invalid.clear();
    @@sender_ids.clear(); @@recipient_ids.clear(); @@token_ids.clear();
    @@sender_first_ms.clear(); @@recipient_first_ms.clear(); @@token_first_ms.clear();
{_when(device, "    @@device_first_ms.clear(); @@ip_first_ms.clear();\n")}\
    @@sender_external.clear(); @@sender_deposit.clear(); @@recipient_external.clear(); @@recipient_deposit.clear();
    @@prior_senders_Transfer.clear(); @@prior_senders_Transaction.clear();
    @@prior_seq_bound_Transfer.clear(); @@prior_ts_bound_Transfer.clear();
    @@prior_seq_bound_Transaction.clear(); @@prior_ts_bound_Transaction.clear();
    @@pair_items.clear(); @@prior_counts.clear();{prior_windows}
    @@prior_last_ts.clear(); @@prior_last_seq.clear(); @@prior_first_ts.clear();
{paths}\
    keep_history = include_flow_timing AND root_type == "Account";
{age}\
    IF include_entity_meta AND root_type == "Account" THEN
      IF @@root_external.get(root_key) THEN @@features += ("is_external" -> 1); END;
      IF @@root_deposit.get(root_key) THEN @@features += ("is_deposit" -> 1); END;
    END;
"""


def _event_accum(stem: str, direction: str, clock: str, groups: frozenset[str]) -> str:
    """Validity, counters, rolling windows and decayed sums of one relation's visible events."""
    parts = [
        f"""        ACCUM @@invalid += ({clock}
            OR t.event_seq == 0 OR t.event_ts_ms == 0 OR t.amount < 0
            OR (NOT t.amount_present AND t.amount != 0)),
          @@visible_count += 1"""
    ]
    if "recency" in groups:
        parts.append(f""",
          @@{direction}_last += t.event_ts_ms""")
    for name, ms in WINDOWS.items() if "rolling_windows" in groups else ():
        parts.append(f''',
          IF include_rolling_windows AND seed_ts_ms - t.event_ts_ms < {ms} THEN
            @@features += ("{name}_{direction}_count" -> 1),
            @@features += ("{name}_{direction}_amount" -> t.amount),
            IF NOT t.amount_present THEN @@features += ("{name}_{direction}_missing" -> 1) END,
            @@features += ("{name}_{direction}_zelle" -> {1 if stem == "zelle" else 0})
          END''')
    for half, duration in HALF_LIVES.items() if "decayed_activity" in groups else ():
        parts.append(f""",
          IF include_decayed_activity THEN
            @@features += ("decay_{half}_{direction}_count" -> exp(-0.6931471805599453 * (seed_ts_ms - t.event_ts_ms) / {duration}.0)),
            @@features += ("decay_{half}_{direction}_amount" -> t.amount * exp(-0.6931471805599453 * (seed_ts_ms - t.event_ts_ms) / {duration}.0)) END""")
    parts.append("""
        POST-ACCUM t.@from_accounts = 0, t.@to_accounts = 0, t.@to_tokens = 0,
          t.@role_invalid = FALSE, t.@peer_key = "";
""")
    return "".join(parts)


def _relation(
    prefix: str,
    vtype: str,
    pid: str,
    rail: str,
    stem: str,
    direction: str,
    reverse: str,
    groups: frozenset[str],
) -> str:
    """One adjacency scan per relation; later passes filter the candidate vertex set.

    Floating-point sums (rolling amounts, decayed activity) depend on accumulation
    order. When they are requested, the Events pass keeps the reviewed v4 adjacency
    traversal so its outputs stay bit-identical; otherwise it filters Candidates. A
    query that computes neither has only the second pass.
    """
    label = f"{stem}_{direction}"
    edge_clock = "e.event_seq != t.event_seq OR e.event_ts_ms != t.event_ts_ms"
    parts = [
        f"""      pass_stamp = pass_stamp + 1;
      Candidates_{label} = SELECT t FROM Visible:s -(({reverse}):e)- {vtype}:t
        WHERE t.event_seq < seed_seq AND t.event_ts_ms <= seed_ts_ms
        ACCUM IF {edge_clock} THEN
          t.@clock_mismatch += pass_stamp END;
""",
        scope_events(f"Candidates_{label}", prefix, label, "      "),
        f"""      @@relation_count = 0;
      Currency_{label} = SELECT t FROM Candidates_{label}:t WHERE NOT t.@scope_blocked
        ACCUM IF t.currency == "USD" THEN @@relation_count += 1 ELSE @@excluded_currency += 1 END;
      IF @@relation_count > max_history THEN
        PRINT "history_capacity_exceeded" AS status, i AS request_index, @@relation_count AS visible_relation_events;
        CONTINUE;
      END;
""",
    ]
    filtered = f"""      Events_{label} = SELECT t FROM Candidates_{label}:t
        WHERE NOT t.@scope_blocked AND t.currency == "USD"
""" + _event_accum(stem, direction, "t.@clock_mismatch == pass_stamp", groups)
    sums = [g for g in ("rolling_windows", "decayed_activity") if g in groups]
    if sums:
        ordered = f"""      Events_{label} = SELECT t FROM Visible:s -(({reverse}):e)- {vtype}:t
        WHERE t.event_seq < seed_seq AND t.event_ts_ms <= seed_ts_ms
          AND NOT t.@scope_blocked AND t.currency == "USD"
""" + _event_accum(stem, direction, edge_clock, groups)
        condition = " OR ".join(f"include_{g}" for g in sums)
        parts += [
            f"      IF {condition} THEN\n",
            _indent(ordered, "  "),
            "      ELSE\n",
            _indent(filtered, "  "),
            "      END;\n",
        ]
    else:
        parts.append(filtered)
    # The canonical counterparty: the recipient Account (Token only without one) for
    # outgoing events, the sender Account for incoming ones.
    peer_role = "To" if direction == "out" else "From"
    for role_name, role_type, count in (
        ("From", "Account", "from_accounts"),
        ("To", "Account", "to_accounts"),
        ("To", "Token", "to_tokens"),
    ):
        unique = ""
        if role_name == peer_role:
            idattr = primary_key(role_type)
            fallback = (
                f' AND t.outdegree("{prefix}_To_Account") == 0' if role_type == "Token" else ""
            )
            windows = "".join(
                f""",
            IF include_rolling_windows AND seed_ts_ms - t.event_ts_ms < {ms} THEN
              @@distinct += ("{name}_{direction}_unique" -> "{role_type}:" + a.{idattr}) END"""
                for name, ms in (WINDOWS.items() if "rolling_windows" in groups else ())
            )
            unique = f""",
          IF e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms{fallback} THEN
            t.@peer_key = "{role_type}:" + a.{idattr}{windows}
          END"""
        parts.append(f"""      Roles_{label}_{role_name}_{role_type} = SELECT t
        FROM Events_{label}:t -({prefix}_{role_name}_{role_type}>:e)- {role_type}:a
        ACCUM t.@{count} += 1, t.@role_invalid += (e.event_seq != t.event_seq OR e.event_ts_ms != t.event_ts_ms
          OR a.first_seen_seq > t.event_seq OR a.first_seen_ts_ms > t.event_ts_ms){unique};
""")
    last_ten = _when(
        "identity_order" in groups,
        "        IF include_identity_order THEN @@last_ten += entry; END;\n",
    )
    parts.append(f'''      Retain_{label} = SELECT t FROM Events_{label}:t
        ACCUM @@invalid += (t.@role_invalid OR t.@from_accounts != 1 OR t.@to_accounts > 1
            OR t.@to_tokens > 1 OR (t.@to_accounts == 0 AND t.@to_tokens == 0)),
          @@{label} += EventRow(t.event_seq, t.event_ts_ms, t.{pid}, "{vtype}",
            "{label}", {rail}, t.amount, t.amount_present, t.channel, t.@peer_key);
      @@ordered.clear(); @@selected.clear(); @@selected_peers.clear();
      WHILE @@{label}.size() > 0 DO
        @@ordered += @@{label}.pop();
      END;
      rank = 0;
      FOREACH entry IN @@ordered DO
        IF keep_history THEN @@history += entry; END;
        IF use_chronology THEN @@chronology += entry; END;
{last_ten}\
        sample_key = entry.relation + ":" + entry.event_id;
        IF rank < per_relation THEN
          @@events += entry; @@strata += (sample_key -> "recent");
          @@selected += entry.event_id; @@selected_peers += entry.peer_key;
        END;
        rank = rank + 1;
      END;
      @@older_ranks.clear();
      IF k_old > 0 AND @@ordered.size() > per_relation THEN
        FOREACH j IN RANGE[1, k_old] DO
          older_rank = per_relation + floor((@@ordered.size() - per_relation - 1) * j / (k_old + 1.0));
          @@older_ranks += older_rank;
        END;
        rank = 0;
        FOREACH entry IN @@ordered DO
          IF @@older_ranks.contains(rank) AND NOT @@selected.contains(entry.event_id) THEN
            @@events += entry; @@strata += (entry.relation + ":" + entry.event_id -> "older");
            @@selected += entry.event_id; @@selected_peers += entry.peer_key;
          END;
          rank = rank + 1;
        END;
      END;
      diverse_taken = 0;
      FOREACH entry IN @@ordered DO
        IF diverse_taken < k_div AND NOT @@selected.contains(entry.event_id)
          AND NOT @@selected_peers.contains(entry.peer_key) THEN
          @@events += entry; @@strata += (entry.relation + ":" + entry.event_id -> "distinct");
          @@selected += entry.event_id; @@selected_peers += entry.peer_key;
          diverse_taken = diverse_taken + 1;
        END;
      END;
''')
    return "".join(parts)


def _chronology(groups: frozenset[str]) -> str:
    parts = [
        """    IF use_chronology THEN
      request_failed = FALSE;
      WHILE @@chronology.size() > 0 DO @@ascending += @@chronology.pop(); END;
      FOREACH chronological_entry IN @@ascending DO
        pair_key = PairKey(chronological_entry.relation, chronological_entry.rail, chronological_entry.peer_key);
        event_key = chronological_entry.relation + ":" + chronological_entry.event_id;
        IF @@pair_counts.containsKey(pair_key) THEN
          IF chronological_entry.ts < @@pair_last_times.get(pair_key) OR chronological_entry.seq <= @@pair_last_seqs.get(pair_key) THEN
            request_failed = TRUE;
            BREAK;
          END;
          @@event_counts += (event_key -> @@pair_counts.get(pair_key));
          @@event_first_times += (event_key -> @@pair_first_times.get(pair_key));
          @@event_previous_times += (event_key -> @@pair_last_times.get(pair_key));
          @@pair_counts += (pair_key -> 1);
        ELSE
          @@pair_counts += (pair_key -> 1);
          @@pair_first_times += (pair_key -> chronological_entry.ts);
        END;
        @@pair_last_times += (pair_key -> chronological_entry.ts);
        @@pair_last_seqs += (pair_key -> chronological_entry.seq);
      END;
      IF request_failed THEN
        PRINT "nonmonotonic_pair_clock" AS status, i AS request_index;
        CONTINUE;
      END;
    END;
"""
    ]
    if "rolling_windows" in groups:
        parts.append("""    FOREACH (key, peers) IN @@distinct DO
      @@features += (key -> peers.size());
    END;
""")
    if "recency" in groups:
        parts.append("""    IF include_recency AND @@out_last > 0 THEN
      @@features += ("out_recency_days" -> (seed_ts_ms - @@out_last) / 86400000.0);
      @@features += ("out_recency_present" -> 1);
    END;
    IF include_recency AND @@in_last > 0 THEN
      @@features += ("in_recency_days" -> (seed_ts_ms - @@in_last) / 86400000.0);
      @@features += ("in_recency_present" -> 1);
    END;
""")
    return "".join(parts)


def _summaries(groups: frozenset[str]) -> str:
    parts = []
    for window in AMOUNT_RATIO_WINDOWS if "amount_ratios" in groups else ():
        parts.append(f'''    IF include_amount_ratios THEN
    incoming_amount = 0.0; outgoing_amount = 0.0;
    IF @@features.containsKey("{window}_in_amount") THEN
      incoming_amount = @@features.get("{window}_in_amount");
    END;
    IF @@features.containsKey("{window}_out_amount") THEN
      outgoing_amount = @@features.get("{window}_out_amount");
    END;
    IF incoming_amount < {AMOUNT_RATIO_FLOOR} THEN incoming_amount = {AMOUNT_RATIO_FLOOR}; END;
    amount_ratio = outgoing_amount / incoming_amount;
    IF amount_ratio > {AMOUNT_RATIO_CAP} THEN amount_ratio = {AMOUNT_RATIO_CAP}; END;
    @@features += ("{window}_out_in_amount_ratio" -> amount_ratio);
    END;
''')
    if "history_support" in groups:
        parts.append("""    IF include_history_support THEN
      @@features += ("visible_event_count" -> @@visible_count);
      IF @@visible_count < 5 THEN @@features += ("history_lt_5_events" -> 1); END;
    END;
""")
    if "identity_order" in groups:
        parts.append("""    IF @@last_ten.size() > 0 THEN
      FOREACH entry IN @@last_ten DO tenth_seq = entry.seq; END;
    END;
""")
    return "".join(parts)


def _association(rel: str, typ: str, source: str, groups: frozenset[str]) -> str:
    """Active tenures at state_seq; only relations leaving the root's type can match."""
    pk = primary_key(typ)
    ext = "t.is_external" if typ == "Account" else "FALSE"
    dep = '(t.account_type == "deposit")' if typ == "Account" else "FALSE"
    scoped = typ in ("Account", "Party")
    candidates = ""
    if scoped:
        candidates = f"""IF scope_id != "" THEN
  CandidateAssoc_{rel} = SELECT t FROM Visible:s -({rel}>:e)- {typ}:t
    WHERE e.valid_from_seq <= state_seq
      AND t.first_seen_seq <= state_seq AND t.first_seen_ts_ms <= seed_ts_ms;
END;
""" + scope_entities(f"CandidateAssoc_{rel}", rel, "")
    scope_filter = '\n    AND (scope_id == "" OR t.@scope_allowed)' if scoped else ""
    identity = ""
    if rel in IDENTITY_ORDER_RELATIONS and "identity_order" in groups:
        identity = f""",
  IF include_identity_order AND tenth_seq > 0 THEN
    IF e.valid_from_seq > tenth_seq THEN @@features += ("{rel}_starts_last10" -> 1) END,
    IF e.valid_to_seq > tenth_seq AND e.valid_to_seq <= state_seq THEN @@features += ("{rel}_ends_last10" -> 1) END
  END"""
    row = f"@@association += AssociationRow(e.valid_from_seq, t.{pk}, t.first_seen_ts_ms, {ext}, {dep})"
    if "association_counts" in groups:
        tenure = f"""  ACCUM IF e.valid_to_seq == 0 OR state_seq < e.valid_to_seq THEN
    IF include_association_counts THEN @@features += ("{rel}_active" -> 1) END,
    {row}
  ELSE IF include_association_counts THEN @@features += ("{rel}_ended" -> 1) END END{identity};"""
    else:
        tenure = f"""  ACCUM IF e.valid_to_seq == 0 OR state_seq < e.valid_to_seq THEN
    {row}
  END{identity};"""
    pair_counts = _when("pair_window_counts" in groups, " 0, 0, 0,")
    device = _when("device_ip_context" in groups, ", 0.0, FALSE, 0.0, FALSE")
    body = f"""{candidates}@@association.clear(); @@associations.clear();
Assoc_{rel} = SELECT t FROM Visible:s -({rel}>:e)- {typ}:t
  WHERE e.valid_from_seq <= state_seq
    AND t.first_seen_seq <= state_seq AND t.first_seen_ts_ms <= seed_ts_ms{scope_filter}
{tenure}
taken = 0;
WHILE @@association.size() > 0 AND taken < k_assoc DO
  @@associations += @@association.pop();
  taken = taken + 1;
END;
FOREACH entry IN @@associations DO
  @@messages += MessageRow("{typ}", entry.node_id, "{rel}", "unknown", "", seed_seq,
    seed_ts_ms, 0.0, FALSE, 0, 0, FALSE,{pair_counts} entry.first_ms, entry.external, entry.deposit, "unknown", "association",
    0, 0.0, FALSE, 0.0, FALSE, FALSE, 0.0, 0.0, FALSE, FALSE{device});
END;
"""
    return f'    IF root_type == "{source}" THEN\n{_indent(body, "      ")}    END;\n'


def _selected_events(groups: frozenset[str]) -> str:
    """One SELECT per role over all sampled events (roles, peers, device/IP, prior bounds)."""
    parts = [
        """    IF @@invalid THEN
      PRINT "invalid_payment_fields" AS status, i AS request_index;
      CONTINUE;
    END;
    FOREACH item IN @@events DO
      IF item.event_type == "Zelle_Transfer" THEN @@selected_zelle += item.event_id;
      ELSE @@selected_payment += item.event_id; END;
    END;
    Selected_Transfer = to_vertex_set(@@selected_zelle, "Zelle_Transfer");
    Selected_Transaction = to_vertex_set(@@selected_payment, "Payment_Transaction");
"""
    ]
    clock = """(e.event_seq != t.event_seq OR e.event_ts_ms != t.event_ts_ms
            OR a.first_seen_seq > t.event_seq OR a.first_seen_ts_ms > t.event_ts_ms)"""
    for prefix, vtype, pid, *_ in EVENT_TYPES:
        key = f'"{vtype}:" + t.{pid}'
        parts.append(f"""    Senders_{prefix} = SELECT t FROM Selected_{prefix}:t -({prefix}_From_Account>:e)- Account:a
      ACCUM @@event_senders += ({key} -> 1), @@sender_ids += ({key} -> a.id),
        @@sender_first_ms += ({key} -> a.first_seen_ts_ms),
        @@sender_external += ({key} -> a.is_external),
        @@sender_deposit += ({key} -> (a.account_type == "deposit")),
        @@event_role_invalid += ({key} -> {clock}),
        IF use_prior_scan THEN
          @@prior_senders_{prefix} += a,
          @@prior_seq_bound_{prefix} += (a.id -> t.event_seq),
          @@prior_ts_bound_{prefix} += (a.id -> t.event_ts_ms)
        END;
    Recipients_{prefix} = SELECT t FROM Selected_{prefix}:t -({prefix}_To_Account>:e)- Account:a
      ACCUM @@event_recipients += ({key} -> 1), @@recipient_ids += ({key} -> a.id),
        @@recipient_first_ms += ({key} -> a.first_seen_ts_ms),
        @@recipient_external += ({key} -> a.is_external),
        @@recipient_deposit += ({key} -> (a.account_type == "deposit")),
        @@event_role_invalid += ({key} -> {clock});
    Tokens_{prefix} = SELECT t FROM Selected_{prefix}:t -({prefix}_To_Token>:e)- Token:a
      ACCUM @@event_tokens += ({key} -> 1), @@token_ids += ({key} -> a.token_id),
        @@token_first_ms += ({key} -> a.first_seen_ts_ms),
        @@event_role_invalid += ({key} -> {clock});
""")
        if "device_ip_context" in groups:
            parts.append("    IF include_device_ip_context THEN\n")
            for typ, field in (("Device", "device"), ("IP", "ip")):
                parts.append(f"""      EventIdentity_{prefix}_{typ} = SELECT t FROM Selected_{prefix}:t -({prefix}_Used_{typ}>:e)- {typ}:a
        WHERE e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms
          AND a.first_seen_seq <= t.event_seq AND a.first_seen_ts_ms > 0 AND a.first_seen_ts_ms <= t.event_ts_ms
        ACCUM @@{field}_first_ms += ({key} -> a.first_seen_ts_ms);
""")
            parts.append("    END;\n")
    parts.append("""    request_failed = FALSE;
    FOREACH item IN @@events DO
      event_ref = item.event_type + ":" + item.event_id;
      IF @@event_role_invalid.get(event_ref) OR @@event_senders.get(event_ref) != 1
         OR @@event_recipients.get(event_ref) > 1 OR @@event_tokens.get(event_ref) > 1
         OR (@@event_recipients.get(event_ref) == 0 AND @@event_tokens.get(event_ref) == 0) THEN
        request_failed = TRUE;
        BREAK;
      END;
      IF use_prior_scan THEN
        recipient_type = "Account"; recipient_key = @@recipient_ids.get(event_ref);
        IF @@event_recipients.get(event_ref) == 0 THEN
          recipient_type = "Token"; recipient_key = @@token_ids.get(event_ref);
        END;
        @@pair_items += (@@sender_ids.get(event_ref) + "|" + item.event_type + "|" + recipient_type + ":" + recipient_key
          -> PairItem(item.relation + ":" + item.event_id, item.seq, item.ts, item.rail));
      END;
    END;
    IF request_failed THEN
      PRINT "invalid_event_roles" AS status, i AS request_index;
      CONTINUE;
    END;
""")
    return "".join(parts)


def _prior_scan(groups: frozenset[str]) -> str:
    """Path B: earlier same-pair events of each sampled event's sender, one scan per sender."""
    parts = ["    IF use_prior_scan THEN\n"]
    for prefix, vtype, *_ in EVENT_TYPES:
        rail = " AND t.payment_rail == pair_item.rail" if prefix == "Transaction" else ""
        parts.append(f"""      PriorSenders_{prefix} = {{@@prior_senders_{prefix}}};
      PriorCandidates_{prefix} = SELECT t FROM PriorSenders_{prefix}:s -({SENDER_EDGES[prefix]}>:e)- {vtype}:t
        WHERE t.event_seq < @@prior_seq_bound_{prefix}.get(s.id) AND t.event_ts_ms <= @@prior_ts_bound_{prefix}.get(s.id)
          AND e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms AND t.currency == "USD"
        ACCUM t.@prior_senders += s.id;
""")
        parts.append(scope_events(f"PriorCandidates_{prefix}", prefix, f"prior_{prefix}", "      "))
        for typ in ("Account", "Token"):
            fallback = f' AND t.outdegree("{prefix}_To_Account") == 0' if typ == "Token" else ""
            windows = "".join(
                f""",
                IF include_pair_window_counts AND pair_item.ts - t.event_ts_ms < {ms} THEN
                  @@prior_{name} += (pair_item.event_key -> 1) END"""
                for name, ms in (PAIR_WINDOWS.items() if "pair_window_counts" in groups else ())
            )
            parts.append(f"""      Pair_{prefix}_{typ} = SELECT t FROM PriorCandidates_{prefix}:t -({prefix}_To_{typ}>:e)- {typ}:a
        WHERE NOT t.@scope_blocked AND e.event_seq == t.event_seq AND e.event_ts_ms == t.event_ts_ms{fallback}
        ACCUM FOREACH sender IN t.@prior_senders DO
            FOREACH pair_item IN @@pair_items.get(sender + "|{vtype}|{typ}:" + a.{primary_key(typ)}) DO
              IF t.event_seq < pair_item.seq AND t.event_ts_ms <= pair_item.ts{rail} THEN
                @@prior_counts += (pair_item.event_key -> 1),
                @@prior_last_ts += (pair_item.event_key -> t.event_ts_ms),
                @@prior_last_seq += (pair_item.event_key -> t.event_seq),
                @@prior_first_ts += (pair_item.event_key -> t.event_ts_ms){windows}
              END
            END
          END;
""")
        parts.append(f"""      Release_{prefix} = SELECT t FROM PriorCandidates_{prefix}:t
        POST-ACCUM t.@prior_senders.clear();
""")
    parts.append("    END;\n")
    return "".join(parts)


def _messages(groups: frozenset[str]) -> str:
    windows, device = "pair_window_counts" in groups, "device_ip_context" in groups
    return f"""    FOREACH item IN @@events DO
      event_ref = item.event_type + ":" + item.event_id;
      event_key = item.relation + ":" + item.event_id;
      recipient_type = "Account"; recipient_key = @@recipient_ids.get(event_ref);
      peer_first_ms = @@recipient_first_ms.get(event_ref);
      peer_external = @@recipient_external.get(event_ref);
      peer_deposit = @@recipient_deposit.get(event_ref);
      IF @@event_recipients.get(event_ref) == 0 THEN
        recipient_type = "Token"; recipient_key = @@token_ids.get(event_ref);
        peer_first_ms = @@token_first_ms.get(event_ref); peer_external = FALSE; peer_deposit = FALSE;
      END;
      peer_type = recipient_type; peer_id = recipient_key;
      IF item.relation == "zelle_in" OR item.relation == "payment_in" THEN
        peer_type = "Account"; peer_id = @@sender_ids.get(event_ref);
        peer_first_ms = @@sender_first_ms.get(event_ref);
        peer_external = @@sender_external.get(event_ref);
        peer_deposit = @@sender_deposit.get(event_ref);
      END;
      previous_ts = 0; previous_seq = 0;
      {
        _when(windows, "pair_1h = 0; pair_1d = 0; pair_7d = 0; ")
    }pair_prior = 0; pair_first = seed_ts_ms;
      IF root_type == "Account" AND @@event_counts.containsKey(event_key) THEN
        pair_prior = @@event_counts.get(event_key);
        pair_first = @@event_first_times.get(event_key);
        previous_ts = @@event_previous_times.get(event_key);
        previous_seq = 1;
      END;
      IF use_prior_scan THEN
        pair_prior = 0; pair_first = seed_ts_ms; previous_ts = 0; previous_seq = 0;
        IF @@prior_counts.containsKey(event_key) THEN
          pair_prior = @@prior_counts.get(event_key);
          previous_ts = @@prior_last_ts.get(event_key);
          previous_seq = @@prior_last_seq.get(event_key);
          IF @@prior_first_ts.get(event_key) < pair_first THEN pair_first = @@prior_first_ts.get(event_key); END;
{
        _when(
            windows,
            "          pair_1h = @@prior_1h.get(event_key); pair_1d = @@prior_1d.get(event_key); pair_7d = @@prior_7d.get(event_key);\n",
        )
    }\
        END;
      END;
      pair_first_age = 0.0;
      IF include_pair_history AND pair_prior > 0 THEN pair_first_age = (item.ts - pair_first) / 1000.0; END;
      flow_seq = 0; flow_ts = 0; flow_amount = 0.0; flow_amount_present = FALSE; flow_rail = "";
      flow_delay = 0.0; flow_ratio = 0.0; flow_ratio_present = FALSE; flow_present = FALSE;
      flow_censored = FALSE; observation_seconds = 0.0;
      IF include_flow_timing AND root_type == "Account" THEN
        FOREACH other IN @@history DO
          IF (item.relation == "zelle_in" OR item.relation == "payment_in")
            AND (other.relation == "zelle_out" OR other.relation == "payment_out")
            AND other.seq > item.seq AND other.ts >= item.ts
            AND (flow_seq == 0 OR other.seq < flow_seq) THEN
            flow_seq = other.seq; flow_ts = other.ts; flow_amount = other.amount;
            flow_amount_present = other.amount_present; flow_rail = other.rail;
          END;
          IF (item.relation == "zelle_out" OR item.relation == "payment_out")
            AND (other.relation == "zelle_in" OR other.relation == "payment_in")
            AND other.seq < item.seq AND other.ts <= item.ts AND other.seq > flow_seq THEN
            flow_seq = other.seq; flow_ts = other.ts; flow_amount = other.amount;
            flow_amount_present = other.amount_present; flow_rail = other.rail;
          END;
        END;
        flow_present = flow_seq > 0;
        IF item.relation == "zelle_in" OR item.relation == "payment_in" THEN
          observation_seconds = (seed_ts_ms - item.ts) / 1000.0;
          flow_censored = NOT flow_present;
          IF flow_present THEN flow_delay = (flow_ts - item.ts) / 1000.0; END;
          incoming_amount = item.amount; outgoing_amount = flow_amount;
        ELSE
          IF flow_present THEN flow_delay = (item.ts - flow_ts) / 1000.0; END;
          incoming_amount = flow_amount; outgoing_amount = item.amount;
        END;
        IF flow_present AND item.amount_present AND flow_amount_present THEN
          flow_ratio_present = TRUE;
          IF incoming_amount < 1.0 THEN incoming_amount = 1.0; END;
          flow_ratio = outgoing_amount / incoming_amount;
          IF flow_ratio > 100.0 THEN flow_ratio = 100.0; END;
        END;
      END;
      gap_ms = 0;
      age_ms = seed_ts_ms - item.ts;
      IF include_time_encoding AND emit_encodings THEN
        @@age_encoding += (event_key -> {FOURIER_QUERY}(age_ms));
      END;
      IF previous_seq > 0 THEN
        gap_ms = item.ts - previous_ts;
        IF include_time_encoding AND emit_encodings THEN
          @@gap_encoding += (event_key -> {FOURIER_QUERY}(gap_ms));
        END;
      END;
{
        _when(
            device,
            '''      device_seen = FALSE; ip_seen = FALSE; device_age = 0.0; ip_age = 0.0;
      IF @@device_first_ms.containsKey(event_ref) THEN
        device_seen = TRUE; device_age = (item.ts - @@device_first_ms.get(event_ref)) / 1000.0;
      END;
      IF @@ip_first_ms.containsKey(event_ref) THEN
        ip_seen = TRUE; ip_age = (item.ts - @@ip_first_ms.get(event_ref)) / 1000.0;
      END;
''',
        )
    }\
      @@messages += MessageRow(peer_type, peer_id, item.relation, item.rail,
        item.event_id, item.seq, item.ts,
        item.amount, item.amount_present,
        age_ms, gap_ms, previous_seq > 0,{_when(windows, " pair_1h, pair_1d, pair_7d,")}
        peer_first_ms, peer_external, peer_deposit, item.channel, @@strata.get(event_key),
        pair_prior, pair_first_age, include_pair_history AND pair_prior > 0,
        flow_delay, flow_present, flow_censored, observation_seconds, flow_ratio,
        flow_ratio_present, flow_present AND flow_rail == item.rail{
        _when(device, ", device_age, device_seen, ip_age, ip_seen")
    });
    END;
    @@diagnostics += ("excluded_non_usd_events" -> @@excluded_currency);
    @@diagnostics += ("visible_payment_participations" -> @@visible_count);
    PRINT "ok" AS status, i AS request_index,
      "__CONTRACT__" AS contract_version, @@diagnostics AS diagnostics,
      root_type AS node_type, root_id AS node_id,
      seed_seq AS cutoff_seq, seed_ts_ms AS cutoff_ms,
      scope_id AS scope_id, visibility_phase AS visibility_phase,
      "log1p_s_400d_32x_sincos_v1" AS basis_id,
      @@features AS features, @@messages AS messages,
      @@age_encoding AS age_encoding, @@gap_encoding AS gap_encoding;
  END;
}}
"""


def _render(name: str, note: str, computed: tuple[str, ...], contract: str) -> str:
    """The text of one context query that computes `computed` and prints `contract`."""
    groups = frozenset(computed)
    parts = [_header(name, note, groups), _root_catalog(), _request_setup(groups)]
    parts.append('    IF root_type == "Account" OR root_type == "Token" THEN\n')
    for prefix, vtype, pid, rail, stem, out_edges, in_edges in EVENT_TYPES:
        for direction, reverse in (("out", out_edges), ("in", in_edges)):
            parts.append(_relation(prefix, vtype, pid, rail, stem, direction, reverse, groups))
    parts.append("    END;\n")
    parts.append(_chronology(groups))
    parts.append(_summaries(groups))
    for pair, targets in zip(ASSOCIATIONS, ASSOCIATION_TARGETS, strict=True):
        sources = (targets[1], targets[0])
        for rel, typ, source in zip(pair, targets, sources, strict=True):
            parts.append(_association(rel, typ, source, groups))
    parts.append(_selected_events(groups))
    parts.append(_prior_scan(groups))
    parts.append(_messages(groups))
    text = "".join(parts).replace("__CONTRACT__", contract)
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def render_context_query(contract: str = CONTEXT_CONTRACT) -> str:
    """The exact text of gsql/queries/training_context.gsql, printing `contract`."""
    return _render(CONTEXT_QUERY, TRAINING_NOTE, TRAINING_GROUPS, contract)


def render_analytics_query(contract: str = ANALYTICS_CONTRACT) -> str:
    """The exact text of gsql/analytics/analytics_context.gsql, printing `contract`."""
    return _render(ANALYTICS_CONTEXT_QUERY, ANALYTICS_NOTE, ANALYTICS_QUERY_GROUPS, contract)


def derived_contract(prefix: str, text: str) -> str:
    """prefix and the first 12 hex digits of the sha256 of text, normalised.

    gsql_text.normalized drops comments and whitespace. text is a query rendered with
    an empty contract literal, so the value covers the whole query but itself.
    """
    return prefix + hashlib.sha256(normalized(text).encode()).hexdigest()[:12]


def context_contract() -> str:
    """The contract the training context query should print (CONTEXT_CONTRACT)."""
    return derived_contract("context_", render_context_query(contract=""))


def analytics_contract() -> str:
    """The contract the analytics context query should print (ANALYTICS_CONTRACT)."""
    return derived_contract("analytics_", render_analytics_query(contract=""))
