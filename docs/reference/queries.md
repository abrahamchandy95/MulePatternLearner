# Queries

Every GSQL query: responsibility, file, callers, reads and writes. Queries are named verb
first after their responsibility, without prefix (the graph is dedicated); files after the
responsibility their queries share. The names are constants of `contract.server`, used
by every adapter, fake and test. All install on the graph `Mule_Pattern_Learner`
(`contract.server.GRAPH_NAME`).

| Folder | Holds | Installed by |
|---|---|---|
| `gsql/queries/` | The training pipeline's queries | `mule install`, and `mule train` when a text differs |
| `gsql/evaluation/` | The oracle, read only by the audits and diagnostics | The same |
| `gsql/analytics/` | Analysis-only queries | Only `mule diagnose`, where a text differs |
| `gsql/schema/` | The graph's DDL and loading job (run by a person), and the scope schema change `mule install` applies | See [Set up a graph](../how-to/set-up-a-graph.md) |

## The queries

| Query | File | Writes | Run by |
|---|---|---|---|
| [`fetch_training_context`](#fetch_training_context) | `queries/training_context.gsql` (generated) | No | Every training batch, proxy evaluations, audits, `mule score`, `mule check`, `mule diagnose` (feature table) |
| [`encode_fourier64`](#encode_fourier64) | `queries/fourier64.gsql` | No | Both context queries, as a subquery |
| [`create_training_scope`](#create_training_scope) | `queries/training_scope.gsql` | The scope | The first preparation, when `scope.id` does not exist |
| [`finalize_training_scope`](#finalize_training_scope) | `queries/training_scope.gsql` | The scope's `ready` flag | Right after creation |
| [`list_scope_accounts`](#list_scope_accounts) | `queries/training_scope.gsql` | No | Preparation, audits, diagnostics |
| [`summarize_scope_policy`](#summarize_scope_policy) | `queries/training_scope.gsql` | No | Preparation, and every run that opens the graph |
| [`resolve_split_cutoffs`](#resolve_split_cutoffs) | `queries/split_cutoffs.gsql` | No | Preparation, `mule score` |
| [`list_hub_accounts`](#list_hub_accounts) | `queries/hub_accounts.gsql` | No | Preparation, `mule score` |
| [`reveal_mule_labels`](#reveal_mule_labels) | `queries/label_reveal.gsql` | The label contract, only on a graph without known labels | Every preparation reaching the graph |
| [`draw_reveal_uniforms`](#draw_reveal_uniforms) | `queries/label_reveal.gsql` | No | The reveal, as a subquery |
| [`validate_label_contract`](#validate_label_contract) | `queries/label_contract.gsql` | No | After the reveal's check |
| [`read_ground_truth`](#read_ground_truth) | `evaluation/ground_truth.gsql` | No | `mule evaluate`, the experiments script, `mule diagnose` |
| [`fetch_analytics_context`](#fetch_analytics_context) | `analytics/analytics_context.gsql` (generated) | No | `mule diagnose` |
| [`encode_zelle_pair_gaps`](#encode_zelle_pair_gaps-and-encode_payment_pair_gaps) | `analytics/zelle_pair_gaps.gsql` | Only with `persist` | By hand, and the pair-gap check of `pytest -m graph` |
| [`encode_payment_pair_gaps`](#encode_zelle_pair_gaps-and-encode_payment_pair_gaps) | `analytics/payment_pair_gaps.gsql` | Only with `persist` | The same |

Only the reveal, the label-contract check and the oracle read the ground truth
(`is_mule`); no feature, population, cutoff or hub query reads a label attribute, and
tests check the rendered context queries for oracle attribute names.

### fetch_training_context

Produces the model's inputs. One call takes 1 to 64 requests, each an entity at its own
cutoff, and returns a row per request with its bounded candidate pool
([Features](features.md#what-the-training-query-returns) lists the row and message
fields).

| Parameter | Meaning |
|---|---|
| `node_types`, `node_ids` | Entity type (Account, Token, Party, Device, IP, Address) and primary id per request |
| `cutoff_seqs`, `cutoff_times` | Exclusive sequence and inclusive millisecond watermark per request |
| `per_relation`, `k_old`, `k_div` | Pool per payment relation: most recent events (1 to 32), older events at evenly spaced recency ranks (0 to 16), recent events with counterparties not yet in the pool (0 to 16) |
| `k_assoc` | Most recent active tenures per association relation (0 to 8; 0 for children) |
| `max_history` | Visible events per payment relation above which the request is rejected (32 to 4,096) |
| `scope_id`, `visibility_phase` | Scope and phase (1 train, 2 validation, 3 test); an empty scope is unscoped |
| `emit_encodings` | Also print the 64 Fourier coordinates of every age and gap, for the client's spot checks |
| `include_entity_meta`, `include_time_encoding`, `include_pair_history`, `include_flow_timing` | Compute each training group TigerGraph supplies (`FeaturePlan.query_flags`); each TRUE by default |

Per request:

1. Typed id lookup; unknown is `missing_entity`.
2. The entity must be first seen by the cutoff and, in a scope, have a partition at most
   the phase; else `invisible_entity`.
3. One scan per payment relation with `event_seq < cutoff_seq AND event_ts_ms <=
   cutoff_ms`: drops events with a From or To account outside the visible partitions
   before any feature or sampling, keeps USD events, validates roles and clocks, and
   rejects with `history_capacity_exceeded` above `max_history` visible events.
4. Keeps the pool: recent, older and distinct strata per payment relation.
5. Pair history (Accounts: one ascending pass over retained history; Tokens: a scan of the
   sender's history) and flow timing (Accounts only).
6. Active association tenures at `cutoff_seq - 1`, the most recent `k_assoc` per relation.
7. Set-based enrichment of every kept event: counterparty type, id, first-seen time,
   external and deposit flags.
8. Prints one row. A failed request prints `{status, request_index}` and the query moves
   on, so one bad key never costs the others.

Per-request statuses, which the client turns into rejected contexts: `invalid_request`,
`missing_entity`, `invisible_entity`, `history_capacity_exceeded`,
`nonmonotonic_pair_clock`, `invalid_payment_fields`, `invalid_event_roles`. Only
`invalid_parameters`, `invalid_visibility_phase` and `scope_not_ready` stop the call.

Generated by `tigergraph.render` (`python scripts/render_queries.py`; do not edit). Every
row prints `CONTEXT_CONTRACT`: `"context_"` plus the first 12 hex digits of the sha256 of
the rendered query without that literal, comments and whitespace removed and text outside
string literals lower-cased (`tigergraph.gsql_text.normalized`), so a case-only change
keeps the contract. A render test keeps the constant equal to the text, so a changed query
cannot ship without a new contract; the client refuses rows of another contract. The
repository text also runs under `INTERPRET` (`tigergraph.render.as_interpreted` swaps only
the header), as the graph tests use.

### encode_fourier64

The fixed time basis shared by GSQL and Python for one non-negative millisecond delta: 64
coordinates, 32 sine and cosine pairs ([Time encoding](../explanation/time-encoding.md)).
A subquery without REST endpoint, called by the context queries only when
`emit_encodings` is set. Changing it reinstalls its callers.

### create_training_scope

Partitions every Account and Party, frozen and label-blind, into train (1), validation (2)
and test (3) by ownership groups, in the client's shares (`train_share`,
`validation_share`, `test_share`, from the scope settings: 0.5, 0.25, 0.25 built in). A
non-positive share, or shares not summing to 1, are `invalid_parameters`.

- **Reads:** every Account and Party, every `Party_Owns_Account` tenure of all time, and
  for the `linked` rule every payment of the unowned internal accounts with their
  counterparty accounts. Never a label.
- **Computes:** ownership components, propagating the smallest internal vertex id over
  ownership edges (at most `max_iterations`, 100). A seeded hash of the component id
  (`split_seed`) picks one of 10,000 buckets; the bucket's middle, as a share, below
  `train_share` is train, below `train_share + validation_share` validation, else test.
  Shares are whole numbers of buckets, so no middle lies near a boundary.
- **Writes:** one `Temporal_Training_Scope` vertex (`ready = false`, source id, split seed,
  the three shares) and one `Entity_In_Training_Scope` edge per Account and Party, with
  `partition` and `group_id`. Runs once, one-hour timeout, single attempt; an existing
  scope is `scope_already_exists`.

Accounts no party owns follow `unowned_policy` (from `scope.unowned`); components with a
Party get the same component and partition under every rule.

| Rule | Placement |
|---|---|
| `independent` (query default) | Each its own component with a hashed partition |
| `shared` | Unowned external and unowned bank ledger accounts (`account_type = "gl"`) get partition 1 (visible in every phase) and group id `shared:<component>`; other unowned internal accounts stay independent |
| `linked` (built in) | As `shared`; also an unowned internal account whose distinct owned internal deposit counterparties (the other account of any payment, all time) are exactly one account takes that account's component, partition and group id |

Reference graph: `strict_mule_v2` (made when shares were fixed at 70, 15 and 15%) has
988,283 members: 414,074 internal accounts linked, 20,709 internal independent, 35,660
external shared. Owned components kept exactly the partitions of the earlier
`strict_mule_v1`.

### finalize_training_scope

`finalize_training_scope(scope_id, expected_members)` reads every membership edge, checks
the count and that each partition is 1 to 3 with a group id, then sets `ready = true`.
Every other scoped query refuses a scope not ready (`scope_not_ready`).

### list_scope_accounts

Pages a ready scope's internal deposit accounts, at most `batch_size` (10,000) per page in
account order after `after_id`: `account_id`, `first_seen_seq`, `first_seen_ts_ms`,
`partition`, `group_id`, `observed_positive`, `known_from_ms`. With
`include_observed = FALSE` (default) the last two are false and 0 for all. With
`include_observed = TRUE` (preparation) an account is an observed positive exactly when
it is the label contract's revealed positive (`pu_label == 1 AND is_mule == 1 AND
mule_label_known AND NOT is_mule_masked`), and only it has a discovery time
(`mule_label_available_ts_ms`). Neither field reveals a withheld label or which accounts
are labelled.

Reference graph: `strict_mule_v2` (70, 15 and 15%) holds 222,337 train, 47,754 validation
and 47,749 test accounts; preparation kept 24,059 (reservoirs and observed positives).

### summarize_scope_policy

Read only, about 0.7 seconds. For a ready scope it prints `members`, `unowned_accounts`,
six counts of unowned member Accounts by class and side, and `shared_ledger` out of
`ledger_accounts`. The six: `shared_internal`, `shared_external` (group id starts
`shared:`); `independent_internal`, `independent_external` (group id is the account's own
component); `linked_internal`, `linked_external` (any other group id). The client infers
the scope's rule (`tigergraph.scope.inferred_scope_policy`) and refuses one differing from
`scope.unowned`:

| Inferred | When |
|---|---|
| `independent` | Nothing shared or linked |
| `shared` | Every unowned external and ledger account shared, nothing linked |
| `linked` | As `shared`, at least one internal account linked |
| No rule | Anything else, such as shared internal customer accounts |

Rules writing identical membership read as the simplest: a `linked` scope where no
internal account qualified reads `shared`; one without unowned external accounts or links
reads `independent`. Set `scope.unowned` to the inferred value to use such a scope (same
membership). The vertex stores no rule, so this is the only check of an existing scope's
rule.

### resolve_split_cutoffs

Converts calendar cutoffs (1 to 24 millisecond times, each midnight UTC minus 1 ms) to
sequence watermarks: the largest event or first-seen sequence visible then, 0 if none.
The client adds 1 for `cutoff_seq`, so history is `event_seq < cutoff_seq AND
event_ts_ms <= cutoff_ms`. It scans every event and entity clock, so it runs once per
preparation, never per batch: about 6.6 seconds on the reference graph, giving
`cutoff_seq` 61,035,552 for 2024-07-01, 89,141,831 for 2024-10-01 and 120,799,198 for
2025-01-01 (covering every event).

### list_hub_accounts

For 1 to 24 cutoff sequences, lists Accounts whose visible history in some payment
relation (events before the cutoff, all currencies) exceeds `threshold`: `account_id`,
`cutoff_seq`, `visibility_phase`, `max_visible`, `max_degree`, `reason` (always
`visible_history`).

- Empty `scope_id`: unscoped counts, every row phase 3 (`mule score`).
- Ready scope: per phase 1, 2 and 3, only events whose From and To Accounts are all
  members with partition at most that phase count (the context query's endpoint rule),
  and the hub must be visible in the phase. No event after the cutoff or hidden in a
  phase can make a hub.
- An all-time outdegree prefilter finds candidates in constant time per account;
  `max_degree`, the largest all-time relation outdegree, is informational.

All-currency counts are an upper bound of the context query's USD-only check, so an
account whose USD history would fit may still be a hub: that costs history, never leaks
it.

### reveal_mule_labels

Decides once which mules a bank would have discovered before each split's cutoff, and
when, and writes them into the label contract
([Label reveal](../explanation/label-reveal.md)). Reads the ground truth and the graph's
per-payment fraud verdicts.

| Parameter | Meaning |
|---|---|
| `scope_id`, `train_cutoff_ms`, `validation_cutoff_ms`, `test_cutoff_ms` | The scope whose partitions are the splits; each split's latest cutoff |
| `budget` | Most mules revealed per split (`scope.reveal_per_split`, 20) |
| `salt` | Seed of the deterministic draws (`scope.reveal_salt`, 42) |
| `apply` | Write the labels; FALSE (default) only prints the plan |
| `force` | Reveal again on a graph with known labels |
| `p_report`, `p_action_first`, `p_action_later`, `proactive_per_day`, `trace_probability`, `propensity_slope`, `propensity_floor` | Discovery model (0.65, 0.5, 0.7, 0.00045, 0.25, 1.0, 0.05) |
| `version` | Recorded in `mule_label_source` (`reveal_v1`) |

Preparation calls it with `apply = TRUE`: without known labels it writes every internal
Account's label fields; with them it changes nothing and says `already_revealed`. A
shortfall is reported, never filled. `contract.discovery` holds the parameters and draws
the job shares with its Python mirror (`reference.label_reveal`).

### draw_reveal_uniforms

The reveal's deterministic uniforms for one key and salt: `n` numbers in (0, 1) from a
modular-arithmetic mixer over the prime 2^31 - 1, as GSQL has no bitwise operators.
`contract.discovery.reveal_uniforms` mirrors it exactly.

### validate_label_contract

Counts the label contract's accounts, known labels, true mules, masked mules, revealed
positives, and five violations: `invalid_mule`, `invalid_pu`, `invalid_unknown`,
`invalid_clocks`, `invalid_ring` ([Labels](labels.md)). Preparation runs it after the
reveal check and fails unless every violation count is zero.

### read_ground_truth

The oracle, the only query exporting truth: pages of at most `batch_size` (1 to 10,000)
Accounts after `after_id`, with `is_mule`, mask and PU label, label clocks,
`mule_ring_id`, `mule_label_source`. Audits and diagnostics read it through
`tigergraph.oracle.TigerGraphTruthReader`, which training cannot import (import contract
"Training never reads ground truth").

### fetch_analytics_context

The training context query with every feature group TigerGraph can compute, for analysis
only: same parameters, statuses and checks, an `include_*` flag for each of its 14 groups
(each TRUE by default), rows printing `ANALYTICS_CONTRACT`. The same renderer generates it
from `contract.analytics_features` ([Features](features.md#the-analytics-groups)).
Rendered with the old flag order it is byte for byte the earlier reviewed all-groups
context query, under a new name and contract. With `include_pair_window_counts` on, its
scan of earlier pair payments supplies the pair clock, so the chronology pass and its
`nonmonotonic_pair_clock` check are skipped; window and decayed sums keep the original
traversal order, so their floating-point values match the earlier query. `mule diagnose`
installs it where its text differs and reads it for the feature table; training never
calls it.

### encode_zelle_pair_gaps and encode_payment_pair_gaps

Exact pair queries for one sender Account and one canonical recipient up to a cutoff: each
payment's predecessor gap, age at the cutoff, both 64-dimensional encodings, and the
pair's counts over the preceding hour, day and week. The payment query also takes the
rail. `max_events` (default 1,000, at most 10,000) bounds the history returned and sorted,
not the adjacency scanned; an oversized history returns `history_limit_exceeded`, writing
nothing. `persist` (default FALSE) stores gaps and encodings on the payment vertices.
[Time encoding](../explanation/time-encoding.md) defines a pair.

## Installation

`mule install` installs `queries/` and `evaluation/`
(`contract.server.TRAINING_QUERY_FILES`); only `mule diagnose` installs the analytics
queries (`tigergraph.installer.install` with `analytics=True`). A dataset records the
hashes of the files preparation runs (`contract.server.QUERY_FILES`), so one prepared from
other query texts is refused. `mule install` and every connecting preparation install
only what is stale:

- **Stale**: `SHOW QUERY` text differs from the repository's (comments, whitespace and
  case outside string literals aside), or the REST endpoint is missing, disabled or has
  other parameters.
- **Callers**: a query calling a stale query installs with it (changing
  `encode_fourier64` reinstalls the context query). Only definitions whose text is missing
  or differs, and their callers, are created again, because `CREATE OR REPLACE` disables
  an installed endpoint until reinstalled. A current text with a stale endpoint
  (compilation unfinished or failed) is only installed.
- **Scope type first**: without a `Temporal_Training_Scope` vertex type,
  `gsql/schema/scope_vertex.gsql` is applied first, since a schema change invalidates
  installed queries. Only `mule install` replaces a differing one
  ([The scope types](#the-scope-types)).
- **The wait**: on TigerGraph 4.2.5 the install request answers only when compilation
  ends, so its read timeout is 90 minutes (`tigergraph.installer.INSTALL_DEADLINE_S`); if
  the client gives up first, the endpoint listing is polled every 30 seconds until all are
  enabled. Success means every endpoint matches the repository text and parameters, not a
  status message. A full install took about 50 minutes, mostly the context query, before
  the training query shrank; the analytics query is the size the context query was.
- **Timeout**: after 90 minutes the command fails, and the server may still be compiling.
  Wait until it ends (`mule check` lists no stale training queries; the GSQL shell's `ls`
  shows each query's state), then rerun: it installs only what is still stale and
  recreates nothing the last run created.
- **Console**: "Installing 3 queries on TigerGraph...", or "Installing all 12 queries on
  TigerGraph (about 50 minutes)..." when all are stale (a fresh graph); the compile wait
  in place on a terminal; then the time taken ("Installed 12 queries in 48.0 min"). The
  `install`, `install_unanswered`, `install_wait` and `installed` events hold the query
  names.
- **Single attempt**: every write (schema change, `CREATE`, install, `DROP`) runs once; a
  failure is reported, never repeated.

### The scope types

Every install first compares the graph's scope vertex and edge types with
`gsql/schema/scope_vertex.gsql` (`tigergraph.installer.scope_schema`): each type's
attributes by name and type in order, the vertex's primary id first (since
`create_training_scope` inserts by position), and the edge's vertex types and reverse
edge; not defaults. Differing types (such as a `Temporal_Training_Scope` made before the
scope recorded its split shares) are outdated: `mule check` reports how, the installs of
`mule train`, `mule diagnose` and the experiments script refuse them, and `mule install`
replaces them (`tigergraph.installer.replace_scope_types`):

1. It refuses, changing nothing, while the graph holds a scope vertex (replacing would
   delete it), an edge type the file does not declare reaches the scope vertex type, or
   an installed query no repository file defines uses the scope types or calls one that
   does. It never touches such types or queries.
2. It drops the repository's server queries that use the scope types (training,
   evaluation, analytics files), callers first, since TigerGraph drops no type a query
   uses: today the context, scope, hub, reveal and analytics context queries.
3. It drops the old scope edge and vertex types in schema change job
   `drop_training_scope`, applies `scope_vertex.gsql`, and checks the types now match.
4. It installs every training query with `-force`, whatever its text, since TigerGraph
   otherwise skips an installed query. The analytics context query waits for
   `mule diagnose`, as on a fresh graph.

The console reports what it replaces and drops (`scope_types` events); a second run finds
the types in place and installs only what is stale.

### The retired names

Once all queries are installed and verified, `mule install` alone drops each installed
query on `contract.server.RETIRED_QUERIES`: the pre-rename names (before queries were named
after their responsibility), plus two retired queries, the public Fourier wrapper and the
removed `shared_history` protocol's population query. It drops callers first, skips names
not installed, checks each drop against the endpoint listing and reports it. It touches no
other query; installed queries no repository file defines are listed as left in place.
Pre-rename code calls the old names, so run `mule install` only when no such job runs
anywhere; until then the installs of `mule train`, `mule diagnose` and the experiments
script put the renamed queries beside the old ones. `mule check` lists retired queries
still installed (`queries.retired` in `results/check.json`).
