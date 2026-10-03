# Queries

Every GSQL query of the repository: what it is responsible for, where it lives, who runs
it and what it reads and writes. A query is named verb first after its responsibility,
with no prefix, since the graph is dedicated; a file is named after the responsibility
its queries share. The names are constants of `contract.server`, which every adapter,
fake and test takes them from. Every query is installed on the graph
`Mule_Pattern_Learner` (`contract.server.GRAPH_NAME`).

| Folder | Holds | Installed by |
|---|---|---|
| `gsql/queries/` | The training pipeline's queries | `mule install`, and `mule train` when a text differs |
| `gsql/evaluation/` | The oracle, read only by the audits and the diagnostics | the same |
| `gsql/analytics/` | Queries used only for analysis | only `mule diagnose`, where a text differs |
| `gsql/schema/` | The graph's DDL and loading job, run by a person, and the scope schema change `mule install` applies | see [Set up a graph](../how-to/set-up-a-graph.md) |

## The queries

| Query | File | Writes | Run by |
|---|---|---|---|
| [`fetch_training_context`](#fetch_training_context) | `queries/training_context.gsql` (generated) | no | every training batch, the proxy evaluations, the audits, `mule score`, `mule check`, and `mule diagnose` (the feature table) |
| [`encode_fourier64`](#encode_fourier64) | `queries/fourier64.gsql` | no | the two context queries, as a subquery |
| [`create_training_scope`](#create_training_scope) | `queries/training_scope.gsql` | the scope | the first preparation, when `scope.id` does not exist |
| [`finalize_training_scope`](#finalize_training_scope) | `queries/training_scope.gsql` | the scope's `ready` flag | right after creation |
| [`list_scope_accounts`](#list_scope_accounts) | `queries/training_scope.gsql` | no | preparation, the audits, the diagnostics |
| [`summarize_scope_policy`](#summarize_scope_policy) | `queries/training_scope.gsql` | no | preparation, and every run that opens the graph |
| [`resolve_split_cutoffs`](#resolve_split_cutoffs) | `queries/split_cutoffs.gsql` | no | preparation, `mule score` |
| [`list_hub_accounts`](#list_hub_accounts) | `queries/hub_accounts.gsql` | no | preparation, `mule score` |
| [`reveal_mule_labels`](#reveal_mule_labels) | `queries/label_reveal.gsql` | the label contract | every preparation that reaches the graph; it writes only on a graph without known labels |
| [`draw_reveal_uniforms`](#draw_reveal_uniforms) | `queries/label_reveal.gsql` | no | the reveal, as a subquery |
| [`validate_label_contract`](#validate_label_contract) | `queries/label_contract.gsql` | no | after the reveal's check |
| [`read_ground_truth`](#read_ground_truth) | `evaluation/ground_truth.gsql` | no | `mule evaluate`, the experiments script, `mule diagnose` |
| [`fetch_analytics_context`](#fetch_analytics_context) | `analytics/analytics_context.gsql` (generated) | no | `mule diagnose` |
| [`encode_zelle_pair_gaps`](#encode_zelle_pair_gaps-and-encode_payment_pair_gaps) | `analytics/zelle_pair_gaps.gsql` | only with `persist` | by hand, and the pair-gap check of `pytest -m graph` |
| [`encode_payment_pair_gaps`](#encode_zelle_pair_gaps-and-encode_payment_pair_gaps) | `analytics/payment_pair_gaps.gsql` | only with `persist` | the same |

Only the reveal, the label-contract check and the oracle read the ground truth
(`is_mule`); no feature, population, cutoff or hub query reads a label attribute, and
the tests check the rendered context queries for oracle attribute names.

### fetch_training_context

The query that produces the model's inputs. One call carries 1 to 64 requests, each an
entity at its own cutoff, and returns one row per request with its bounded candidate
pool. [Features](features.md#what-the-training-query-returns) lists the row and its
message fields.

| Parameter | Meaning |
|---|---|
| `node_types`, `node_ids` | Entity type (Account, Token, Party, Device, IP, Address) and primary id per request |
| `cutoff_seqs`, `cutoff_times` | The exclusive sequence and the inclusive millisecond watermark per request |
| `per_relation`, `k_old`, `k_div` | The candidate pool per payment relation: most recent events (1 to 32), older events at evenly spaced recency ranks (0 to 16), recent events with counterparties not yet in the pool (0 to 16) |
| `k_assoc` | Most recent active tenures per association relation (0 to 8; 0 for children) |
| `max_history` | Visible events per payment relation above which the request is rejected (32 to 4,096) |
| `scope_id`, `visibility_phase` | The experiment scope and phase (1 train, 2 validation, 3 test); an empty scope is unscoped |
| `emit_encodings` | Also print the 64 Fourier coordinates of every age and gap (for the client's spot checks) |
| `include_entity_meta`, `include_time_encoding`, `include_pair_history`, `include_flow_timing` | Whether to compute each training group TigerGraph supplies (`FeaturePlan.query_flags`); each is TRUE by default |

For each request the query:

1. Resolves the id with a typed lookup; an unknown id is `missing_entity`.
2. Checks that the entity was first seen at or before the cutoff and, in a scope, that
   its partition is at most the phase; otherwise `invisible_entity`.
3. Scans each payment relation once with `event_seq < cutoff_seq AND event_ts_ms <=
   cutoff_ms`, drops every event with a From or To account outside the visible
   partitions before any feature or sampling, keeps USD events, validates roles and
   clocks, and rejects the request with `history_capacity_exceeded` when more than
   `max_history` events are visible.
4. Keeps the pool: the recent, older and distinct strata per payment relation.
5. Computes pair history (for Accounts one ascending pass over the retained history; for
   Tokens a scan of the sender's history) and flow timing (Accounts only).
6. Reads the active association tenures at `cutoff_seq - 1` and keeps the most recent
   `k_assoc` per relation.
7. Enriches every kept event in set-based selects: counterparty type, id, first-seen time,
   and external and deposit flags.
8. Prints one row. A failed request prints `{status, request_index}` and the query goes
   on with the next request, so one bad key never costs the others.

Per-request statuses, which the client turns into a rejected context: `invalid_request`,
`missing_entity`, `invisible_entity`, `history_capacity_exceeded`,
`nonmonotonic_pair_clock`, `invalid_payment_fields` and `invalid_event_roles`. Only
`invalid_parameters`, `invalid_visibility_phase` and `scope_not_ready` stop the whole
call.

The file is generated by `tigergraph.render` (`python scripts/render_queries.py`; do not
edit it). Every row prints `CONTEXT_CONTRACT`: `"context_"` and the first 12 hex digits of
the sha256 of the rendered query without that literal, with comments and whitespace
removed and the text outside string literals lower-cased (`tigergraph.gsql_text.normalized`),
so a change of case alone keeps the contract. A render test keeps the constant equal to the text, so a changed query cannot
ship without a new contract, and the client refuses a row of another contract. The
repository text also runs under `INTERPRET` (`tigergraph.render.as_interpreted` swaps
only the header), which the tests against the graph use.

### encode_fourier64

The fixed time basis shared by GSQL and Python, for one non-negative millisecond delta:
64 coordinates, 32 sine and cosine pairs ([Time encoding](../explanation/time-encoding.md)).
It is a subquery with no REST endpoint; the context queries call it only when
`emit_encodings` is set. A change to it reinstalls the queries that call it.

### create_training_scope

Creates a frozen, label-blind partition of every Account and Party into train (1, 70%),
validation (2, 15%) and test (3, 15%), by ownership groups.

- **Reads:** every Account and Party, every `Party_Owns_Account` tenure of all time, and,
  for the `linked` rule, every payment of the unowned internal accounts with their
  counterparty accounts. Never a label.
- **Computes:** ownership components, by propagating the smallest internal vertex id over
  ownership edges (at most `max_iterations`, 100), each with a partition from a seeded
  hash of its id (`split_seed`).
- **Places the accounts no party owns** (`unowned_policy`, from `scope.unowned`):
  - `independent` (the query's default): each is its own component with a hashed
    partition;
  - `shared`: unowned external accounts and unowned bank ledger accounts (`account_type =
    "gl"`) get partition 1, visible in every phase, and the group id
    `shared:<component>`; other unowned internal accounts stay independent;
  - `linked` (the built-in run's rule): as `shared`, and an unowned internal account whose
    distinct owned internal deposit counterparties (the other account of any payment, all
    time) are exactly one account takes that account's component, partition and group id.
  Components with a Party keep the same component and partition under every rule.
- **Writes:** one `Temporal_Training_Scope` vertex (`ready = false`) and one
  `Entity_In_Training_Scope` edge per Account and Party, with `partition` and
  `group_id`. It runs once, with a one-hour timeout and a single attempt; an existing
  scope is `scope_already_exists`.

On the reference graph, `strict_mule_v2` has 988,283 members: 414,074 internal accounts
linked, 20,709 internal accounts left independent and 35,660 external accounts shared.
The owned components kept exactly the partitions of the earlier `strict_mule_v1`.

### finalize_training_scope

`finalize_training_scope(scope_id, expected_members)` reads every membership edge,
checks the count and that each partition is 1 to 3 with a group id, then sets `ready =
true`. Every other scoped query refuses a scope that is not ready (`scope_not_ready`).

### list_scope_accounts

Pages the internal deposit accounts of a ready scope, at most `batch_size` (10,000) per
page in account order after `after_id`: `account_id`, `first_seen_seq`,
`first_seen_ts_ms`, `partition`, `group_id`, `observed_positive` and `known_from_ms`.
With `include_observed = FALSE` (the default) the last two are false and 0 for every
account. With `include_observed = TRUE`, which preparation uses, an account is an
observed positive exactly when it is the label contract's revealed positive (`pu_label ==
1 AND is_mule == 1 AND mule_label_known AND NOT is_mule_masked`), and only it has a
discovery time (`mule_label_available_ts_ms`). Neither field reveals a withheld label or
which accounts are labelled.

On the reference graph the built-in scope (`strict_mule_v2`) holds 222,337 train, 47,754
validation and 47,749 test accounts, and preparation kept 24,059 of them: the reservoirs
and the observed positives.

### summarize_scope_policy

Read-only, about 0.7 seconds. For a ready scope it prints `members`, `unowned_accounts`
and six counts of unowned member Accounts by class and side: `shared_internal` and
`shared_external` (group id starts with `shared:`), `independent_internal` and
`independent_external` (group id is the account's own component), `linked_internal` and
`linked_external` (any other group id), plus `shared_ledger` out of `ledger_accounts`.
The client infers the rule the scope was created with from these counts
(`tigergraph.scope.inferred_scope_policy`) and refuses a scope whose rule differs from
`scope.unowned`:

- `independent`: nothing shared and nothing linked;
- `shared`: every unowned external and ledger account shared, nothing linked;
- `linked`: as `shared`, with at least one internal account linked;
- no rule: anything else, such as shared internal customer accounts.

Rules that write identical membership read as the simplest of them: a `linked` scope in
which no internal account qualified reads as `shared`, and a scope without unowned
external accounts or links reads as `independent`. Set `scope.unowned` to the inferred
value to use such a scope; its membership is the same. The vertex stores no rule, so this
inference is the only check that an existing scope was created with the configured rule.

### resolve_split_cutoffs

Converts calendar cutoffs (1 to 24 millisecond times, each midnight UTC minus 1 ms) to
sequence watermarks: for each, the largest event or first-seen sequence visible at that
time, 0 when nothing is. The client uses that value plus 1 as `cutoff_seq`, so history
is `event_seq < cutoff_seq AND event_ts_ms <= cutoff_ms`. It scans every event and entity
clock, so it runs once per preparation, never per batch (about 6.6 seconds on the
reference graph). There it gave the `cutoff_seq` 61,035,552 for 2024-07-01, 89,141,831 for
2024-10-01 and 120,799,198 for 2025-01-01, which covers every event.

### list_hub_accounts

For 1 to 24 cutoff sequences, lists the Accounts whose visible history in some payment
relation (events before the cutoff, all currencies) exceeds `threshold`: rows of
`account_id`, `cutoff_seq`, `visibility_phase`, `max_visible`, `max_degree` and `reason`
(always `visible_history`).

- With an empty `scope_id` the counts are unscoped and every row has phase 3 (`mule
  score`).
- With a ready scope it counts per phase 1, 2 and 3 only the events whose From and To
  Accounts are all members with a partition at most that phase, the endpoint rule of
  the context query, and the hub itself must be visible in the phase. No event after the
  cutoff and no event hidden in a phase can make an account a hub.
- An all-time outdegree prefilter finds the candidates in constant time per account;
  `max_degree`, the largest all-time relation outdegree, is informational only.

The counts cover all currencies, an upper bound of the context query's USD-only check,
so an account whose USD history would fit may still be a hub: that costs history, never
leaks it.

### reveal_mule_labels

Decides once which mules a bank would have discovered before each split's cutoff, and
when, and writes them into the label contract ([Label reveal](../explanation/label-reveal.md)).
It reads the ground truth and the graph's per-payment fraud verdicts.

| Parameter | Meaning |
|---|---|
| `scope_id`, `train_cutoff_ms`, `validation_cutoff_ms`, `test_cutoff_ms` | The scope whose partitions are the splits, and each split's latest cutoff |
| `budget` | Mules revealed per split at most (`scope.reveal_per_split`, 20) |
| `salt` | Seed of the deterministic draws (`scope.reveal_salt`, 42) |
| `apply` | Write the labels; FALSE (the default) only prints the plan |
| `force` | Reveal again on a graph that already has known labels |
| `p_report`, `p_action_first`, `p_action_later`, `proactive_per_day`, `trace_probability`, `propensity_slope`, `propensity_floor` | The discovery model's parameters (0.65, 0.5, 0.7, 0.00045, 0.25, 1.0, 0.05) |
| `version` | Recorded in `mule_label_source` (`reveal_v1`) |

Preparation calls it with `apply = TRUE`: on a graph without known labels it writes
every internal Account's label fields; on a graph with them it changes nothing and says
`already_revealed`. A shortfall is reported, never filled. `contract.discovery` holds
the parameters and draws the job and its Python mirror (`reference.label_reveal`) share.

### draw_reveal_uniforms

The reveal's deterministic uniforms for one key and salt: `n` numbers in (0, 1) from a
modular-arithmetic mixer over the prime 2^31 - 1, since GSQL has no bitwise operators.
`contract.discovery.reveal_uniforms` mirrors it exactly.

### validate_label_contract

Counts the label contract's accounts, known labels, true mules, masked mules and revealed
positives, and five violations: `invalid_mule`, `invalid_pu`, `invalid_unknown`,
`invalid_clocks` and `invalid_ring` ([Labels](labels.md)). Preparation runs it after the
reveal check and fails unless every violation count is zero.

### read_ground_truth

The oracle: pages of at most `batch_size` (1 to 10,000) Accounts after `after_id`, with
`is_mule`, the mask and PU label, the label clocks, `mule_ring_id` and
`mule_label_source`. It is the only query that exports truth. The audits and the
diagnostics read it through `tigergraph.oracle.TigerGraphTruthReader`, which training
cannot import (the import contract "Training never reads ground truth").

### fetch_analytics_context

The training context query with every feature group TigerGraph can compute, for
analysis only: the same parameters, statuses and checks, an `include_*` flag for each of
its 14 groups (each TRUE by default), and rows that print `ANALYTICS_CONTRACT`. It is
generated by the same renderer from `contract.analytics_features`
([Features](features.md#the-analytics-groups)). Rendered with the old flag order it is
the reviewed all-groups context query of before byte for byte, under a new name and
contract. When `include_pair_window_counts` is on, its scan of earlier pair payments
supplies the pair clock, so the chronology pass and its `nonmonotonic_pair_clock` check
are skipped; its window and decayed sums keep the original traversal order, so their
floating-point values are those of the earlier query. `mule diagnose` installs it where
its text differs and reads it for its feature table; training never calls it.

### encode_zelle_pair_gaps and encode_payment_pair_gaps

Exact pair queries for one sender Account and one canonical recipient up to a cutoff:
each payment's predecessor gap, its age at the cutoff, both 64-dimensional encodings, and
the pair's counts over the preceding hour, day and week. The payment query also takes the
rail. `max_events` (1,000 by default, at most 10,000) bounds the history returned and
sorted, not the adjacency scanned; an oversized history returns
`history_limit_exceeded` and writes nothing. `persist` (FALSE by default) stores the gaps
and encodings on the payment vertices. [Time encoding](../explanation/time-encoding.md)
describes what a pair is.

## Installation

`mule install`, and every preparation that connects, installs only what is stale:

- A query is stale when its `SHOW QUERY` text differs from the repository's (comments,
  whitespace and the case outside string literals aside), its REST endpoint is missing or
  disabled, or the endpoint's parameters differ.
- A query that calls a stale query is installed with it, so a change to
  `encode_fourier64` also reinstalls the context query. Only the definitions whose text
  is missing or differs, and the queries that call them, are created again, because
  `CREATE OR REPLACE` disables an installed endpoint until it is installed again. A query
  whose text is current but whose endpoint is not (a compilation that has not finished,
  or failed) is only installed.
- When the graph has no `Temporal_Training_Scope` vertex type,
  `gsql/schema/scope_vertex.gsql` is applied first, since a schema change invalidates
  installed queries.
- On TigerGraph 4.2.5 the install request answers only when compilation finishes, so it
  runs with a 90-minute read timeout (`tigergraph.installer.INSTALL_DEADLINE_S`); when
  the client gives up first, the endpoint listing is polled every 30 seconds until every
  query is enabled. Success is decided by checking every endpoint against the repository
  text and parameters, not by a status message. Installing every query took about 50
  minutes, most of it the context query, before the training query shrank; the analytics
  query is the size the context query was.
- If the 90 minutes pass, the command fails, and the server may still be compiling. Wait
  until it has finished (`mule check` no longer lists the training queries under
  `queries.stale`; the GSQL shell's `ls` shows every query's state), then run the same
  command again: it installs only what is still stale, and creates nothing the last run
  created.
- Every write (the schema change, `CREATE`, the install, `DROP`) runs once: a failed one
  is reported, never repeated.

`mule install` installs the folders `queries/` and `evaluation/`
(`contract.server.TRAINING_QUERY_FILES`). The analytics queries are installed only by
`mule diagnose` (`tigergraph.installer.install` with `analytics=True`). A dataset records
the hashes of the files preparation runs (`contract.server.QUERY_FILES`), so a dataset
prepared from other query texts is refused.

### The retired names

Once every query is installed and verified, `mule install`, and no other command, drops
each installed query named on `contract.server.RETIRED_QUERIES`: the names the queries had before they were named
after their responsibility, and two retired queries, the public Fourier wrapper and the
population query of the removed `shared_history` protocol. It drops callers before the
queries they call, skips the names that are not installed, checks each drop against the
endpoint listing and lists what it dropped under `dropped`. It never touches any other
query: installed queries that no repository file defines are only listed
(`not_defined`). Code from before the rename calls the old names, so run `mule install`
only when no such job runs anywhere; until then the installs of `mule train`, `mule
diagnose` and the experiments script put the renamed queries beside the old names.
`mule check` lists the retired queries still installed under `queries.retired`.
