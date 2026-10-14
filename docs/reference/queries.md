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
| `gsql/schema/` | The graph's DDL and loading job (run by a person), and the scope schema change `mule install` applies | See [Set up a graph](../how-to/set-up-a-graph.md) |

## The queries

| Query | File | Writes | Run by |
|---|---|---|---|
| [`fetch_training_context`](#fetch_training_context) | `queries/training_context.gsql` | No | Every training batch, proxy evaluations, audits, `mule score`, `mule check`, `mule diagnose` (feature table) |
| [`encode_fourier64`](#encode_fourier64) | `queries/fourier64.gsql` | No | The context query, as a subquery |
| [`create_training_scope`](#create_training_scope) | `queries/training_scope.gsql` | The split | The first preparation, when the graph has no split |
| [`finalize_training_scope`](#finalize_training_scope) | `queries/training_scope.gsql` | The split's `ready` flag | Right after creation |
| [`list_scope_accounts`](#list_scope_accounts) | `queries/training_scope.gsql` | No | Preparation, audits, diagnostics |
| [`resolve_split_cutoffs`](#resolve_split_cutoffs) | `queries/split_cutoffs.gsql` | No | Preparation, `mule score` |
| [`list_hub_accounts`](#list_hub_accounts) | `queries/hub_accounts.gsql` | No | Preparation, `mule score` |
| [`reveal_mule_labels`](#reveal_mule_labels) | `queries/label_reveal.gsql` | The Account labels, only on a graph without known labels | Every preparation reaching the graph |
| [`draw_reveal_uniforms`](#draw_reveal_uniforms) | `queries/label_reveal.gsql` | No | The reveal, as a subquery |
| [`read_ground_truth`](#read_ground_truth) | `evaluation/ground_truth.gsql` | No | `mule evaluate`, the experiments script, `mule diagnose` |

Only the reveal and the oracle read the ground truth (`is_mule`); no feature, population,
cutoff or hub query reads a label attribute, and tests check the context query file for
oracle attribute names.

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
| `visibility_phase` | 0 (default): no split, everything visible (`mule score`); 1 train, 2 validation, 3 test: only the graph's split's partitions up to it |
| `emit_encodings` | Also print the 64 Fourier coordinates of every age and gap, for the client's spot checks; the client asks for them only when its plan has the time encoding |

The query computes every group TigerGraph supplies, whatever the model: the client
validates the groups of its plan and reads only those (`FeaturePlan.node_names`,
`FeaturePlan.edge_names`).

Per request:

1. Typed id lookup; unknown is `missing_entity`.
2. The entity must be first seen by the cutoff and, in a phase of the split, have a
   partition at most the phase; else `invisible_entity`.
3. One scan per payment relation for USD events with `event_seq < cutoff_seq AND
   event_ts_ms <= cutoff_ms`, then one pass over the four that drops events with a From or
   To account outside the visible partitions, before any feature or sampling. More than
   `max_history` visible events in a relation is `history_capacity_exceeded`. Each kept
   event takes its counterparty from its role edges: type, id, first-seen time, external
   and deposit flags.
4. Keeps the pool of each payment relation: its most recent events, older ones at evenly
   spaced recency ranks, then recent events with counterparties not yet in the pool.
5. Pair history (Accounts: one ascending pass over retained history; Tokens: a scan of the
   sender's history) and flow timing (Accounts only).
6. Active association tenures at `cutoff_seq - 1`, the most recent `k_assoc` per relation:
   one read per target type, since a root has at most one relation to each.
7. Prints one row. A failed request prints `{status, request_index}` and the query moves
   on, so one bad key never costs the others.

Per-request statuses, which the client turns into rejected contexts: `missing_entity`,
`invisible_entity`, `history_capacity_exceeded`. Only `invalid_parameters` (a request or
pool outside its bounds, or a phase outside 0 to 3) and `scope_not_ready` (a phase of the
split while the graph does not hold exactly one ready split) stop the call. The query
assumes, without checking, what held on the reference data: positive event clocks, role
edges whose clocks equal their event's, one sending account and a recipient per event,
and a strictly increasing clock within each counterparty pair.

Every row prints `CONTEXT_CONTRACT`: `"context_"` plus the first 12 hex digits of the
sha256 of the query file without that literal, comments and whitespace removed and text
outside string literals lower-cased (`tigergraph.gsql_text.query_contract`), so a layout or
case-only change keeps the contract. A test keeps the constant equal to the text, so a
changed query cannot ship without a new contract; the client refuses rows of another
contract. The repository text also runs under `INTERPRET`
(`testing.gsql_blocks.as_interpreted` swaps only the header), as the graph tests use.

### encode_fourier64

The fixed time basis shared by GSQL and Python for one non-negative millisecond delta: 64
coordinates, 32 sine and cosine pairs ([Time encoding](../explanation/time-encoding.md)).
A subquery without REST endpoint, called by the context query only when
`emit_encodings` is set. Changing it reinstalls its callers.

### create_training_scope

Creates the graph's one split: every Account and Party, frozen and label-blind, into train
(1), validation (2) and test (3) by ownership groups, in the client's shares
(`train_share`, `validation_share`, `test_share`, from the scope settings: 0.5, 0.25, 0.25
built in) with `split_seed` (`dataset.split_seed`, 42).

- **Reads:** every Account and Party, every `Party_Owns_Account` tenure of all time, and
  every payment of the unowned internal accounts with its counterparty accounts. Never a
  label.
- **Computes:** ownership components, propagating the smallest internal vertex id over
  ownership edges (at most 100 iterations). A seeded hash of the component id
  (`split_seed`) picks one of 10,000 buckets; the bucket's middle, as a share, below
  `train_share` is train, below `train_share + validation_share` validation, else test.
  Shares are whole numbers of buckets, so no middle lies near a boundary.
- **Writes:** the `Temporal_Training_Scope` vertex `"split"` (`ready = false`, split seed,
  the three shares) and one `Entity_In_Training_Scope` edge per Account and Party, with
  its `partition`. Runs once, one-hour timeout, single attempt.
- **Prints:** `status` (`ok`) and `expected_members`, the count `finalize_training_scope`
  checks. It refuses with `invalid_parameters` (a share not positive, or shares not adding
  up to 1), `scope_already_exists` (the graph holds a scope vertex) or
  `ownership_components_did_not_converge` (still changing after 100 iterations).

A split is never changed or deleted by the code. A run whose split seed or shares differ
from the split's is refused. To split again, delete the `Temporal_Training_Scope` vertices
by hand (their membership edges go with them) and move the old split's datasets out of
`data/`: the next `mule train` creates the configured one.

Accounts no party owns follow one rule; they never change a component with a Party:

- unowned external accounts and unowned bank ledger accounts (`account_type = "gl"`) are
  shared: partition 1, visible in every phase;
- an unowned internal account whose distinct owned internal deposit counterparties (the
  other account of any payment, all time) are exactly one account joins that account's
  ownership group and partition;
- any other unowned internal account is its own group, with a hashed partition.

Reference graph, split 70, 15 and 15% (before the shares were settings): 988,283 members,
414,074 unowned internal accounts linked, 20,709 in groups of their own, 35,660 external
shared. Owned components kept exactly the partitions of the split made before unowned
accounts were placed by this rule.

### finalize_training_scope

`finalize_training_scope(expected_members)` reads every membership edge of the graph's
one scope vertex (`invalid_scope` unless there is exactly one, or with `expected_members`
0), checks the count and that each partition is 1 to 3, then sets `ready = true` and
prints `ok` and `members`; otherwise it prints `incomplete_membership` and `members`.
Every other query of the split refuses unless the graph holds exactly one ready split
(`scope_not_ready`).

### list_scope_accounts

Pages the ready split's internal deposit accounts, at most `batch_size` (1 to 10,000;
10,000 by default, else `invalid_batch_size`) per page in account order after `after_id`:
`account_id`, `first_seen_ts_ms`, `partition`, `observed_positive`, `known_from_ms`. An
account is an observed positive exactly when its label is revealed (`pu_label == 1`), and
only it has a discovery time, its `mule_label_available_ts_ms`; every other account has
false and 0. Neither field reveals a withheld label or which accounts are labelled.

Reference graph, split 70, 15 and 15%: 222,337 train, 47,754 validation and 47,749 test
accounts; preparation kept 24,059 (reservoirs and observed positives).

### resolve_split_cutoffs

Converts calendar cutoffs (1 to 24 millisecond times, each midnight UTC minus 1 ms) to
sequence watermarks: the largest event or first-seen sequence visible then, 0 if none.
The client adds 1 for `cutoff_seq`, so history is `event_seq < cutoff_seq AND
event_ts_ms <= cutoff_ms`. It scans every event and entity clock, so it runs once per
preparation, never per batch: about 6.6 seconds on the reference graph, giving
`cutoff_seq` 61,035,552 for 2024-07-01, 89,141,831 for 2024-10-01 and 120,799,198 for
2025-01-01 (covering every event).

### list_hub_accounts

For 1 to 24 cutoff sequences (`cutoff_seqs`, none 0), lists Accounts whose visible
history in some payment relation (events before the cutoff, all currencies) exceeds
`threshold` (2,048 by default). It prints `status` and `hubs`, one unordered row per hub,
cutoff and phase: `account_id`, `cutoff_seq`, `visibility_phase`, `max_visible` (the
largest relation's visible count). Other cutoffs, or a `threshold` below 1, are
`invalid_parameters`.

- `by_phase = FALSE` (default): every event counts, every row phase 0 (`mule score`).
- `by_phase = TRUE` (preparation; else `scope_not_ready` without a ready split): rows for
  phases 1, 2 and 3. `finalize_training_scope` gives every Account a partition from 1 to
  3; an event's phase is the largest partition among its Account endpoints (the context
  query's endpoint rule), and it counts in that phase and every later one. A row of phase
  p needs the hub's own partition at most p. No event after the cutoff or hidden in a
  phase can make a hub.
- An all-time outdegree prefilter picks the accounts to count in constant time per
  account; one `SELECT` over the four relations' edge types then counts each relation's
  events per cutoff and phase.

All-currency counts are an upper bound of the context query's USD-only check, so an
account whose USD history would fit may still be a hub: that costs history, never leaks
it.

### reveal_mule_labels

Decides which mules a bank would have discovered before each split's cutoff, and when,
and writes them into the Account labels
([Label reveal](../explanation/label-reveal.md)). Reads the ground truth and the graph's
per-payment fraud verdicts. Every mule discovered before its split's cutoff is revealed;
nothing caps a split.

| Parameter | Meaning |
|---|---|
| `train_cutoff_ms`, `validation_cutoff_ms`, `test_cutoff_ms` | Each split's latest cutoff, increasing (else `invalid_cutoffs`); the splits are the partitions of the graph's ready split (else `scope_not_ready`) |
| `salt` | Seed of the deterministic draws (`scope.reveal_salt`, 42) |
| `apply` | Write the labels; FALSE (default) only prints the plan |
| `force` | Reveal again on a graph with known labels |

The discovery model's parameters are constants of the query: `p_report` 0.65,
`p_action_first` 0.5, `p_action_later` 0.7, `proactive_per_day` 0.00045,
`trace_probability` 0.25. A mule's label is available at the end of its discovery's UTC
day, never after the last event's time and never before the account's `first_seen_ts_ms`.

- **Prints:** `status` (`dry_run`, or `ok` with `apply`), `revealed` (the mules revealed
  per partition), `data_end_ts_ms` (the last event's time) and `revealed_mules`
  (`account_id`, `known_ts_ms`, `channel`).
- **Writes**, with `apply = TRUE`, every internal Account: `mule_label_known = TRUE`; for
  a mule, `pu_label` (1 revealed, 0 hidden), its availability as
  `mule_label_available_ts_ms`, and `mule_label_source`, the reveal's record plus
  `;revealed:<channel>` or `;hidden`; for a non-mule, `pu_label = 0`, its
  `first_seen_ts_ms` as availability and the source `phantomledger_role`. External
  accounts stay unknown.

The record is `phantomledger_role;reveal_v1;salt=<salt>;budget=1000`, so a revealed
mule's source reads
`phantomledger_role;reveal_v1;salt=42;budget=1000;revealed:victim_report`. It is the
record earlier code wrote, whose `budget=1000` meant no cap, kept byte for byte so the
labels already on a graph count as this reveal's. On a graph with known labels and without
`force` the query changes nothing and prints `status`, `revealed_labels`, `other_reveal`
(internal mules whose label source is not this reveal's), `mules` (internal mules) and
`reveal_record`: `already_revealed` when `other_reveal` is 0, else `revealed_differently`.

Preparation calls it with `apply = TRUE`; on `revealed_differently` it calls it again with
`force`, and so it does on `already_revealed` with `mules` when the run has just created
the split, whose partitions the labels follow. Labels on a graph with no internal mule are
another source's and stay. Before a run reads the graph, a dry run with the dataset's
settings must say `already_revealed`. `contract.discovery` holds the salt default, the model constants (a
test compares them with the query) and the draws the job shares with its Python mirror
(`reference.label_reveal`).

### draw_reveal_uniforms

The reveal's deterministic uniforms for one key and salt: `n` numbers in (0, 1) from a
modular-arithmetic mixer over the prime 2^31 - 1, as GSQL has no bitwise operators.
`contract.discovery.reveal_uniforms` mirrors it exactly.

### read_ground_truth

The oracle, the only query exporting truth: pages of at most `batch_size` (1 to 10,000;
1,000 by default, else `invalid_batch_size`) Accounts in account order after `after_id`:
`account_id`, `is_mule`, `mule_label_known`, `mule_ring_id`, `mule_label_source`. Audits
and diagnostics read it through `tigergraph.oracle.TigerGraphTruthReader`, which training
cannot import (import contract "Training never reads ground truth").

## Installation

`mule install` installs `queries/` and `evaluation/`
(`contract.server.TRAINING_QUERY_FILES`). A dataset records the hashes of the files
preparation runs (`contract.server.QUERY_FILES`), so one prepared from other query texts
is refused. `mule install` and every connecting preparation install
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
  the training query shrank.
- **Timeout**: after 90 minutes the command fails, and the server may still be compiling.
  Wait until it ends (`mule check` lists no stale training queries; the GSQL shell's `ls`
  shows each query's state), then rerun: it installs only what is still stale and
  recreates nothing the last run created.
- **Console**: "Installing 3 queries on TigerGraph...", or "Installing all 10 queries on
  TigerGraph (about 50 minutes)..." when all are stale (a fresh graph); the compile wait
  in place on a terminal; then the time taken ("Installed 10 queries in 48.0 min"). The
  `install`, `install_unanswered`, `install_wait` and `installed` events hold the query
  names.
- **Single attempt**: every write (schema change, `CREATE`, install, `DROP`) runs once; a
  failure is reported, never repeated.

### The scope types

Every install first compares the graph's scope vertex and edge types with
`gsql/schema/scope_vertex.gsql` (`tigergraph.installer.scope_schema`): each type's
attributes by name and type in order, the vertex's primary id first (since
`create_training_scope` inserts by position), and the edge's vertex types and reverse
edge; not defaults. Differing types (such as a `Temporal_Training_Scope` keyed by a scope
id, from when a graph held several splits, or a membership edge with `group_id`) are
outdated: `mule check` reports how, the installs of `mule train`, `mule diagnose` and the
experiments script refuse them, and `mule install` replaces them
(`tigergraph.installer.replace_scope_types`):

1. It refuses, changing nothing, while the graph holds a scope vertex (replacing would
   delete it, which only its owner does, by hand, as the refusal says), an edge type the
   file does not declare reaches the scope vertex type, or a query on the server,
   installed or only created (`SHOW QUERY *` lists both), that is neither the
   repository's nor retired uses the scope types or calls one that does. It never touches
   such types or queries.
2. It drops the repository's and the [retired](#the-retired-names) queries on the server
   that use the scope types, callers first, since TigerGraph drops no type a query uses:
   today the context, scope, hub and reveal queries, and `summarize_scope_policy` and
   `fetch_analytics_context` where they are still on the server, installed or not.
3. It drops the old scope edge and vertex types in schema change job
   `drop_training_scope`, applies `scope_vertex.gsql`, and checks the types now match.
4. It installs every training query with `-force`, whatever its text, since TigerGraph
   otherwise skips an installed query; the retired ones stay dropped.

The console reports what it replaces and drops (`scope_types` events); a second run finds
the types in place and installs only what is stale.

A graph that holds its split keeps outdated types until the owner deletes the split by
hand ([After the data changes](../how-to/set-up-a-graph.md#after-the-data-changes)) and
runs `mule install`. The next `mule train` creates the split again: on the same loaded
graph it has the same partitions, since the seed, shares and rule are the same and the
hash uses TigerGraph's internal vertex ids, and it reveals the same labels again
([On the CUDA host](../how-to/train-and-evaluate.md#on-the-cuda-host) has the steps).

### The retired names

Once all queries are installed and verified, `mule install` alone drops each installed
query on `contract.server.RETIRED_QUERIES`: the pre-rename names (before queries were named
after their responsibility), plus two retired queries, the public Fourier wrapper and the
removed `shared_history` protocol's population query, then two that nothing reads any
more, the split's unowned-rule summary `summarize_scope_policy` and the label-contract
check `validate_label_contract`, and last the deleted analytics queries,
`fetch_analytics_context` first, then `encode_zelle_pair_gaps` and
`encode_payment_pair_gaps`. It drops callers first, skips names not installed, checks
each drop against the endpoint listing and reports it. It touches no other query;
installed queries no repository file defines are listed as left in place. Earlier code
calls the old names, so stop its jobs everywhere before `mule install` ([On the CUDA
host](../how-to/train-and-evaluate.md#on-the-cuda-host)). `mule check` lists retired
queries still installed (`queries.retired` in `results/check.json`).
