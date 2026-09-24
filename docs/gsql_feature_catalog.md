# GSQL feature and query catalog

The [feature-group redesign](feature_redesign.md) documents the window-free feature groups, optional summaries, sampler, and migration. Fixed 83/135 dimensions below describe the legacy control profile. The v5 default profile and its candidate pools are described in [training from the live temporal graph](live_temporal_training.md#candidate-pools-and-resampling).

This describes `temporal/live`, the live TGAT-style path. The older snapshot
queries are a separate implementation. The live strict path applies server-side
ownership-group partitions before sampling and feature aggregation; see
[leakage and scaling](leakage_and_scaling.md).

## Queries used by preparation and training

| Query | Inputs and output | When used |
|---|---|---|
| `temporal_create_training_scope` | Creates a frozen, label-blind Account/Party ownership-group partition. `unowned_policy` places Accounts whose ownership component has no Party: `"independent"` (the query's default, the original behaviour) gives each its own hash partition; `"shared"` gives unowned external accounts partition 1 (visible in every phase) and group ID `shared:<component>`, while unowned internal accounts keep their own hash partition; `"linked"` (the client default) is `"shared"` plus: an unowned internal account whose distinct owned internal deposit counterparties (the other endpoint of any Payment_Transaction or Zelle_Transfer, all time) are exactly one account takes that account's component, partition and group ID. Any other value is `invalid_parameters`. Components with a Party keep the same component and hash partition under every policy. Also prints `unowned_policy`, `shared_accounts` and `linked_accounts`. | Once per strict experiment scope, only with `create_scope` or `prepare --create-scope`; writes only experiment membership. |
| `temporal_finalize_training_scope` | Checks committed membership count/attributes and marks the scope ready. | After scope creation; incomplete scopes fail closed. |
| `temporal_scope_policy` | Read-only. For a ready scope, prints `members` (all member Accounts and Parties), `unowned_accounts` and six counts of unowned member Accounts (no `Account_Owned_By_Party` edge) by class and side: `shared_internal`, `shared_external` (group ID starts with `shared:`), `independent_internal`, `independent_external` (group ID is the account's own component), `linked_internal`, `linked_external` (any other group ID). The client infers the creation policy from them. `scope_not_ready` otherwise. | When a strict preparation reuses or creates a scope, and when every streamed run opens. |
| `temporal_scope_population` | Pages internal deposit accounts with preassigned partition, first-seen clocks and optional observed supervision (`include_observed`, default FALSE): `observed_positive` is the label contract's revealed positive, and only such an account has a nonzero `known_from_ms`. | Strict preparation; bounded reservoir selection, never model features. |
| `temporal_training_population` | Legacy metadata pages including ownership IDs, with the same optional observed supervision (`include_observed`, default FALSE). | Optional shared-history preparation only; client metadata is capped. |
| `temporal_training_cutoffs` | Converts exclusive calendar cutoffs to sequence watermarks using payments and entity first observations. Every requested cutoff key is present, 0 when nothing is visible. | Preparation and `score-new`. Scans history; not a constant-time lookup. |
| `temporal_hub_registry` | For 1 to 24 cutoff sequences, lists Accounts whose visible history (events before the cutoff, all currencies) in some payment relation exceeds `threshold`; the only reason is `visible_history`. With `scope_id` empty the counts are unscoped and every row has `visibility_phase` 3. With a `scope_id` (the scope must be ready, else `scope_not_ready`) it counts per phase 1, 2 and 3 only events whose From/To Account endpoints are all members with partition at most that phase, the endpoint rule of `temporal_training_context`, and the hub itself must be allowed in the phase. Rows are `(account_id, cutoff_seq, visibility_phase, max_visible, max_degree, reason)`; `max_degree`, the all-time relation outdegree, is informational only. The response echoes `cutoff_seqs`, `threshold` and `scope_id`. O(1) outdegree prefilter; read-only. | Preparation (dataset cutoffs; the scope for `strict_inductive`) and `score-new` (the requested cutoff, unscoped). |
| `temporal_training_context` | Accepts 1 to 64 entity/time contexts plus scope and phase, and returns one candidate pool per context. Filters excluded Account/Party contributions before rolling features, neighbor selection and pair history. | Every batch in streaming mode (roots, then children); once per context in optional SQLite staging. |
| `temporal_fourier64_values` | Encodes a nonnegative millisecond delta into 32 sine/cosine pairs. | Called inside temporal queries. |
| `temporal_fourier64` | Public validation wrapper for the same calculation. | Diagnostics and parity checks. |

`training_context.gsql` is generated by
`python scripts/temporal/render_training_queries.py` from the shared Python
relation/window contract. Edit the generator and regenerate when changing the
feature contract. Preparation fingerprints the query sources and feature contract;
old prepared datasets cannot silently reuse changed definitions.

The population queries default to `include_observed = FALSE`, which reads no
graph label attribute. Only `label_policy = "graph_observed"` turns it on; then a
positive is the revealed positive of the
[account label contract](account_mule_labels.md),
`pu_label == 1 AND is_mule == 1 AND mule_label_known AND NOT is_mule_masked`,
accompanied by its discovery timestamp (`mule_label_available_ts_ms`). Every other
account, including masked mules and labeled non-mules, has `observed_positive`
FALSE and `known_from_ms` 0, so neither field reveals a withheld label or which
accounts are labeled. The client fails fast on a nonzero `known_from_ms` without a
positive, the sign of an older installed query.
The query never exports raw oracle truth or the synthetic mask. The old
`temporal_training_accounts` oracle export is local experiment material and is
not installed or called by the production path.

## Context query contract

`temporal_training_context` parameters, in signature order:

| Parameter | Bounds | Meaning |
|---|---|---|
| `node_types`, `node_ids`, `cutoff_seqs`, `cutoff_times` | 1 to 64 entries each | One request per index. |
| `per_relation`, `k_old`, `k_div` | 1..32, 0..16, 0..16 | Candidate pool per payment relation: most recent, rank quantiles, new peers. |
| `k_assoc` | 0..8 | Valid-time associations per association relation (0 for payments-only children). |
| `max_history` | 32..4096 | Visible events per relation above which the request is rejected. |
| `emit_encodings` | default FALSE | When TRUE, `age_encoding` and `gap_encoding` hold Fourier vectors; otherwise they are empty maps and no `temporal_fourier64_values` call runs. |
| `include_*` | 14 flags | Exactly the non-categorical, non-client groups of `FeaturePlan.query_flags()`; there is no `include_hub_indicator`. |
| `scope_id`, `visibility_phase` | phase 1..3 | Strict scope filtering, applied before any feature or sampling. |

A failed request never aborts the call: it prints `{status, request_index}` and
the query continues with the next request, so every index gets exactly one row.
Per-request statuses are `invalid_request`, `missing_entity` (unknown IDs are
resolved with typed lookups, never a runtime error), `invisible_entity`,
`history_capacity_exceeded`, `nonmonotonic_pair_clock`, `invalid_payment_fields`
and `invalid_event_roles`; the client maps them to a rejected context. Only
`invalid_parameters`, `invalid_visibility_phase` and `scope_not_ready` stop the
whole call.

The client computes Fourier features on the training device from `age_ms` and
`gap_ms`. It asks for encodings only on the first request of a source and every
`encoding_check_every`-th request after it, and then checks every vector against
numpy `fourier64` (tolerance 1e-5).

Roles, peer metadata, device/IP context and the prior-pair scan are set-based
SELECTs over all sampled events of a request. When `include_pair_window_counts`
is on, the prior-pair scan supplies the pair clock, so the chronology pass (and
its `nonmonotonic_pair_clock` check) is skipped. Rolling-window and decayed sums
keep the original traversal order, so their floating-point values are identical
to the earlier query text. The exact repository text also runs under INTERPRET:
`queries.as_interpreted(text)` swaps only the header.

## The 83 entity/context features

| Family | Count | Source |
|---|---:|---|
| Entity type, external/deposit indicators and age since first observation | 9 | GSQL metadata; Python creates six type indicators and transforms age. |
| Activity over 1 hour, 1 day, 7 days and 30 days | 40 | GSQL calculates ten fields per window. |
| Incoming/outgoing payment recency and two presence indicators | 4 | GSQL. |
| Active/ended association counts | 28 | Seven relationship families, both directions, two counts each, in GSQL. |
| Outgoing/incoming amount ratios over 1 day and 7 days | 2 | GSQL derives these from scoped window sums, capped at 100. |

Each window contains incoming and outgoing **payment count, amount sum,
missing-amount count, Zelle count and distinct counterparty count**. Missing
amounts remain distinct from genuine zeros. Unresolved Zelle recipients remain
Tokens. The current implementation requires USD for amount aggregation; it does
not silently mix currencies.

The seven association families are party ownership, party-token use,
token-account binding, party-device use, account-device use, party-IP use and
party-address registration. Counts and sampled associations respect their valid
interval and experiment visibility. Repeated tenures can therefore contribute separately. These are
association counts, not necessarily counts of distinct people or devices.

For each ratio window, GSQL computes
`min(outgoing_amount / max(incoming_amount, 1.0), 100.0)`. The denominator floor
and cap preserve the original model-input semantics, including no-incoming-payment
cases. Python does not recompute these ratios; it applies the same fixed `log1p`
transform as before. A missing ratio field is a query-contract error.

Python applies fixed `log1p` transforms where appropriate. No learned account-ID
embedding, label, synthetic mask, ring ID, whole-history PageRank or stored
FastRP vector enters these features. The canonical ordered names are in
`src/mule_pattern_learner/temporal/live/contract.py`.

## The 135 numerical message features

Each sampled payment contributes:

- Seven scalars: amount, amount-present, event-present, predecessor-present,
  and prior same-pair payment counts over 1 hour, 1 day and 7 days.
- 64 coordinates for **age = context cutoff time minus payment time**.
- 64 coordinates for **gap = payment time minus previous payment time** for
  the same directed sender, canonical recipient and rail.

Relation and rail embeddings are learned separately in Python. Association
messages do not invent elapsed time from sequence numbers: they have no exact
association-start timestamp and carry no payment time encoding.

For a payment at sequence `s`, the neighbor's recursive context excludes that
payment (`event_seq < s`), so its older behavior can inform the current event.
A calendar context uses both `event_seq < cutoff_seq` and
`event_ts_ms <= cutoff_ms`. Association visibility is evaluated at
`seed_seq = cutoff_seq - 1` with the valid-time predicate. Known-time corrections
require additional source history; valid-time filtering cannot recover it.

## What the cutoff and 64 coordinates mean

The cutoff is the **as-of time of an account-risk assessment**. It is chosen by
the training/evaluation schedule or the caller of the scoring API. It is not the
last payment time, the extract time or a fixed property of an account. For a
calendar cutoff such as `2024-07-01`, the CLI uses history strictly before that
UTC midnight. Changing the scoring date changes the age of every prior event.

Consider this history for the same directed A-to-B Zelle pair:

| Moment | Meaning |
|---|---|
| Monday 10:00 | Previous A-to-B payment |
| Tuesday 10:00 | Payment being represented |
| Tuesday 12:00 | Account assessment cutoff |

For Tuesday's payment, age is approximately **2 hours**, and same-pair gap is
**24 hours**. A payment from A to a different recipient does not reset this pair
gap. The separate incoming/outgoing recency features measure time since the
account's most recent visible payment in each direction, across counterparties.

Each duration is mapped independently to 32 sine/cosine pairs, yielding 64
coordinates. These are different mathematical frequencies applied to the same
log-scaled duration, not 64 time buckets, 64 payments or an FFT of the payment
sequence. Nearby durations have smoothly changing representations; attention
learns how combinations of these coordinates relate to risk. Gap and age answer
different questions: cadence within a pair and relevance to the current assessment.
Explicit 1h/1d/7d pair counts supply frequency information as well.

When attention traverses a historical payment, the neighbor's context moves back
to that payment's timestamp and exclusive sequence boundary. Its earlier payments
are aged relative to that historical context, rather than the root assessment
time. This prevents the neighbor's subsequent activity entering the earlier
message. The time/partition predicates apply before features are calculated.

The frequencies and 400-day normalization below are fixed design choices, not
parameters fitted to labels. They do not create time-of-day/week features or
prove that 64 dimensions is optimal; the no-Fourier ablation measures their value.

## Can all payments have a delta-time encoding?

**Both Zelle and ordinary payment types are supported. The current training
query calculates vectors only for its selected payment messages.** It does not
materialize vectors for every transaction in the database.

`zelle_pair_time64` and `payment_pair_time64` independently query an exact
sender/recipient pair up to a cutoff. They return predecessor gaps, cutoff ages
and prior pair counts. The ordinary-payment query also separates rails.
`max_events` defaults to 1,000 and cannot exceed 10,000; oversized pair histories
are rejected. This bounds results/sorting, not necessarily the adjacency scanned.

Both pair queries default to `persist=false`. Their optional persistence stores
pair-gap values and encoding metadata on event vertices. The trainer recomputes
its inputs rather than trusting persisted vectors. There is no all-pairs,
checkpointed bulk encoding job in this repository.

A cutoff-age vector changes at every scoring moment, so a single persisted value
cannot represent all cutoffs. Pair gaps can be reused only while predecessor
history and visibility scope remain unchanged; backfills or a different training
scope can change them. A missing predecessor has `gap_present=0` and zero
coordinates. A real simultaneous predecessor has `gap_present=1` and the real
zero-delta sine/cosine vector.

The fixed basis is `log1p_s_400d_32x_sincos_v1`. For delta in milliseconds:

```
u = log1p(delta_ms / 1000) / log1p(34_560_000)
f[k] = 0.125 * 16 ** (k / 31), k = 0..31
encoding[2*k]     = sin(2*pi*f[k]*u)
encoding[2*k + 1] = cos(2*pi*f[k]*u)
```

The 400-day value is a normalization scale, not a cutoff or clipping rule.
These are deterministic time features, not trained account embeddings. The model
learns how to combine them. Computing them in GSQL works today; sending scalar
deltas and expanding them on the GPU is also a viable bandwidth optimization.

## Other queries and the main-branch distinction

`temporal_get_account_supervision` and
`temporal_validate_account_supervision` are optional synthetic-label audit
queries. They expose complete supervision and do not belong in feature extraction
or ordinary production training. The live trainer's normal installer excludes them.

The main snapshot path uses `sample_khop_neighborhood`,
`fetch_account_features`, `fetch_has_paid_features`, `derive_reference_epoch`
and `derive_max_bins`. Offline preparation writes account money-flow statistics,
identity-sharing counts, time bins, PageRank, triangle/clustering statistics and
FastRP embeddings. Those stored full-snapshot statistics must be recomputed with
appropriate time and split boundaries before making temporal or strict-inductive
claims. Merely filtering held-out neighbors at training time does not sanitize
already-computed features. These legacy stored statistics are not inputs to the
live temporal model.

The live context query returns a bounded candidate pool per relation, then Python
selects the layer fanouts (default 16 and 4 in the v5 profile). A small output is
not proof of a cheap query: rolling summaries and predecessor searches still
traverse candidate history, which is why hub accounts from `temporal_hub_registry`
are never expanded as children. The sampler enforces strict scopes.
Time-organized adjacency and server-side rollups remain required work for larger
deployments. The bounded client response is not a bound on server scan memory or
latency.
