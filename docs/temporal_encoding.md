# GSQL temporal encoding

These queries calculate fixed 64-dimensional Fourier time features. They do not
train a GNN or calculate learned account embeddings. Both payment vertex types
have optional attributes for caching **pair-gap** features. Scoring-cutoff **age**
features are returned by the query and are never stored on a payment.

## Run in TigerGraph

The standalone calculator needs no graph data:

```gsql
USE GRAPH Mule_Pattern_Learner
RUN QUERY temporal_fourier64(60000)
```

It returns a 64-value encoding for a gap of 60,000 milliseconds. Negative input
returns `invalid_delta_t`. The shared subquery `temporal_fourier64_values` accepts
an unsigned millisecond delta and returns the vector for use by other GSQL queries.

For an account pair, replace the example IDs and cutoffs with actual values:

```gsql
// sender, recipient type, recipient ID, seed sequence, seed milliseconds,
// persist, maximum complete pair-history size
RUN QUERY zelle_pair_time64("account-A", "Account", "account-B",
                           1000000, 1800000000000, false, 1000)

// Unresolved external recipient: use its opaque token ID.
RUN QUERY zelle_pair_time64("account-A", "Token", "opaque-token-id",
                           1000000, 1800000000000, false, 1000)

// The same calculation for a non-Zelle rail, kept separate from other rails.
RUN QUERY payment_pair_time64("account-A", "Account", "account-B",
                             1000000, 1800000000000, "ach", false, 1000)
```

The default `persist=false` reads facts and returns features without updating
vertices. Set `persist=true` to replace the pair-gap cache on the selected
payments. Each payment result includes:

- Event ID, event sequence, and timestamp.
- Previous event ID, `pair_delta_t_ms`, and `pair_delta_t_present`.
- `pair_time_encoding`: 64 coordinates when a predecessor exists.
- `age_ms` and `age_time_encoding`: elapsed time to the requested scoring cutoff.

The final result object contains `status`, `event_count`, `basis_id`, `dimensions`,
`persisted`, cutoffs, and pair counts over the preceding hour, 24 hours, and seven
days. Consumers must inspect this final status. An error status means no feature
rows were calculated or saved. Empty history returns `ok` and zero events.

## Exact meaning of a pair

The pair is an ordered sender Account and a canonical recipient, within one
payment rail. An explicit recipient Account recorded on the payment takes
precedence over its routing token. If no recipient Account was recorded, its
explicit recipient Token is the canonical recipient. Consequently, a transfer
with both account and token roles appears in the account-pair query only. It is
not counted twice. Token-pair queries represent unresolved recipients only;
they are not all traffic addressed to a token.

These queries use **observed event roles**, not today's `Token_Bound_To_Account`
mapping. They do not traverse association histories, and thus cannot accidentally
reassign old payments when a token moves. Association feature extraction is a
separate query that must use the documented validity predicate. No label field
is used by the encoding queries.

History includes only events with `event_seq < seed_seq` and
`event_ts_ms <= seed_ts_ms`. Both cutoffs are required; sequences must belong to
the common chronological domain. Two historical events may have the same
timestamp and different authoritative sequences, producing a legitimate zero
gap. If source data does not establish order within an equal-timestamp group,
use a strict timestamp boundary (one millisecond before that group) instead of
assuming that synthetic sequence numbers establish causality.

The first observed payment has `pair_delta_t_present=false`, delta zero, no
previous event ID, and an empty vector. A genuine zero gap has presence true
and a 64-value vector. The first observed event does not prove that no earlier
event exists outside the retained dataset. Do not treat missing gaps as zero.

All edge event clocks must agree with their event vertex. Sender and recipient
first-seen sequences must be positive and no later than the event; first-seen
timestamps must not be later than the event. Each selected event needs exactly
one sender Account, at most one recipient Account, and at most one recipient
Token. The queries reject ambiguous roles, mismatched clocks, duplicate
sequences within a pair, and timestamps that decrease in sequence order. They
validate the entire selected history **before any writes**.

## Basis and persisted attributes

The fixed basis identifier is `log1p_s_400d_32x_sincos_v1`:

```text
s = delta_t_ms / 1000
u = ln(1 + s) / ln(1 + 400 * 86400)
frequency[i] = 0.125 * 16^(i / 31), i = 0..31
encoding[2*i]     = sin(2*pi*frequency[i]*u)
encoding[2*i + 1] = cos(2*pi*frequency[i]*u)
```

The 400-day value is a normalization scale, **not a clipping limit**. Frequencies
cover 0.125 to 2 cycles per normalized log-time unit. These are documented POC
defaults, not tuned model parameters. The basis ID describes the numerical
recipe; it does not introduce entity or relationship versioning.

Integer timestamps are subtracted before converting elapsed time to floating
point. Axes are appended sequentially in a fixed order, never from concurrent
`ACCUM` matches. GSQL's trigonometric functions return FLOAT values; storing the
result in `LIST<DOUBLE>` does not restore precision lost inside those functions.
Live tests compare all coordinates against a Python double-precision reference
with absolute tolerance `1e-5`.

The additive attributes on both event vertex types are:

| Attribute | Meaning |
| --- | --- |
| `pair_delta_t_ms` | Elapsed integer milliseconds since the prior pair event |
| `pair_delta_t_present` | Whether that preceding event exists in the retained history |
| `pair_time_encoding` | Empty when missing; otherwise 64 ordered coordinates |
| `time_encoding_basis_id` | Numerical recipe identifier |
| `pair_previous_event_id` | Predecessor used for this calculation |
| `pair_sender_id` | Sender that defines the cached pair |
| `pair_recipient_type`, `pair_recipient_id` | Canonical recipient that defines the cached pair |

These are derived caches. If an older payment arrives late, run the affected
pair again through the latest required cutoff. If event roles are corrected,
recompute both affected pairs and clear caches on records that lose a valid pair.
Do not use stale persisted vectors for training. Query mode recalculates from
current event facts and does not read those caches. As documented for the
schema, retrospective corrections still require knowledge history or frozen
extracts to support knowledge-safe historical evaluation.

## Limits and deployment

These are exact, bounded pair queries for the POC. They traverse the sender's
candidate history, validate the matching pair, then sort that pair by sequence.
`max_events` defaults to 1,000 and cannot exceed 10,000. The limit bounds returned
history and sorting, not the sender adjacency traversal. `history_limit_exceeded`
returns no rows and performs no writes; the query never silently truncates away
the predecessor. High-degree or long-lived accounts need staged temporal indexes
and a batch pipeline; do not call this once per pair per training epoch at scale.

Installation uses the project's `.env`, without printing credentials:

```sh
.venv/bin/python scripts/temporal/install_time_encoding.py
```

The installer adds attributes once, checks for running loading jobs, repairs the
affected payment loading definitions, and installs the shared encoder plus the
three public queries. The fresh schema file includes the same attributes. Do not
rerun the earlier empty-graph migration against this populated graph.

Explicit live verification creates isolated synthetic records and deletes only
those records in a `finally` block:

```sh
.venv/bin/python scripts/temporal/verify_time_encoding.py
```

Verification evidence is recorded in `temporal_encoding_deployment.json`.
The live graph's application data is not bulk materialized by installation or
verification. Save mode is available when running the pair queries.

References: [GSQL mathematical functions](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/func/mathematical-functions),
[subqueries](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/operators-and-expressions),
[schema changes and loading-job impact](https://www.tigergraph.com/docs/gsql-ref/4.2/ddl-and-loading/modifying-a-graph-schema).
