# Time encoding

How time reaches the model: every payment message carries its age at the cutoff and the
gap since its pair's previous payment, each turned into 64 fixed Fourier coordinates.
These are deterministic features, not learned embeddings; the model learns how to combine
them.

## What the cutoff means

The cutoff is the as-of time of an account's risk assessment, chosen by the training
schedule or by the caller of `mule score`. It is not the last payment time, the extract
time or a property of the account. For a calendar cutoff such as `2024-07-01` the history
is what happened strictly before that UTC midnight: the date becomes the preceding
millisecond, and its sequence watermark comes from `resolve_split_cutoffs`. Changing the
scoring date changes the age of every earlier event.

Consider the same directed A-to-B Zelle pair:

| Moment | Meaning |
|---|---|
| Monday 10:00 | The previous A-to-B payment |
| Tuesday 10:00 | The payment being represented |
| Tuesday 12:00 | The assessment cutoff |

For Tuesday's payment the **age** is about 2 hours and the **pair gap** 24 hours. A
payment from A to someone else does not reset this pair's gap. Age says how relevant the
payment is to the current assessment; the gap says the pair's cadence.

When attention follows a historical payment to its counterparty, the neighbour's context
moves back to that payment's time and exclusive sequence, so the neighbour's earlier
payments are aged against that moment, not the root's assessment, and its later activity
never enters the earlier message.

## What a pair is

A pair is an ordered sender Account and a canonical recipient, within one rail. An
explicit recipient Account on the payment takes precedence over its routing token; when
no recipient Account was recorded, the recipient Token is the canonical recipient. A
transfer with both an account and a token role is therefore counted once, as an account
pair, and token pairs represent unresolved recipients only, not all traffic addressed to
a token. Pairs use the event's observed roles, never today's token binding, so a token
that moves cannot reassign old payments.

The first observed payment of a pair has no predecessor: `gap_present` is 0 and its
coordinates are zero. A genuine simultaneous predecessor has `gap_present` 1 and the real
zero-delta vector. A first observation does not prove that no earlier event exists
outside the retained data, so a missing gap is never read as zero. Two events may share a
timestamp with different sequences, a legitimate zero gap; when the source cannot order
an equal-timestamp group, use a strict timestamp boundary rather than trusting synthetic
sequence numbers for causality.

## The basis

The basis is `log1p_s_400d_32x_sincos_v1` (`contract.time_basis.BASIS_ID`). For a delta
in milliseconds:

```text
s = delta_ms / 1000
u = ln(1 + s) / ln(1 + 400 * 86400)
frequency[i] = 0.125 * 16^(i / 31), i = 0..31
encoding[2*i]     = sin(2*pi*frequency[i]*u)
encoding[2*i + 1] = cos(2*pi*frequency[i]*u)
```

The 400 days is a normalisation scale, not a cut-off or a clip. The 32 frequencies cover
0.125 to 2 cycles per unit of log time: different frequencies applied to the same
log-scaled duration, not 64 time buckets, 64 payments or a transform of the payment
sequence. Nearby durations get smoothly changing coordinates. The frequencies and the
scale are fixed design choices, not fitted to labels; they create no time-of-day or
day-of-week features, and nothing shows that 64 dimensions is the right number (the
`drop_time_encoding` variant measures what the encoding is worth).

## Where it is computed

TigerGraph and the client share the basis: `encode_fourier64` in GSQL, and
`contract.time_basis.fourier64` in numpy and `batching.time_encoding` in torch. Every
payment can be encoded, Zelle or not, but only the messages of a context's pool ever are;
nothing materialises encodings for every payment in the graph. The
training query sends only `age_ms` and `gap_ms` per message, and the client expands them
on the training device, which cuts the REST payload by about 70%. The first request of
every context source and every 64th after it (`transport.encoding_check_every`) also asks
TigerGraph for the vectors (`emit_encodings`), and the run fails if any coordinate
differs from the client's by more than 1e-5 + 1e-5 x |value| (measured: 3.7e-7). GSQL's
trigonometric functions return FLOAT values, so storing them as doubles does not restore
the precision lost inside them; integer timestamps are subtracted before any conversion
to floating point, and the coordinates are appended in a fixed order, never from
concurrent accumulators.

## The pair-gap queries

`encode_zelle_pair_gaps` and `encode_payment_pair_gaps` (in `gsql/analytics/`) compute,
for one exact pair up to a cutoff, every payment's predecessor gap, its age at the cutoff,
both encodings, and the pair's counts over the preceding hour, day and week; the payment
query keeps rails apart. They are exact, bounded queries for analysis: they traverse the
sender's candidate history, validate the pair, then sort it by sequence, so `max_events`
(1,000 by default, at most 10,000) bounds what they return and sort, not what they scan.
An oversized history returns `history_limit_exceeded`, no rows and no writes; a query that
finds ambiguous roles, mismatched clocks, duplicate sequences within a pair or timestamps
that decrease in sequence order fails before any write. Training never calls them, so only
`mule diagnose` installs them.

```gsql
// sender, recipient type, recipient id, seed sequence, seed milliseconds,
// persist, largest pair history
RUN QUERY encode_zelle_pair_gaps("account-A", "Account", "account-B",
                                 1000000, 1800000000000, false, 1000)

// An unresolved external recipient: its opaque token id.
RUN QUERY encode_zelle_pair_gaps("account-A", "Token", "opaque-token-id",
                                 1000000, 1800000000000, false, 1000)

// The same calculation for another rail, kept apart from the other rails.
RUN QUERY encode_payment_pair_gaps("account-A", "Account", "account-B",
                                   1000000, 1800000000000, "ach", false, 1000)
```

Each result row holds the event id, sequence and timestamp, the previous event id,
`pair_delta_t_ms`, `pair_delta_t_present`, `pair_time_encoding`, `age_ms` and
`age_time_encoding`; the final object holds `status`, `event_count`, `basis_id`,
`dimensions`, `persisted`, the cutoffs and the pair counts. Read the final status: an
error means nothing was calculated or saved, and an empty history is `ok` with no
events.

With `persist = true` they store the gaps and encodings on the payment vertices (the
[pair-gap cache](../reference/schema.md#the-pair-gap-cache)). Training never reads that
cache: an age changes with every cutoff, so no stored value can serve them all, and a gap
is reusable only while the predecessor history and the scope stay as they were; a
backfill or another scope changes it.
