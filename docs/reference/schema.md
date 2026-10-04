# Schema

The graph `Mule_Pattern_Learner`, as `gsql/schema/schema.gsql` creates it for TigerGraph
4.2.5: eight vertex types, nineteen directed edge types with a named reverse edge each
(38 edge types in the catalog), and seven valid-time association types. The scope vertex
type that experiments use is added by `gsql/schema/scope_vertex.gsql`. The account
supervision fields are described in [Labels](labels.md), and how a graph is created and
loaded in [Set up a graph](../how-to/set-up-a-graph.md).

## Vertex types

| Vertex | Contents |
|---|---|
| `Party` | Stable id, party type, first-seen sequence and time |
| `Account` | Stable id, account type, external flag, first-seen clocks, and the ten supervision fields of the label contract |
| `Token` | Tokenised id, first-seen clocks, token kind (`phone`, `email`, `handle`) and network |
| `Address` | Tokenised address id, first-seen clocks, country code |
| `Device`, `IP` | Opaque ids and first-seen clocks; a Device also has a device type |
| `Zelle_Transfer` | One Zelle payment: transfer id, event time, milliseconds and sequence, amount and its presence flag, currency, channel, the transfer's fraud label fields, and the optional pair-gap cache |
| `Payment_Transaction` | One payment on another rail, with the same event clock, amount, currency, rail (`payment_rail`), channel and pair-gap cache |

A Zelle transfer is stored only as a `Zelle_Transfer`, never also as a
`Payment_Transaction`. An unknown external account needs no invented Account: keep the
token the payment actually used and leave the unknown account role out. Known account
roles and the tokens a payment used can coexist. Set the network of Zelle routing tokens
to `zelle`, and namespace and normalise opaque token ids consistently before loading. A
party's contact token is no evidence of enrolment or account ownership, and conflicting
confirmed Zelle enrolments across accounts need their own validation. Entity metadata
is the value at first observation and is never overwritten by later state. No
whole-history aggregate, PageRank, community, embedding, label-derived feature or split
flag is stored.

## Events and their participation

Each event has six participation edges with explicit roles: `Transfer_From_Account`,
`Transfer_To_Account`, `Transfer_From_Token`, `Transfer_To_Token`, `Transfer_Used_Device`
and `Transfer_Used_IP` for Zelle, and the matching `Transaction_*` edges for other
payments. All twelve repeat the event's `event_seq` and `event_ts_ms` and describe one
occurrence, not a tenure. Their reverse edges from the account
(`Account_Initiated_Transaction`, `Account_Received_Transaction`,
`Account_Sent_Zelle_Transfer`, `Account_Received_Zelle_Transfer`) are the four payment
relations the model sees: `payment_out`, `payment_in`, `zelle_out` and `zelle_in`.

The context query requires exactly one sending account and at most one recipient account
and one recipient token per event, at least one of the two. A recorded recipient Account
takes precedence over its routing token. On the reference graph every payment has one
From and one To account and no token edges, and every Zelle transfer has one edge of each
of the four roles.

`Transfer_Used_Device` records an observation at one transfer; `Account_Uses_Device` is
an association interval. A device observation alone is no proof of a continuing
association. For historical account resolution use the event's own account role or the
token binding valid at the cutoff, never today's token mapping.

## Associations and valid time

The seven association types are directed, each with an explicitly named reverse edge:

| Association | Reverse |
|---|---|
| `Party_Owns_Account` | `Account_Owned_By_Party` |
| `Party_Uses_Token` | `Token_Used_By_Party` |
| `Token_Bound_To_Account` | `Account_Bound_From_Token` |
| `Party_Uses_Device` | `Device_Used_By_Party` |
| `Account_Uses_Device` | `Device_Used_By_Account` |
| `Party_Uses_IP` | `IP_Used_By_Party` |
| `Party_Has_Address` | `Address_Used_By_Party` |

These fourteen are the association relations of the model. Each edge has
`DISCRIMINATOR(valid_from_seq UINT)`, `valid_to_seq` (0 while open), `confidence` and
`source_system`, so its key is the edge type, the endpoints and the start sequence, and
repeated tenures of one pair never overwrite each other. Every traversal, in either
direction, uses one rule:

```gsql
WHERE rel.valid_from_seq <= @@seed_seq
  AND (rel.valid_to_seq == 0 OR @@seed_seq < rel.valid_to_seq)
```

Intervals are `[valid_from_seq, valid_to_seq)`. The bindings `(Token X, Account A, start
10, end 20)` and `(Token X, Account A, start 30, end 0)` are two rows: the token is bound
to A at sequences 15 and 30, but not at 20 or 25. Deregistration closes the first row and
re-enrolment adds the second; the first is never deleted. A shared seed sequence suits
only seeds of one cutoff: a batch of mixed cutoffs carries each seed's cutoff, since one
later cutoff for all would admit future relationships. The context query evaluates
associations at `cutoff_seq - 1`.

The schema supports this contract but does not enforce it. The loader must give positive
start sequences, ends after starts, idempotent event ids, non-overlapping tenures where
that applies, edge clocks equal to their event's clocks and the canonical role counts. It
must not impose exclusive ownership on joint accounts or shared devices. If the source
cannot order an equal-timestamp group, exclude those peers with a strict timestamp
cutoff rather than inventing an order from ids.

## Clocks

Payments and association changes share one chronological sequence domain: `event_seq`
and the association bounds come from a single counter, never one per rail, and sequence
differences are never read as elapsed time. Sequences are positive (0 means unavailable
or an open end) and need not be dense. Millisecond timestamps measure elapsed time.
History at a cutoff is `event_seq < cutoff_seq AND event_ts_ms <= cutoff_ms`, and a
target transfer never enters its own history.

When a fact became known is deferred, not inferred: the graph has no `known_from_seq`, so
valid time cannot tell whether a backdated correction was available to a historical
prediction. A knowledge-safe backtest needs arrival or discovery history, or frozen
extracts; the current discriminator could not hold two corrections of one tenure with the
same start without a new key design.

## The pair-gap cache

Both payment types have optional attributes that the pair queries of `gsql/analytics/`
fill with `persist = true`: `pair_delta_t_ms`, `pair_delta_t_present`,
`pair_time_encoding` (empty when missing, else 64 coordinates), `time_encoding_basis_id`,
`pair_previous_event_id`, `pair_sender_id`, `pair_recipient_type` and
`pair_recipient_id`. They are derived caches. Training never reads them: it recomputes
every gap at its cutoff, and ages depend on the cutoff, so they are never stored. A late
payment or a corrected role makes the cached gaps of its pairs stale until the pair query
runs again.

## The experiment scope

`scope_vertex.gsql` adds the vertex type `Temporal_Training_Scope` (`scope_id`,
`source_id`, `split_seed`, `train_share`, `validation_share`, `test_share`, `ready`) and
the edge `Entity_In_Training_Scope` from Account and Party (`partition`, `group_id`;
reverse `Training_Scope_Has_Entity`). It holds experiment membership apart from the
business data, and `mule install` applies it when the graph lacks the type. A scope
records the shares of its partition, and every run refuses a scope whose shares differ
from its settings, or which records none. The vertex type keeps its name because it is
part of the graph's schema; the scope id of the built-in run is `strict_mule_v3`.

## Jobs

`schema.gsql` defines its types in the schema-change job `create_payment_schema` and
`scope_vertex.gsql` in `add_training_scope`; each runs and drops its job.
`account_loading.gsql` defines the loading job `load_accounts`, which stays installed
once created ([Labels](labels.md#loading-accounts)); a graph set up before the job had
this name keeps it under its first name, and no command runs either. The other vertex and
edge types are loaded by the data producer.

TigerGraph discriminators cannot be changed with `ALTER ... ADD ATTRIBUTE`, so changing
an association's identity means replacing and reloading it. The populated reference graph
reached this schema through three one-off migrations (the valid-time upgrade of the
empty graph, the pair-gap attributes and the Account supervision fields), kept in git
history. Never run `schema.gsql` on a populated graph.

## The reference graph

Counted on 24 September 2026 (the scope vertices are experiment metadata):

| Vertex type | Count | Role |
|---|---|---|
| `Payment_Transaction` | 114,515,477 | Non-Zelle payments (rails card, unknown, cash, internal, ach, check) |
| `Zelle_Transfer` | 1,225,864 | Zelle payments |
| `Account` | 788,283 | 317,840 internal deposit accounts (the scored population), 434,783 internal credit, 35,660 external |
| `Party` | 200,000 | Account owners |
| `Token` | 144,910 | Zelle tokens (email or phone aliases) |
| `Device` | 374,192 | Devices seen on payments and tenures |
| `IP` | 1,311,576 | IP addresses seen on payments and tenures |
| `Address` | 196,409 | Party addresses |
| `Temporal_Training_Scope` | 2 | `strict_mule_v1` and `strict_mule_v2` |

Events run from 2024-01-01 to 2024-12-31, about 8.4 to 11.0 million payments a month,
every amount present and every currency USD. The data is an export of the PhantomLedger
simulator.

Some accounts are enormous: 4,988 have more than 2,048 incoming payments and the largest
has 3,814,933, all of them external. In a sample of 2,000 internal deposit accounts, 95%
had such a hub among their eight most recent payments, so hubs are neighbours in almost
every batch; [Sampling](../explanation/sampling.md#hubs-and-stubs) describes how the
pipeline keeps them out of child requests.
