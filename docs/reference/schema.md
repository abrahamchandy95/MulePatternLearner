# Schema

The graph `Mule_Pattern_Learner` from `gsql/schema/schema.gsql` (TigerGraph 4.2.5): 8 vertex
types and 19 directed edge types, 12 event participations and 7 valid-time associations,
each with a named reverse (38 edge types in the catalog). `gsql/schema/scope_vertex.gsql`
adds the graph's split ([The split](#the-split)). Supervision fields: [Labels](labels.md).
Creating and loading: [Set up a graph](../how-to/set-up-a-graph.md).

## Vertex types

| Vertex | Contents |
|---|---|
| `Party` | Stable id, first-seen sequence and time |
| `Account` | Stable id, account type, external flag, first-seen clocks, and six supervision fields ([The Account attributes](labels.md#the-account-attributes)) |
| `Token` | Tokenised id and first-seen clocks |
| `Address` | Tokenised address id and first-seen clocks |
| `Device`, `IP` | Opaque ids and first-seen clocks |
| `Zelle_Transfer` | One Zelle payment: transfer id, event milliseconds and sequence, amount and its presence flag, currency, and the transfer's fraud label fields |
| `Payment_Transaction` | One payment on another rail, with the same event clock, amount, currency and rail (`payment_rail`) |

- A Zelle payment is a `Zelle_Transfer` only, never also a `Payment_Transaction`.
- An unknown external account gets no invented Account: keep the token the payment used
  and omit the unknown account role. Known account roles and used tokens can coexist.
- Namespace and normalise opaque token ids consistently before loading.
- A party's contact token proves no enrolment or account ownership. Conflicting confirmed
  Zelle enrolments across accounts need their own validation.
- Entity metadata is its value at first observation, never overwritten.
- Never stored: whole-history aggregates, PageRank, communities, embeddings, label-derived
  features, split flags.

## Events and their participation

Each event has six role edges that repeat its `event_seq` and `event_ts_ms` and record
one occurrence, not a tenure: `Transfer_From_Account`, `Transfer_To_Account`,
`Transfer_From_Token`, `Transfer_To_Token`, `Transfer_Used_Device` and `Transfer_Used_IP`
for Zelle, the matching `Transaction_*` edges for other payments. The account's reverse
edges are the model's payment relations: `Account_Initiated_Transaction` (`payment_out`),
`Account_Received_Transaction` (`payment_in`), `Account_Sent_Zelle_Transfer`
(`zelle_out`), `Account_Received_Zelle_Transfer` (`zelle_in`).

- The context query assumes per event exactly one sending account, at most one recipient
  account and one recipient token with at least one of the two, and role edges whose
  clocks equal the event's; it no longer checks them, as no check fired on the reference
  data. A recorded recipient Account takes precedence over its routing token.
- On the reference graph every payment has one From and one To account and no token
  edges; every Zelle transfer has one edge of each of the four roles.
- `Transfer_Used_Device` is one observation; `Account_Uses_Device` is an association
  interval. An observation proves no continuing association.
- Resolve a historical account by the event's own account role or the token binding valid
  at the cutoff, never today's token mapping.

## Associations and valid time

| Association | Reverse |
|---|---|
| `Party_Owns_Account` | `Account_Owned_By_Party` |
| `Party_Uses_Token` | `Token_Used_By_Party` |
| `Token_Bound_To_Account` | `Account_Bound_From_Token` |
| `Party_Uses_Device` | `Device_Used_By_Party` |
| `Account_Uses_Device` | `Device_Used_By_Account` |
| `Party_Uses_IP` | `IP_Used_By_Party` |
| `Party_Has_Address` | `Address_Used_By_Party` |

These fourteen are the model's association relations. Each edge has
`DISCRIMINATOR(valid_from_seq UINT)` and `valid_to_seq` (0 while open); its key is edge
type, endpoints and start sequence, so repeated tenures of one pair never overwrite each
other. Every traversal, in either direction:

```gsql
WHERE rel.valid_from_seq <= @@seed_seq
  AND (rel.valid_to_seq == 0 OR @@seed_seq < rel.valid_to_seq)
```

- Intervals are `[valid_from_seq, valid_to_seq)`. `(Token X, Account A, start 10, end 20)`
  and `(Token X, Account A, start 30, end 0)` are two rows: bound at sequences 15 and 30,
  not at 20 or 25. Deregistration closes the first row, re-enrolment adds the second, and
  the first is never deleted.
- One shared seed sequence suits only seeds of one cutoff. A batch of mixed cutoffs
  carries each seed's cutoff, since one later cutoff for all would admit future
  relationships. The context query evaluates associations at `cutoff_seq - 1`.
- The schema does not enforce this. The loader must give positive start sequences, ends
  after starts, idempotent event ids, non-overlapping tenures where that applies, edge
  clocks equal to their event's and the canonical role counts, and must not impose
  exclusive ownership on joint accounts or shared devices.
- Exclude an equal-timestamp group the source cannot order with a strict timestamp
  cutoff; never invent an order from ids.

## Clocks

- Payments and association changes share one chronological sequence domain: `event_seq`
  and the association bounds come from one counter, never one per rail.
- Sequences are positive (0 means unavailable or an open end) and need not be dense.
  Their differences are never elapsed time; millisecond timestamps measure that.
- History at a cutoff is `event_seq < cutoff_seq AND event_ts_ms <= cutoff_ms`. A target
  transfer never enters its own history.
- When a fact became known is deferred, not inferred: with no `known_from_seq`, valid time
  cannot tell whether a backdated correction was available to a historical prediction. A
  knowledge-safe backtest needs arrival or discovery history, or frozen extracts, and two
  corrections of one tenure with the same start would need a new key design.

## The split

`scope_vertex.gsql` adds the vertex type `Temporal_Training_Scope` (`id`, `split_seed`,
`train_share`, `validation_share`, `test_share`, `ready`) and the edge
`Entity_In_Training_Scope` from Account and Party (`partition`; reverse
`Training_Scope_Has_Entity`), keeping the graph's one train, validation and test split
apart from the business data: one vertex, `"split"`, and one membership edge per Account
and Party.

- `mule install` applies it when the graph lacks the type, or replaces a differing type
  while the graph holds no scope vertex ([Queries](queries.md#the-scope-types)).
- The split records its seed and shares. Every run refuses a split whose seed or shares
  differ from its settings, or which records none; nothing deletes a split but its owner.
- The type keeps its name as part of the graph's schema.

## Jobs

- `schema.gsql` and `scope_vertex.gsql` run and drop their schema change jobs,
  `create_payment_schema` and `add_training_scope`.
- `loading.gsql` creates the loading job `load_graph`, which loads every vertex and edge
  type from PhantomLedger's export by column name, stays installed and is run by a person,
  never a command ([Set up a graph](../how-to/set-up-a-graph.md#2-load-the-data)).
- `ALTER ... ADD ATTRIBUTE` cannot change a discriminator, so changing an association's
  identity means replacing and reloading it.
- Never run `schema.gsql` on a populated graph: a new schema means a new graph, loaded
  again.

## The reference graph

Counted on 24 September 2026 (scope vertices are experiment metadata):

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
| `Temporal_Training_Scope` | 2 | Two splits, from when a graph held several; one now |

- Events run from 2024-01-01 to 2024-12-31, about 8.4 to 11.0 million payments a month,
  every amount present and every currency USD. The data is a PhantomLedger simulator
  export.
- Hubs: 4,988 accounts have more than 2,048 incoming payments, the largest 3,814,933,
  all external. In a sample of 2,000 internal deposit accounts, 95% had such a hub among
  their eight most recent payments, so hubs are neighbours in almost every batch.
  [Sampling](../explanation/sampling.md#hubs-and-stubs) keeps them out of child requests.
