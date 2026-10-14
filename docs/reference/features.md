# Features

What the model reads. Reasons: [Feature design](../explanation/feature-design.md); the
Fourier coordinates: [Time encoding](../explanation/time-encoding.md).

- Training keeps only the built-in run's groups, the registry
  `contract.feature_groups.FEATURE_GROUPS`. `fetch_training_context` computes those
  TigerGraph supplies; the client computes the rest.
- Model columns follow the registry's order, whatever order a configuration lists them in.
- Labels, masks, ring ids and learned account embeddings are never features.

## The training groups

| Group | Computed by | Kind | Columns | Needs |
|---|---|---|---|---|
| `entity_meta` | TigerGraph (the type indicators by the client from the row's type) | node | `type_Account`, `type_Token`, `type_Party`, `type_Device`, `type_IP`, `type_Address`, `is_external`, `is_deposit` | |
| `hub_indicator` | the client, from the hub registry | node | `history_withheld`: 1 for a hub child built as a stub, without a request | |
| `message_core` | TigerGraph | message | `amount`, `amount_present`, `is_event` (1 for a payment, 0 for an association); relation and rail embeddings | |
| `time_encoding` | TigerGraph sends `age_ms` and `gap_ms`; the client expands them on the device | message | `gap_present`, `age_fourier_0` to `age_fourier_63`, `gap_fourier_0` to `gap_fourier_63` | |
| `pair_history` | TigerGraph | message | `pair_prior_count`, `pair_first_age_seconds`, `pair_first_present` | |
| `flow_timing` | TigerGraph | message | `flow_delay_seconds`, `flow_present`, `flow_censored`, `flow_observation_seconds`, `flow_amount_ratio`, `flow_ratio_present`, `flow_same_rail` | |
| `pool_activity` | the client, from the root's payment messages | summary | 12 counts, below | `pair_history`, `flow_timing` |
| `pool_internal_inflows` | the client, from the root's payment messages | summary | 3 counts, below | `pair_history` |

The built-in run reads all eight (`BUILT_IN_GROUPS`); without the pool groups it is
`CORE_GROUPS`. The graph model needs `message_core`.

- **Age**: the context's cutoff minus the event's time. **Pair gap**: the event's time
  minus the previous event of the same directed pair and rail (`gap_present` 0 if none).
- **Pair history**: earlier payments of the same directed pair (relation, rail and
  counterparty) strictly before the sampled event, and the first one's age. An Account
  recipient wins over a Token; an unresolved token stays its own counterparty.
- **Flow timing**, for Account contexts only, measures coincidence, not the movement of
  the same dollars (the schema has no balances). For an incoming payment: the delay to
  the account's next outgoing payment before the cutoff, censored if none yet. For an
  outgoing payment: the time since the previous incoming payment. Both carry the
  outflow's amount over the inflow's and whether both used the same rail.

### Transforms

Counts, amounts, durations and the flow amount ratio get `log1p`. Flags, the entity-type
indicators a group lists under `identity` in the registry, `gap_present` and every
Fourier coordinate pass through unchanged (`batching.features.IDENTITY` and its
`_fourier_` rule). Missing amounts stay apart from real zeros (`amount_present`).

### The pool groups

Counts over the payment messages of the root's own candidate pool (at most `recent +
older + distinct` per payment relation, 16 in the built-in run), not its whole history,
computed by `batching.pool_counts.pool_activity`. The TGAT model reads them for roots
only, in its summary branch; children and stubs hold zeros. Every count gets `log1p`.

| Group | Column | Meaning |
|---|---|---|
| `pool_activity` | `pool_<relation>_count`, `pool_<relation>_unique` | Candidate payments and distinct counterparties (entity type and id) of each payment relation: `zelle_out`, `zelle_in`, `payment_out`, `payment_in` |
| `pool_activity` | `pool_in_unique`, `pool_out_unique` | Distinct payers over `zelle_in` and `payment_in`; distinct payees over `zelle_out` and `payment_out` |
| `pool_activity` | `pool_first_in` | Inflows from a first-time payer (`pair_prior_count` is 0) |
| `pool_activity` | `pool_pass_through_1d` | Inflows whose next outflow followed within 24 hours and moved 50 to 100 percent of the inflow: `flow_present`, `flow_ratio_present`, `flow_delay_seconds` of at most 86,400 and `flow_amount_ratio` from 0.5 to 1 |
| `pool_internal_inflows` | `pool_first_in_internal` | First-time inflows with an internal payer (`peer_external` false) |
| `pool_internal_inflows` | `pool_first_in_internal_100`, `pool_first_in_internal_1000` | Internal first-time inflows whose amount is present and at least 100, or at least 1,000 |

The first-time and pass-through tests read the per-message `pair_*` and `flow_*` fields,
which the query computes over the account's whole visible history, so the counts are
cutoff-safe. The bands, 24 hours and 50 to 100 percent are `FIRST_INFLOW_BANDS`,
`PASS_THROUGH_SECONDS` and `PASS_THROUGH_RATIO` of `contract.feature_groups`.

## The model's inputs

For B roots and N distinct contexts of a built-in run batch:

| Tensor | Shape | Content |
|---|---|---|
| `x` | N x 24 | The 9 node columns (`entity_meta` and `history_withheld`), then the 15 pool counts (roots only; other contexts hold zeros) |
| `first_edge` | B x 16 x 142 | The hop-1 message columns: `amount`, `amount_present`, `is_event`, `gap_present`, 64 age and 64 pair-gap coordinates, 3 pair-history and 7 flow-timing values |
| `second_edge` | N x 4 x 142 | The hop-2 message columns |
| `second_x` | N x 4 x 9 | The 9 node columns of each hop-2 message's counterparty, from that message: the type from `node_type`, `is_external` and `is_deposit` from `peer_external` and `peer_deposit`, and `history_withheld` 1 for a hub |
| `*_relation`, `*_rail` | slot indices | Categorical codes the model embeds |
| `*_mask`, `root_positions`, `neighbor_positions` | | Padding masks and batch-local positions |

Positions are batch-local: the same account at two cutoffs is two contexts, and there is
no global id table. Without the pool groups `x` is N x 9.

## What the training query returns

`fetch_training_context` returns one row per requested context. An `ok` row holds
`status` (`ok`), `request_index`, `contract_version` (`CONTEXT_CONTRACT`), `basis_id`, the request key
(`node_type`, `node_id`, `cutoff_seq`, `cutoff_ms`, `visibility_phase`),
`features` (node features), `messages` (the candidate pool), and `age_encoding` and
`gap_encoding` (empty unless the request asked for the Fourier vectors). A rejected
request gets only `status` and `request_index`; [Queries](queries.md) lists the statuses.

Each message has 25 fields, each of which batches read:

| Fields | Content |
|---|---|
| `node_type`, `node_id`, `relation`, `rail`, `event_id`, `event_seq`, `event_ts_ms` | The counterparty, the relation and rail, and the event |
| `amount`, `amount_present` | The payload |
| `age_ms`, `gap_ms`, `gap_present` | The age at the cutoff and the gap since the pair's previous event |
| `pair_prior_count`, `pair_first_age_seconds`, `pair_first_present` | Pair history |
| `flow_delay_seconds`, `flow_present`, `flow_censored`, `flow_observation_seconds`, `flow_amount_ratio`, `flow_ratio_present`, `flow_same_rail` | Flow timing |
| `peer_first_ms`, `peer_external`, `peer_deposit` | The counterparty's first observation and flags |

Association messages carry the tenure's target with the parent's cutoff clocks and no
payment time. The query computes every field whatever the model; a model reads those of
its groups.

## Fingerprints

- **Contract fingerprint** (`contract.feature_groups.contract_fingerprint`): what
  TigerGraph returns, meaning `CONTEXT_CONTRACT`, the time basis, the relations, the
  rails and the registry's groups other than the pool groups. A saved model records it;
  a model of another contract is refused.
- **Input fingerprint** (`FeaturePlan.fingerprint`): a model's inputs, meaning the
  contract, its groups and architecture, and for a pool group the pool definitions (the
  bands, the pass-through thresholds and `POOL_ACTIVITY_VERSION`). A model trained under
  other pool definitions is refused once they change.
