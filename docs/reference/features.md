# Features

What the model reads, and what the analytics query computes beside it. Training keeps
only the feature groups of the built-in run: they are the registry
`contract.feature_groups.FEATURE_GROUPS`, the training context query
`fetch_training_context` computes the ones TigerGraph supplies, and the client computes
the rest. Every other group is analytics: `contract.analytics_features.ANALYTICS_GROUPS`,
computed only by `fetch_analytics_context` for `mule diagnose`, and no training module may
import it (the import contract "Training never reads the analytics features").
[Feature design](../explanation/feature-design.md) explains why the groups are what they
are; [Time encoding](../explanation/time-encoding.md) explains the Fourier coordinates.

Model columns follow the registry's order, whatever order a configuration lists its
groups in. Labels, masks, ring ids and learned account embeddings are never features.

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

The built-in run reads all eight (`BUILT_IN_GROUPS`); the graph model needs
`message_core`. Without the pool groups the same model is `CORE_GROUPS`.

What the message groups mean:

- **Age** is the context's cutoff minus the event's time; the **pair gap** is the event's
  time minus the previous event of the same directed pair and rail (`gap_present` is 0
  when there is none).
- **Pair history** counts the earlier payments of the same directed pair (relation, rail
  and counterparty), strictly before the sampled event, and gives the age of the first
  one. An Account recipient wins over a Token; an unresolved token stays its own
  counterparty.
- **Flow timing** measures coincidence, not the movement of the same dollars (the schema
  has no balances). For an incoming payment it is the delay to the account's next
  outgoing payment before the cutoff, censored when there is none yet; for an outgoing
  payment, the time since the previous incoming payment. Both carry the outflow's
  amount over the inflow's and whether both used the same rail. Only Account contexts
  have it.

### Transforms

Counts, amounts, durations and the flow amount ratio get `log1p`. Flags and the
entity-type indicators that a group lists under `identity` in the registry pass through
as they are, and so do `gap_present` and every Fourier coordinate
(`batching.features.IDENTITY` and its `_fourier_` rule). Missing amounts stay apart from
real zeros (`amount_present`).

### The pool groups

Counts over the payment messages of the root's own candidate pool (at most `recent +
older + distinct` per payment relation, 16 in the built-in run), not over its whole
history, computed by `batching.pool_counts.pool_activity`. The TGAT model reads them for
the roots only, in its summary branch, so children and stubs hold zeros there. Every
count gets `log1p`.

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
cutoff-safe. The bands, the 24 hours and the 50 to 100 percent are
`FIRST_INFLOW_BANDS`, `PASS_THROUGH_SECONDS` and `PASS_THROUGH_RATIO` of
`contract.feature_groups`.

## The model's inputs

For B roots and N distinct contexts of a batch of the built-in run:

| Tensor | Shape | Content |
|---|---|---|
| `x` | N x 24 | The 9 node columns (`entity_meta` and `history_withheld`), then the 15 pool counts (roots only; other contexts hold zeros) |
| `first_edge` | B x 16 x 142 | The hop-1 message columns: `amount`, `amount_present`, `is_event`, `gap_present`, 64 age and 64 pair-gap coordinates, 3 pair-history and 7 flow-timing values |
| `second_edge` | N x 4 x 142 | The hop-2 message columns |
| `second_x` | N x 4 x 9 | The node columns of the outermost counterparties, from the connecting message (`peer_external`, `peer_deposit`) |
| `*_relation`, `*_rail`, `*_channel`, `*_stratum` | slot indices | Categorical codes; the model embeds the relation and the rail |
| `*_mask`, `root_positions`, `neighbor_positions` | | Padding masks and batch-local positions |

Positions are batch-local: the same account at two cutoffs is two contexts, and no global
id table exists. Without the pool groups `x` is N x 9.

## What the training query returns

`fetch_training_context` returns one row per requested context. An `ok` row holds
`request_index`, `contract_version` (`CONTEXT_CONTRACT`), `basis_id`, the request key
(`node_type`, `node_id`, `cutoff_seq`, `cutoff_ms`, `scope_id`, `visibility_phase`),
`diagnostics` (counts of non-USD and visible participations, never model inputs),
`features` (the node features), `messages` (the candidate pool) and `age_encoding` and
`gap_encoding` (empty unless the request asked for the Fourier vectors). A rejected
request gets only its `status` and `request_index`; [Queries](queries.md) lists the
statuses.

Each message has 27 fields:

| Fields | Content |
|---|---|
| `node_type`, `node_id`, `relation`, `rail`, `channel`, `event_id`, `event_seq`, `event_ts_ms`, `stratum` | The counterparty, the relation and rail, the event, and the sampling stratum (`recent`, `older`, `distinct` or `association`) |
| `amount`, `amount_present` | The payload |
| `age_ms`, `gap_ms`, `gap_present` | The age at the cutoff and the gap since the pair's previous event |
| `pair_prior_count`, `pair_first_age_seconds`, `pair_first_present` | Pair history |
| `flow_delay_seconds`, `flow_present`, `flow_censored`, `flow_observation_seconds`, `flow_amount_ratio`, `flow_ratio_present`, `flow_same_rail` | Flow timing |
| `peer_first_ms`, `peer_external`, `peer_deposit` | The counterparty's first observation and flags |

Association messages carry the tenure's target with the parent's cutoff clocks and no
payment time. Every message carries its channel and stratum, though no model reads them
(the loaded data's channels map one to one onto rails).

## The analytics groups

`fetch_analytics_context` takes the training query's parameters with one `include_*`
flag per group TigerGraph computes (14), computes the training groups exactly as the
training query does, and adds these. They are hypotheses and controls for analysis; no
model reads them.

| Group | Kind | Columns | Meaning |
|---|---|---|---|
| `entity_age` | node | `age_days` | Age at the cutoff since the entity's first observation |
| `history_support` | summary | `visible_event_count`, `history_lt_5_events` | Visible USD payment participations, and whether there are fewer than five (a self-transfer counts in both directions) |
| `rolling_windows` | summary | 40: `<window>_<field>` for the windows `1h`, `1d`, `7d`, `30d` | Per window, incoming and outgoing payment count, amount sum, missing-amount count, Zelle count and distinct counterparties (`out_count`, `in_count`, `out_amount`, `in_amount`, `out_missing`, `in_missing`, `out_zelle`, `in_zelle`, `out_unique`, `in_unique`) |
| `amount_ratios` | summary | `1d_out_in_amount_ratio`, `7d_out_in_amount_ratio` | `min(outgoing / max(incoming, 1.0), 100.0)` over the window's sums; needs `rolling_windows` |
| `recency` | summary | `out_recency_days`, `in_recency_days`, `out_recency_present`, `in_recency_present` | Days since the most recent outgoing and incoming payment, across counterparties |
| `association_counts` | summary | 28: `<relation>_active`, `<relation>_ended` | Active and ended tenures of each of the 14 association relations (7 families, both directions); repeated tenures count separately |
| `decayed_activity` | summary | 16: `decay_<half-life>_<direction>_<count or amount>` | Incoming and outgoing counts and amounts, each weighted by `2**(-age / half_life)` for the half-lives `1d`, `7d`, `30d`, `90d` |
| `identity_order` | summary | 6: `<relation>_starts_last10`, `<relation>_ends_last10` | Starts and ends of owner, token and device tenures since the tenth most recent payment: an order, not a duration |
| `pair_window_counts` | message | `pair_count_1h`, `pair_count_1d`, `pair_count_7d` | Earlier payments of the same pair within each window before the message (age less than the window) |
| `device_ip_context` | message | `device_age_seconds`, `device_present`, `ip_age_seconds`, `ip_present` | Age of the device and IP the payment used, at the payment; the youngest eligible observation when there are several |

The windows, half-lives, ratio bounds and identity relations are constants of
`contract.analytics_features`. The rows of the analytics query print
`ANALYTICS_CONTRACT` instead of `CONTEXT_CONTRACT`. `reference.gsql_features` mirrors the
account aggregates a payment history determines (association counts and identity order
need the root's associations and are not mirrored), and `pytest -m graph` compares the
mirror with the installed query.

## Fingerprints

- **The contract fingerprint** (`contract.feature_groups.contract_fingerprint`) covers
  what TigerGraph returns: `CONTEXT_CONTRACT`, the time basis, the relations, the rails
  and the registry's groups other than the pool groups. A saved model records it, and a
  model of another contract is refused.
- **The input fingerprint** (`FeaturePlan.fingerprint`) covers a model's inputs: the
  contract, its groups and its architecture, and, when it reads a pool group, the pool
  definitions (the bands, the pass-through thresholds and `POOL_ACTIVITY_VERSION`). A
  model trained under other pool definitions is refused once they change.
