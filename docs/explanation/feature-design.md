# Feature design

Why the model reads what it reads, and the limits; [Features](../reference/features.md)
lists the groups and their columns. Every feature is a hypothesis: no group has an
established mule-detection lift until the control experiments measure it over seeds.

## Window-free messages

The model reads payments as messages, each with its own amount, rail, age and pair gap,
and has no hard-window input: the counts and sums over the last hour, day, week and month
of earlier designs are gone (below), and time reaches the model only through each
message's own clocks.

| Group | Why, and where it stops |
|---|---|
| `entity_meta` | The only node input from TigerGraph (type, external and deposit flags), so a control can drop it and leave the model only events |
| `message_core` | Direction and rail stay on each message (relation and rail embeddings). Only USD events: others are excluded one by one, never summed with USD, and never abort an otherwise valid context; there is no currency conversion |
| `time_encoding` | Age at the cutoff and the gap since the same directed pair's previous payment on the same rail, as 64 fixed Fourier coordinates each ([Time encoding](time-encoding.md)); never compared with today's date |
| `pair_history` | Earlier payments of the same pair, and the first one's age, strictly before the sampled event: a first contact is a classic mule signal |
| `flow_timing` | Timing coincidence (delay from an inflow to the next outflow, censored when none came yet, or since the previous inflow; amount ratio; same rail), not tracing of the same funds: the schema has no balances, and missing outgoing history is censored, not a negative finding. Account contexts only |
| `hub_indicator` | Flags a stub, so the model can tell a hub whose history was withheld from a dormant account |

### How TigerGraph computes them

- One bounded heap, ordered by relation, is filled from the account's participation
  scans, with canonical counterparty keys serving both the distinct-counterparty part of
  the pool and the pair keys.
- For Accounts one ascending pass over the retained history gives each payment's
  predecessor, prior count and first observation, without scanning the sender's history
  per sampled event; Token contexts keep that scan.
- Flow timing searches the bounded retained history (at most the selected events times
  the history operations), with no extra adjacency traversal.
- Each candidate takes its counterparty and the counterparty's metadata in the select
  that reads its role edge; Token contexts read their sampled events' sender and
  recipient again, set-based, for the scan.
- Per the [GSQL accumulators](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/accumulators),
  numeric `MapAccum` values add, so pair counts increment by one; last-event times use
  `MaxAccum` values.
- No silent truncation: more than `max_history` visible USD events in one relation
  rejects the context (`history_capacity_exceeded`) rather than calling a truncated count
  all-time, losing the older events to a burst, or inventing a missing predecessor. That
  is an admission guard, not a solution for high-degree accounts, which the hub registry
  keeps out of child requests.
- A neighbour first observed on the payment that connects it contributes its immutable
  metadata with an empty prior history; refusing it would break first-interaction
  inference.

## The pool groups

Without summary inputs a root's node vector was its entity type and three flags,
identical for every internal deposit account, so a mule could show only through attention
over at most 16 sampled payments, a weighted average that cannot count. [The diagnostic
study](../research/diagnostic-study.md) found the strongest signals in counts over the
payments TigerGraph already returns for the account (distinct payers, incoming payments,
inflows from first-time payers): distinct payers and internal first-time inflows alone
ranked test mules at a weighted ROC AUC of 0.88 and 0.92, against the model's 0.78 then.
So the built-in run adds two client-computed groups, read for the roots by the TGAT
model's summary branch:

- **`pool_activity`**: candidate payments and distinct counterparties per relation,
  distinct payers and payees, first-time inflows and rapid pass-throughs (an inflow
  forwarded within 24 hours at 50 to 100 percent of its amount);
- **`pool_internal_inflows`**: first-time inflows from internal payers, all and of at
  least 100 and 1,000.

They come from the per-message pair and flow fields, so TigerGraph never sends them, the
query knows nothing of them, and changing them leaves the query and prepared datasets
alone; their definitions are in a model's input fingerprint, so a model trained with
other definitions is refused. Their limits:

- **They count the pool, not the history:** at most `recent + older + distinct` (16)
  payments per relation, so `pool_payment_out_count` sits at its cap for most roots and
  the distinct part of the pool (events with new counterparties) supplies many first-time
  inflows.
  Other pools mean other counts. The cap bounds drift between cutoffs, though Zelle
  counts still roughly doubled between the study's train and test cutoffs.
- **Chosen after reading the data:** the bands, 24 hours and 50 to 100 percent are round
  numbers, but the counts were picked by a study that had read the generator's mule
  typology and test-split mules, so test audits are optimistic for them; report
  validation too.
- **The internal counts suit the generator,** which places scam victims inside the bank,
  where a real bank mostly does not see them: on the study's data a logistic regression
  on `pool_activity` alone reached a test AP of about 0.04, against 0.20 to 0.24 with
  `pool_internal_inflows`. Hence a group of their own, which `drop_pool_internal_inflows`
  drops.
- **Revealed mules are louder:** mules with more victim reports, which come from the
  inflows these counts see, are revealed more often, so observed-label metrics overstate
  the counts; compare revealed and hidden mules in the audit.
- **A lead, not a setting:** the 0.5 lower ratio bound flags almost half of the test
  non-mules; a narrower band (0.8 to 0.99) separated mules better on the study's data,
  but was found after looking at labels.

## Groups of earlier designs

The other ten groups of earlier designs (entity age, history support, rolling windows,
amount ratios, recency, association counts, decayed activity, identity order, pair window
counts, and device and IP ages) are gone. Training keeps only the built-in run's groups,
so its query computes nothing no model reads (it shrank from 1,833 to 1,282 lines, a
message from 34 fields to 27), and the analytics query that still computed them for `mule
diagnose` was deleted. A group returns to training only by adding it to the training
query on purpose, with a new contract, and then a drop variant measures it like any
other.

## Deferred

Not built until the first measured wave says they are worth it: full counterparty
concentration and reciprocity summaries, thresholded forwarding summaries, root-only
causal walks, neighbours at the root's clock, external-peer stubs, association time
brackets and caches shared across samplers. Real association timestamps, mule onset and
discovery clocks, and accounts opened after the cutoff need source or generator work; no
timestamp was invented for them.
