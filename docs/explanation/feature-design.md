# Feature design

Why the model reads what it reads. The groups and their columns are listed in
[Features](../reference/features.md); this page gives the reasons and the limits. Every
feature is a hypothesis: no group has an established mule-detection lift until the
control experiments measure it over seeds.

## Window-free messages

The model reads payments as messages, each with its own amount, rail, age and pair gap,
and has no hard-window input: the counts and sums over the last hour, day, week and
month of earlier designs are analytics now (see below), and time reaches the model only
through each message's own clocks. The training groups are:

- **`entity_meta`**: the entity's type and its external and deposit flags. It is the only
  node input from TigerGraph, so a control can drop it and leave the model only events.
- **`message_core`**: the amount, its presence and whether the message is an event, with
  relation and rail embeddings. Direction and rail stay on each message. Only USD events
  are included; others are excluded one by one, counted in diagnostics, never summed with
  USD, and never abort an otherwise valid context. There is no currency conversion.
- **`time_encoding`**: the payment's age at the cutoff and the gap since the same directed
  pair's previous payment on the same rail, each as 64 fixed Fourier coordinates
  ([Time encoding](time-encoding.md)). Never compared with today's date.
- **`pair_history`**: how many earlier payments the same pair made, and how long ago the
  first was, strictly before the sampled event. A first contact is a classic mule signal.
- **`flow_timing`**: for an incoming payment, the delay to the account's next outgoing
  payment, censored when none came yet; for an outgoing one, the time since the previous
  incoming payment, with the amount ratio and whether the rails match. It measures timing
  coincidence, not the tracing of the same funds: the schema has no balances, and missing
  outgoing history is censored, not a negative finding. Only Account contexts have it.
- **`hub_indicator`**: a flag on a stub, so the model can tell a hub whose history was
  withheld from a dormant account.

### How TigerGraph computes them

The query fills bounded heaps per relation from the account's participation scans, with
canonical counterparty keys that serve both the distinct-counterparty stratum and the
pair keys. For Accounts one ascending pass over the retained history computes each
payment's predecessor, prior count and first observation without scanning the sender's
history per sampled event; Token contexts keep that scan. Flow timing searches the
bounded retained history (at most the selected events times the history operations) with
no extra adjacency traversal. Roles, counterparty metadata and the enrichment of sampled
events are set-based selects over all sampled events of a request. The accumulators
follow the [GSQL reference](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/accumulators):
numeric `MapAccum` values add, so pair counts increment by one, and last-event times use
`MaxAccum` values.

There is no silent truncation: more than `max_history` visible USD events in one relation
rejects the context (`history_capacity_exceeded`) instead of calling a truncated count
all-time, losing an older stratum to a burst, or inventing a missing predecessor. That is
an admission guard, not a solution for high-degree accounts; the hub registry keeps those
out of child requests.

A neighbour first observed on the very payment that connects it contributes its immutable
metadata with an empty prior history; refusing such a peer would break first-interaction
inference.

## The pool groups

Without summary inputs, a root's node vector held only its entity type and three flags,
identical for every internal deposit account, so a mule could show only through attention
over at most 16 sampled payments, a weighted average that cannot count. A diagnostic study
([the diagnostic study](../research/diagnostic-study.md)) found the strongest signals in
counts over the payments TigerGraph already returns for the account: distinct payers,
incoming payments, inflows from first-time payers. Distinct payers and internal first-time
inflows alone ranked test mules at a weighted ROC AUC of 0.88 and 0.92, against the
model's 0.78 then. So the built-in run adds two client-computed groups, read by the TGAT
model's summary branch for the roots:

- **`pool_activity`**: candidate payments and distinct counterparties per relation,
  distinct payers and payees, first-time inflows and rapid pass-throughs (an inflow
  forwarded within 24 hours at 50 to 100 percent of its amount);
- **`pool_internal_inflows`**: first-time inflows from internal payers, all and of at
  least 100 and 1,000.

They are computed from the per-message pair and flow fields, so TigerGraph never sends
them, the query has no flag for them, and adding or changing them leaves the query and a
prepared dataset as they are. Their definitions are part of a model's input fingerprint,
so a model trained with other definitions is refused.

Their limits:

- **They count the pool, not the history.** At most `recent + older + distinct` payments
  per relation (16), so `pool_payment_out_count` sits at its cap for most roots, and the
  distinct stratum, which picks events with new counterparties, supplies many of the
  first-time inflows. Changing the pools changes what the counts mean. The cap bounds
  drift between cutoffs, though Zelle counts still roughly doubled between the train and
  test cutoffs of the study.
- **They were chosen after reading the data.** The bands, the 24 hours and the 50 to 100
  percent are round numbers, but the choice of counts followed a study that had read the
  generator's mule typology and test-split mules, so test audits are optimistic for them;
  report validation as well.
- **The internal counts suit the generator.** PhantomLedger places scam victims inside the
  bank, which a real bank mostly does not see: on the study's data a logistic regression
  on `pool_activity` alone reached a test AP of about 0.04, against 0.20 to 0.24 with
  `pool_internal_inflows` as well. That is why the internal counts are a group of their own
  that the `drop_pool_internal_inflows` variant drops.
- **Revealed mules are louder.** Mules are revealed more often the more victim reports they
  have, and the reports come from the same inflows these counts see, so observed-label
  metrics overstate them; compare revealed and hidden mules in the audit.
- **A lead, not a setting.** The 0.5 lower ratio bound flags almost half of the test
  non-mules; a narrower band (0.8 to 0.99) separated mules better on the study's data, but
  it was found after looking at labels.

## The analytics groups

The other ten groups (entity age, history support, rolling windows, amount ratios,
recency, association counts, decayed activity, identity order, pair window counts, and
device and IP ages) are computed only by the analytics context query, for `mule
diagnose`. They were optional controls of earlier designs; training keeps only the
built-in run's groups, so the training query computes nothing no model reads (it shrank
from 1,833 to 1,282 lines, and a message from 34 fields to 27). Their value as analysis
remains: the diagnostics baselines answer, from these account aggregates, how well a
table of the account's own activity ranks mules with no neighbour, association or pool
input. A group comes back into training only by moving it into the training query on
purpose, with a new contract, after which a drop variant measures it like any other.

What they mean and where they stop:

- Windows count incoming and outgoing payments, amount sums, missing amounts, Zelle
  payments and distinct counterparties; the amount ratio is outflow over inflow with a
  floor of 1 and a cap of 100, which is not pass-through speed.
- Decayed sums weigh each payment by `2**(-age / half_life)` at half-lives of 1, 7, 30
  and 90 days, smooth summaries whose half-lives were never tuned.
- Identity order counts owner, token and device tenure starts and ends since the tenth
  most recent payment: an order, never converted to time.
- Association counts respect their valid interval and the scope; repeated tenures count
  separately, so they are counts of tenures, not of distinct people or devices.
- Device and IP ages use the payment's own participation edges and the device's first
  observation, not the age of an association tenure.
- Pair window counts scan each sender's full outgoing history, so they are a control,
  not something to compute for large runs.

## Deferred

Ideas not built until the first measured wave says they are worth it: full counterparty
concentration and reciprocity summaries, thresholded forwarding summaries, root-only
causal walks, neighbours at the root's clock, external-peer stubs, association time
brackets and caches shared across samplers. Real association timestamps, mule onset and
discovery clocks, and accounts opened after the cutoff need source or generator work; no
timestamp was invented for them.
