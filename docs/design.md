# MulePatternLearner design

The specification the code on branch `redesign` builds. It was agreed with the owner in
October 2026 and revised on October 17 to follow the owner's direction: the model learns
from behaviour over time and from graph structure (shared devices and IPs, payment flows,
communities), never from metadata such as when an account was first seen. The research
behind the decisions is in the owner's research folder
(`MulePatternLearner-backup-2026-10-14/research/`). Values marked *default* are starting
points the owner may change; everything else is decided.

## Changes in the October 17 revision

Features:

- Every feature is behaviour of the data or graph structure; the ages from a first sighting
  (account, party, token, device, IP and address) are gone, and `opened_ms` only decides
  which accounts exist on a date.
- No feature depends on when the data starts: every duration that runs from a start is cut
  at the lookback start, with a flag that says so, and links that ended before the lookback
  are not visible.
- Short-term behaviour is added: per-payment gaps since the sender's previous payment and
  the receiver's previous receipt, the most payments in any hour and any day, the shortest
  gap between inflows, and money passed on within an hour.
- New payees are added beside new payers, both counted as distinct accounts, with a year's
  distinct counterparties to measure growth.
- Connectivity change is added: each customer's new device and IP links, each device's and
  IP's new users, and each account's device and IP co-users and how many of them are new.
- A community block per seed is added: a bounded device-and-IP community computed in the
  query as of t, with an ablation that trains with and without it.
- Token and Address leave the model; Device and IP stay. The `at_address` relation goes.
- Closure is no longer a feature: the account's closed flag, the party's open and closed
  account counts and the ownership edge's end are gone. The `owned_by` edge has no features.
- Devices and IPs that no customer is enrolled on (carrier NAT, public terminals) are leaves.
- The category values are listed in the design.

Splits and labels:

- A split by account and date replaces the customer hold-out: each account belongs to one of
  training, validation and test (50%, 25%, 25% by a seeded hash), and every account is
  visible as a neighbour in every phase.
- The split is not written on the graph, so `write_split` and the `split` attributes go and
  the queries lose `held_out_visible`.
- A mule is positive while it is active: from its first ring payment to a week after its
  last, not until its discovery; `mule_last_ms` moves from the answer key to supervision.
- Training and validation read their labels once, at the first day of the test window.
- A revealed mule is never an unlabelled example; its dates outside its activity are
  dropped.
- Validation and test rank a true mule only on the dates it is active.
- The maturity gap is now the wait before the labels are read, estimated from training's
  revealed mules (right-truncated), never below 60 days.
- A sparse-label setting keeps a seeded share of the revealed mules, and an experiment
  reports the test results at 100%, 50%, 25% and 10%.
- The population of a date is the open accounts a customer owns; external and ledger
  accounts are never examples.

Model, training and evaluation:

- nnPU weights the positive risk by 1 minus the prior by default (balanced); the textbook
  weight stays a setting.
- The five seeds vary the model only.
- The label-input confirmations come from supervision, joined in training; the query adds
  the ownership edge of every account of a seed's owner it reaches.
- A new comparison, trees with hand-made graph features, tells what message passing adds
  beyond graph aggregates.
- Test dates score a sample of 5,000 other accounts (validation keeps 2,000).
- The subgraph cache is keyed by the loaded world's identity, holds one file per request,
  and no longer depends on the phase.
- A `bench` command measures fetch, build and training cost at growing world sizes, and a
  recommended training architecture is written down.

## What we are testing

A temporal heterogeneous graph neural network finds money-mule accounts that a bank has not
found yet, from what accounts do over time and how they are connected: payments, their
speed and order, and devices and IPs shared with other customers. It must work on accounts
it never trained on, including accounts with no payments yet that share devices or IPs with
mules, and keep working when only a small share of mules is labelled. It must beat models
without graph context given the same information.

## The data

PhantomLedger (`../PhantomLedger`, use case `mule-temporal`) simulates one US bank. Every
behaviour that matters here is backed by published research and recorded in PhantomLedger's
citation ledger, `docs/fraud_model_audit.md`, including the declared choices where no
research gives a number. The design is built for this data:

- A mule is an **account** that receives proceeds of crime. The label comes from what the
  account did: `mule_onset_ms` is its first ring payment received, `mule_last_ms` its last
  ring payment received or sent after it became a mule. Columns record where it sat in the
  chain (`mule_layer`) and what it did with the money (`mule_flow`).
- Discovery is simulated inside PhantomLedger in time order: victim reports, bank
  monitoring from the first laundering payment, and tracing downstream along the money. On
  the day a mule is found its customer is offboarded, so nothing it would have done later
  exists.
- Ordinary customers also open and close accounts all year, so neither a late opening nor
  a closure marks a mule.
- Rings share devices and IPs while they operate; ordinary sharing happens inside real
  households.

**Timeline (default).** One world covers 2023-01-01 to 2024-12-31. 2023 is a warm-up with
no laundering: it is loaded so every example has a year of history, and it is never scored.

**Worlds.** Development uses one 30,000-person world. Reported numbers come from three
independent 300,000-person worlds (different seeds), plus one sensitivity world in which
ring members share devices and IPs at 10% instead of the declared 70% and 60%. The bench
also runs on 100,000-person worlds.

**What PhantomLedger must still provide.** Its money flow and behaviour labels are done.
Until the rest lands, the data differs from the list above, and training cannot run, since
no mule has a discovery date:

- Discovery and offboarding: `discovered_ms` set to the midnight that ends the discovery's
  UTC day, `discovery_channel` one of `victim_report`, `monitoring` and `trace`, the whole
  customer closed that day with its links ended, and no later activity, so
  `mule_last_ms <= discovered_ms`.
- Ordinary opening and closing all year, shipped with discovery, so that a closure or a
  link ending is not a catch marker; mule accounts that are opened for the ring; ring bursts
  kept after their members join.
- Households for ordinary device, IP and address sharing, with ring organizers in them like
  everyone else, and link starts that do not differ in pattern between households and rings.
- Scam payees and account-takeover drops as accounts with a mule role or external accounts.
- A setting that starts laundering after the warm-up.
- Card cash-outs on the `card` rail. Today they export as `unknown`, while every
  legitimate card purchase is `card`, which would make the rail a mule marker.
- Camouflage payments only while a ring member is recruited. Today they run all year, so a
  year of counts marks ring membership.

## The graph

Same vertex and edge types as before; attributes cut to what is used. One millisecond clock
everywhere; no sequence numbers. The single copy of the DDL is `gsql/schema/schema.gsql`;
PhantomLedger's export matches it.

| Type | Attributes |
|---|---|
| Party | id, joined_ms |
| Account | id, account_type, is_external, opened_ms, closed_ms; supervision: discovered_ms, discovery_channel, mule_onset_ms, mule_last_ms; answer key: is_mule (-1 unknown), mule_ring_id, mule_layer, mule_flow |
| Token, Device, IP, Address | id |
| Payment_Transaction | id, amount, payment_rail, event_ts_ms |
| Zelle_Transfer | id, amount, event_ts_ms |
| 7 identity and ownership links | valid_from_ms, valid_to_ms (0 while open) |
| Payment role and observation edges | event_ts_ms |

Everything comes from PhantomLedger; MulePatternLearner writes nothing to the graph. Each
account has at most one owner (PhantomLedger never writes joint accounts).

**What the model reads.** Party, Account, Device and IP; both payment types with their
sender, receiver and observed device and IP; `Party_Owns_Account`, `Party_Uses_Device` and
`Party_Uses_IP`. Token, Address, `Account_Uses_Device` (a copy of the owner's device links),
the token links and the token roles of payments are loaded but not read. A token is one
opaque handle per account, never shared or rebound, so it carries no graph signal; Zelle
shows as the payment rail `zelle`. Device and IP sharing is the identity signal for this
proof of concept. Addresses are left out: rings share none, and PhantomLedger's address
tokens today collide between strangers by chance. They can return once households exist, if
an ablation shows that telling households from rings needs them.

**Who may read what.**

- *Supervision* (`discovered_ms`, `discovery_channel`, `mule_onset_ms`, `mule_last_ms`) is
  what a bank knows after an investigation: when the account was found and the dates of its
  suspicious activity, which an investigation reads from the account's history. It is read
  only through `read_supervision`, which returns the mules discovered before a cutoff.
- *Answer key* (`is_mule`, ring, layer, flow) is read only by evaluation.
- Neither is ever a feature. The one exception is the label-input variant, which reads
  whether a neighbour was confirmed by t (its `discovered_ms`). `is_mule` defaults to -1, so
  an account with no loaded truth is unknown, never normal.

## Time

Every example is an (account, as-of time t) pair, with t a Monday at 00:00 UTC. Everything
the model sees is computed from what existed before t, within a lookback of 365 days
(*default*):

- **Events** count when `t - lookback <= event_ts_ms < t`. Payments in the same second have
  no order: none is another's predecessor, and all of them count in a window.
- **Links** are visible when `valid_from_ms <= t` and they had not ended before the
  lookback began (`valid_to_ms` is 0 or after `t - lookback`). A link is active at t when
  `valid_to_ms` is 0 or after t: PhantomLedger's tenures are half-open,
  `[valid_from_ms, valid_to_ms)`. A link that ended by t stays visible as a past fact with
  the time since it ended, because ring links are short and dropping them would erase the
  evidence. An end after t is invisible.
- **Confirmations** (label-input variant only) count when
  `t - lookback < discovered_ms <= t`. Since `discovered_ms` is the midnight that ends the
  day of discovery, a mule found on a Sunday reads as confirmed on the Monday.
- **Supervision** for training and validation is the mules with
  `0 < discovered_ms < R`, R the labels read date.
- **Activity.** A mule is active at t when `mule_onset_ms < t <= mule_last_ms + 7 days`
  (*default* 7) and it was not discovered by t (`discovered_ms` is 0 or after t). Since
  events must be before t, its first ring payment is visible on every date it is active.
  The week after its last ring payment gives every mule at least one active Monday, the
  first after its onset, unless it was discovered before that Monday.

**No feature depends on when the data starts.** Every duration is relative: to t, or
between two events. One that runs from a start (a link's start, a pair's first payment, a
node's first use) never reaches back past the lookback start: a start before it is taken as
the lookback start, and the flag `before_lookback` says so. The warm-up is as long as the
lookback, so on every as-of date the lookback lies inside the data (`prepare` checks it), a
link PhantomLedger clamped to the data start reads exactly like one older than the lookback,
and a customer looks the same in January and in December unless its behaviour changed. No
absolute time reaches the model, so a model trained in 2024 scores 2027 the same way.

**Every duration is Fourier-encoded.** The basis is fixed: u = log1p(seconds) / log1p(10
years), 32 log-spaced frequencies from 0.125 to 2 cycles per unit of u, sine and cosine, 64
numbers per duration, computed on the GPU, with a presence bit when a duration can be
absent. It resolves a second (u = 0.035), a minute (0.21), an hour (0.42), a day (0.58) and
a year (0.88), so timing from seconds to months reaches the model. TigerGraph computes the
durations themselves.

## Features

A feature is either **behaviour** (what an account or customer did: its payments, their
amounts, speed and order) or **graph structure** (who is linked to whom through payments,
owners, devices and IPs, and how those links change). Both kinds come in a temporal form
(over windows before t, or as durations) and a non-temporal form (the state of the graph at
t). TigerGraph computes every feature as of t from what is visible; Python only encodes
them:

- categories one-hot over their values: `account_type` is `deposit`, `credit`,
  `brokerage`, `gl` or `unknown`; `rail` is `ach`, `card`, `cash`, `check`, `internal`,
  `unknown` or `zelle` (a `Zelle_Transfer`'s rail is `zelle`);
- flags 0 or 1;
- counts and amounts as log1p;
- shares (numbers in [0, 1]) as they are;
- durations as a presence bit and the 64 Fourier numbers.

**Left out, and why.**

- First-seen ages of accounts, customers, tokens, devices, IPs and addresses. PhantomLedger
  clamps them to the data start, so they measure the window, not the customer; and the
  owner rules out an account's age even where it would be real.
- Closure: the account's closed flag, the customer's open and closed accounts, and the
  ownership edge's end. A closure is the bank's action, not mule behaviour; once
  PhantomLedger offboards found mules it would mark caught neighbours.
- The clock of day and the second within a minute. PhantomLedger schedules many rows on
  whole minutes and at fixed UTC hours, choices of the generator in one naive time zone.
- Tokens and addresses, as said under the graph.

The ends of identity links stay (since a link ended, how long it was held): a customer
stopping use of a device is behaviour, and ring links end soon after a ring's payments.
Until PhantomLedger simulates ordinary closures, an offboarded customer's links all end on
its discovery day; this is listed under the limitations.

### The feature table

"Last N days" means `t - N days <= event_ts_ms < t`. Every count is over visible payments
and links. A *hub* and an *observed-only* node are defined under "Hubs and observed-only
nodes" below; *shares* is defined under "Shared devices and IPs" below.

| Where | Feature | Temporal | Kind | How TigerGraph computes it, as of t |
|---|---|---|---|---|
| Account | `account_type` (5 one-hot) | no | structure | the account's attribute |
| Account | `is_external` | no | structure | the account's attribute |
| Account | `hub` | no | structure | the account is a counterparty hub |
| Account | `in_7d`, `in_30d`, `in_365d` | yes | behaviour | payments received in the last 7, 30 and 365 days |
| Account | `out_7d`, `out_30d`, `out_365d` | yes | behaviour | payments sent in the last 7, 30 and 365 days |
| Account | `amount_in_7d`, `amount_in_30d`, `amount_out_7d`, `amount_out_30d` | yes | behaviour | sum of the amounts received and sent in the last 7 and 30 days |
| Account | `counterparties_in_30d`, `counterparties_in_365d`, `counterparties_out_30d`, `counterparties_out_365d` | yes | structure | distinct accounts it received from and sent to in the last 30 and 365 days |
| Account | `new_payers_7d`, `new_payers_30d` | yes | structure change | distinct senders whose earliest payment to it in the lookback is in the last 7 or 30 days |
| Account | `new_payees_7d`, `new_payees_30d` | yes | structure change | distinct receivers whose earliest payment from it in the lookback is in the last 7 or 30 days |
| Account | `burst_in_1h`, `burst_in_24h` | yes | behaviour | the most payments received within [x, x + 1 hour) or [x, x + 24 hours), x the time of one of them, over the last 30 days |
| Account | `burst_out_1h`, `burst_out_24h` | yes | behaviour | the same over payments sent |
| Account | `burst_new_payers_24h` | yes | behaviour | the most new payers of the last 30 days whose first payment falls within any 24 hours |
| Account | `passed_on_1h`, `passed_on_24h` | yes | behaviour | the amount of the last 30 days' inflows followed, at or after them and within 1 hour or 24 hours, by a payment it sent |
| Account | `shortest_in_gap_ms` (present with at least two inflows) | yes | behaviour | the shortest time between consecutive payments received in the last 30 days |
| Account | `since_last_in_ms`, `since_last_out_ms` (each with presence) | yes | behaviour | t minus its latest payment received, and sent, in the lookback |
| Account | `device_co_users`, `ip_co_users` | no | structure | distinct other customers that share a device, or an IP, with its owner |
| Account | `new_device_co_users_30d`, `new_ip_co_users_30d` | yes | structure change | those co-users whose sharing with its owner began in the last 30 days |
| Party | `accounts` | no | structure | accounts it owns that were opened by t, open or closed |
| Party | `devices_now`, `ips_now` | no | structure | devices and IPs it has an active link to |
| Party | `new_devices_7d`, `new_devices_30d`, `new_ips_7d`, `new_ips_30d` | yes | structure change | its device and IP links that began in the last 7 or 30 days |
| Device, IP | `parties_now` | no | structure | customers with an active link to it |
| Device, IP | `parties_lookback` | no | structure | customers with a link to it visible in the lookback |
| Device, IP | `new_parties_7d`, `new_parties_30d` | yes | structure change | customers whose earliest link to it in the lookback began in the last 7 or 30 days, after the lookback start |
| Device, IP | `accounts` | no | structure | accounts its lookback customers own, and accounts observed sending from it in the lookback |
| Device, IP | `payments_7d`, `payments_30d` | yes | behaviour | payments observed on it in the last 7 and 30 days |
| Device, IP | `senders_30d` | yes | structure | distinct accounts observed sending from it in the last 30 days |
| Device, IP | `observed_only` | no | structure | observed on a payment in the lookback, with no customer linked to it in the lookback |
| Device, IP | `hub` | no | structure | the node is a hub |
| `pays` edge | `amount` | no | behaviour | the payment's amount |
| `pays` edge | `rail` (7 one-hot) | no | behaviour | `payment_rail`, or `zelle` |
| `pays` edge | `age_ms` | yes | behaviour | t minus the payment's time |
| `pays` edge | `prior_count`, `first_in_pair` | yes | structure | payments of the same sender to the same receiver earlier in the lookback, and whether there are none |
| `pays` edge | `gap_ms`, `pair_age_ms` (present when `prior_count` > 0) | yes | behaviour | time since the latest and the earliest of those earlier payments |
| `pays` edge | `next_outflow_ms` (with presence) | yes | behaviour | time to the receiver's earliest payment sent at or after it, before t |
| `pays` edge | `sender_gap_ms` (with presence) | yes | behaviour | time since the sender's latest earlier payment sent to anyone in the lookback; absent when the sender is a hub |
| `pays` edge | `receiver_gap_ms` (with presence) | yes | behaviour | time since the receiver's latest earlier payment received from anyone in the lookback; absent when the receiver is a hub |
| `enrolled_on_device`, `enrolled_on_ip` edges | `since_start_ms`, `before_lookback` | yes | structure change (relationship age) | t minus the later of the link's start and the lookback start, and whether the start was at or before the lookback start |
| `enrolled_on_device`, `enrolled_on_ip` edges | `active`, `since_end_ms` (present when not active) | yes | structure change | whether the link is active at t, and t minus its end |
| `enrolled_on_device`, `enrolled_on_ip` edges | `held_ms` | yes | structure change | from the later of its start and the lookback start to its end, or to t while active |
| `owned_by` edge | none | no | structure | the edge itself |
| `sent_from_device`, `sent_from_ip` edges | `uses` | yes | behaviour | the account's payments observed on the node in the lookback |
| `sent_from_device`, `sent_from_ip` edges | `since_first_ms`, `since_last_ms` | yes | behaviour | t minus the first and the last of them |
| Community of the seed | `parties`, `capped` | no | structure | members of the seed's community, and whether the cap left some out |
| Community of the seed | `nodes`, `links` | no | structure | devices and IPs shared by at least two members, and member-node sharing pairs on them |
| Community of the seed | `new_links_30d` | yes | structure change | those sharing pairs that began in the last 30 days |
| Community of the seed | `accounts` | no | structure | accounts the members own that were opened by t |
| Community of the seed | `internal_payments_30d`, `internal_amount_30d` | yes | behaviour | payments in the last 30 days between accounts of two different members, and their amount |
| Community of the seed | `external_payments_30d` | yes | behaviour | the members' payments in the last 30 days whose other account no member owns |
| Community of the seed | `density` | no | structure | `links / (parties * nodes)`, 0 with no nodes |
| Community of the seed | `isolation` | yes | structure | `internal / (internal + external)` payments of the last 30 days, 0 with none |
| Account (label-input variant) | `confirmed`, time since confirmation | yes | label input | joined in training from supervision (see "The model") |

Every account in a seed's subgraph carries the account features, so a hop-2 neighbour's
behaviour and sharing reach the seed. The per-payment gaps, bursts, pass-on within an hour,
new payees and co-users are what a pass-through mule does in minutes and hours: by value,
PhantomLedger's mules pass on 28% of a credit within 15 minutes, 53% within an hour and
86% within a day, and nearly every victim is a first-time payer.

### Hubs and observed-only nodes

- An account is a **counterparty hub** when more than `hub_threshold` distinct customers own
  the other accounts of its payments in the lookback (billers, merchants, the bank's
  ledgers). One with no more payments than that in all time cannot be one and is not
  counted.
- A device or IP is a **hub** when more than `hub_threshold` distinct customers are linked
  to it, or own accounts observed on it, in the lookback.
- A device or IP is **observed-only** when it appears on payments but no customer is linked
  to it in the lookback: carrier NAT lines and public terminals, which link strangers.

A hub or an observed-only node stays in the subgraph as a leaf, with its features, but is
not expanded and does not count for co-users or communities. `hub_threshold` is 200
(*default*), above the largest ring PhantomLedger can make (about 105 device holders) and
far above ordinary groups (2 to 6). `mule bench` reports the distribution of distinct
customers per counterparty, device and IP, by enrolled and observed-only, so the owner can
lower it.

### Shared devices and IPs

A customer **shares** a device or IP at t when it has a link to it that was active at some
time in the last 30 days, or one of its accounts sent a payment observed on it in the last
30 days, and the node is neither a hub nor observed-only. Thirty days keeps a ring's links
in view on the dates after its last payment, since ring tenures end a few days after it.
Two customers that share the same node are **co-users**; their sharing began at the later of
the two starts (a link's start or a first observed use in the lookback).

`device_co_users`, `ip_co_users` and their new counts follow this rule for an account's
owner. They carry the sharing of accounts whose own devices and IPs are not in the subgraph
(hop-2 accounts and those reached through an identity node).

### The community of a seed

The owner asked for "clustered connections with isolated communities". A global algorithm
(TigerGraph's connected components, Louvain or label propagation) runs on the whole graph
with no as-of time, so it would carry future ring members into training, and the observed
hubs would merge strangers. The community is therefore local, bounded and computed inside
`fetch_subgraphs` for each seed as of t:

1. The members start as the seed's owners.
2. A first round adds every customer that shares a node with a member; a second round does
   the same from the customers the first round added. There is no third round.
3. At most `community_parties` members per seed (*default* 64): the seed's owners first,
   then the first round's, then the second round's, within a round the newest sharing
   first, ties by party id. `capped` says whether a reachable customer was left out.

The measures are those of the feature table. A ring holding a device and an IP gives a small,
dense community whose links began recently and whose members pay each other; an ordinary
group gives older links and little money inside. The cost is bounded by the cap: up to 64
members per seed, each with a handful of shared nodes and 30 days of payments. The block is
per seed, so it is printed once per seed and goes to the model's head (see "The model").
Whether it adds anything to what two message-passing layers already see is measured by the
community ablation (see "Experiments").

## Splits and labels

**By date (defaults).**

| Window | As-of dates (Mondays) | Accounts | Labels |
|---|---|---|---|
| Training | 2024-01-01 to 2024-06-30, ending the maturity gap before R | the training split | mules discovered before R |
| Validation | 2024-07-01 to 2024-09-30 | the validation split | mules discovered before R |
| Test | 2024-10-01 to 2024-12-31 | the test split | the full answer key |

R, the **labels read date**, is 00:00 UTC of the test window's first day (2024-10-01).
Training and validation read their labels once, at R, as a bank would after the activity
it trains on. Every mule they label was discovered before the test begins, so in
PhantomLedger it was offboarded before any test date. A mule active in the training window
but found in August is a training positive, which reading the labels at the window's end
would have missed.

**By account.** Each account belongs to exactly one split: the 64-bit blake2b hash of its id,
keyed by the data seed, falls in the training split's first 50%, the validation split's next
25%, or the test split's last 25% (*default* shares). No account is an example in more than
one window. The split is computed in Python and never written on the graph, so no query
writes, and one customer's accounts can fall in different splits.

Every account is visible in every phase as a neighbour: its payments, its owner, its devices
and IPs, and its features. Only its labels are kept to its own split: the training split's
revealed mules are training positives, and other splits' accounts are never examples in
training. A test account is scored from its relationships with accounts the model trained
on; sharing devices, IPs and other identity with several mules is the evidence the model is
meant to use, so a neighbour that was a training positive is not leakage. The model has no
per-account parameters, so having seen an account as a neighbour does not let it remember
that account.

The design still reports how the model does on **customers it has never trained on**: the
test accounts whose customer owns no account in the training split, as a group of their own.

**Labels from mule activity.** A mule is positive while it is active (see "Time"), from
just after its first ring payment to a week after its last, not from when it was caught.
Supervision therefore holds `mule_onset_ms` and `mule_last_ms` for each discovered mule: an
investigation that catches an account reads its whole history, as a suspicious activity
report records the date range of the activity. `read_supervision` refuses a row unless
`0 < mule_onset_ms <= mule_last_ms <= discovered_ms`, because a mule is found only for what
it has done and nothing happens after offboarding. Quiet weeks between a mule's bursts lie
inside its activity and are positive: PhantomLedger exports no per-burst clocks, and real
mule accounts are reused. The catch decides only when the account stops existing: an
account closed by t is no example on t.

**A caught mule stays a mule.** Practice works forward from the catch and dates the
activity separately: a Cifas marker is kept for six years from its filing, a bank exits the
customer, and production AML systems time their positives on the activity's start and end.
None relabels the account's earlier life, and doing so would teach the model who was caught,
not what mules do. So "permanently" means:

- A revealed mule is never an unlabelled example, on any date. It is positive on the dates
  it is active and dropped on every other date: before its onset, and between a week after
  its last activity and its discovery.
- For the label-input variant it stays confirmed for its neighbours on every later date,
  with the time since its confirmation, so the model can learn how a confirmation's weight
  fades.
- The label belongs to the account that carried the money, where the behaviour is. The
  customer's other accounts are labelled by their own behaviour; in PhantomLedger a mule's
  own second deposit account that the money reached is a mule itself.
- Evaluation removes a mule from every date's ranking once it is discovered, since a bank
  does not alert on an account it has already caught.

**The maturity gap.** Training's last as-of date stays the maturity gap before R, because a
mule whose activity began just before R has had no time to be found and would sit among the
unlabelled. The gap is the delay within which the estimate finds 90% of mules, from onset to
discovery, estimated from the training split's mules revealed at R, never from the answer
key, and never less than 60 days (*default*). Revealed mules are only those found before R,
so their delays are right-truncated, not right-censored; the estimate is the product-limit
estimator for right-truncated data (Lynden-Bell). With R three months after the training
window, the 60-day floor does not bind unless discovery is slow; `prepare` reports the
estimate.

**Hidden labels come from discovery, not from masking.** On any date only the mules found
by R are known; the rest are unlabelled, which is the positive-unlabelled setting the nnPU
loss needs. Discovery is not random (victim reports find the busiest mules first), which
violates nnPU's "selected completely at random" assumption; we report this as a limitation,
and the full-answer-key test measures its effect.

**Sparse labels.** `revealed_share` (*default* 1) keeps a share of the revealed mules: a
mule is kept when the 64-bit blake2b hash of its account id, keyed by the data seed and the
word `revealed`, falls in the lowest `revealed_share` of its range. The kept sets are nested
(every mule kept at 10% is kept at 25%), and independent of the account split. A dropped mule
is unrevealed everywhere: it is no positive, it may be drawn as unlabelled, it is no
validation positive, it is not confirmed for the label-input variant, and it does not enter
the maturity estimate. Test always uses the full answer key.

## Examples

**The population of a date** is every account open at t (`opened_ms <= t`, and
`closed_ms` 0 or after t) that is not external and whose owner's ownership link is active at
t. External accounts and the bank's ledger accounts (no owner) are never examples, so no
account with `is_mule` -1 is ever ranked. Each window takes the accounts of its split.

**Training.**

- *Positives.* On each date, the training split's revealed mules (after `revealed_share`)
  active that date. Each mule carries a total weight of 1, shared equally among its dates.
- *Unlabelled.* On each date, a uniform draw of the training split's population, leaving
  out every revealed mule, 15 per positive on that date (*default*), including accounts with
  no payments. They share the weight of that date's positives, so the date cannot give the
  answer away.
- *Dropped.* A revealed mule on every date it is not active.

**Validation and test.** On each weekly date of the window:

- every account of the split active that date is positive, with weight 1: the revealed mules
  for validation, the answer key's `is_mule` 1 for test;
- a true mule (revealed, for validation) that is not active that date, or that was
  discovered by t, is not ranked that date;
- the other accounts are a uniform sample, 2,000 for validation and 5,000 for test
  (*defaults*), each weighted by the accounts it stands for.

Every draw is seeded by the data seed and the date, so it depends only on the accounts it
draws from.

## The subgraph query

One installed GSQL query, `fetch_subgraphs`, takes a batch of seed accounts that share one
as-of time t and the subgraph settings. It returns, for each seed, a typed 2-hop subgraph as
of t (nodes by type with TigerGraph-computed features, edges by relation with theirs) and the
seed's community block. Python turns a batch into one PyG `HeteroData`.

The subgraph is not a static snapshot: every payment is its own edge with its age and its
gaps, and every identity link carries its start and end, so the attention reads when things
happened and in what order, as temporal graph attention does with a time encoding on its
edges.

A node reached by two paths appears once in a seed's subgraph, so cycles (Z, device, A,
payment, Z) are visible. Nodes are not merged across the seeds of a batch: each seed's
subgraph is a component of its own, so a seed's score never depends on its batch, at the
cost of copying shared neighbours. Seeds are grouped by as-of date so each node's features
are computed once per request.

**Relations** (each has a reverse, so two layers carry hop-2 information to the seed; caps
are *defaults*):

| Relation | Cap per node |
|---|---|
| Account sent payment to Account (one edge per payment) | 32 newest per direction for a seed, 16 for others |
| Account owned by Party; Party owns Account (siblings) | 1; 8 |
| Account enrolled on Device (through its owner) | 8; reverse 16 other customers' deposit accounts |
| Account enrolled on IP (through its owner) | 8; reverse 16 |
| Account sent from Device (observed on its own sent payments) | 8; reverse 16 |
| Account sent from IP (observed on its own sent payments) | 8; reverse 16 |

At most 48 hop-2 accounts reached through identity per seed, and 64 community members.

**The contract lock.** The query prints a version, and the model file stores it with a
fingerprint of the feature layout (the community block and the label inputs included) and
the subgraph settings it was trained on (the caps, `community_parties`, `hub_threshold` and
`lookback_days`). Scoring refuses a model built for another version, layout or settings:
with another lookback, features such as `in_365d` or `gap_ms` would mean something else.

### Response format

The reference graph of the tests produces exactly this, and `tigergraph/queries.py` parses
it.

**Request.** `seed_ids` (distinct accounts) share one `as_of_ms`, t: callers group seeds by
date. The caps, `community_parties`, `hub_threshold` and `lookback_days` come from config
under the names of the fields below. There is no phase: every customer is visible.

**Rows.** A refused request prints one row, `{"status": reason}`: `invalid_parameters`,
`duplicate_seed`, or `unknown_seed` and `invisible_seed` (not opened by t) with the offending
`seeds`. Otherwise the first row is `{"status": "ok", "version", "as_of_ms"}`, then one row
per seed in request order with `seed`, the node lists `accounts`, `parties`, `devices`,
`ips`, the object `community`, and the edge lists `pays`, `owned_by`, `enrolled_on_device`,
`enrolled_on_ip`, `sent_from_device`, `sent_from_ip`, each edge from an account. Lists are
unordered; a node appears once, the seed is among the accounts, and both ends of every edge
are nodes of the same subgraph. Durations are milliseconds and never negative; one whose
presence flag is false is 0. Python adds the reverse edges.

**Visibility.** A payment is visible when `t - lookback_days <= event_ts_ms < t`; a link
when `valid_from_ms <= t` and `valid_to_ms` is 0 or after `t - lookback_days`. A device or
IP is observed on the visible payments an account sent.

**Expansion.** Every cap keeps the newest first, ties broken by id.

- The seeds, then each counterparty of a seed's payments that is not a hub (hop 1), expand:
  their sent and received payments (`seed_payments` per direction for a seed,
  `other_payments` otherwise), one edge each, whose other account becomes a node; their
  owners (their `owners` latest ownership links); their owners' links to devices and IPs
  (`enrolled_devices`, `enrolled_ips` by start, one edge per link); and the devices and IPs
  observed on their own payments (`sent_devices`, `sent_ips` by last use, one edge per
  node).
- Each owner of a seed adds its `siblings` newest accounts by ownership start. Every account
  of the subgraph that a seed's owner owns carries its `owned_by` edge, however it was
  reached, so the label-input variant can always tell it apart.
- Each device or IP a seed reached that is neither a hub nor observed-only adds other
  customers' deposit accounts (owned by a party that is not an owner of the seed): those
  whose owner links to it, one candidate per link, and those observed sending from it, one
  per account. Per seed, candidates are taken newest first (link start or last use), ties by
  account, node and relation: at most `reverse_accounts` per node and relation, and at most
  `identity_accounts` distinct accounts.
- The community of each seed is built as "The community of a seed" says, with at most
  `community_parties` members.

**Node fields.**

| List | Fields |
|---|---|
| `accounts` | `id`, `account_type`, `is_external`, `hub`; `in_7d`, `in_30d`, `in_365d`, `out_7d`, `out_30d`, `out_365d`; `amount_in_7d`, `amount_in_30d`, `amount_out_7d`, `amount_out_30d`; `counterparties_in_30d`, `counterparties_in_365d`, `counterparties_out_30d`, `counterparties_out_365d`; `new_payers_7d`, `new_payers_30d`, `new_payees_7d`, `new_payees_30d`; `burst_in_1h`, `burst_in_24h`, `burst_out_1h`, `burst_out_24h`, `burst_new_payers_24h`; `passed_on_1h`, `passed_on_24h`; `shortest_in_gap_present`, `shortest_in_gap_ms`; `last_in_present`, `since_last_in_ms`, `last_out_present`, `since_last_out_ms`; `device_co_users`, `ip_co_users`, `new_device_co_users_30d`, `new_ip_co_users_30d` |
| `parties` | `id`, `accounts`, `devices_now`, `ips_now`, `new_devices_7d`, `new_devices_30d`, `new_ips_7d`, `new_ips_30d` |
| `devices`, `ips` | `id`, `parties_now`, `parties_lookback`, `new_parties_7d`, `new_parties_30d`, `accounts`, `payments_7d`, `payments_30d`, `senders_30d`, `observed_only`, `hub` |
| `community` (one object) | `parties`, `capped`, `nodes`, `links`, `new_links_30d`, `accounts`, `internal_payments_30d`, `internal_amount_30d`, `external_payments_30d`, `density`, `isolation` |

**Edge fields.**

| Lists | Fields |
|---|---|
| `pays` | `id` (the payment), `from_id` (sender), `to_id` (receiver), `amount`, `rail`, `age_ms`; `prior_count`, `first_in_pair`, `gap_present`, `gap_ms`, `pair_age_ms`; `next_outflow_present`, `next_outflow_ms`; `sender_gap_present`, `sender_gap_ms`, `receiver_gap_present`, `receiver_gap_ms` |
| `owned_by` | `from_id`, `to_id` |
| `enrolled_on_device`, `enrolled_on_ip` | `from_id`, `to_id`; `since_start_ms`, `before_lookback`; `active`; `since_end_ms` (0 while active); `held_ms` |
| `sent_from_device`, `sent_from_ip` | `from_id`, `to_id`, `since_first_ms`, `since_last_ms`, `uses` |

The query reads no supervision and no answer key: the label-input variant's confirmations
are joined in Python (see "The model").

## The model

- Heterogeneous temporal graph attention: two `HeteroConv` layers of `TransformerConv` per
  relation, taking the edge features (Fourier encodings included) as edge attributes, summed
  across relations. `owned_by` and its reverse have no edge features, so their attention
  reads the nodes alone.
- *Defaults*: hidden size 64, 4 heads, dropout 0.2.
- The seed's final representation, joined with its community block (11 numbers), goes
  through a small head (to 32, then to 1) that gives one score. Without the community block
  (`community` off, for the ablation) the head reads the representation alone.
- No per-account embedding and no memory, so any account, including a new one, can be
  scored.

**Label-input variant.** The same model also sees, for each account node, whether it was a
confirmed mule by t, with the Fourier encoding of the time since its confirmation; every
relation has its own attention, so a confirmation counts according to how the account is
linked. The seed and every account of its owner always read as unknown: a bank that had
confirmed the seed's own customer would have closed it. The confirmations are read with
`read_supervision` (with R as cutoff for training and validation, and the day after the last
date scored for test and scoring), kept by `revealed_share`, and joined in training by
account id under the rule of "Time"; confirmed accounts of every split count, as a bank
knows all its confirmed mules.
Masking (*default*: each confirmed neighbour hidden with probability 0.5 during training,
never at evaluation) happens on the GPU, so cached subgraphs are reused across epochs and
serve both variants.

## Training

- **Loss.** nnPU (kept from the old code), with the class prior chosen on validation from
  {0.001, 0.003, 0.01} (*default* grid) and its sensitivity reported. The positive risk is
  weighted by 1 minus the prior (*default* `pu_weighting` balanced), which is imbalanced nnPU
  with a balanced target (Su, Chen and Xu, 2021). The textbook weight, the prior itself,
  stays a setting (`pu_weighting` prior): with it, training collapses to a constant scorer
  (on the reference world the loss settles at exactly the prior by the fourth epoch, and the
  earlier runs on NVIDIA measured a PR AUC of 0.001). The negative-risk correction keeps the
  prior either way.
- **Weights.** Before training, each label's weights are divided by their mean over all
  the training batches, once, because the loss divides each label's weighted sum by its
  count in the batch. Each mule then keeps its equal share however its dates fall into
  batches.
- **Selection.** The epoch and prior are chosen by recall of revealed validation mules
  within the top 1% and 5% of validation accounts, per weekly date, averaged over dates.
  Only revealed labels are used, as in a bank. A later epoch or prior must be strictly
  better.
- **Seeds.** Five seeds per world (*default*), 0 to 4, vary the model only: the initial
  weights, dropout, the label masks and the batch order. The data seed (the account split,
  the revealed share and every example draw) stays fixed, since the three worlds already
  vary the data. Results are reported per seed, with their mean and range; `score` uses the
  seed with the best validation score.
- *Defaults*: Adam, learning rate 0.001, weight decay 0.0001, 128 examples per step, at most
  30 epochs, patience 5.

## Scaling

**Sampling and batching.** The query's caps are the sampler: deterministic, newest first,
so a seed's subgraph is the same in every run and its cost is bounded whatever its degree.
Hubs and observed-only nodes are leaves, which keeps billers, ledgers, carrier NAT and
terminals from pulling in thousands of strangers. Seeds are fetched by date, 32 to a request
(*default*, a transport setting, since it never changes a result); training steps take 128
examples.

**Extraction and caching.** TigerGraph does the feature work as of t, so only the subgraphs
leave the database. The cache keeps each request's response, as printed, in one compressed
file per (world, as-of date, request), written once and never rewritten, with an index of
the seeds each holds. The key is the world's identity, the query version and the subgraph
settings. The world's identity is a hash of what `read_world` returns: the count of every
vertex and edge type, the earliest `joined_ms`, the earliest and latest `event_ts_ms`, and
integer checksums of the payments (sums of their seconds and of their amounts in cents per
payment type). A regenerated world reloaded under the same name has another identity, so its
stale subgraphs are never read. Subgraphs no longer depend on the phase, so one cache serves
training, validation, test, both variants, the controls and every experiment. `prepare` also
checks the lookback against the data start the identity gives.

**Building.** Responses are parsed into one component per seed, and durations are
Fourier-encoded on the GPU. Built batches stay on the training device for every epoch when
they fit (*default*), otherwise in pinned host memory, copied each step.

**`mule bench`.** On the loaded world, for a seeded sample of accounts on a few dates
(*default* 256 seeds on each of 4 dates) and at 8, 32 and 128 seeds per request, it measures
the time and response size of each request, the nodes and edges per subgraph by type
(median, 90th percentile and largest), the community sizes, how many nodes are hubs or
observed-only and the distinct customers per counterparty, device and IP, the parse and build
time per training batch, the forward and backward time and peak device memory per training
step, and the cache bytes per seed. It writes one table per world, with the world's size, to
`results/bench/`. It is run on worlds of 30,000, 100,000 and 300,000 people, and the tables
side by side show how cost grows.

**Recommended training architecture.** TigerGraph computes the features and samples the
subgraphs as of each date in one query per date and batch of seeds; requests for different
dates are independent and can run in parallel. The responses are cached once per world and
reused by every epoch, prior, seed, variant and control. Python builds one component per
seed, and the GPU encodes time, masks labels and trains on 128 examples a step. The bench
decides the request size and whether the built batches fit on the device; this paragraph
is updated from its tables.

## Comparisons

Every ranker is trained on the same examples with the same weights and scored on the same
dates and groups. The controls read the features of the fetched subgraphs, so they see
exactly what TigerGraph gives the GNN.

| Ranker | What it is |
|---|---|
| Rule | Ranks by the count of confirmed accounts linked to the seed at t (payment counterparties, shared devices and IPs; not its own customer), ties by the newest confirmation |
| Trees, own behaviour | Gradient-boosted trees (scikit-learn's `HistGradientBoostingClassifier`) on the seed's own account row, without its `hub` and co-user fields |
| Trees, graph features | The same on the seed's full account row, its owner's party row and its community block |
| Trees, graph features and labels | The same plus the counts of confirmed accounts linked to the seed, by how they are linked |
| GNN | The behavioural GNN, without label inputs |
| GNN with labels | The label-input variant |

The trees learn revealed against unlabelled with the examples' weights, rescaled so the two
labels weigh the same in total, as balanced nnPU does. When labels are selected completely
at random, a classifier of labelled against unlabelled ranks accounts as one of positive
against negative would (Elkan and Noto, 2008).

The claims we make:

- what graph learning adds: the GNN against trees on own behaviour;
- what message passing adds beyond graph features made by hand: the GNN against trees with
  graph features;
- what the GNN adds beyond guilt by association: the GNN with labels against trees with
  graph features and labels, and the rule.

## Experiments

- **Community ablation.** The GNN, and the trees with graph features, trained with and
  without the community block, on the same examples and the five seeds, compared at test
  seed by seed.
- **Sparse labels.** `prepare`, `train` and `evaluate` run at `revealed_share` 1, 0.5, 0.25
  and 0.1, for the GNN, the GNN with labels and the trees, still positive-unlabelled, and
  reported at test against the full answer key.
- **Sensitivity world.** The results on the world whose rings share devices and IPs at 10%.

## Evaluation

- **Scoring.** On every weekly date in the test window, every true mule of the test split
  active that date is scored, plus the weighted sample of the other accounts, so the metrics
  estimate the full ranking without scoring every account every week.
- **Per-date alert budgets** (top 1% and 5%): precision and recall of the hidden mules active
  that date.
- **Account level.** The first alert date of each mule among its active dates: time to
  detection (first alert minus onset) and lead time (discovery minus first alert). An alert
  counts only before discovery.
- **Cost.** Distinct accounts alerted per week.
- **Groups.** Customers never trained on; accounts opened after the training window;
  accounts with no payments in the lookback; accounts linked to a confirmed mule at t (by
  link type and direction) and not linked; mules never discovered; mules by `mule_flow`.
- **Uncertainty.** A ring-level bootstrap within each world, and the spread across the five
  seeds and across the three worlds.

## Scoring new accounts

`mule score --date D` scores any account at any date with the same query and the saved
model, including accounts opened after training. An account with no payments is scored from
its links and its community.

## Package layout and commands

| Package | Holds |
|---|---|
| `config.py`, `paths.py` | frozen settings; output locations |
| `tigergraph/` | connection, client with retries (kept), executor, query installer, query names and parsers |
| `data/` | the world's identity, the windows and the account split, the population per date, supervision with the revealed share, the examples, the maturity gap; the answer-key reader, which only evaluation imports |
| `subgraph/` | the query request, the sharded disk cache, response to `HeteroData` |
| `model/` | Fourier time encoding, the heterogeneous attention model, nnPU loss (kept) |
| `training/` | trainer, selection, the runs over priors and seeds |
| `controls/` | the rule, the gradient-boosted trees, and the features they read from fetched subgraphs |
| `evaluation/` | weekly scoring, groups, metrics and the ring bootstrap (kept), the experiments |
| `bench.py` | the cost measurements |
| `cli.py` | the commands below |

| Command | What it does |
|---|---|
| `mule install` | installs the queries |
| `mule prepare` | reads the world's identity and checks the lookback against its start, reads the populations and supervision, writes every window's examples, and reports their counts and the maturity estimate |
| `mule fetch` | fetches and caches the subgraph of every example |
| `mule train` | trains the GNN (both variants, five seeds) and the rankers it is compared with |
| `mule evaluate` | scores every weekly test date and reports the metrics |
| `mule experiment` | runs the community ablation or the sparse-label series |
| `mule score` | scores accounts at a date with the saved model |
| `mule bench` | measures cost on the loaded world |

GSQL: `gsql/schema/` (schema and loading job), `gsql/queries/` (`fetch_subgraphs`,
`list_population`, `read_supervision`, `read_world`), `gsql/evaluation/`
(`read_answer_key`). Results are tables and files, no charts.

## Questions raised while building

| Question | Settled |
|---|---|
| The audit sample was unused and refused more than one split | Removed; the examples draw the per-date samples |
| External accounts have `is_mule` -1 | The population holds only accounts a customer owns, never external ones |
| "Active that date" was undefined for test mules | Defined under "Time", the same for training, validation and test |
| Where the answer-key reader lives | `data/answer_key.py`, imported only by evaluation; an import contract keeps training and the rest of data from it |
| Time boundaries | Stated under "Time": events before t, links from their start with half-open tenures, confirmations at or before t |
| How confirmations reach the label-input variant | From `read_supervision`, joined in training; the query reads no supervision |
| No query listed the ids for `write_split`, and chunked writes could mis-assign joint accounts | The split is a hash in Python and is never written |
| The hub threshold had no measured value | 200 stays, above any ring PhantomLedger makes; observed-only nodes are always leaves; the bench reports the distributions |
| The maturity estimate's data | Right-truncated, estimated with Lynden-Bell's estimator from the training split's revealed mules, floor 60 days |
| The cache key had no load identity | The world's identity from `read_world` |
| What the five seeds vary | The model only |
| An account of the seed's owner reached only at hop 2 had no ownership edge | The query adds the ownership edge of every subgraph account a seed's owner owns |
| The controls' weights | The scaled weights (each label's mean 1 over all training batches, once, not per batch), with the two labels rescaled to equal totals |
| The category values | Listed under "Features" |
| Nodes are not merged across the seeds of a batch | Stated under "The subgraph query" |
| torch_geometric prints thousands of DeprecationWarnings per test run | Not a design matter: filtering that one warning in the test settings waits for the owner's agreement |
| Joint accounts | PhantomLedger writes one owner per account, and the design assumes at most one |

## Limitations we state up front

- Synthetic data, however well cited, is not a bank's data.
- Most mule research is British; PhantomLedger simulates a US bank.
- Discovery is not completely at random, so nnPU's assumption does not hold exactly.
- The sharing rates of ring devices and IPs are declared, not measured; the sensitivity world
  shows how much the result depends on them.
- A mule's onset and last activity come from PhantomLedger's ground truth, so they are
  sharper than an investigation's reconstruction.
- Label noise: organizers' accounts that ring money reached, and a mule's own second
  account, are positives; recruits whose payments the fraud budget cut hold ring devices but
  are not mules; until PhantomLedger adds scam payees, account-takeover drops and impostor
  payees receive fraud money unlabelled.
- Traits of the generator that a model could learn in place of behaviour, until
  PhantomLedger changes them: camouflage payments all year, card cash-outs on the `unknown`
  rail, organizers outside ordinary sharing groups, ring links starting on staggered days
  where ordinary groups start on one day, and an offboarded customer's links all ending on
  its discovery day.
- Each world has few test mules (roughly a quarter of the year's mules, in a quarter of the
  accounts), so the bootstrap intervals matter.
