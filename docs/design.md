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

- Every feature is behaviour of the data or graph structure. The ages from a first sighting
  (account, party, token, device, IP and address) are gone, and so is the age of a link that
  began when its customer joined: `opened_ms` and `joined_ms` only decide what exists on a
  date and which link starts are shown. `account_type` is the one attribute kept that is not
  behaviour, until the owner decides.
- No feature depends on when the data starts or when a customer joined: a link's start is
  shown only when the link was added after its customer joined and within the lookback, and
  links that ended before the lookback are not visible.
- Short-term behaviour is added: per-payment gaps since the sender's previous payment and the
  receiver's previous receipt, the most payments in any hour and any day, the shortest gap
  between inflows, and money passed on within an hour.
- New payees are added beside new payers, both counted as distinct accounts, with a year's
  distinct counterparties to measure growth.
- Connectivity change is added: each customer's new device and IP links, each device's and
  IP's new users, and each account's device and IP co-users and how many of them are new.
- A community block per seed is added: a bounded device-and-IP community computed in the
  query as of t.
- Token and Address leave the model; Device and IP stay. The `at_address` relation goes.
- Closure is no longer a feature: the account's closed flag, the party's open and closed
  account counts and the ownership edge's end are gone, and a link end caused by its customer
  leaving the bank is not shown. The `owned_by` edge has no features.
- Hubs (billers, ledgers, very busy devices and IPs) and crowds (large devices and IPs no
  customer is enrolled on, such as carrier NAT and public terminals) are leaves whose rows
  hold only their flags; small devices and IPs no customer is enrolled on are expanded like
  any other, so a ring device no member enrolled still links the ring. A new query,
  `read_leaves`, finds them once per date, and another, `read_sharing`, counts every
  customer's co-users once per date.
- Amounts are integer cents, every time boundary is strictly before t, and the category
  values are listed.

Splits and labels:

- A split by account replaces the customer hold-out: each account belongs to one of
  training, validation and test (50%, 25%, 25% by a seeded hash), and every account is
  visible as a neighbour in every phase. The split is not written on the graph, so
  `write_split`, the `split` attributes and `held_out_visible` go.
- A mule is positive while it is active: from its first ring payment to a week after its
  last, not until its discovery; `mule_last_ms` moves from the answer key to supervision.
- Labels are read after the activity they label, on a rolling origin: a selection run trains
  on labels read when validation begins and chooses the prior and the epochs; the final run
  trains on labels read when the test begins, with those choices.
- Validation's dates, like training's, end the maturity gap before their labels are read,
  and selection pools recall over the dates.
- A revealed mule is never an unlabelled example; its dates outside its activity are
  dropped, and it stays confirmed for its neighbours on every later date.
- Validation and test rank a true mule only on the dates it is active; test also reports
  early warning in the four weeks before a mule's onset.
- The maturity gap is estimated from training's revealed mules (right-truncated), never
  below 60 days.
- A sparse-label series keeps a seeded share of the revealed mules (100%, 50%, 25% and
  10%) and holds everything else fixed.
- The population of a date is the open accounts a customer owns; external and ledger
  accounts are never examples. Every draw takes the accounts with the lowest seeded hashes,
  so draws nest when what they draw from changes.

Model, training and evaluation:

- nnPU weights the positive risk by 1 minus the prior by default (balanced); the textbook
  weight stays a setting. The prior grid is scaled from the revealed active share, and the
  correction term counts positives per date, as the unlabelled draw does.
- The five seeds vary the model only.
- The label-input confirmations come from supervision, joined in training; the query adds
  the ownership edge of every account of a seed's owner it reaches.
- New comparisons: trees with hand-made graph features, and ablations of the GNN without the
  community block, without devices and IPs, with the seed alone, without link ends, without
  year-long counts and without `account_type`.
- New groups: test mules of rings that gave training no positive, and test accounts that are
  in no training subgraph, in place of the never-seen customers.
- Test dates score a sample of 5,000 other accounts (validation 2,000).
- The subgraph cache is keyed by the loaded world's identity, holds one file per request,
  and no longer depends on the phase.
- A `parity` command checks every query on a live TigerGraph against the reference graph,
  and no query runs for real until its version has passed. A `bench` command measures cost
  at growing world sizes, and a recommended training architecture is written down.

## What we are testing

A temporal heterogeneous graph neural network finds money-mule accounts that a bank has not
found yet, from what accounts do over time and how they are connected: payments, their
speed and order, and devices and IPs shared with other customers. It must work on accounts
it never trained on and on rings none of whose members it trained on, keep working when
only a small share of mules is labelled, and beat models without graph context given the
same information. How often it flags a mule before its first ring payment, from the devices
and IPs it shares with active mules, is reported apart, as early warning.

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
  monitoring from the first laundering payment, and tracing downstream along the money. Only
  mules are discovered. On the day a mule is found its customer is offboarded, so nothing it
  would have done later exists.
- Ordinary customers also open and close accounts and leave the bank all year, so neither a
  late opening nor a closure marks a mule.
- Rings share devices and IPs while they operate; ordinary sharing happens inside real
  households.

**Timeline (default).** One world covers 2023-01-01 to 2024-12-31. 2023 is a warm-up with
no laundering: it is loaded so every example has a year of history, and it is never scored.

**Worlds.** Development uses one 30,000-person world. Reported numbers come from three
independent 300,000-person worlds (different seeds), plus one sensitivity world in which
ring members share devices and IPs at 10% instead of the declared 70% and 60%. The bench
also runs on 100,000-person worlds, and the parity check on a 5,000-person world.

**What PhantomLedger must still provide.** Its money flow and behaviour labels are done.
Until discovery lands no mule has a discovery date, so training cannot run; reported numbers
wait for every item:

- Discovery and offboarding: `discovered_ms` set to the midnight that ends the discovery's
  UTC day, `discovery_channel` one of `victim_report`, `monitoring` and `trace`, only for
  accounts with `is_mule` 1, the whole customer closed that day with every link ending at
  the same instant as its accounts close, and no later activity, so
  `mule_last_ms <= discovered_ms`.
- Ordinary opening, closing and leaving all year, shipped with discovery, with the same
  link ends when a customer leaves; mule accounts that are opened for the ring; ring bursts
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
`Party_Uses_IP`. `joined_ms`, `opened_ms` and `closed_ms` are read only to decide what
exists on a date, when a customer left, and which link starts are shown; none is a feature.
Token, Address, `Account_Uses_Device` (a copy of the owner's device links), the token links
and the token roles of payments are loaded but not read. A token is one opaque handle per
account, never shared or rebound, so it carries no graph signal; Zelle shows as the payment
rail `zelle`. Device and IP sharing is the identity signal for this proof of concept.
Addresses are left out: rings share none, and PhantomLedger's address tokens today collide
between strangers by chance. They can return once households exist, if an ablation shows
that telling households from rings needs them.

**Who may read what.**

- *Supervision* (`discovered_ms`, `discovery_channel`, `mule_onset_ms`, `mule_last_ms`) is
  what a bank knows after an investigation: when the account was found and the dates of its
  suspicious activity, which an investigation reads from the account's history. Training
  and validation read it only through `read_supervision`, which returns the mules
  discovered by a date.
- *Answer key* (`is_mule`, ring, layer, flow) is read only by evaluation, through
  `read_answer_key`. That query also returns every account's `mule_onset_ms`,
  `mule_last_ms` and `discovered_ms`, discovered or not, because evaluation needs the
  active dates of the test mules nobody found.
- `read_supervision` and `read_answer_key` are the only queries that read supervision
  fields, and `read_answer_key` the only one that reads the answer key.
- Neither is ever a feature. The one exception is the label-input variant, which reads
  whether a neighbour was confirmed by t (its `discovered_ms`). `is_mule` defaults to -1, so
  an account with no loaded truth is unknown, never normal.

## Time

Every example is an (account, as-of time t) pair, with t a Monday at 00:00 UTC. Everything
the model sees existed strictly before t, within a lookback of 365 days (*default*):

- **Events** count when `t - lookback <= event_ts_ms < t`. Payments in the same second have
  no order: none is another's predecessor, and all of them count in a window.
- **Windows.** "In the last N days" means `t - N days <= x < t`, for an event's time and for
  a link's start alike. "Within w of a time x" means from x, inclusive, to `x + w`,
  exclusive. A link is "active at some time in the last 30 days" when
  `[valid_from_ms, valid_to_ms)` meets `[t - 30 days, t)`.
- **Links** are visible when `valid_from_ms < t` and they had not ended before the lookback
  began (`valid_to_ms` is 0 or after `t - lookback`). A link is active at t when
  `valid_to_ms` is 0 or after t: PhantomLedger's tenures are half-open,
  `[valid_from_ms, valid_to_ms)`. A link that ended by t stays visible as a past fact with
  the time since it ended, because ring links are short and dropping them would erase the
  evidence. An end after t is invisible.
- **A customer's exit.** A customer has left by t when every account it owns has
  `0 < closed_ms <= t`; it left at the latest of those times. An end of one of its links at
  or after the time it left comes from the bank's offboarding or the customer's departure,
  not from its use of a device, so it is not shown: such a link reads as never ended, on
  every later date. Ends before it left stay.
- **Link starts.** A link's start is *known* when the link began after its customer joined
  (`valid_from_ms > joined_ms`) and after the lookback start. A link that began when its
  customer joined is its enrolment, which dates the customer, not its behaviour; one that
  began before the lookback is older than anything else the model sees. Neither shows its
  start: durations from it are absent, and it never counts as new.
- **Existence.** An account exists at t when `opened_ms < t`; it is open at t when
  `closed_ms` is also 0 or after t.
- **Confirmations** (label-input variant only) count when `discovered_ms <= t`, on every
  later date. Since `discovered_ms` is the midnight that ends the day of discovery, a mule
  found on a Sunday reads as confirmed on the Monday. The time since a confirmation is cut
  at the lookback, like every duration from a start.
- **Labels read at a date L** are the mules with `0 < discovered_ms <= L`, the mules a bank
  has found by then.
- **Activity.** A mule is active at t when `mule_onset_ms < t <= mule_last_ms + 7 days`
  (*default* 7) and it was not discovered by t (`discovered_ms` is 0 or after t). Since
  events must be before t, its first ring payment is visible on every date it is active.
  The week after its last ring payment gives every mule at least one active Monday, the
  first after its onset, unless it was discovered before that Monday.

**No feature depends on when the data starts or when a customer joined.** Every duration is
relative: to t, or between two events. One that runs from a start (a pair's first payment, a
node's first observed use, a confirmation) never reaches back past the lookback start, and a
link start is shown only when it is known. The warm-up is as long as the lookback, so on
every as-of date the lookback lies inside the data (`prepare` checks it). PhantomLedger
clamps customers who joined before the data to its start, and their links start on that day;
such links read like links older than the lookback, and a customer who joined last month
reads like one who joined years ago, until its behaviour differs. No absolute time reaches
the model, so a model trained in 2024 scores 2027 the same way.

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
t). TigerGraph computes every feature as of t from what is visible; Python only joins the
per-date tables to the rows by id and encodes:

- categories one-hot over their values: `account_type` is `deposit`, `credit`,
  `brokerage`, `gl` or `unknown`; `rail` is `ach`, `card`, `cash`, `check`, `internal`,
  `unknown` or `zelle` (a `Zelle_Transfer`'s rail is `zelle`);
- flags 0 or 1;
- counts as log1p, and amounts, printed as integer cents, as log1p of dollars;
- shares (numbers in [0, 1]) as they are;
- durations as a presence bit and the 64 Fourier numbers.

**Left out, and why.**

- First-seen ages of accounts, customers, tokens, devices, IPs and addresses, and the age of
  a link that began when its customer joined. PhantomLedger clamps them to the data start,
  so they measure the window, not the customer; and the owner rules out an account's age
  even where it would be real.
- Closure: the account's closed flag, the customer's open and closed accounts, the
  ownership edge's end, and link ends caused by a customer's exit (see "Time"). A closure is
  the bank's action, not mule behaviour; once PhantomLedger offboards found mules it would
  mark caught neighbours.
- The clock of day and the second within a minute. PhantomLedger schedules many rows on
  whole minutes and at fixed UTC hours, choices of the generator in one naive time zone.
- Tokens and addresses, as said under "The graph".

The other ends of identity links stay (since a link ended, how long it was held): a
customer stopping use of a device is behaviour, ordinary devices turn over in
PhantomLedger, and ring links end soon after a ring's payments. A customer that left still
goes silent: its payments stop. That cannot be hidden, and until ordinary customers also
leave in PhantomLedger it is rarer for them than for caught mules; it is listed under the
limitations, and evaluation measures it (see "Evaluation").

`account_type` is not behaviour: it is what kind of product the account is. It stays until
the owner decides, because it shapes which payments an account can make, and an ablation
measures what the model takes from it.

### The feature table

"Last N days" follows "Time". Every count is over visible payments and links. *Hub*,
*crowd* and *leaf* are defined under "Hubs, crowds and leaves" below; *shares*, *co-user*
and a *known* start under "Shared devices and IPs" below and "Time" above. The row of a hub
account that is not a seed, and a leaf's row, hold only their ids and flags (`account_type`
and `is_external` included): every other field is 0 and every presence flag false.

| Where | Feature | Temporal | Kind | How TigerGraph computes it, as of t |
|---|---|---|---|---|
| Account | `account_type` (5 one-hot) | no | attribute | the account's attribute, kept until the owner decides |
| Account | `is_external` | no | structure | the account's attribute |
| Account | `hub` | no | structure | the account is a counterparty hub (from `read_leaves`) |
| Account | `in_7d`, `in_30d`, `in_365d` | yes | behaviour | payments received in the last 7, 30 and 365 days |
| Account | `out_7d`, `out_30d`, `out_365d` | yes | behaviour | payments sent in the last 7, 30 and 365 days |
| Account | `amount_in_7d`, `amount_in_30d`, `amount_out_7d`, `amount_out_30d` | yes | behaviour | sum in cents of the amounts received and sent in the last 7 and 30 days |
| Account | `counterparties_in_30d`, `counterparties_in_365d`, `counterparties_out_30d`, `counterparties_out_365d` | yes | structure | distinct accounts it received from and sent to in the last 30 and 365 days |
| Account | `new_payers_7d`, `new_payers_30d` | yes | structure change | distinct senders whose earliest payment to it in the lookback is in the last 7 or 30 days |
| Account | `new_payees_7d`, `new_payees_30d` | yes | structure change | distinct receivers whose earliest payment from it in the lookback is in the last 7 or 30 days |
| Account | `burst_in_1h`, `burst_in_24h` | yes | behaviour | the most payments received within 1 hour or 24 hours of x, x the time of one of them, over its newest `recent_payments` received in the last 30 days |
| Account | `burst_out_1h`, `burst_out_24h` | yes | behaviour | the same over payments sent |
| Account | `burst_new_payers_24h` | yes | behaviour | the most new payers of the last 30 days whose first payment falls within 24 hours of one of them, over the newest `recent_payments` of those first payments |
| Account | `passed_on_1h`, `passed_on_24h` | yes | behaviour | sum in cents of its newest `recent_payments` inflows of the last 30 days that it followed, within 1 hour or 24 hours, by one of its newest `recent_payments` payments sent |
| Account | `shortest_in_gap_ms` (present with at least two inflows) | yes | behaviour | the shortest time between consecutive payments among its newest `recent_payments` received in the last 30 days |
| Account | `since_last_in_ms`, `since_last_out_ms` (each with presence) | yes | behaviour | t minus its latest payment received, and sent, in the lookback |
| Account | `device_co_users`, `ip_co_users` | no | structure | its owner's co-users on devices, and on IPs (from `read_sharing`, joined by owner) |
| Account | `new_device_co_users_30d`, `new_ip_co_users_30d` | yes | structure change | those co-users whose sharing with its owner began at a known start in the last 30 days |
| Party | `accounts` | no | structure | accounts it owns that exist at t, open or closed |
| Party | `devices_now`, `ips_now` | no | structure | devices and IPs it has a link to that is active at t |
| Party | `new_devices_7d`, `new_devices_30d`, `new_ips_7d`, `new_ips_30d` | yes | structure change | its device and IP links whose known start is in the last 7 or 30 days |
| Device, IP | `leaf`, `observed_only` | no | structure | the node is a leaf; no customer has a visible link to it |
| Device, IP | `parties_now` | no | structure | customers with a link to it active at t |
| Device, IP | `parties_lookback` | no | structure | customers with a visible link to it |
| Device, IP | `new_parties_7d`, `new_parties_30d` | yes | structure change | customers whose earliest visible link to it has a known start in the last 7 or 30 days |
| Device, IP | `accounts` | no | structure | accounts that exist at t owned by its linked customers, and accounts observed sending from it in the lookback |
| Device, IP | `payments_7d`, `payments_30d` | yes | behaviour | payments observed on it in the last 7 and 30 days |
| Device, IP | `senders_30d` | yes | structure | distinct accounts observed sending from it in the last 30 days |
| `pays` edge | `amount` | no | behaviour | the payment's amount in cents |
| `pays` edge | `rail` (7 one-hot) | no | behaviour | `payment_rail`, or `zelle` |
| `pays` edge | `age_ms` | yes | behaviour | t minus the payment's time |
| `pays` edge | `prior_count`, `first_in_pair` | yes | structure | payments of the same sender to the same receiver earlier in the lookback, and whether there are none |
| `pays` edge | `gap_ms`, `pair_age_ms` (present when `prior_count` > 0) | yes | behaviour | time since the latest and the earliest of those earlier payments |
| `pays` edge | `next_outflow_ms` (with presence) | yes | behaviour | time to the receiver's earliest payment sent at or after it, before t; absent when the receiver is a hub |
| `pays` edge | `sender_gap_ms` (with presence) | yes | behaviour | time since the sender's latest earlier payment sent to anyone in the lookback; absent when the sender is a hub |
| `pays` edge | `receiver_gap_ms` (with presence) | yes | behaviour | time since the receiver's latest earlier payment received from anyone in the lookback; absent when the receiver is a hub |
| `enrolled_on_device`, `enrolled_on_ip` edges | `start_known`, `since_start_ms`, `held_ms` (both present with a known start) | yes | structure change (relationship age) | whether the link's start is known; t minus it; from it to the link's end, or to t while active |
| `enrolled_on_device`, `enrolled_on_ip` edges | `active`, `since_end_ms` (present when not active) | yes | structure change | whether the link is active at t, and t minus its end |
| `owned_by` edge | none | no | structure | the edge itself |
| `sent_from_device`, `sent_from_ip` edges | `uses` | yes | behaviour | the account's payments observed on the node in the lookback |
| `sent_from_device`, `sent_from_ip` edges | `since_first_ms`, `since_last_ms` | yes | behaviour | t minus the first and the last of them |
| Community of the seed | `parties`, `capped` | no | structure | members of the seed's community, and whether the cap left a candidate out |
| Community of the seed | `nodes`, `links` | no | structure | devices and IPs shared by at least two members, and member-node sharing pairs on them |
| Community of the seed | `new_links_30d` | yes | structure change | those pairs whose sharing start is known and in the last 30 days |
| Community of the seed | `accounts` | no | structure | accounts the members own that exist at t |
| Community of the seed | `internal_payments_30d`, `internal_amount_30d` | yes | behaviour | payments in the last 30 days between accounts of two different members, and their sum in cents |
| Community of the seed | `external_payments_30d` | yes | behaviour | the members' payments in the last 30 days whose other account no member owns |
| Community of the seed | `density` | no | structure | `links / (parties * nodes)`, 0 with no nodes |
| Community of the seed | `isolation` | yes | structure | `internal / (internal + external)` payments of the last 30 days, 0 with none |
| Account (label-input variant) | `confirmed`, time since confirmation | yes | label input | joined in training from supervision (see "The model") |

Every account in a seed's subgraph that is not a hub carries the account features, so a
hop-2 neighbour's behaviour and sharing reach the seed. The per-payment gaps, bursts,
pass-on within an hour, new payees and co-users are what a pass-through mule does in
minutes and hours: by value, PhantomLedger's mules pass on 28% of a credit within 15
minutes, 53% within an hour and 86% within a day, and nearly every victim is a first-time
payer. `recent_payments` (*default* 256 per direction) bounds the work of the time-ordered
features; a mule rarely makes that many payments in 30 days.

### Hubs, crowds and leaves

- An account is a **counterparty hub** when more than `hub_threshold` distinct customers own
  the other accounts of its payments in the lookback (billers, merchants, the bank's
  ledgers); accounts with no owner are not counted. One with no more payments than that in
  all time cannot be one and is not tested.
- A device or IP is **observed-only** when it appears on a visible payment but no customer
  has a visible link to it.
- A device or IP that is not observed-only is a **hub** when more than `hub_threshold`
  distinct customers have a visible link to it or own accounts observed sending from it in
  the lookback. An observed-only one is a **crowd** when more than `observed_threshold`
  distinct customers own accounts observed sending from it in the lookback: carrier NAT
  lines and public terminals, which link strangers.
- Hubs and crowds are **leaves**: each stays in the subgraph with a row of its flags only,
  is not expanded, and counts for neither co-users nor communities. A seed is the one
  exception: it is always expanded and its own row is always full, even when it is a hub,
  because it is the account being scored.

`read_leaves` finds every hub and crowd once per date (see "Date tables"). `hub_threshold`
is 200 (*default*), above the largest ring PhantomLedger can make (about 105 device
holders) and far above ordinary groups (2 to 6). `observed_threshold` is 8 (*default*),
below carrier NAT (about 96 people a line) and the average public terminal (24 people), and
above an observed-only ring device or IP: one that no holder has on file has almost always
two or three holders, since each holder lacks it on file with chance 0.28 for a device and
0.39 for an IP. A small observed-only node is expanded like any other, so a ring device no
member enrolled still links the ring. `mule bench` reports the exact distribution of
distinct customers per account, device and IP, by enrolled and observed-only, and the share
of positive examples that are hubs, which must be 0, so the owner can set both thresholds.

### Shared devices and IPs

A customer **shares** a device or IP at t when the node is not a leaf and the customer has a
link to it active at some time in the last 30 days, or one of its accounts sent a payment
observed on it in the last 30 days. Thirty days keeps a ring's links in view on the dates
after its last payment, since ring tenures end a few days after it.

Its **sharing start** on the node is the earliest of its links' starts and its first
observed use in the lookback; it is known unless it is a link start that is not known (see
"Time"). Two customers that share the same node are **co-users**; on that node their
sharing began at the later of their two sharing starts, and a pair that shares several
nodes of a kind began sharing at the earliest of those. A start that is not known counts as
the lookback start, so it is never new.

`device_co_users`, `ip_co_users` and their new counts follow this rule for an account's
owner, so they carry the sharing of accounts whose own devices and IPs are not in the
subgraph (hop-2 accounts and those reached through an identity node). `read_sharing`
computes them once per date for every customer (see "Date tables").

### The community of a seed

The owner asked for "clustered connections with isolated communities". A global algorithm
(TigerGraph's connected components, Louvain or label propagation) runs on the whole graph
with no as-of time, so it would carry future ring members into training, and hubs and crowds
would merge strangers. The community is therefore local, bounded and computed inside
`fetch_subgraphs` for each seed as of t:

1. The members start as the seed's owners (round 0).
2. Round 1's candidates are the customers, not yet members, that share a node with a round-0
   member; round 2's are those that share a node with a member kept in round 1. A customer
   belongs to the first round that reaches it, and there is no third round.
3. A candidate's sharing time is the latest, over the nodes it shares with members kept in
   the previous round, of the later of its sharing start and that member's.
4. Each round keeps its candidates newest sharing time first, ties by party id, until the
   community holds `community_parties` members (*default* 64). `capped` says whether a
   candidate of either round was left out.

The measures are those of the feature table. A ring holding a device and an IP gives a small,
dense community whose links began recently and whose members pay each other; an ordinary
group gives older links and little money inside. The work per seed is at most the kept
members, times the nodes each shares, times `hub_threshold`, and the measures read 30 days
of the members' payments. The block is per seed, so it is printed once per seed and goes to
the model's head (see "The model"). Whether it adds anything to what two message-passing
layers already see is measured by an ablation (see "Experiments").

## Splits and labels

**Runs and dates (defaults).** Labels are read after the activity they label, on a rolling
origin. V, the selection run's labels read date, is 00:00 UTC of 2024-06-01; R, the final
run's, is 00:00 UTC of 2024-10-01, the first day of the test. "Revealed at L" means read at L
(see "Time") and kept by `revealed_share`.

| Run | Examples of | As-of dates (Mondays) | Labels |
|---|---|---|---|
| Selection | the training split | 2024-01-01 to the maturity gap before V | mules revealed at V |
| Selection | the validation split | V to the maturity gap before R | mules revealed at R |
| Final | the training split | 2024-01-01 to the maturity gap before R | mules revealed at R |
| Final | the test split | R to 2024-12-31 | the full answer key |

With 60-day gaps that is 14 training Mondays (to 2024-04-01) and 9 validation Mondays
(2024-06-03 to 2024-07-29) for the selection run, and 31 training Mondays (to 2024-07-29)
and 13 test Mondays (2024-10-07 to 2024-12-30) for the final run. The validation window
starts in June so that about two months of Mondays remain after the gap.

The selection run is validated only on dates after its labels were read, as the final run is
tested only on dates after its own: no model is ever scored on a date earlier than the
labels it trained on. The final run reads its labels as late as a bank could before the
test, so a mule active in June but found in September is a training positive, and every
mule it labels was offboarded before any test date. The selection run chooses the prior and
the number of epochs (see "Training"); the final run is never validated.

**The maturity gap.** A run's dates end the maturity gap before their labels are read,
because a mule whose activity began just before that date has had no time to be found: in
training it would sit among the unlabelled, and in validation the late dates would hold only
the mules found fastest. The gap before a date L is the delay within which the estimate
finds 90% of mules, from onset to discovery, estimated from the training split's mules
revealed at L, never from the answer key or another split, and never less than 60 days
(*default*). Revealed mules are only those found by L, so their delays are right-truncated,
not right-censored; the estimate is the product-limit estimator for right-truncated data
(Lynden-Bell). The 60-day floor binds when discovery is fast. `prepare` reports both gaps,
and refuses when a run's dates hold no Monday or validation's dates hold fewer than
`min_validation_mules` (*default* 5) distinct revealed mules, so no run fails later for want
of them.

**By account.** Each account belongs to exactly one split: the 64-bit blake2b hash of its id,
keyed by the data seed and the word `split`, falls in the training split's first 50%, the
validation split's next 25%, or the test split's last 25% (*default* shares). No account is
an example in more than one split. The split is computed in Python and never written on the
graph, so no query writes, and one customer's accounts can fall in different splits.

Every account is visible in every run as a neighbour: its payments, its owner, its devices
and IPs, and its features. Only its labels are kept to its own split: the training split's
revealed mules are training positives, and other splits' accounts are never training
examples. A test account is scored from its relationships with accounts the model trained
on; sharing devices, IPs and other identity with several mules is the evidence the model is
meant to use, so a neighbour that was a training positive is not leakage. The model has no
per-account parameters, so having seen an account as a neighbour does not let it remember
that account.

A ring's members fall in all three splits, so most test mules belong to rings the model
trained on. Evaluation therefore reports, as groups of their own, the test mules of **new
rings** (no mule of the ring was a training positive) and the test accounts **never in
training** (in no training example's subgraph); see "Evaluation".

**Labels from mule activity.** A mule is positive while it is active (see "Time"), from
just after its first ring payment to a week after its last, not from when it was caught.
Supervision therefore holds `mule_onset_ms` and `mule_last_ms` for each discovered mule: an
investigation that catches an account reads its whole history, as a suspicious activity
report records the date range of the activity. `read_supervision` refuses a row unless
`0 < mule_onset_ms <= mule_last_ms <= discovered_ms`, because only mules are discovered, a
mule is found only for what it has done, and nothing happens after offboarding. Quiet weeks
between a mule's bursts lie inside its activity and are positive: PhantomLedger exports no
per-burst clocks, and real mule accounts are reused. The catch decides only when the account
stops existing: an account closed by t is no example on t.

**A caught mule stays a mule.** Practice works forward from the catch and dates the
activity separately: a Cifas marker is kept for six years from its filing, a bank exits the
customer, and production AML systems time their positives on the activity's start and end.
None relabels the account's earlier life, and doing so would teach the model who was caught,
not what mules do. So "permanently" means:

- A revealed mule is never an unlabelled example, on any date. It is positive on the dates
  it is active and dropped on every other date: before its onset, and between a week after
  its last activity and its discovery.
- For the label-input variant it stays confirmed for its neighbours on every later date,
  with the time since its confirmation (cut at the lookback), so the model can learn how a
  confirmation's weight fades.
- The label belongs to the account that carried the money, where the behaviour is. The
  customer's other accounts are labelled by their own behaviour; in PhantomLedger a mule's
  own second deposit account that the money reached is a mule itself.
- Evaluation removes a mule from every date's ranking once it is discovered, since a bank
  does not alert on an account it has already caught.

**Hidden labels come from discovery, not from masking.** On any date only the mules found
by the labels read date are known; the rest are unlabelled, which is the positive-unlabelled
setting the nnPU loss needs. Discovery is not random (victim reports find the busiest mules
first), which violates nnPU's "selected completely at random" assumption; we report this as
a limitation, and the full-answer-key test measures its effect.

**Sparse labels.** `revealed_share` (*default* 1) keeps a share of the revealed mules: a
mule is kept when the 64-bit blake2b hash of its account id, keyed by the data seed, the
word `revealed` and `revealed_key` (*default* 0), falls in the lowest `revealed_share` of
its range. For one key the kept sets are nested (every mule kept at 10% is kept at 25%), and
they are independent of the account split. A dropped mule is unrevealed everywhere: it is
no positive, it may be drawn as unlabelled, and it is not confirmed for the label-input
variant. Below a share of 1, everything that is not a label stays as at 1: the runs' dates
and maturity gaps, each training date's unlabelled count and weight, and each seed's prior
and number of epochs (see "Experiments"). Test always uses the full answer key.

## Examples

**The population of a date** is every account open at t (see "Time") that is not external
and whose owner's ownership link is active at t. External accounts and the bank's ledger
accounts (no owner) are never examples, so no account with `is_mule` -1 is ever ranked.
Each set of examples takes the accounts of its split.

**Draws.** Every draw on a date takes the accounts with the lowest 64-bit blake2b hash of
their id and the date, keyed by the data seed and the word `draw`, among the accounts it may
draw. A draw is therefore uniform, depends only on what it draws from, and nests: when a few
accounts join or leave what it draws from, the rest of the draw stays, and so do their
cached subgraphs.

**Training** (a run with labels read at L):

- *Positives.* On each date, the training split's mules revealed at L that are active that
  date. Each mule carries a total weight of 1, shared equally among its dates.
- *Unlabelled.* On each date, a draw from the training split's population, leaving out every
  mule revealed at L, of 15 accounts (*default*) per positive that date at `revealed_share`
  1, accounts with no payments included. Together they weigh what that date's positives
  weigh at `revealed_share` 1, so the date cannot give the answer away.
- *Dropped.* A mule revealed at L on every date it is not active.

**Validation and test.** On each date:

- every account of the split active that date is positive, with weight 1: the mules revealed
  at R for validation, the answer key's `is_mule` 1 for test;
- a true mule (revealed at R, for validation) that is not active that date, or that was
  discovered by t, is not ranked that date;
- the other accounts are a draw of 2,000 for validation and 5,000 for test (*defaults*),
  each weighted by the accounts it stands for.

**Early warning** (test only). On each test date, every test-split mule of the answer key
whose onset is in the next four weeks (`mule_onset_ms - 28 days < t <= mule_onset_ms`) and
that was not discovered by t is scored too, apart from the ranking: its first ring payment
has not happened, so only its links and its neighbours can flag it.

## The subgraph query

One installed GSQL query, `fetch_subgraphs`, takes a batch of seed accounts that share one
as-of time t, the subgraph settings and the date's leaves. It returns, for each seed, a typed
2-hop subgraph as of t (nodes by type with TigerGraph-computed features, edges by relation
with theirs) and the seed's community block. Python joins the date's co-user counts and turns
a batch into one PyG `HeteroData`.

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

**The contract lock.** Every query prints a version, and the model file stores the subgraph
query's with a fingerprint of the feature layout (the community block, the label inputs and
any ablation included) and the subgraph settings it was trained on (the caps,
`community_parties`, `hub_threshold`, `observed_threshold`, `recent_payments` and
`lookback_days`). Scoring refuses a model built for another version, layout or settings:
with another lookback, features such as `in_365d` or `gap_ms` would mean something else.

### Response format

The reference graph of the tests produces exactly this, and `tigergraph/queries.py` parses
it.

**Request.** `seed_ids` (distinct accounts) share one `as_of_ms`, t: callers group seeds by
date. The caps, `community_parties`, `recent_payments` and `lookback_days` come from config
under the names of the fields below, and the date's leaves from `read_leaves`:
`hub_accounts`, `hub_devices`, `crowd_devices`, `hub_ips` and `crowd_ips`. There is no
phase: every customer is visible.

**Rows.** A refused request prints one row, `{"status": reason}`: `invalid_parameters`,
`duplicate_seed`, or `unknown_seed` and `invisible_seed` (not existing at t) with the
offending `seeds`. Otherwise the first row is `{"status": "ok", "version", "as_of_ms",
"work"}`, then one row per seed in request order with `seed`, the node lists `accounts`,
`parties`, `devices`, `ips`, the object `community`, and the edge lists `pays`, `owned_by`,
`enrolled_on_device`, `enrolled_on_ip`, `sent_from_device`, `sent_from_ip`, each edge from
an account. `work` counts what the request read (payments, identity nodes, community
candidates) for the bench; the reference graph does not mirror it and parity ignores it.
Lists are unordered; a node appears once, the seed is among the accounts, and both ends of
every edge are nodes of the same subgraph. Durations are milliseconds and never negative;
one whose presence flag is false is 0. Amounts are integer cents,
`floor(amount * 100 + 0.5)`, summed as integers. The row of a hub account that is not a
seed, and a leaf's row, follow "The feature table". Python adds the reverse edges. Inside the
query every per-seed mark is kept in an accumulator keyed by the seed, never in a vertex flag
that two seeds of a request could share.

**Visibility** follows "Time". A device or IP is observed on the visible payments an account
sent.

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
- Each device or IP a seed reached that is not a leaf adds other customers' deposit accounts
  (owned by a party that is not an owner of the seed): those whose owner links to it, one
  candidate per link, and those observed sending from it, one per account. Per seed,
  candidates are taken newest first (link start or last use), ties by account, node and
  relation: at most `reverse_accounts` per node and relation, and at most
  `identity_accounts` distinct accounts.
- The community of each seed is built as "The community of a seed" says.

**Node fields.**

| List | Fields |
|---|---|
| `accounts` | `id`, `owner_id` (its owner, `""` with none; for the join, not a feature), `account_type`, `is_external`, `hub`; `in_7d`, `in_30d`, `in_365d`, `out_7d`, `out_30d`, `out_365d`; `amount_in_7d`, `amount_in_30d`, `amount_out_7d`, `amount_out_30d`; `counterparties_in_30d`, `counterparties_in_365d`, `counterparties_out_30d`, `counterparties_out_365d`; `new_payers_7d`, `new_payers_30d`, `new_payees_7d`, `new_payees_30d`; `burst_in_1h`, `burst_in_24h`, `burst_out_1h`, `burst_out_24h`, `burst_new_payers_24h`; `passed_on_1h`, `passed_on_24h`; `shortest_in_gap_present`, `shortest_in_gap_ms`; `last_in_present`, `since_last_in_ms`, `last_out_present`, `since_last_out_ms` |
| `parties` | `id`, `accounts`, `devices_now`, `ips_now`, `new_devices_7d`, `new_devices_30d`, `new_ips_7d`, `new_ips_30d` |
| `devices`, `ips` | `id`, `leaf`, `observed_only`, `parties_now`, `parties_lookback`, `new_parties_7d`, `new_parties_30d`, `accounts`, `payments_7d`, `payments_30d`, `senders_30d` |
| `community` (one object) | `parties`, `capped`, `nodes`, `links`, `new_links_30d`, `accounts`, `internal_payments_30d`, `internal_amount_30d`, `external_payments_30d`, `density`, `isolation` |

**Edge fields.**

| Lists | Fields |
|---|---|
| `pays` | `id` (the payment), `from_id` (sender), `to_id` (receiver), `amount`, `rail`, `age_ms`; `prior_count`, `first_in_pair`, `gap_present`, `gap_ms`, `pair_age_ms`; `next_outflow_present`, `next_outflow_ms`; `sender_gap_present`, `sender_gap_ms`, `receiver_gap_present`, `receiver_gap_ms` |
| `owned_by` | `from_id`, `to_id` |
| `enrolled_on_device`, `enrolled_on_ip` | `from_id`, `to_id`; `start_known`, `since_start_ms`, `held_ms`; `active`, `since_end_ms` |
| `sent_from_device`, `sent_from_ip` | `from_id`, `to_id`, `since_first_ms`, `since_last_ms`, `uses` |

The query reads no supervision and no answer key: the label-input variant's confirmations
are joined in Python (see "The model").

### Date tables

Two queries compute, once per as-of date, what every request of the date would otherwise
repeat. Both are cached by (world identity, date, settings), and both follow "Time".

- `read_leaves(as_of_ms, lookback_days, hub_threshold, observed_threshold)` prints the ids of
  the date's hub accounts, hub devices, crowd devices, hub IPs and crowd IPs. `fetch` passes
  them to every request of the date.
- `read_sharing(as_of_ms, lookback_days, the four device and IP leaf lists, after_id,
  batch_size)` prints, a page at a time by party id, one row per customer with a co-user:
  `party_id`, `device_co_users`, `ip_co_users`, `new_device_co_users_30d`,
  `new_ip_co_users_30d`. A customer with no row has none. Python joins them to each account
  row by `owner_id`.

### Other queries

- `list_population(as_of_ms, after_id, batch_size)`: the date's population a page at a time
  by id, with `account_id`, `owner_id`, `account_type` and `opened_ms` (for evaluation's
  groups, never a feature). Reads no label.
- `read_supervision(known_by_ms)`: every account with `0 < discovered_ms <= known_by_ms`,
  with `discovered_ms`, `discovery_channel`, `mule_onset_ms` and `mule_last_ms`.
- `read_world()`: what identifies the loaded world (see "Scaling").
- `profile_hubs(as_of_ms, lookback_days, buckets)`: for the bench, histograms of the exact
  measures behind the hub and crowd tests, over every account, device and IP of the world,
  with devices and IPs split into enrolled and observed-only.
- `read_answer_key(after_id, batch_size)`, under `gsql/evaluation/`: every account's
  `is_mule`, ring, layer, flow, `mule_onset_ms`, `mule_last_ms` and `discovered_ms`, for
  evaluation only.

## The model

- Heterogeneous temporal graph attention: two `HeteroConv` layers of `TransformerConv` per
  relation, taking the edge features (Fourier encodings included) as edge attributes, summed
  across relations. `owned_by` and its reverse have no edge features, so their attention
  reads the nodes alone.
- *Defaults*: hidden size 64, 4 heads, dropout 0.2.
- The seed's final representation, joined with its community block (11 numbers), goes
  through a small head (to 32, then to 1) that gives one score. Without the community block
  (for the ablation) the head reads the representation alone.
- No per-account embedding and no memory, so any account, including a new one, can be
  scored.
- The ablations of "Experiments" are switches of the same model: emptied relations and
  zeroed columns of the built batches.

**Label-input variant.** The same model also sees, for each account node, whether it was a
confirmed mule by t, with the Fourier encoding of the time since its confirmation, cut at the
lookback; every relation has its own attention, so a confirmation counts according to how
the account is linked. The seed and every account of its owner always read as unknown: a
bank that had confirmed the seed's own customer would have closed it. A run reads the
confirmations once with `read_supervision`, known by the last date it scores, keeps them by
`revealed_share`, and joins them in training by account id when `discovered_ms <= t`;
confirmed accounts of every split count, as a bank knows all its confirmed mules.
Masking (*default*: each confirmed neighbour hidden with probability 0.5 during training,
never at evaluation) happens on the GPU, so cached subgraphs are reused across epochs and
serve both variants.

## Training

- **Loss.** nnPU (kept from the old code). The positive risk is weighted by 1 minus the
  prior (*default* `pu_weighting` balanced), which is imbalanced nnPU with a balanced target
  (Su, Chen and Xu, 2021). The textbook weight, the prior itself, stays a setting
  (`pu_weighting` prior): with it, training collapses to a constant scorer (on the reference
  world the loss settles at exactly the prior by the fourth epoch, and the earlier runs on
  NVIDIA measured a PR AUC of 0.001). The negative-risk correction subtracts the prior times
  the positives' risk as negatives; there each positive example counts once, one per mule and
  date, because a hidden mule enters the unlabelled draw once per date it is active, while
  the positive risk keeps each mule's equal weight.
- **Prior.** The prior is the share of hidden active mules in the pool the unlabelled are
  drawn from. `prepare` reports the revealed active share of the selection run: on each of
  its training dates, the training split's mules revealed at V that are active, divided by
  the training split's population, averaged over the dates. The grid is 1, 3 and 10 times it
  (*default*), so a hidden mule is assumed as common as, three times or ten times a revealed
  one; the choice and its sensitivity are reported.
- **Weights.** Before training, each label's weights are divided by their mean over all
  the training batches, once, because the loss divides each label's weighted sum by its
  count in the batch. Each mule then keeps its equal share however its dates fall into
  batches.
- **Selection run.** For each prior of the grid, train for at most 30 epochs (patience 5)
  and score validation after each epoch. The score is the recall of the revealed validation
  mules within the top 1% and within the top 5% of their date's accounts, pooled over the
  dates (mules found on all dates divided by positives on all dates), averaged over the two
  budgets. Only revealed labels are used, as in a bank. The run keeps the prior and the
  epoch with the best score; a later epoch or prior must be strictly better.
- **Final run.** Train once with the chosen prior for exactly the chosen number of epochs, on
  the final run's training examples. The final run is the model that is tested and saved.
- **Seeds.** Five seeds per world (*default*), 0 to 4, vary the model only: the initial
  weights, dropout, the label masks and the batch order. Each seed has its own selection run
  and final run. The data seed (the account split, the revealed share and every draw) stays
  fixed, since the three worlds already vary the data. Results are reported per seed, with
  their mean and range; `score` uses the final model of the seed whose selection run scored
  best.
- *Defaults*: Adam, learning rate 0.001, weight decay 0.0001, 128 examples per step.

## Scaling

**Sampling and batching.** The query's caps are the sampler: deterministic, newest first,
so a seed's subgraph is the same in every run. Hubs and crowds are leaves whose rows hold
only their flags, which keeps billers, ledgers, carrier NAT and terminals from pulling in
thousands of strangers or a year of their payments, and the time-ordered account features
read at most `recent_payments` payments per direction. A request's cost is therefore
bounded by the caps and by the lookback's payments of its seeds and of the non-hub accounts
it holds, read once each. Seeds are fetched by date, 32 to a request (*default*, a transport
setting, since it never changes a result); training steps take 128 examples.

**Date tables.** The leaves and the co-user counts are computed once per date by
`read_leaves` and `read_sharing` (see "Date tables"), not once per request: a co-user count
needs every device and IP an owner touched, each with its own users, and the hub test reads
a hub's whole year of payments.

**Extraction and caching.** TigerGraph does the feature work as of t, so only the subgraphs
and the date tables leave the database. The cache keeps each request's response, as printed,
in one compressed file per (world, as-of date, request), written once and never rewritten,
with an index of the seeds each holds; `fetch` requests only seeds the index does not hold.
The key is the world's identity, the query version and the subgraph settings. The world's
identity is a hash of what `read_world` returns:

- the count of every vertex and edge type, and of each value of `account_type`,
  `payment_rail` and `is_external`;
- sums in whole seconds of `joined_ms`, `opened_ms`, `closed_ms`, each link type's
  `valid_from_ms` and `valid_to_ms`, and each payment type's `event_ts_ms`;
- sums of the amounts in cents per payment type;
- per edge type, the sum over its source vertices of their squared degree, so rewiring
  shows.

All are integers, so the hash is the same on every run. A world regenerated or changed in
any count, time, amount, category or wiring has another identity, so its stale subgraphs and
date tables are never read. Subgraphs no longer depend on the phase, so one cache serves
every run, both variants, the controls and every experiment. `prepare` also checks the
lookback against the data start the identity gives.

**Response size.** The executor asks TigerGraph for responses of at most 64 MB
(`sizeLimit`). `seeds_per_request` is set so that the 90th percentile of bytes per seed,
times the seeds per request, stays under half of it. A request refused for its size is split
in two and retried, down to one seed; a single seed over the limit is an error.

**Parity.** No query has run on a TigerGraph server yet: the GSQL is checked by text tests
and mirrored in Python by the reference graph. `mule parity` takes a live TigerGraph that
holds a small export (*default* the 5,000-person parity world) and the reference graph
reading the same export. On a seeded sample of dates and seeds it runs every installed query
on both and compares the canonical responses, and it checks that a seed fetched alone equals
the same seed fetched among 31 others. It records the query versions that passed under
`results/parity/`. Every command that runs a query, except `install` and `parity`, refuses a
version with no passing record.

**Building.** Responses are parsed into one component per seed, the co-user counts are
joined by owner, and durations are Fourier-encoded on the GPU. Built batches stay on the
training device for every epoch when they fit (*default*), otherwise in pinned host memory,
copied each step.

**`mule bench`.** On the loaded world, after `prepare`, it draws seeds from the prepared
examples of every run in their proportions (*default* 256 seeds on each of 4 dates), and at
8, 32 and 128 seeds per request, with 1, 2 and 4 requests at once, it measures:

- the time and response size of each request, and its `work` counts, by positives and
  others, a refusal for size counted as a result;
- the time of `read_leaves` and `read_sharing` per date;
- the nodes and edges per subgraph by type, and the community sizes (median, 90th
  percentile and largest);
- TigerGraph's memory in use, from its metrics endpoint;
- the parse and build time per training batch, the forward and backward time and peak device
  memory per training step, and the cache bytes per seed;
- `profile_hubs`'s histograms on the bench dates, and the share of positive examples that
  are hub accounts;
- a projection of a full run: the examples per date divided by the seeds per request, times
  the time per request, plus the date tables, and the cache bytes.

It writes one table per world, with the world's size, to `results/bench/`. It is run on
worlds of 30,000, 100,000 and 300,000 people, and the tables side by side show how cost
grows and which part drives it.

**Recommended training architecture.** For each as-of date, TigerGraph computes the date
tables once, then the features and subgraphs of the date's seeds in requests of 32; requests
for different dates are independent and can run in parallel. The responses are cached once
per world and reused by every epoch, prior, seed, run, variant, ablation and control. Python
builds one component per seed, and the GPU encodes time, masks labels and trains on 128
examples a step. The bench decides the request size, the parallel requests and whether the
built batches fit on the device; this paragraph is updated from its tables.

## Comparisons

Every ranker is trained on the final run's examples with the same weights and scored on the
same dates and groups. The controls read the features of the fetched subgraphs, so they see
exactly what TigerGraph gives the GNN.

| Ranker | What it is |
|---|---|
| Rule | Ranks by the count of confirmed accounts linked to the seed at t (payment counterparties, shared devices and IPs; not its own customer), ties by the newest confirmation |
| Trees, own behaviour | Gradient-boosted trees (scikit-learn's `HistGradientBoostingClassifier`, its default settings, early stopping off, seeded by the model seed) on the seed's own account row, without its `hub` and co-user fields |
| Trees, graph features | The same on the seed's full account row, its owner's party row and its community block |
| Trees, graph features and labels | The same plus the counts of confirmed accounts linked to the seed, by how they are linked |
| GNN | The behavioural GNN, without label inputs |
| GNN with labels | The label-input variant |

The trees learn revealed against unlabelled with the examples' weights, rescaled so the two
labels weigh the same in total, as balanced nnPU does. When labels are selected completely
at random, a classifier of labelled against unlabelled ranks accounts as one of positive
against negative would (Elkan and Noto, 2008). The trees have no settings to select, so they
train only once, in the final run.

The claims we make:

- what graph learning adds: the GNN against trees on own behaviour, and against the GNN with
  the seed alone (its own payment sequence without its neighbours);
- what devices and IPs add: the GNN against the GNN without them;
- what message passing adds beyond graph features made by hand: the GNN against trees with
  graph features;
- what the GNN adds beyond guilt by association: the GNN with labels against trees with
  graph features and labels, and the rule.

## Experiments

- **Ablations.** The GNN without labels, trained without one family of inputs at a time, on
  the same examples and five seeds, each seed with the prior and number of epochs its full
  model chose, and compared with the full model at test seed by seed. None needs a new fetch.
  - Without the community block; the trees with graph features also run without it.
  - Without devices and IPs: no Device or IP nodes, no `enrolled_on_*` or `sent_from_*`
    relations, no co-user fields and no community block, so only payments and owners remain.
  - The seed alone: the seed's own payment edges and their features stay, and every other
    node's features are zeroed.
  - Without link ends: `active`, `since_end_ms`, `held_ms`, `devices_now`, `ips_now` and
    `parties_now` zeroed.
  - Without year-long counts: `in_365d`, `out_365d`, `counterparties_in_365d` and
    `counterparties_out_365d` zeroed.
  - Without `account_type`.
- **Sparse labels.** The final run at `revealed_share` 1, 0.5, 0.25 and 0.1, for the GNN,
  the GNN with labels and the trees, still positive-unlabelled, reported at test against the
  full answer key. Every share keeps the dates, the maturity gaps, the unlabelled counts and
  weights, and each seed's prior and number of epochs of share 1, so only the labels change;
  this borrows share 1's validation labels for selection, which a bank with sparse labels
  would not have, and the report says so. Below 1, the run is repeated for `revealed_key` 0
  to 4 with model seed 0, beside model seeds 0 to 4 with key 0, so the spread over which
  mules are kept is reported apart from the spread over seeds. The report gives each share's
  positives and the labelled share of all mules (the share kept times the share discovered by
  R).
- **Sensitivity world.** The results on the world whose rings share devices and IPs at 10%.

## Evaluation

- **Scoring.** On every weekly date in the test window, every true mule of the test split
  active that date is scored, plus the weighted draw of the other accounts, so the metrics
  estimate the full ranking without scoring every account every week.
- **Per-date alert budgets** (top 1% and 5%): precision and recall of the hidden mules active
  that date.
- **Account level.** The first alert date of each mule among its active dates: time to
  detection (first alert minus onset) and lead time (discovery minus first alert). An alert
  counts only before discovery.
- **Early warning.** The share of the early-warning mules (see "Examples") scored above their
  date's top-1% and top-5% thresholds, reported apart from the ranking metrics.
- **Cost.** Distinct accounts alerted per week.
- **Groups.**
  - New rings: test mules whose ring (`mule_ring_id`) gave no positive to the final run's
    training examples.
  - Never in training: test accounts in no subgraph of the final run's training examples.
  - Accounts opened after the final run's last training date.
  - Accounts linked to a confirmed mule at t (by link type and direction) and not linked,
    for every ranker including the GNN without labels: since a confirmed mule's customer was
    offboarded, a lift of the GNN without labels concentrated there would come from what
    offboarding leaves in the graph, not from behaviour.
  - Mules never discovered; mules by `mule_flow`.
- **Uncertainty.** A ring-level bootstrap within each world, and the spread across the five
  seeds, the revealed keys and the three worlds.

## Scoring new accounts

`mule score --date D` scores any account at any date with the same queries and the saved
model, including accounts opened after training. An account with no payments is scored from
its links and its community.

## Package layout and commands

| Package | Holds |
|---|---|
| `config.py`, `paths.py` | frozen settings; output locations |
| `tigergraph/` | connection, client with retries (kept), executor, query installer, query names and parsers, the world's identity |
| `data/` | the runs and their dates, the account split and the draws, the population per date, supervision with the revealed share, the training and validation examples, the maturity gap; the answer-key reader, which only evaluation imports |
| `subgraph/` | the date tables, the query request, the sharded disk cache, response to `HeteroData` |
| `model/` | Fourier time encoding, the heterogeneous attention model and its ablation switches, nnPU loss |
| `training/` | trainer, selection, the selection and final runs over priors and seeds |
| `controls/` | the rule, the gradient-boosted trees, and the features they read from fetched subgraphs |
| `evaluation/` | the test and early-warning examples, weekly scoring, groups, metrics and the ring bootstrap (kept), the experiments |
| `testing/` | the reference graph, the test world and the parity check |
| `bench.py` | the cost measurements |
| `cli.py` | the commands below |

| Command | What it does |
|---|---|
| `mule install` | installs the queries |
| `mule parity` | checks every installed query against the reference graph on the parity world and records the versions that pass |
| `mule prepare` | reads the world's identity and checks the lookback against its start, reads the populations and supervision, estimates the maturity gaps, writes every run's examples (the test and early-warning examples through evaluation), and reports their counts, the gaps, the revealed active share and the labelled share of mules |
| `mule fetch` | reads the date tables and fetches and caches the subgraph of every example |
| `mule train` | runs the selection and final runs of the GNN (both variants, five seeds) and trains the trees |
| `mule evaluate` | scores every weekly test date and reports the metrics |
| `mule experiment` | runs the ablations or the sparse-label series |
| `mule score` | scores accounts at a date with the saved model |
| `mule bench` | measures cost on the loaded world |

GSQL: `gsql/schema/` (schema and loading job), `gsql/queries/` (`fetch_subgraphs`,
`read_leaves`, `read_sharing`, `list_population`, `read_supervision`, `read_world`,
`profile_hubs`), `gsql/evaluation/` (`read_answer_key`). Results are tables and files, no
charts.

## Questions raised while building

| Question | Settled |
|---|---|
| The audit sample was unused and refused more than one split | Removed; the examples draw the per-date samples |
| External accounts have `is_mule` -1 | The population holds only accounts a customer owns, never external ones |
| "Active that date" was undefined for test mules | Defined under "Time", the same for training, validation and test |
| Where the answer-key reader lives | `data/answer_key.py`, imported only by evaluation; it also returns, for evaluation only, every account's onset, last activity and discovery |
| Time boundaries | Stated under "Time": everything strictly before t, half-open tenures, labels and confirmations at or before their date |
| How confirmations reach the label-input variant | From `read_supervision`, joined in training; the query reads no supervision |
| No query listed the ids for `write_split`, and chunked writes could mis-assign joint accounts | The split is a hash in Python and is never written |
| The hub threshold had no measured value | 200 stays, above any ring PhantomLedger makes; large observed-only nodes are crowds; `profile_hubs` gives the bench the exact distributions |
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
  rail, organizers outside ordinary sharing groups, and ring links starting on staggered days
  where ordinary groups start on one day.
- A customer that left the bank goes silent. Its link ends at the exit are hidden, but until
  ordinary customers also leave in PhantomLedger, a silent neighbour is more often a caught
  mule than anything else; the groups by link to a confirmed mule show how much the models
  draw on it.
- Each world has few test mules (roughly a quarter of the year's mules, in a quarter of the
  accounts), and few validation mules, so the bootstrap intervals matter and selection is
  noisy.
- The final run trains on more dates than the selection run that chose its epochs.
- Most test mules belong to rings the model trained on; only the new-rings group measures
  rings it never saw.
