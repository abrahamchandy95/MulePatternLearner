# MulePatternLearner design

The agreed design for the rebuild on branch `redesign`. Every decision here was settled with
the owner in October 2026; the research behind each one is in the owner's research folder
(`MulePatternLearner-backup-2026-10-14/research/`). Values marked *default* are starting
points the owner may change; everything else is decided.

## What we are testing

A temporal heterogeneous graph neural network finds money-mule accounts that a bank has not
found yet, including accounts opened after the model was trained and accounts with no
payments yet that share devices, IPs or addresses with mules. It must beat models without
graph context given the same information, and it must work on customers it has never seen.

## The data

PhantomLedger (`../PhantomLedger`, use case `mule-temporal`) simulates one US bank. Every
behaviour that matters here is backed by published research and recorded in PhantomLedger's
citation ledger, `docs/fraud_model_audit.md`, including the declared choices where no
research gives a number. In particular:

- A mule is an **account** that receives proceeds of crime. The label comes from what the
  account did, not from its owner's role. Columns record where it sat in the chain
  (`mule_layer`) and what it did with the money (`mule_flow`).
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
ring members share devices and IPs at 10% instead of the declared 70% and 60%.

## The graph

Same vertex and edge types as before; attributes cut to what is used. One millisecond clock
everywhere; no sequence numbers. The single copy of the DDL is `gsql/schema/schema.gsql`;
PhantomLedger's export and `tf_gnn_loader_v2` match it.

| Type | Attributes |
|---|---|
| Party | id, joined_ms, split |
| Account | id, account_type, is_external, opened_ms, closed_ms, split; supervision: discovered_ms, discovery_channel, mule_onset_ms; answer key: is_mule (-1 unknown), mule_ring_id, mule_layer, mule_flow, mule_last_ms |
| Token, Device, IP, Address | id |
| Payment_Transaction | id, amount, payment_rail, event_ts_ms |
| Zelle_Transfer | id, amount, event_ts_ms |
| 7 identity and ownership links | valid_from_ms, valid_to_ms (0 while open) |
| Payment role and observation edges | event_ts_ms |

`split` is written by MulePatternLearner; everything else comes from PhantomLedger.

**Who may read what.**

- *Supervision* (`discovered_ms`, `discovery_channel`, `mule_onset_ms`) is what a bank knows
  after an investigation. Training reads it only for accounts whose `discovered_ms` is
  before the relevant cutoff, through one query that applies that test.
- *Answer key* (`is_mule`, ring, layer, flow, last activity) is read only by evaluation.
- Neither is ever a feature. `is_mule` defaults to -1, so an account with no loaded truth is
  unknown, never normal.

## Time

Every example is an (account, as-of time t) pair, with t a Monday at 00:00 UTC. Everything
the model sees is computed from what existed before t:

- events with `event_ts_ms < t`;
- links with `valid_from_ms <= t`. A link that ended before t stays visible as a past fact
  with "time since it ended", because ring links are short and dropping them would erase the
  evidence. An end after t is invisible.
- confirmations with `discovered_ms <= t` (only for the label-input variant).

At most 365 days of payment history are used (*default*), so every example sees the same
amount.

**Every duration is Fourier-encoded**, and every duration is relative: to t, or between two
events. No absolute time reaches the model, so a model trained in 2024 scores 2027 the
same way. The basis is fixed: u = log1p(seconds) / log1p(10 years), 32 log-spaced
frequencies from 0.125 to 2 cycles per unit of u, sine and cosine, 64 numbers per duration,
computed on the GPU. TigerGraph computes the durations themselves.

## Splits and hidden labels

**By time (defaults).**

| Window | As-of dates | Labels the model may use |
|---|---|---|
| Training | 2024-01-01 to 2024-06-30, minus a maturity gap | mules discovered before 2024-07-01 |
| Validation | 2024-07-01 to 2024-09-30 | mules discovered before 2024-10-01 |
| Test | 2024-10-01 to 2024-12-31 | the full answer key |

**By customer.** 25% of customers (a seeded hash of the party id, written as `split`) are
held out. They are invisible during training and validation, even behind a shared device or
IP, and are reported as their own group at test.

**Hidden labels come from discovery, not from masking.** On any date only mules found by then
are known; the rest are unlabelled, which is the positive-unlabelled setting the nnPU loss
needs. Discovery is not random (victim reports find the busiest mules first), which violates
nnPU's "selected completely at random" assumption; we report this as a limitation, and the
full-answer-key test measures its effect.

**The maturity gap** stops training as-of dates early, because the newest mules have not had
time to be found and would sit among the unlabelled. Its length is chosen from revealed mules
only (a right-censored estimate of the delay from onset to discovery), never from the answer
key. *Default* 60 days.

## Training examples

- **Positives.** A revealed mule is positive on every grid date from its onset
  (`mule_onset_ms`, the investigation's earliest suspicious activity) to its discovery,
  within the training window. Each mule carries equal total weight however many dates it
  spans.
- **Unlabelled.** On each grid date, a fixed sample of accounts open that day in training
  customers (*default* 15 per positive on that date), including accounts with no payments.
  Positives and their unlabelled accounts share the same date, so the date cannot give the
  answer away.
- **Before onset.** A revealed mule's dates in the 30 days before onset are dropped; earlier
  dates are ordinary unlabelled examples.

## The subgraph query

One installed GSQL query, `fetch_subgraphs`, takes a batch of (account, t) seeds, a phase
(which customers are visible) and the caps. It returns, for each seed, a typed 2-hop
subgraph as of t: nodes by type with TigerGraph-computed features, and edges by relation
with their durations. Python turns a batch into one PyG `HeteroData`.

A node reached by two paths appears once, so cycles (Z, device, A, payment, Z) are visible.
Batches are grouped by as-of date so each identity node's features are computed once per
batch.

**Relations** (each has a reverse, so two layers carry hop-2 information to the seed; caps
are *defaults*):

| Relation | Cap per node |
|---|---|
| Account sent payment to Account (one edge per payment) | 32 most recent per direction for a seed, 16 for others |
| Account owned by Party; Party owns Account (siblings) | 1; 8 |
| Account enrolled on Device (through its owner, as of t) | 8; reverse 16 other customers' deposit accounts |
| Account enrolled on IP (through its owner) | 8; reverse 16 |
| Account at Address (through its owner) | 4; reverse 16 |
| Account sent from Device (observed on its own sent payments) | 8; reverse 16 |
| Account sent from IP (observed on its own sent payments) | 8; reverse 16 |

At most 48 hop-2 accounts reached through identity per seed. Token gets no relation: each
token belongs to exactly one account, so its age becomes an account feature.

**Hubs.** An identity node or counterparty with more distinct visible owners than a measured
threshold (carrier NAT, busy terminals, billers, the bank's ledger accounts) stays as a leaf
node with its degree and age as features and is not expanded. The count is computed in the
query as of t, stopping early at the threshold.

**Node features** (computed in TigerGraph as of t, only over customers visible in the phase):

- *Account*: account type, external flag, age, token age, payments in and out over 7, 30 and
  365 days, amounts in and out over 30 days, distinct counterparties in and out over 30 days,
  first-time inflows over 7 and 30 days, money passed on within 24 hours over 30 days, time
  since the last payment in and out, and closed-before-t. These are features of every
  account in the subgraph, so a hop-2 neighbour's behaviour reaches the seed.
- *Party*: age, open accounts, closed accounts.
- *Device, IP, Address*: age (first visible use), parties now and ever, accounts, new parties
  in the last 30 days, observed-only and hub flags.

**Edge features.**

- *Payment*: amount, rail, and the Fourier encodings of its age, the gap since the pair's
  previous payment, the pair's age when it happened and the delay to the receiving account's
  next outflow; the pair's prior count and a first-in-pair flag; presence bits.
- *Identity link*: time since it started, time since it ended (and an active flag), how long
  it was held; for observed links, time since first and last use and use counts.

**The contract lock.** The query prints a version, and the model file stores it with a
fingerprint of the feature layout. Scoring refuses a model built for another version.

## The model

- Heterogeneous temporal graph attention: two `HeteroConv` layers of `TransformerConv` per
  relation, taking the edge features (Fourier encodings included) as edge attributes, summed
  across relations.
- *Defaults*: hidden size 64, 4 heads, dropout 0.2.
- The seed's final representation goes through a small head (64 to 32 to 1) that gives one
  score.
- No per-account embedding and no memory, so any account, including a new one, can be
  scored.

**Label-input variant.** The same model also sees whether each neighbour account was a
confirmed mule by t, with the Fourier encoding of time since confirmation, split by how it is
linked. The seed and every account of its owner always read as unknown. The query returns the
visible confirmations as data, and masking (*default*: each confirmed neighbour hidden with
probability 0.5 during training, never at evaluation) happens on the GPU, so cached
subgraphs are reused across epochs.

## Training

- **Loss.** nnPU (kept from the old code), with the class prior chosen on validation from
  {0.001, 0.003, 0.01} (*default* grid) and its sensitivity reported.
- **Selection.** The epoch and prior are chosen by recall of revealed validation mules
  within the top 1% and 5% of validation accounts, per weekly date, averaged over dates.
  Only revealed labels are used, as in a bank.
- *Defaults*: Adam, learning rate 0.001, weight decay 0.0001, at most 30 epochs, patience
  5, five seeds per world.
- **Caching.** Subgraphs are cached on disk per (world, phase, as-of date), keyed by the
  query version.

## Comparisons

Every ranker is trained on the same examples with the same weights and scored on the same
dates and groups.

| Ranker | What it is |
|---|---|
| Rule | Ranks by counts of confirmed mules linked to the account |
| Trees, account only | Gradient-boosted trees on the seed's own TigerGraph features |
| Trees, with links | The same plus the confirmed-neighbour counts |
| GNN | The behavioural GNN, without label inputs |
| GNN with labels | The label-input variant |

The claims we make: the GNN against trees on the account alone (what graph learning adds),
and the GNN with labels against trees with links (what the GNN adds beyond guilt by
association).

## Evaluation

- **Scoring.** Every test account is scored on every weekly date in the test window. Mules
  already discovered before a date are removed from that date's ranking.
- **Per-date alert budgets** (top 1% and 5%): precision and recall of hidden mules active on
  that date.
- **Account level.** The first alert date: time to detection (first alert minus onset) and
  lead time (discovery minus first alert). An alert counts only before discovery.
- **Cost.** Distinct accounts alerted per week.
- **Groups.** Held-out customers; accounts opened after training; accounts with no payments
  yet; linked to a confirmed mule (by link type and direction) and not linked; mules never
  discovered.
- **Uncertainty.** A ring-level bootstrap within each world, and the spread across the three
  worlds.

## Scoring new accounts

`mule score --date D` scores any account at any date with the same query and the saved
model, including accounts opened after training. An account with no payments is scored from
its links.

## Package layout

| Package | Holds |
|---|---|
| `config.py`, `paths.py` | frozen settings; output locations |
| `tigergraph/` | connection, client with retries (kept), query installer, query names |
| `data/` | windows and customer hold-out, the weekly example grid, supervision and answer-key readers |
| `subgraph/` | the query request, response to `HeteroData`, the disk cache |
| `model/` | Fourier time encoding, the heterogeneous attention model, nnPU loss (kept) |
| `training/` | trainer, selection |
| `controls/` | the rule and the gradient-boosted trees |
| `evaluation/` | weekly scoring, groups, metrics and the ring bootstrap (kept) |
| `cli.py` | `install`, `prepare`, `fetch`, `train`, `evaluate`, `score` |

GSQL: `gsql/schema/` (schema and loading job), `gsql/queries/` (`fetch_subgraphs`, the
population per date, the supervision reader), `gsql/evaluation/` (the answer-key reader).
Results are tables and files, no charts.

## Limitations we state up front

- Synthetic data, however well cited, is not a bank's data.
- Most mule research is British; PhantomLedger simulates a US bank.
- Discovery is not completely at random, so nnPU's assumption does not hold exactly.
- The sharing rates of ring devices and IPs are declared, not measured; the sensitivity world
  shows how much the result depends on them.
