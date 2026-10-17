# MulePatternLearner design

The specification the code builds. Every decision in it was agreed with the owner in the
grilling session of 2026-10-17, and it replaces the October 17 revision of
this file (commits 5637d9c and ed62329), which was written before that session. Nothing here
is built yet.

- [The features](features.md) is part of this design: every number that reaches the model in
  a batch, with its exact definition, encoding, width, leak guard and the generator gates it
  depends on. This file says why each query and field exists; that one says exactly what
  each is.
- [Future design work](future_design.md) holds what the owner has raised but not decided.
- The research behind the decisions lives in PhantomLedger's `docs/research/` folder, where
  the dossiers move from the owner's research folder: the generator shortcut dossier
  (`generator_shortcuts_2026-10-17.md`, 63 shortcuts with their fixes, its implementation
  plan and the citation checks), the realism and burst dossiers
  (`generator_realism_2026-10-15.md`, `mule_bursts_2026-10-15.md`), the mule activity research
  (`mule_activity_2026-10-17.md`), and the checked customer lifecycle and GSQL dwell research
  of 2026-10-17.

## Standing rules

| Rule | Where it comes from |
|---|---|
| Leaks are absolutely not allowed. | The owner |
| TigerGraph and GSQL are the main deliverable and Python the additional one. TigerGraph computes every number the model reads; Python only joins rows by id, encodes them and builds tensors. | The owner: "our whole deliverable is TigerGraph and GSQL with the python as an additional deliverable" |
| Query names start with a verb. | The owner's standing rule, confirmed on 2026-10-17 |
| Commands take no options; dates and settings live in config. | The same |
| GSQL computes nothing that nothing reads. | The same |
| Every table and every column PhantomLedger exports is loaded, so the schema holds every exported column and the export holds nothing the schema lacks. | The owner: "when I look at the uploaded data, i dont want to see missing fields in the load data page" |
| PhantomLedger gets no new flags. It simulates the whole universe of behaviour, and a use case only decides what its label means. | The owner: "we are defining the behavior of the mule usecase now"; "we code in behavoir of the whole universe" |
| Every behaviour in PhantomLedger is backed by research, or declared as a choice, in its citation ledger. | The owner: "every line in phantomledger is research backed" |
| One branch, `main`, the same on both remotes; nothing is pushed without the owner's confirmation; `redesign` is merged into `main` at that push. | The owner's standing rule, confirmed on 2026-10-17 |

## What we are testing

A temporal heterogeneous graph neural network learns the behaviour of money-mule accounts
from what accounts do over time and how they are connected: their payments with their speed
and order, their owners, and the devices and IPs they share with other customers. On every
Monday it ranks a bank's accounts by how likely each is to be acting as a mule.

- **Behaviour, not detection.** The owner's framing (2026-10-16): a model trained today can
  learn from a period long past, whose identified mules are all known by now, so "the date
  of discovery adds unnecessary complexity". Training uses a period whose identified mules
  are all in, and nothing reads when a mule was found.
- **Sparse labels.** Only some mules are identified by the bank (the owner: "we cannot go
  back to pretending it was fully labeled"). Training is positive-unlabelled, with the nnPU
  loss.
- **Inductive.** The model has no per-account embedding and no memory, so it scores accounts
  it never saw, including accounts opened after training. Training, selection and test use
  three independent worlds that share no account, ring, device or IP.
- **What success means.** In the test world every true mule is scored on every Monday it is
  active, beside every ordinary account, and should rank high. The owner: "we need to
  score every one of the true mules that were hidden from the model. the model needs to be
  able to identify all of them, or atleast score them high. if not, we need to investigate
  why. maybe theres a bug, or we are missing some features that need to be added." The
  headline group is mules in rings the bank never surfaced, found from behaviour and shared
  devices alone. Short-lived mules must never be missed (the owner: "these are most
  important").
- **What the graph adds.** The GNN is compared with XGBoost given the same information, and
  ablations measure what neighbours, devices and IPs, the community block and per-payment
  timing add.
- **Early warning.** How often a mule is flagged on the Monday before its first ring payment,
  from the devices and IPs it shares and from its neighbours, is reported apart.
- **The scored unit is an account at a date.** The owner: "we want to detect mule
  accounts. not customers." Customer, device, IP and community features describe the
  account's surroundings; they never turn the model into a customer score.

## The data

### PhantomLedger's mule use case

PhantomLedger (`../PhantomLedger`, use case `mule-temporal`) simulates one US bank. It
simulates the whole universe of behaviour, criminal flows included; the mule use case only
decides which accounts its label calls mules:

- A mule is an internal account that ring money landed in, where the money's chain starts
  with a victim payment. Drug-cash structuring and organizers moving their own
  money (the bipartite, cycle and invoice playbooks) stay in the world as real criminal
  flows and are unlabelled in this use case, because a mule moves victims' money.
- Account-takeover drops and impostor payees become mules with no ring (`mule_ring_id` -1)
  once PhantomLedger's scam payees step lands.
- PhantomLedger also decides which mules the bank identifies (see
  [Identification](#identification-as-phantomledger-defines-it)).

### Worlds and calendar

| Item | Decision | Reason |
|---|---|---|
| Worlds | Three independent worlds from different seeds: one trains, one selects, one tests. Reported runs rotate the roles (A/B/C, B/C/A, C/A/B); development uses one rotation. | Every test account, ring, device and IP is new to the model, which is "mule accounts created after the GNN was trained", with no split machinery. Rotation gives three test numbers and a spread across worlds. |
| Sizes | Development worlds have 30,000 people, reported worlds 300,000. Development worlds use other seeds, never those of A, B or C. | Thresholds and caps are tuned on development worlds only and frozen before the reported worlds are generated. |
| Burn-in | 365 days are simulated with laundering running and are not exported. The burn-in is defined in the use case, not a flag. | The scored year starts in steady state: rings mid-life, device tenures open. Rings born at a constant rate and living about 730 days reach only about 12% of steady state by the first scored Monday without it. |
| Export window | 455 days: 90 days of history only, then 365 scored days holding 52 Mondays. Laundering may run from day 0. There is no laundering-free warm-up. The window may grow if research shows a need. | Every scored Monday has a full 90-day lookback, for mules and ordinary accounts alike, and nobody loads a year nobody reads (the owner: "we would have our window of 365 days + 90 days"). |
| Scored Mondays | The 52 Mondays from the first Monday at or after day 90 of the export, at 00:00 UTC, listed in config. | The calendar is the same in every world. Which calendar does not matter, since no feature reads the clock. |
| Scoring frequency | Weekly, on Mondays, everywhere. Daily scoring is a later option once the benchmark shows its cost (see [future design work](future_design.md)). | A mule active Tuesday to Thursday is still scored the next Monday with its whole burst in view. |

### What PhantomLedger must provide

Each item is research-backed or declared in PhantomLedger's citation ledger, follows the
shortcut dossier's implementation plan as amended by the owner's answers, and is checked by
a gate (see [Gates](#gates)).

| Area | What the world must do |
|---|---|
| Shortcuts | Every one of the dossier's 63 shortcuts that a model could exploit is fixed with research-backed behaviour or declared. The owner: "get rid of all the shortcuts". |
| Mule activity | One active period per account per ring; a second episode only through a second ring (0.06 of mule persons). Victims over the account's life are the sum of today's per-burst draws, so credits per mule stay about the same. The first victim arrives at onset plus the recruitment lag; later victims after a renewal gap LogNormal with median 2 days and sigma 1.3, kept to [10 minutes, 21 days], all declared choices. The active period belongs to the account and is shared by every playbook. Long quiet belongs before the first misuse. No source supports quiet gaps of weeks to months between bursts in one account (`mule_activity_2026-10-17.md`, committed as PhantomLedger 77e0c61). |
| Ring devices and IPs | They belong only to controller-operated accounts (online handover and organizers' own ring accounts), about 12% of mules instead of 70%. A tenure starts at the first attempted ring payment minus a lead (median 1 day, at most 14 days) and is built from attempted payments; no ring tenure has a quiet stretch longer than 21 days. A controller who logs in and never pays out is not generated, since no research gives a rate. |
| Usage edges | Every observed device and IP, carrier networks, public terminals and hotspots included, gets usage edges at the declared coverage; `read_hubs` marks the big ones as crowds. |
| Onward money | A declared 0.6 of mules' onward value goes to accounts at other banks, following RUSI's direction. The share of victims at the same bank is pinned in the dossier. |
| Identification | The mechanism of [Identification](#identification-as-phantomledger-defines-it), written as `pu_label` only. Identification closes nothing. |
| Customer lifecycle | One research-backed change: relationship starts backdated years before the data by the customer's age (the owner: a world of 2030 to 2031 must not have its first account opened in 2030); each account's own opening date; small-business and other secondary accounts; device, IP and token starts; ordinary openings and closures all year; mule account origins; no exported row before its account opens. |
| Lifecycle numbers | New to the bank: at least 0.033 of customers a year, a floor (NAFBS 2024: 3.3% of US adults have held their primary account under a year; 0.014 at 60 and over), by age, so that about 0.8% or more of customers joined in the last 90 days on any date. Closures from ordinary churn and death only: charge-off about 0.06 of accounts a year, switch-outs 0.02 to 0.03 a year. Small-business accounts for 0.045 of adults (adults only), mostly owners without employees, with owner transfers in both directions. One mule account per customer at one bank, a second with chance 0.02 (UK 2025: 1.02 accounts per offboarded customer). |
| Mule origins | Three: the mule's own established account; an account the person opens for the purpose; and, new, accounts a controller opens under a stolen or synthetic identity (about 0.10 of mule accounts, declared, below Experian's bound of about 0.23), run from the controller's devices, with legitimate look-alikes (new, quiet, single-device customers) so that "new and quiet" is not a mule-only marker. |
| Device and IP link lives | The phone keeps its 40 months; a web device is keyed on a cookie or login token (its life uncited); home IPs are a two-class mixture of about 0.4 static lines and spells of one to four months, so 30% to 40% of home-IP links start inside a 90-day lookback. Legitimate new links must be common, or "new link" becomes mule-only. |
| Export | The contract under [The graph](#the-graph). Payment ids are exported as BLAKE2b pseudonyms like entity ids, not row numbers. No self-payments. Links made at sign-up start exactly at `joined_ms`. A NaN amount fails the export. |
| Clocks | The Clocks step (no rows stamped at second 0, no same-second couplings, victim, mule and organizer clocks like their holders' own, legitimate night ATM use, the account-takeover clock, legitimate rails on every day of the month) must pass before a reported run, because durations measured to t carry the clock of their event. |
| Rails | Card cash-outs on rail `card` and own-account moves on rail `internal`, so that the `unknown` rail is honest. |

Data for PhantomLedger's research: the Federal Reserve's 2025 Diary of Consumer Payment Choice
(`dcpc-2025-public-csv.zip`, 2.9 MB, downloaded with the owner's approval into the research
folder's `data/dcpc_2025/`) anchors the hour-of-day profiles. The RBIH mule data on Kaggle is
optional and the owner's to download. The NAFBS and SCF tenure files are to be located, with
file, source and size shown to the owner before any download.

**Expected label counts.** Ring births are calibrated so identified mule customers come to
about 25 per 10,000 customers per scored year: about 75 in a 30,000-person world and 750 in a
300,000-person one. The identified share of true mules is then measured and reported, not set.
Development worlds test the pipeline; priors, epochs and features are not chosen from them
except where this design says so.

### Gates

A value is only as honest as PhantomLedger's behaviour. [Generator gates each feature depends
on](features.md#generator-gates-each-feature-depends-on) maps every dossier entry to the
features its cue reaches, and adds gates of its own. A reported run refuses a world unless:

- every PhantomLedger gate mapped to a feature passed on it or on its gate world, from
  PhantomLedger's gate report for that world (`mule prepare` checks it);
- `mule audit` passed on it: the clock probe shows chance and the gates that need the answer
  key (the newness, same-owner move and recent shared link lifts) stay under the ceilings
  declared before the run (`mule train` checks it);
- the [feature audit](#the-feature-audit) passed on the development rotation for the same
  query versions and settings (`mule train` checks it).

Development worlds run with a warning, and their results are marked development only.

## The graph

### The export contract

PhantomLedger's mule use case exports exactly these 22 tables, and `tf_gnn_loader_v2` loads
every table and every column of them. One UTC millisecond clock, 0 for none. Each link holds
from `valid_from_ms` until `valid_to_ms` (0 while open), and `DISCRIMINATOR(valid_from_ms)`
keeps repeated links of one pair. Every event edge repeats its payment's `event_ts_ms`.

| Table | Columns | Read by |
|---|---|---|
| `Party` | `id`, `joined_ms` | `joined_ms` only in the known-start test |
| `Account` | `id`, `account_type`, `is_external`, `opened_ms`, `closed_ms`, `is_mule`, `pu_label`, `mule_ring_id`, `mule_onset_ms`, `mule_last_ms` | see [Who may read what](#who-may-read-what) |
| `Device`, `IP` | `id` | the identity queries |
| `Token` | `id` | loaded only |
| `Payment_Transaction` | `id`, `amount`, `payment_rail`, `event_ts_ms` | every behaviour query |
| `Zelle_Transfer` | `id`, `amount`, `event_ts_ms` | every behaviour query (rail `zelle`) |
| `Party_Owns_Account` | `from_id`, `to_id`, `valid_from_ms`, `valid_to_ms` | ownership |
| `Party_Uses_Device`, `Party_Uses_IP` | the same | enrolment |
| `Party_Uses_Token`, `Token_Bound_To_Account` | the same | loaded only |
| `Transaction_From_Account`, `Transaction_To_Account`, `Transfer_From_Account`, `Transfer_To_Account` | `from_id`, `to_id`, `event_ts_ms` | sender and receiver |
| `Transaction_Used_Device`, `Transaction_Used_IP`, `Transfer_Used_Device`, `Transfer_Used_IP` | the same | devices and IPs seen on payments |
| `Transfer_From_Token`, `Transfer_To_Token` | the same | loaded only |

`account_type` is `deposit`, `credit`, `brokerage`, `gl` (the bank's ledgers) or `unknown`;
`payment_rail` is `ach`, `card`, `cash`, `check`, `internal` or `unknown`. Every account has at
most one owner and one ownership tenure; the loader refuses otherwise, since a second tenure
filled two device cap slots in the rebuild.

**Token** stays in the export because Zelle addresses money by the recipient's token: an
email or US mobile number enrolled once and resolved by Early Warning's directory to a
deposit account (CFPB v. Early Warning Services, 2024; New York v. Early Warning Services,
2025). The money still moves between the two deposit accounts, and in PhantomLedger each token
copies one account and never moves, so no query reads it; Zelle use shows as the rail. Its
enrolment is backdated with the accounts. Token re-registration is in
[future design work](future_design.md).

**Left out of the export**, each for its reason:

| Left out | Reason |
|---|---|
| `mule_layer`, `mule_flow`, `discovered_ms`, `discovery_channel` | Noise for this design (the owner). `mule_flow` is PhantomLedger's after-the-fact tag of what a mule did with the money, so as a column it would be the answer; the discovery fields belong to a detection model this design does not build. |
| `Address`, `Party_Has_Address` | Rings share none, and PhantomLedger's address tokens collide between strangers by chance. Address returns only once households exist (see [future design work](future_design.md)). |
| `Account_Uses_Device` | A copy of the owner's device links; no query reads it. The owner-enrolment relation is built in the query from the customer's links. |
| `Transaction_From_Token`, `Transaction_To_Token` | PhantomLedger never fills them. |
| `device_type`, `country_code`, `fraud_label`, the audit ledger's `is_fraud` | Never in the contract. |

### Schema and loading

`gsql/schema/schema.gsql` is the single copy of the DDL, for TigerGraph 4.2.5. The loader keeps
a byte-identical copy (a test compares them) and loads every table by header name with one
command. MulePatternLearner writes nothing to the graph.

Before loading, the loader refuses an export with: a repeated id; a dangling edge end; a
repeated link of one pair and start, which `DISCRIMINATOR(valid_from_ms)` would merge; a
payment without exactly one sender and one receiver; an event edge time unlike its payment's;
a payment before either of its accounts opens; a payment between two external accounts; an
account with two owners or two ownership tenures; a link made at sign-up that does not start
at `joined_ms`; payment ids whose order follows event time. It counts self-payments, which
must be 0.

### Who may read what

| Columns | Meaning | Read only by |
|---|---|---|
| `pu_label` | the bank identified the account as a mule (1) or not (0) | `read_labels` (training, selection, and evaluation's groups) |
| `mule_onset_ms`, `mule_last_ms` | the account's first and last ring payment: the activity period an investigation records | `read_labels`, which prints them only for `pu_label` 1; `read_answer_key` |
| `is_mule` | the truth: 1 mule, 0 not, -1 external (unknown) | `read_answer_key` (evaluation) |
| `mule_ring_id` | the ring of the account's first landing payment, -1 for none | `read_answer_key` (evaluation: ring groups and ring-resampled uncertainty) |
| `opened_ms`, `closed_ms`, `joined_ms` | dates of record | existence, the scored population and the known-start test; `opened_ms` is printed only by `read_answer_key`, for an evaluation group |
| `account_type` | the product | the scored population and the deposit-only co-user accounts; never a feature |
| `is_external` | another bank's account | a feature on neighbours |

## Time

Every example is an (account, t) pair, with t a Monday at 00:00 UTC. The lookback is 90 days:
it sees the current burst and the account's ordinary life around it, which needs a few months.

**Bounds.** Every boundary is strictly before t (an event at t would leak, and
PhantomLedger starts about one ring tenure in seven exactly on a Monday 00:00).

- A payment counts when `t - 90 d <= ts < t`. "Last w" means `t - w <= ts < t`. "Within w of
  x" means `[x, x + w)`.
- A link (ownership, device or IP enrolment) is **visible** when `valid_from_ms < t` and
  (`valid_to_ms = 0` or `valid_to_ms > t - 90 d`).
- A link is **current** when `valid_from_ms < t` and (`valid_to_ms = 0` or
  `valid_to_ms > t - 30 d`): held at some time in the last 30 days.
- No other comparison reads `valid_to_ms`. A link end is never a value, an order or a test
  against t, because PhantomLedger plans a ring tenure's end from the ring's future payments,
  so "still active at t" would say whether more ring payments are coming.
- An account exists at t when `opened_ms < t`, and is open at t when it exists and
  (`closed_ms = 0` or `closed_ms >= t`). Openness is used only to choose the scored
  population.

**Known starts.** A date of record dates the customer or the data, never its behaviour:

- A link's start is known when `valid_from_ms > joined_ms` and `valid_from_ms > t - 90 d`. A
  start at sign-up dates the customer; one before the lookback dates the data, and once
  `joined_ms` is backdated a clamped link would pass "after joining" and its age would measure
  the calendar.
- A first seen use of a device or IP is a known start when, in the lookback, an account the
  customer owns sent an earlier payment that was not seen on that node. Otherwise it is only
  the start of the customer's activity in the lookback, which dates the customer.
- An unknown start shows no time since start, never counts as new, and counts as `t - 90 d`
  wherever a sharing start is needed.

**Ties.** Payments in one second have no order, because PhantomLedger's row order sorts
by fraud flag and ring id within a second.

- Every count, window, burst, gap, B1 walk and community value is order-free: swapping the
  ids of same-second payments changes no B1, B2, B3 or B6 value.
- A gap is to the latest strictly earlier payment. Windows are anchored at payments, never at
  clock bins.
- The time-ordered caps keep whole seconds: when a list is cut, b is the second of the newest
  payment left out, and only payments after b are kept.
- Subgraph caps order by activity, newest first, and break ties within one second by the
  payment id, which is a pseudonym. Other ties (customers, devices, IPs, accounts) break by
  their pseudonymous ids.

**Caps.** Recorded in the model's contract lock, tuned and frozen on development worlds:

| Cap | Value | What it bounds |
|---|---|---|
| Time-ordered lists | 256 per list | B1's inflows and outflows of the last 30 days; B2's first-payment times of new payers; B3's payments per direction of the last 30 days |
| Scored account's payments | 32 each way | B4 |
| Expanded counterparty's payments | 16 each way | B4 |
| Other accounts of the scored account's customer | 8 | B4 |
| Devices and IPs | 8 of each per kind of link, per customer (enrolment) and per account (seen) | B5 |
| Co-user candidates | 48 per scored account | B5 |
| Community members | 64 | B6 |

Without a cap on both directions, an account with thousands of outflows overflows a
fixed-capacity heap and breaks parity with the reference.

**Money.** Every amount is integer cents, `floor(amount x 100 + 0.5)`, and every sum and share
is computed in cents in INT, never mixed with UINT: float sums flipped the sixth decimal in 3%
to 35% of large sums and failed parity though both answers were right. Dollars appear only at
encoding. Every unsigned time subtraction is guarded.

**Self-payments.** A payment with the same account at both ends is read by no query.

**No clock.** No feature reads the time of day, day of week, day of month or second of the
minute, the Monday's index or days since the data start. A model that reads a
simulator's clock habits learns the simulator, and this also keeps the model calendar-free
across worlds: one trained on one year scores another the same way.

## Labels

### Identification as PhantomLedger defines it

Identification is defined once, in PhantomLedger's mule use case, with no new flag, and written
as `pu_label` only: no channel column and no uniform alternative. It works
on two levels. In some rings no member is ever surfaced; in rings that surface, a victim's
report identifies the account that received its money, tracing along the money finds some
onward accounts and organizers, and monitoring flags a few. The numbers come from the shortcut
dossier's identification step:

| Channel | Rule |
|---|---|
| Victim report | One draw per deceived victim's case: p 0.53 for authorised scams (PSR 2024; range 0.31 to 0.56), 0.67 for account takeover. Only a deceived victim reports, never a ring participant or a structuring depositor. |
| Bank action | The first reported case on a mule account is acted on with p 0.5, each later one with 0.7. An acted-on case identifies every internal mule account the victim paid in it. |
| Monitoring | A hazard on laundering-active days, fitted so monitoring finds a quarter of identified mules (declared). |
| Trace | Downstream along ring payments from each identified account: 0.25 in the first round, then 0.10, up to four rounds (declared). |
| Same customer | Every mule account of a customer with an identified account is identified. |

- **Fully hidden rings** arise from the mechanism (no acted-on report and no member found by
  monitoring) and are measured and reported. If their share turns out near zero, a declared
  share is revisited. The owner: "we need to be able to hide a proportion of entire
  rings (so the whole ring remains hidden in a few cases). this is why we need the GNN to be
  able to identify a few members of these mules".
- **No identification through shared devices or IPs**: labels would then lean on the
  very evidence the GNN is tested on and inflate its results.
- **No identified non-mules**: every `pu_label` 1 is a true mule. Real offboarding lists
  hold some; this is a stated limitation.
- **Identification closes nothing**. Closures come only from ordinary churn and death,
  independent of mule status, so closure or silence caused by a catch never exists in the
  world. The owner: "i just don't want the act of "closing" or inactivity to leak into the GNN
  model. Therefore, it doesn't close." There is no exit-hiding rule and no link-ends ablation;
  a test shows a caught neighbour reads exactly like an uncaught one.

### The sparse labels are the pu_labels

The owner: "Sparse labels is the pu_labels. we dont know as much as the ground truth
mules ... once we compute pu_labels, that's our sparse labels, these would contains some hidden
rings as well." There is no sparse-label series that thins them further. `mule prepare`
measures and reports the identified share, and the identified share by onset quarter, since
identification after the money moved would label late mules less often.

### Positives, dropped Mondays and the activity period

- A **positive** is an identified mule (`pu_label` 1) on a Monday t on which it is active:
  `mule_onset_ms < t <= mule_last_ms + 7 days`, and it is in the scored population that
  Monday. Since events must be before t, its first ring payment is in view on every positive
  Monday, and the week after its last payment gives every mule at least one positive Monday.
- Its other Mondays are **dropped**, never unlabelled, because ring participants carry ring
  signatures outside their activity.
- Labels follow mule activity, not the catch: an investigation reads the account's whole
  history and records the activity's date range (a US SAR has a field for it), which is why
  `mule_onset_ms` and `mule_last_ms` are the only label attributes besides `is_mule` and
  `pu_label`. With PhantomLedger's activity model the range carries almost no silent
  weeks.
- Boundary tests: an onset exactly at t is not positive; `mule_last_ms + 7 days` exactly at t
  is.

### No label inputs

No feature says whether any account is known or caught, and there is no rule ranker and no
trees-with-links control. The owner: "we don't want the knowledge of caught mules to
affect some scores." Guilt by association is a later experiment (see
[future design work](future_design.md)). Graph-structure features, device and IP sharing above
all, are computed identically for every account whatever its label; no feature is ever
conditioned on `pu_label` or any label. The owner: "if you link devices only to pu_label
== 1, this leaks data and may dilute the effectiveness."

### Checks on every world

`mule prepare` refuses a world where: a `pu_label` 1 account has `is_mule` other than 1 or
`mule_onset_ms` 0; `read_labels` sees a row without `0 < mule_onset_ms <= mule_last_ms`; an
`is_mule` 1 account lacks its activity dates or a ring value (a ring id or -1); a scorable
account lacks truth; a truth join shares a column other than the account id. It counts
`is_mule` 1 by `account_type` and owner presence and reports any mule outside the scored
population as a mule never scored: the agreed population assumes every mule is an owned
deposit account, and this checks it on every world. External accounts have `is_mule` -1 and
`pu_label` 0.

## Worlds, examples and weights

### The scored population

On each Monday t the **scorable accounts** are the internal deposit accounts open at t with a
visible ownership link: owned by a customer. Ring money moves between deposit accounts. Other
accounts (credit cards, brokerage, the bank's ledgers, other banks' accounts) appear only as
neighbours. `account_type` chooses the population and is not a feature: the rail on each
payment says what kind of payment it was.

### Training world

| Kind | Which | Weight |
|---|---|---|
| Positive | each identified mule active at t (see [Positives](#positives-dropped-mondays-and-the-activity-period)) | 1 divided by its number of active Mondays, so every mule counts once however long it ran |
| Dropped | identified mules on their other Mondays | none |
| Unlabelled | a uniform keyed draw of the other scorable accounts of that Monday: 15 per positive and at least 500. Hidden mules can be drawn; identified mules never | the number of accounts it was drawn from divided by the number drawn, so each Monday's draw stands for that Monday's scorable population |

- The draw takes the accounts with the lowest keyed blake2b hash of (account id, Monday), so
  it is uniform and nests: when a few accounts join or leave the pool, the rest of the draw and
  their cached rows stay. Hash outputs are pinned by a golden test.
- Each label's weights are then divided by their mean over all training batches, once, since
  the loss divides each label's weighted sum by its count in the batch. Without it a 26-date
  mule pulled 2.4 to 3.9 times as hard as a 1-date one.
- `mule prepare` reports the per-Monday ratio of positive to unlabelled weight, positives and
  weight per Monday (if identified mules cluster in some months, seasonal spending could act
  as a clock), and the identified active share. It refuses a selection world with fewer than
  5 identified active mules.

### Selection world

Scored whole: every scorable account on every Monday. Only `pu_label` is read, as a bank
would: the positives are the identified mules active that Monday, and identified mules'
inactive Mondays are left out of the ranking. Hidden mules are ranked like anyone else.
The answer key never chooses the model: if every true mule were scored in selection, the
truth would pick the epoch and prior.

### Test world

Scored whole: every scorable account on every Monday. Every true mule is scored on every Monday
of its activity and on its early-warning Monday, beside every normal account.
The answer key is read only after scoring. True mules on Mondays outside their activity are
left out of the rankings, being neither mule nor clean; their scores before onset count as
early warning and after activity are reported apart.

### If whole-world scoring is too slow

`mule bench` decides on the 30,000-person worlds, before any 300,000-person world is
generated. If a fallback is needed it is written per world:

- **Selection world:** every identified mule on its active Mondays, plus a uniform keyed sample
  of every other scorable account, hidden mules included at the same rate, each weighted by the
  inverse of its chance of inclusion, so that no sampled account stands for more than about 10.
  The answer key is never read.
- **Test world:** every true mule on its active and early-warning Mondays, plus a uniform
  weighted sample of the rest under the same bound. Evaluation builds this list after
  selection is frozen and fetches it through the fetch code, which never imports evaluation;
  the cache stores rows only, never why an account was chosen. One account's subgraph and
  score never depend on its batch, so the list cannot change a feature or a score.

Scoring one Monday in four was rejected: most mule lives are under 30 days, so it would catch a
short-lived mule's active Monday with chance about 1 in 4. The old 2,000-account sample,
each standing for about 60 accounts, made PR AUC fragile.

## The queries

Each query has one job, and its reason is what it extracts that the model needs (the owner:
"each query must have a dedicated scope of what behavior we want to extract from the dataset
and pass into the model. you need justification for each query as to why we need it"). The
lookback is 90 days and no query reads clock time.

| Query | Job | Runs | Prints |
|---|---|---|---|
| `list_accounts` | the scorable accounts at t | once per (world, Monday) | account id and owner id |
| `read_labels` | the identified mules and their activity period | once per world | accounts with `pu_label` 1 and their `mule_onset_ms`, `mule_last_ms` |
| `read_answer_key` | the truth | evaluation only; fetch and train refuse it | `is_mule`, `mule_ring_id`, the activity period, `opened_ms` |
| `read_hubs` | the accounts, devices and IPs too connected to expand | once per (world, Monday) | account hubs, device and IP crowds and hubs, each with its measure |
| `read_world` | the loaded world's identity | once per world | counts and sums (see [Cache](#cache)) |
| `profile_hubs` | the distributions behind the hub and crowd thresholds | `mule bench` only | histograms |
| `read_money_flow` (B1) | velocity | per (world, Monday, account), cached | the velocity columns, and one row per tracked inflow lot |
| `read_counterparties` (B2) | who pays it and whom it pays | per (world, Monday, account), cached | the counterparty columns |
| `read_activity_change` (B3) | the account against its own past | per (world, Monday, account), cached | the activity columns |
| `read_payment_events` (B4) | the payment graph | per scored account | the subgraph's accounts with `is_external`, payment edges and their fields, ownership edges |
| `read_shared_identity` (B5) | the device and IP graph | per scored account; node rows cached per (world, Monday, node) | customer, device and IP rows; enrolment, owner-enrolment and seen edges |
| `read_community` (B6) | how the account's network formed | per scored account | the community block |

B1 to B3 run on every account of every subgraph, not only the scored one, so a neighbour's
behaviour reaches the score. They are computed on demand for the accounts the batches
need and cached per (world, Monday, account), so an account shared by many subgraphs costs
once. A hub that is not the scored account gets a flag-only row and no B1 to B3.

### Shared rules

**Hubs and crowds** (`read_hubs`, one rule for every account):

| Kind | Rule | Examples |
|---|---|---|
| Account hub | more than 200 distinct customers own the other accounts of its payments in `[t - 90 d, t)`; ownerless counterparties are not counted; an account with at most 200 payments in all time skips the test, since it cannot pass it | billers, merchants, ledgers |
| Crowd | a device or IP with more than 8 distinct customers in the lookback, registered or not: those with a visible enrolment link to it plus the owners of deposit accounts that exist at t and sent a payment seen on it in the lookback | carrier networks, public terminals |
| Device or IP hub | more than 200 such customers | |

- Leaves are the account hubs and the crowds (device and IP hubs among them). A leaf stays in
  the subgraph as a flagged node with its kind and measure, so it never looks like an empty
  account (the old identical stubs filled 6.4 of a root's 16 slots). It is never expanded,
  never counted as sharing and never in a community. The scored account is always expanded
  and keeps its full row, even when it is a hub.
- Why: otherwise one biller or carrier IP floods every subgraph, and "shares an IP with 5,000
  people" means nothing. 200 sits above the largest ring PhantomLedger makes (at most 150
  members); 8 sits above
  households (2 to 6) and below public terminals (at least about 15 users) and carrier lines
  (about 90 to 128). The crowd counts registered and unregistered customers alike: once
  every observed endpoint has usage edges, terminals and carrier lines count as registered,
  and expanding them would make about 8,000 sharing pairs per carrier line.
- Both thresholds are checked against the bench histograms on development worlds built after
  PhantomLedger's households and ring timing steps, counting customers, never accounts, and
  frozen before any reported run. On development worlds, read from the answer key in
  evaluation: no positive is an account hub, and the share of ring devices and IPs that are
  leaves on their ring's active Mondays is near 0.

**Sharing a device or IP.** A customer shares a device or IP that is not a leaf when it has a
current enrolment link to it, or an account it owns (visible ownership) sent a payment seen on
it in `[t - 30 d, t)`. Its sharing start there is the earliest of its link starts and its first
seen use in the lookback, each counted as `t - 90 d` when unknown. Two **co-users** began
sharing a node at the later of their two starts, and began sharing at all at the earliest of
those over the nodes they share. A pair, a link or a customer is new over w when its sharing
start is at or after `t - w`. Co-users are counted as customers before any cap, because each
owned account copies its owner's links and a person holds several accounts. Thirty days keeps
a ring's links in view after its last payment.

### `list_accounts`

Prints the scorable accounts of t with their owner, a page at a time by strictly increasing
id. Examples must come from accounts that existed at t, or the future leaks in. It prints no
`opened_ms`: a date of record in every world's cache is one edit away from a feature.

### `read_labels`

Prints every account with `pu_label` 1 and its `mule_onset_ms` and `mule_last_ms`, paged by id,
and refuses a row unless `0 < mule_onset_ms <= mule_last_ms`. It is the only training signal
and the only reader of `pu_label`; it prints the dates only through a selection filtered on
`pu_label` 1, so a hidden mule's dates never leave the graph. Evaluation also reads it, for the
test world's groups.

### `read_answer_key`

Under `gsql/evaluation/`. Prints every internal account's `is_mule`, `mule_ring_id`,
`mule_onset_ms`, `mule_last_ms` and `opened_ms` (for the group of accounts opened within 90
days of t), paged by id. Only the evaluation package imports its reader, and the fetch executor
refuses to run it. Evaluation needs the activity dates of every mule, hidden ones included, to
build its groups and early warning.

### `read_hubs`

Prints the Monday's account hubs, crowds and device and IP hubs with their measure (distinct
customers over 90 days), under the rules above. Fetch passes the hub lists to B4 to B6 as
`SET<VERTEX>` parameters. A hub needs to be known before a subgraph is built, and the test
reads a hub's whole lookback, so it runs once per Monday rather than once per request.

### `read_world`

Prints what identifies the loaded world, for the cache key and the lookback check (see
[Cache](#cache)). It reads no label.

### `profile_hubs`

For `mule bench` only: exact whole-world histograms of distinct customers per account, device
and IP at a 90-day lookback, devices and IPs split into registered and observed-only, so the
owner can set the two thresholds.

### `read_money_flow` (B1): velocity

**Job.** Match each account's recent money in to the money out that followed it, last in,
first out, and say how fast it left (the owner: "money going in vs coming out - for this
behavior that gives us velocity, we need to figure out how to give this in as a feature and
how to encode this as a time").

**Mule behaviour.** A mule forwards 90% to 95% of what arrives and keeps 5% to 10%. By value,
in PhantomLedger 0.30 leaves within 15 minutes, 0.57 within an hour and 0.95 within a day; in
RUSI's UK bank data 0.28, 0.53 and at least 0.85, by transfer 0.72, card 0.18 and cash 0.10.

**Ordinary look-alike.** A direct-deposit split moves 10% to 35% of a salary 5 to 30 minutes
later; paycheck-to-paycheck spending leaves over days; bills go right after payday.

**Why a query, not the GNN alone.** Matching money in to money out walks hundreds of payments
in time order; attention over 16 sampled edges cannot do it.

**How velocity reaches the model.** Both as a dwell curve per account and as a dwell per
incoming payment. The per-payment dwell puts the timing on the event itself, so the GNN sees
which payer's money moved and how fast; the curve gives the account's whole speed in a few
numbers. Both encode velocity as time, not as a ratio: ratios of money out to money in over
windows lose order, and a saver paying a big bill at month end would look like a mule.

**Last in, first out.** An outflow spends the money that arrived most recently, which is
what passing money through means. First in, first out would say victim money "waited" until
the owner's old salary was used up, and hide the pass-through.

**How it runs.** The newest 256 inflows and the newest 256 outflows of the last 30 days, within
a whole-second span (exact walk in [The features](features.md#account-node-309-columns)).
Inflows of one second form one lot, so same-second order never matters. Only outflows before t
count: an inflow received less than w before t shows only what left by t, and the payment's
age on its edge tells the model how much of its curve it could see. On development
worlds, a check confirms that busy ordinary accounts' card spending does not fill the short
dwell shares; if it does, the dwell shares count transfer and cash outflows only.

**In GSQL.** The owner chose to write the walk in TigerGraph: "we need to try b". It is
plain GSQL, with no user-defined function and no subquery: local scalars, FOREACH and WHILE in
POST-ACCUM over ListAccums an earlier SELECT filled, heaps ordered (ts DESC, id DESC) with
explicit capacities, and a suffix-minimum early stop. Lot k occupies cents `[L, L + c)` of a
running level and still holds `max(0, min(lowest level since k, L + c) - L)` cents; each fall
of that minimum inside its band is a piece that left at that outflow's time. A global OrAccum
records a heap that did not iterate in order, and the query then stops with a status. A port
of this sketch matched full-history last-in, first-out matching in 20,000 random worlds with no
mismatch. The Python reference is an independent explicit stack of lots written from the
specification. The walk is compiled and verified on a live TigerGraph; its worst account (an
account later emptied by a sweep, up to about 32,896 steps; a retaining mule about n + m) is
measured by `mule bench`.

**Fields** (definitions in [The features](features.md#account-node-309-columns)):

| Field | Why |
|---|---|
| `in_amount_7d`, `_30d`, `_90d`; `out_amount_7d`, `_30d`, `_90d` | the scale of the flow; a mule moves victim-sized sums in and out within days, a salary account over a month, and 7 against 90 days shows the change |
| `flow_present` | tells "nothing came in" from "everything left after 7 days" |
| `left_15m`, `left_1h`, `left_6h`, `left_1d`, `left_3d`, `left_7d` | the velocity curve: a mule's jumps near 1 within a day, a salary split's sits at 0.1 to 0.35, ordinary spending leaves over days |
| `held_share` | a mule keeps a small cut; savers and salary accounts hold most |
| `median_dwell` | the curve's centre as a time: seconds for a pass-through, weeks for spending |
| `left_by_transfer`, `left_by_card`, `left_by_cash` | onward transfer and cash-out against ordinary card spending |
| per incoming payment `dwell_tracked`, `left_24h`, `dwell`, on its B4 edge | which payer's money moved and how fast; replaces "delay to the receiver's next outflow", which matched any unrelated purchase |

Transfer means `ach`, `check`, `internal` and `zelle`; `unknown` counts in the denominator
only.

### `read_counterparties` (B2): who pays it and whom it pays

**Job.** Count the account's payers and payees, how new they are, and where its money goes.

**Mule behaviour.** Fan-in from first-time payers: victims, about 2.6 a burst and up to 30.
Fan-out to one fixed onward account (a pinned primary; a secondary with chance 0.12) in pieces
of about USD 660.

**Ordinary look-alike.** A landlord (the same tenants monthly), a business (many payers by
card), family transfers, a new job's payroll (one new external payer), a new tenant (one new
payer a month).

**Why a query, not the GNN alone.** "How many new payers in 24 hours" is a count, and attention
averages rather than counts.

**Internal and external payers.** A bank knows which payers are its own customers, and
it matters more now that 0.6 of onward value leaves the bank. New payers are split by the
payer's `is_external`. The dossier pins the share of victims at the same bank, or the split
would learn a generator choice. "New" means the first payment in the lookback: a payer older
than 90 days that pays again reads new, for everyone alike.

| Field | Why |
|---|---|
| `payers_30d`, `payees_30d` | breadth: fan-in from many victims, fan-out to few onward accounts; a landlord or small business also has many payers |
| `new_internal_payers_7d`, `_30d`; `new_external_payers_7d`, `_30d` | victims are first-time payers; the internal and external split is what a bank knows |
| `new_payees_7d`, `_30d` | new onward accounts; a new merchant or biller is a hub and shows in `hub_payee_share_30d` |
| `new_payer_share_30d` | a mule's money comes from strangers, a landlord's from repeat tenants |
| `max_new_payers_24h` | a burst of victims, against a seller's sale day or a group collection |
| `top_payee_share_30d` | a mule forwards to one fixed account; a saver's sweep to its own savings shows as the same customer through the ownership edge |
| `hub_payee_share_30d` | ordinary money goes to merchants and billers, mule money to people |

### `read_activity_change` (B3): the account against its own past

**Job.** Compare the account's recent activity with its own previous weeks.

**Mule behaviour.** An account becomes active suddenly: in PhantomLedger's adopted activity
model each mule account has one active period per ring, with victims arriving minutes to hours
apart within a session and about every two days across the period (renewal gap median 2 days,
kept to 10 minutes to 21 days); 0.695 of mule lives are under 30 days.

**Ordinary look-alike.** Dormancy wake-ups (about 35% of accounts a year), holiday spending,
payday bill runs.

**Why a query, not the GNN alone.** It compares the account with its own past, which needs all
of its 90 days, not a capped sample. The model reads the change from 7 days against 90 as a
difference of logs.

| Field | Why |
|---|---|
| `in_7d`, `in_30d`, `in_90d`; `out_7d`, `out_30d`, `out_90d` | payment counts, whose 7-day against 90-day difference shows a sudden start |
| `burst_in_1h`, `burst_in_24h`, `burst_out_1h`, `burst_out_24h` | victim sessions and a mule's pieces, against payday bill runs; windows anchored at payments, never clock hours |
| `shortest_in_gap` | the seconds-level rhythm of victim payments, against split salary credits |
| `since_last_in`, `since_last_out` | how long it has been quiet: bursts against dormancy |

**Account newness from behaviour.** Ninety days of payments can show that an account is
new (every payer new, `in_90d` equal to `in_30d`) though `opened_ms` is never read. That is
behaviour a bank sees, not the first-seen time the owner ruled out, so it stays. Scoring only
accounts older than 90 days would lose the short-lived, purpose-opened mules the owner called
most important. PhantomLedger's lifecycle keeps the cue as strong as in reality and no
stronger: a gate holds the `is_mule` lift of "no payment in the first 60 days of the lookback,
some after" under its ceiling (today it is a 37 to 85 times marker against 5 to 10 times in
reality).

### `read_payment_events` (B4): the payment graph

**Job.** Build the payment edges around the scored account, one edge per payment, each with
its own timing.

**Mule behaviour.** Order and pairing: a victim pays (a first contact), the mule forwards
within minutes, an onward account collects; paths run victim to mule to batch-mate to
organizer.

**Ordinary look-alike.** Ordinary money chains (parent to student to school), rent (a repeat
payment), a new merchant (a first contact, usually a hub).

**Why a query, not the GNN alone.** These are the edges the GNN passes messages over. Money
passed on within 15 minutes, a burst from new payers and a victim paying in escalating amounts
exist only at the level of single payments; one edge per pair with totals would erase them.

**Structure.** The scored account's newest 32 payments each way; each expanded counterparty's
newest 16 each way (hubs not expanded); up to 8 other accounts of the scored account's
customer, most recently active first. The full composition of an example's subgraph is in
[The features](features.md#what-one-examples-subgraph-holds).

| Field | Why |
|---|---|
| `amount`, `rail` | size and kind of payment: victim-sized credits, onward pieces of about USD 660, Zelle and ATM cash-out against card spending and ordinary P2P |
| `age` | recency and order of the events around the account |
| `pair_prior_count`, `pair_gap`, `pair_age` (its absent bit is "first in pair") | first contact against an established relationship: victims are first contacts, the onward account a repeat; rent is a repeat |
| `sender_gap`, `receiver_gap` | rapid repeat payments and a mule's fast pieces, against a business's steady flow |
| `dwell_tracked`, `left_24h`, `dwell` (from B1) | velocity on the event itself |

### `read_shared_identity` (B5): the device and IP graph

**Job.** Build the customer's devices and IPs, those seen on payments, and who else uses each
node and since when, as enrolment, owner-enrolment and seen edges with customer, device and IP
rows.

**Mule behaviour.** Controller-operated accounts, about 12% of mules, hold the ring's device
and IP from shortly before the first ring payment (median lead 1 day). A ring device nobody
registered still shows on members' payments: a holder lacks it on file with chance 0.28 for a
device and 0.39 for an IP. Controller-opened accounts under stolen or synthetic identities run
from the controller's devices.

**Ordinary look-alike.** Households of 2 to 6 share devices and IPs, with old links that
started together; carrier networks and public Wi-Fi are crowds; a new phone, a new browser or
a move adds a new link.

**Why a query, not the GNN alone.** The device and IP edges plus who else uses each node and
since when. Counting a customer's co-users across all its devices, and when each pair began
sharing, reads nodes and links the subgraph does not hold.

**Both kinds of link.** Enrolment (the customer registered it) and seen (it appeared on
the account's own payments), because a ring device no member registered still shows on their
payments.

**Every account gets owner-enrolment edges.** Each account in a subgraph gets edges to
its customer's kept devices and IPs, so an account with no payments that shares a device with
an active mule reads the mule account's behaviour within two layers: the owner's Account Z
(2026-10-14), which must be scored differently from a payment-less account that shares
nothing. The alternative, a third layer, was rejected.

**Link ends.** `active`, `since_end` and `held`, agreed earlier in the session, are dropped,
and the "now" counts become 30-day counts: PhantomLedger plans a ring tenure's end from the
ring's future payments, so they read the future. `since_last_seen`, the time since the
customer's accounts last used the node on a payment, replaces them with observed evidence.

**Deposit-only co-user accounts.** Accounts reached through a shared device or IP are deposit
accounts only, one candidate per customer, deduplicated by customer before the caps.
The cap of 16 other customers per node was dropped: a node with more than 8 customers is a
crowd and never expanded, so a non-leaf node has at most 7 other customers.

| Where | Field | Why |
|---|---|---|
| enrolment edge (customer and owner-enrolment) | `since_start` (only with a known start) | ring tenures are recent; links made at sign-up date the customer, not behaviour |
| enrolment edge | `since_last_seen` | a controller's device on file is used in bursts; the observed stand-in for the dropped link end |
| seen edge | `uses_90d`, `since_first_use`, `since_last_use` | a ring device nobody registered still shows on members' payments, used briefly and recently; a new phone looks alike but has no co-users |
| device or IP node | `customers_30d`, `customers_90d` | how shared it is; households share small numbers, carrier lines and terminals are leaves |
| device or IP node | `new_customers_7d`, `_30d` | ring tenures arrive staggered and recently; household members arrive together and long ago |
| device or IP node | `accounts_seen_30d`, `payments_seen_30d` | activity on the endpoint |
| device or IP node | `observed_only`, `crowd`, `hub` | a ring device nobody registered; the kind of leaf |
| customer node | `accounts_owned` | a recruited mule's second account or a purpose-opened one, against small-business and secondary accounts; counts open and closed accounts alike, since counting open ones only would rebuild closure |
| customer node | `devices_30d`, `ips_30d` | how many endpoints a customer uses; busy families |
| customer node | `new_devices_30d`, `new_ips_30d` | a ring handover adds a device or IP shortly before the first ring payment, against a new phone or a move |
| customer node | `co_users`, `new_co_users_30d` | its sharing across all its devices in one number; rings form fast, households are old |
| customer node | `co_users_both` | a ring shares a device and an IP. In PhantomLedger today only rings do, which is a shortcut; once households land, families share both too and it must separate a family from a ring |

### `read_community` (B6): how the mule network formed

**Job.** Find the cluster of customers around the scored account's owner, as of t, and measure
its size, density, how recently its links formed, the money moving inside it and how cut off
it is. It answers the owner's "formation of mule networks" and "clustered connections with
isolated communities" (2026-10-16).

**Mule behaviour.** A ring is a small, dense cluster formed recently that pays itself; its
victims and onward accounts surround a mule account through payments.

**Ordinary look-alike.** A long-standing household: old links, mostly paying the world.

**Why a query, not the GNN alone.** How a cluster formed is a property of the cluster, beyond
the GNN's two hops. It joins the scored account at the model's head and is ablated.

**Members.** Every step is computed as of t, and a customer belongs to the first round that
reaches it:

1. Round 0 is the scored account's owner.
2. Round 1 adds customers that share a non-leaf device or IP with it.
3. Round 2 adds customers that share one with a member kept in round 1.
4. The payment round adds the owners of non-hub internal accounts that paid, or were
   paid by, an account of a kept member in the last 30 days.

Candidates of each round are kept newest first, ties by customer id, until the community holds
64 members. A sharing candidate's time is the latest, over the shared (node, member) pairs, of
the later of the two sharing starts; a payment candidate's time is its latest such payment. A
member's accounts are those that exist at t with a visible ownership link, open or closed,
leaving out account hubs.

**The payment round** was added because the device and IP rounds see a ring only through its
endpoints, which about 88% of mules do not hold; for them the block would read like a customer
who shares nothing, and a large ring's device can become a crowd. The owner: "just ensure
that it is a meaningful feature". So the block is checked on development worlds by the
[feature audit](#the-feature-audit) (a column with no separation is researched or dropped),
by the "without community" ablation, and by a report of the share of mules, by whether they
hold a ring endpoint, whose block differs from that of a customer who shares nothing.

**Never a library algorithm.** Connected components, Louvain and label propagation run on the
whole graph with no as-of time, so future ring members would leak into training, and hubs and
crowds would merge strangers. The demo query `demo_account_web` also reads `is_mule`.

| Field | Why |
|---|---|
| `members`, `capped` | size, and whether the cap cut it |
| `shared_nodes`, `density` | a ring is small and dense |
| `new_link_share_30d` | rings form fast, households are old |
| `internal_amount_30d`, `isolation` | rings pay each other, families mostly pay the world |
| `member_payments_30d` | shows when `isolation` has an empty denominator |

### Query plumbing

- Every query prints a version: a hash of its normalised text. The model file and the manifest
  record them; fetch refuses a version with no passing parity record.
- Every per-account query refuses invalid parameters and duplicate, unknown or invisible
  (not existing at t) accounts with a status row. `list_accounts`, `read_labels` and
  `read_answer_key` page by strictly increasing id.
- Every PRINT of a vertex set is a projection: a bare `PRINT Accounts;` once leaked every
  attribute while all 160 tests passed.
- Per-request state is kept in accumulators keyed by the scored account, never in a vertex
  flag two scored accounts of one request could share.
- Work counters are printed only because `mule bench` reads them; the reference graph does not
  mirror them and parity ignores them.
- Each query is checked offline by `gsql-lsp check` against `gsql/schema/` as it is written,
  and passes a compile probe on the TigerGraph 4.2.5 workspace before the Python above it is
  built. None has been compiled on a server yet.
- The executor keeps the rebuild's failure classes with an outage budget, one shared backoff,
  finite timeouts, HTTP status raised before pyTigerGraph hides it, POST with orjson, and
  reinstalls changed queries only. Requests are sized from the measured 90th percentile of
  bytes per account to stay under half of TigerGraph's 64 MB response limit; a request refused
  for its size or timing out is halved, down to one account, and one account still failing is
  an error, never a gap. Accounts per request (32 by default) is a transport setting that never
  changes a result.

## Encoding

| Kind | Encoding |
|---|---|
| Counts | `log(1 + n)` |
| Amounts | `log(1 + dollars)`, from integer cents |
| Shares | as they are, in `[0, 1]`; 0 when the denominator is 0, with another column showing the empty denominator |
| Flags | 0 or 1 |
| Rail | one-hot over `ach`, `card`, `cash`, `check`, `internal`, `unknown`, `zelle` (a `Zelle_Transfer` is `zelle`) |
| Durations | a presence bit plus 64 Fourier numbers: `u = log1p(seconds) / log1p(10 years)`, 32 log-spaced frequencies from 0.125 to 2 cycles per unit of u, sine and cosine. An absent duration is bit 0 and 64 zeros. |

The basis resolves a second (u = 0.035), a minute (0.21), an hour (0.42), a day (0.58) and 90
days (0.81). It is invertible, which is why durations measured to t carry the clock of their
event (see the Clock guard). The cache and the built batches keep raw integer milliseconds;
the GPU encodes them each step. The `unknown` rail column stays because PhantomLedger's rail
fixes make it honest, and a reported run refuses a world that fails those gates. On an
earlier task the Fourier encoding added little on its own (average precision 0.829 against
0.823), so no gain is credited to it without the per-payment timing ablation.

## Leak rules and how tests enforce them

Leaks are absolutely not allowed. Each guard below is named once; [the
features](features.md) cites them by name for every feature.

| Guard | Rule | Enforced by |
|---|---|---|
| Labels | Only `read_labels` reads `pu_label`, `mule_onset_ms` and `mule_last_ms`, printing the dates only for `pu_label` 1; only `read_answer_key` reads `is_mule` and `mule_ring_id` and prints `opened_ms`. No feature query reads any of them, and no feature says whether any account is known or caught. Labels and weights never join a node row. | erasing every label column leaves every query's answer identical; projection-only PRINTs; text tests fail any other reader; the fetch executor refuses `read_answer_key`; import contracts |
| Selection by labels only | Selection reads `pu_label`, never the answer key, in every scoring mode. | with the answer key erased, every selection input is byte-identical |
| Before t | Values come only from payments with `t - 90 d <= ts < t` and links with `valid_from_ms < t`; an account exists only when `opened_ms < t`. | fixtures with a payment, a link start and an account opening exactly at t, a link end exactly at `t - 90 d` and at `t - 30 d`, and a closure exactly at t; text tests that require one exact allowed form for every bound; an account alone equals itself among 31 others |
| No link ends as values | `valid_to_ms` appears only in the visible and current tests, both at least 30 days back; `closed_ms` only in the population test. These are decided by the past only while no ring tenure is quiet for more than about 26 days, so a gate keeps PhantomLedger's renewal gap at most 21 days and tenures built from attempted payments. | a fixture of two holders with identical pasts, one with a later ring payment, giving identical rows at t; text tests allowing only the two forms |
| Lookback | Every window and duration lies inside `[t - 90 d, t)`, which with 90 days of history is always inside the export, so no value reaches the data start, a clamped date or the burn-in boundary. | `mule prepare` checks the first Monday against the first payment, never the first join |
| Known starts | A link start or first seen use is used only when known. | fixtures: a joiner and a long-standing customer with the same behaviour read the same; a pre-data customer reads the same on two Mondays far apart |
| No dates of record | `opened_ms`, `joined_ms` and `closed_ms` decide existence, the population and the known-start test, never a value or an order; no token is read; caps order by activity. | a perturbation test moves every date of record and link start before `t - 90 d` by keyed amounts that keep each on the same side of every bound it meets, on the reference graph and in live parity, and requires every answer byte-identical; a test that the batch builder never reads `opened_ms` |
| No identification closure | Identification closes nothing in PhantomLedger, so no end, silence or missing account is caused by a catch. | a caught neighbour reads exactly like an uncaught one |
| Clock | No column holds a clock value. Durations measured to t (payment `age`, `since_last_in`, `since_last_out`, enrolment `since_start` and `since_last_seen`, seen `since_first_use` and `since_last_use`) carry their event's clock time, since t is always Monday 00:00, and a monthly payment in the subgraph can fix the day of month. | a reported run refuses a world unless PhantomLedger's Clocks gates passed (`mule prepare`) and the clock probe of `mule audit` (a classifier on the residues of every such duration modulo 60 seconds, an hour, a day and a week) predicts `is_mule` no better than chance, its ROC AUC interval containing 0.5 (`mule train`); until then results are development only |
| Order-free ties | Same-second payments have no order (see [Time](#time)). | a fixture swapping same-second ids changes no B1, B2, B3 or B6 value; the loader refuses payment ids that follow event time |
| Same rule for all | Caps, orders, hub handling and subgraph shape are identical for positives, unlabelled draws, neighbours and accounts scored in selection or test; one query with one set of settings builds all of them; thresholds are tuned on development worlds only, and on reported worlds a check only verifies and a failure stops the run. | the cache key holds query versions and settings; the contract lock |
| No label conditioning | Graph-structure features are computed identically for every account whatever its label. | erasing the labels (above) |
| Evaluation never feeds back | Time to first flag and every other evaluation output never become features (the owner: "dont sent this time to the model"). No query, data or training code imports evaluation, and nothing reads evaluation's results back. Gate results (`mule audit`'s clock probe, lift gates and feature audit) are pass records holding only pass or fail, query versions, settings and world identity; `mule train` reads them, so they can stop a run but never change an input. | import contracts (the XGBoost package included) |
| Gates | A value is only as honest as PhantomLedger's behaviour. | a reported run refuses a world unless every gate mapped to a feature passed (see [Gates](#gates)) |

**Other tests.** Every query ignores a self-payment; two payment-less accounts, one sharing a
node with a busy mule, get different subgraphs and scores (Account Z); about 60 one-line
mutants and a planted-slip harness (seven planted slips once passed all 16 text tests) are run
in every review; every printed field is an edge end, a feature or a presence flag; unknown
categories are refused; the reference-world fixtures include the behaviour catalogue's
look-alikes (direct-deposit splits, self-transfers, dormancy wake-ups, landlords, family
chains).

**Leak paths found and how each is closed.**

| Path | Concrete case | Closed by |
|---|---|---|
| A label in a feature query | the rebuild's bare `PRINT Accounts;` | Labels guard |
| A date of record leaving a query | `list_accounts` printing `opened_ms` into every cache | `opened_ms` only in `read_answer_key` |
| The answer key in selection | a fallback scoring every hidden mule at weight 1 against 10 for others | Selection by labels only |
| An event at t | a ring tenure starting exactly on Monday 00:00 | Before t |
| A link end read against t | two ring-mates last paid on 3 March; one resumes on 20 March; on 16 March one read active, the other ended | No link ends as values, while its gate holds |
| A future account in the subgraph | a co-user's fallback account opened at t + 3 days | every account node passes `opened_ms < t` and visible ownership |
| A date of record as a value | a backdated join makes a clamped link's age measure the calendar | Known starts, No dates of record |
| A first seen use dating the customer | a joiner's first payment makes a new co-user pair | the known first-seen-use rule |
| Account newness rebuilt from behaviour | an account opened 20 days before t | a bank sees it; its strength is gated |
| Closure or silence from a catch | a caught mule's links end and payments stop | No identification closure |
| Clock phase | a `since_last_out` of 16 to 17 hours is a Sunday 07:xx outflow, and only mules make night ATM withdrawals | Clock |
| Same-second ties by row order | whether a victim edge shows a 9-minute dwell followed the generator's row order | Order-free ties |
| Generator cues | a Zelle credit from a first-time payer and an internal move of USD 662 to the same owner 9 minutes later is mule-only today | Gates |
| Settings tuned on reported worlds | raising the hub threshold after seeing world C's collector | Same rule for all |
| Payments a bank cannot see | an external neighbour's payments to another external account | the loader refuses them |

## The model

| Part | Decision |
|---|---|
| Node types | account, customer, device, IP |
| Relations | 16: payment (`pays`), ownership (`owned_by`), device and IP enrolment (customer to node), device and IP owner-enrolment (account to node), device and IP seen on payments (account to node), each with its reverse |
| Layers | two `HeteroConv` layers of one `TransformerConv` per relation, summed over relations, reading each edge's fields (every Fourier-encoded duration included) as edge attributes; `owned_by` and its reverse have none |
| Sizes | hidden 64, 4 heads of 16, dropout 0.2 |
| Inputs | account 309 columns, customer 11, device 12, IP 12; payment edge 401, enrolment edge 130, seen edge 131 ([widths](features.md#widths)) |
| Degrees | every node carries the log of its number of edges per incoming relation in the example's subgraph, because attention's softmax averages over neighbours and cannot count them |
| Head | the scored account's final 64 numbers joined with its 8-number community block: Linear(72, 32), ReLU, Linear(32, 1) |
| Inductive | no per-account embedding and no memory; no batch-statistics layer, so a score never depends on its batch |

An example is one (account, Monday) with its own copy of its two-hop subgraph, so no node is
shared between examples and a node reached by two paths appears once, which keeps cycles
visible. The ablations are switches on built batches ([exact
rules](features.md#ablation-and-comparison-switches)).

## Training and selection

- **Loss.** nnPU with balanced weighting (imbalanced nnPU with a balanced target, Su, Chen and
  Xu 2021). The textbook weight collapsed: the loss settled at the prior by the fourth epoch,
  and PR AUC was 0.001 on the earlier NVIDIA runs. The positive risk keeps each mule's weight
  of 1 divided by its active Mondays; the negative-risk correction uses unit weights over
  positives, because a hidden mule enters the unlabelled draw once per active Monday. The loss
  for a positive uses `1 - sigmoid(f)` for float32, and a test drives the negative risk below
  zero.
- **Prior.** The share of hidden active mules among the unlabelled is unknown, as in a bank.
  The grid is 1, 3 and 10 times the identified active share measured in the training world
  (identified active mules divided by scorable accounts, averaged over Mondays). The grid is
  run on the development rotation; if the three are within noise, the prior is fixed at 1
  times for reported runs and that is stated, saving two thirds of selection training. It is
  never set from PhantomLedger's identified-share report, which reads the truth.
- **Settings.** Adam, learning rate 0.001, weight decay 0.0001, 128 examples a step (the
  rebuild's setting), at most 30 epochs, patience 5.
- **Selection.** After each epoch the selection world is scored whole. The score is the mean
  of the recall of identified active mules within the top 1% and within the top 5% of each
  Monday's ranking, pooled over Mondays (mules found on all Mondays divided by positives on
  all Mondays), using `pu_label` only. Each seed keeps the prior and epoch with the best
  score; a later epoch or prior must be strictly better. Pooled recall never fails on a
  Monday with no positive.
- **Seeds.** Five model seeds, 0 to 4, vary only the initial weights, dropout and batch order;
  the draw seed is a separate setting and stays fixed, since the three worlds vary the data.
- **No refit.** With separate worlds the selected model is final.
- **Which model scores.** `mule score` uses the seed with the best selection score, never
  chosen by test; its test metrics are reported beside the mean and spread over seeds.
- **Determinism.** Batches are built once, shuffled across Mondays and labels, and reused;
  per-step reseeding and deterministic flags; PyG's scatter softmax is checked under
  `torch.use_deterministic_algorithms` on CUDA. A per-epoch checkpoint is written atomically
  so a run can resume.
- **Contract lock.** The model file holds every query version, every layout with its count
  transform and Fourier basis, the community, degree and ablation flags, and the settings
  (lookback 90; caps 32, 16, 8, 8 devices and IPs, 48; time-ordered lists 256; community 64;
  hub 200; crowd 8). `mule score` refuses a model built for other query versions or settings
  and names the fields that differ.

## Comparisons

Every ranker trains on the same examples, weights and labels as the GNN and is scored on the
same Mondays and groups. They read the features of the fetched subgraphs, so they see exactly
what TigerGraph gives the GNN.

| Ranker | What it reads | Question it answers |
|---|---|---|
| XGBoost, own behaviour | the scored account's own row (50 numbers, durations as presence and seconds) | what graph learning adds at all |
| XGBoost, graph features | the same plus its customer row (8), its owner's device and IP rows aggregated by mean and maximum, and its community block (8) | what message passing adds beyond hand-made graph features |
| GNN | everything | the model |
| GNN, scored account alone | its own row and own payment edges; every other node zeroed, no other edge, no community | what neighbours add |
| GNN without devices and IPs | no device and IP nodes or relations; customer rows keep `accounts_owned` only; no community | what shared identity adds |
| GNN without community | head Linear(64, 32) | whether two hops already see the ring |
| GNN without per-payment timing | every payment-edge duration and dwell field zeroed | what the temporal encoding adds |

- XGBoost replaced gradient-boosted trees at the owner's request ("just train it against
  industry standard xgboost"). It learns identified against unlabelled with each label's
  weights rescaled to equal totals: when labels are selected completely at random, a classifier
  of labelled against unlabelled ranks like one of positive against negative (Elkan and Noto,
  2008). It uses XGBoost's default settings, seeded by the model seed, with the number of
  rounds chosen on the selection world by the same selection score, as the GNN chooses its
  epochs.
- Each ablation trains and selects its epoch like the full model, at the prior the full model
  chose for that seed, with the same five seeds.
- A difference is called shown only when its paired, ring-resampled interval excludes zero;
  an ablation whose interval straddles zero gets more seeds.
- What to expect: in the old runs a model without message passing matched the GNN in the top
  1% (35% against 34%) once counts were given, and the old Fourier ablation was null. With a
  ring device on about 12% of mules, the device and IP ablation may move little; the share of
  positives with any ring-device link is reported beside it. Plan for a close result, and fix
  PhantomLedger's shortcuts first, since a shortcut favours the simple models.

## Evaluation

Evaluation is the only code that reads the answer key. It scores the test world from the cache
after selection is frozen.

### Rankings

On each test Monday the ranking holds every scorable account except true mules outside their
activity. Positives are the true mules active that Monday; the rest are non-mules. A mule is
**flagged** on a Monday when it ranks within that Monday's top 1% or top 5% (two budgets,
tie-aware). Scores are ranked as raw logits in float64, since a float32 sigmoid tied every
logit above about 17.

### Metrics

| Metric | Definition |
|---|---|
| Recall and precision in the top 1% and 5% | per Monday and pooled over Mondays; these lead the report |
| ROC AUC, PR AUC | over all ranked account-Mondays, with ring-resampled intervals |
| Share ever flagged while active | per mule, flagged on any active Monday |
| Time to first flag | days from onset to the first Monday it is flagged while active |
| Early warning | the share of mules flagged on their early-warning Monday, the one Monday with `mule_onset_ms - 7 days < t <= mule_onset_ms`, against that Monday's thresholds; also among mules whose ring link is visible that Monday. Reported as conditional on PhantomLedger's researched lead law, since tenures with no payment are not generated |
| After activity | flags on Mondays after `mule_last_ms + 7 days`, reported apart |
| Distinct accounts alerted | over the scored year at each budget, beside the share ever flagged, which otherwise rewards lists that churn |

Lift is never printed beside recall at the same budget. In the old runs about 90% of PR AUC
came from hidden mules ranked above every sampled normal account, which is why the top of the
list leads.

### Groups

Each group is ranked against non-mules with the other groups removed, as an investigator
removes known cases:

- mules in fully hidden rings (no member identified): the headline;
- unidentified members of rings that surfaced;
- identified mules (`pu_label` 1 in the test world; the model never saw them either);
- mules with no ring (account-takeover drops and impostor payees);
- mules whose account opened within 90 days of t (from `read_answer_key`'s `opened_ms`):
  purpose-opened mules are the hard case;
- mules scored with no payment in the lookback (on an early-warning Monday, or an account
  opened for a ring and not yet used), which only shared devices and IPs can reveal: the
  owner's Account Z, reported beside the enrolment ceiling (72% of device and 61% of IP
  associations are on file);
- recall by hop distance from the nearest identified mule.

### Uncertainty

A ring bootstrap within each world resamples whole rings with all their account-Mondays, each
ringless mule as its own cluster, and whole non-mule accounts with all their Mondays, since an
account's 52 Mondays are correlated. Comparisons use paired replicates. The spread over the
five seeds and the three rotations is reported beside it.

### The missed-mule report

The owner: "if not, we need to investigate why. maybe theres a bug, or we are missing
some features that need to be added." For every true mule never flagged in the top 5% on any
of its active Mondays, evaluation writes its decoded feature rows on each active Monday, its
subgraph (nodes and edges with their fields), its group, its ring, and how each ring-mate
scored and whether it was identified. Each missed mule is explained as a bug or a missing
feature before the next run. The report is written to the run's results and read by nothing
else.

### The feature audit

On development worlds, `mule audit` (evaluation code, reading the cache and the answer key)
measures each feature's univariate separation of active mules from non-mules over scored
account-Mondays: its ROC AUC against `is_mule`, with a ring-resampled interval. Every column
of the scored account's row, its customer row, its own device and IP rows (aggregated by mean
and maximum), its own edge fields (likewise), its degree columns and its community block is
audited.

- A feature with no separation (its interval contains 0.5) is researched or dropped.
- A feature that separates almost perfectly (ROC AUC at least 0.95, or at most 0.05) is a
  suspected leak or shortcut and stops the run: `mule train` refuses a reported run until a
  development audit of the same query versions and settings passes.

On development worlds the same command also checks that no positive is an account hub, that
the share of ring devices and IPs that are leaves on their ring's active Mondays is near 0,
and that busy ordinary accounts' card spending does not fill B1's short dwell shares.

### Identification bias

Victim reports may identify more often the mules whose victims bank here, so
`new_internal_payers_*` could learn who gets caught rather than what mules do.
Evaluation reports identified against hidden mules by victim origin: the internal share of
their new payers over their active Mondays, from their own B2 rows, and recall of hidden mules
by that share. It also reports how well the features separate identified from hidden mules,
and the correlation of the selection score with recall of all mules per prior and epoch,
reported, never used to select.

## The pipeline

**Where each step runs.** PhantomLedger, then `tf_gnn_loader_v2`, then
MulePatternLearner, one command per repository for generation and loading. TigerGraph is the
MulePatternLearner workspace on TigerGraph Cloud that the repository's `.env` points to
(host, graph name `Mule_Pattern_Learner`, secret); `TransactionFraud_GNN` lives in a different
workspace and is never touched. The 5,000-person test runs from this laptop; the
300,000-person worlds are generated and loaded from the NVIDIA training machine, which also
fetches and trains.

1. **Code** reaches the training machine by git: pushed to `main` after the owner's code
   review of all three repositories, then pulled.
2. **PhantomLedger** generates a world.
3. **tf_gnn_loader_v2** loads it. It pins pyTigerGraph below 2 while MulePatternLearner runs
   2.0.4, so the two keep separate environments on the training machine.
4. **`mule install`** installs the queries; **`mule parity`** checks them on a 5,000-person
   world against the reference graph and records the versions that pass, on every machine
   that fetches, before its first `mule fetch`.
5. **`mule prepare`** reads the world's identity, checks the lookback, the gates and the
   labels, and writes the training examples.
6. **`mule fetch`** fills the cache. Worlds are loaded one at a time into the one graph, and a
   world is cleared only after everything each of its roles needs is cached.
7. **`mule audit`** runs the checks that need the answer key and writes pass records.
8. **`mule train`** trains from the cache and never queries TigerGraph.
9. **`mule evaluate`** scores the test world from the cache and writes the results.

### Cache

- One folder per (world, Monday) holds the population, that Monday's labels and the hubs, and
  one Parquet table per query: one row per node or edge, every field named as in [the
  features](features.md). Parquet tables can be opened and checked by hand.
- Each table is a set of part files written once (a temporary file, then rename) with an index
  of their accounts; fetch computes only missing accounts. Only 00:00 UTC dates are accepted,
  and cached rows are reparsed with the live checks.
- The key, held in the path so other settings never overwrite, is the world's identity, the
  query versions and the settings, plus a label identity (a hash of `read_labels`' output) for
  the label tables.
- **World identity** (`read_world`): counts per vertex and edge type and per value of
  `account_type`, `payment_rail` and `is_external`; second sums of `joined_ms`, `opened_ms`,
  `closed_ms`, each link type's starts and ends and each payment type's event times; cent
  sums per payment type; per edge type the sum of squared source degrees and the sum of the
  product of both ends' degrees (one link moving between vertices of degrees 2 and 1 leaves
  the squared sums unchanged); the first join and the first and last payment. All are
  integers. A world regenerated or changed has another identity, so stale rows are never read.
  The lookback check uses the first payment, since backdated joins make the first join years
  earlier than the data.
- The cache, PhantomLedger's exports and `.env` are never in git.

### Results

`results/<run>/` holds the result tables, the model files and a manifest, and is never
committed: the owner zips it on the training machine and copies it to Downloads on this laptop.
`.gitignore` stays as it is.

The manifest records the code commit and whether the tree was dirty (a reported run refuses a
dirty tree: the NVIDIA results came from a commit with 71 files differing), the query versions,
the world identities, the GPU name and count, the CUDA, torch, PyG and XGBoost versions, wall
time per stage, TigerGraph calls and cache hits. Every result is written through a temporary
file and a rename, and each command appends to an `events.jsonl` with the error that stopped
it. Results are tables and files, no charts. There is no baseline paragraph from the old
NVIDIA numbers, which came from a pipeline with known leaks.

### The demo

When the time comes, the TigerGraph workspace is wiped. Only the two demo queries that show an
account's network and its timeline (`gsql/demo/demo_account_web.gsql` and
`demo_account_timeline.gsql`, the owner's untracked files) are ported to the new schema, and
the demos are redone. They read the answer key and are never a training input.

## The 5,000-person test on TigerGraph Cloud

The MulePatternLearner workspace is its own TigerGraph Cloud workspace, and the repository's
`.env` points only to it. Once PhantomLedger, the loader and MulePatternLearner are built:

1. Run `CLEAR GRAPH STORE -HARD` and drop every query, loading job and schema object in that
   workspace.
2. Generate the 5,000-person world with PhantomLedger and load it with `tf_gnn_loader_v2`
   (one command per repository), using the same graph name and secret.
3. Install and run every query, the live B1 verification and the reference-graph checks.
4. Run `CLEAR GRAPH STORE -HARD` again and tell the owner.

If after the wipe the `.env` secret no longer authenticates, stop and ask the owner. The owner
then generates and loads the 300,000-person worlds on the NVIDIA machine and trains there,
after the code review.

## Commands

Commands take no options: worlds, roles, dates and settings live in config.

| Command | What it does |
|---|---|
| `mule install` | installs the changed queries of `gsql/queries/` and `gsql/evaluation/` |
| `mule parity` | runs every installed query on the loaded world and the reference graph on the same export, on a seeded sample of Mondays and accounts with mules oversampled; compares the canonical answers and refusals, and that an account fetched alone equals itself among 31 others; records the versions that pass under `results/parity/` |
| `mule prepare` | reads the world's identity, checks the lookback and, for a reported run, the gate records; reads the populations and labels; writes the training examples and weights; reports counts, the identified active share, the identified share by onset quarter and the weight balance per Monday |
| `mule fetch` | fills the cache for every example the configured roles need: the training draw, and every scorable account of every Monday of a selection or test world |
| `mule audit` | evaluation code, on the loaded world from the cache: the clock probe and the lift gates that need the answer key; on development worlds also the feature audit and the hub, leaf and card-spending checks; writes pass records |
| `mule train` | refuses a reported world without passing records; trains the GNN over the prior grid and five seeds with selection after each epoch, the ablations and the two XGBoost comparisons |
| `mule evaluate` | scores the test world and writes the metrics, groups, uncertainty, missed-mule report and identification bias |
| `mule score` | scores the accounts and dates set in config with the model of the best-selection seed |
| `mule bench` | measures cost on the loaded world and writes `results/bench/` |

Every long command first runs a read-only readiness check: installed versions, the world's
identity, one account fetched, one training step.

## Scaling and the benchmark

**Size.** A 300,000-person world has about 1.6 million accounts and about 46 million payments
in any 90 days. Whole-world scoring is about 23 million account-Mondays (about 2.3 million at
30,000 people); at the old rate of about 80 subgraphs a second that is 80 to 90 hours per
300,000-person world. Each is fetched once and cached, after which scoring is a cheap GPU
pass. A training Monday needs a few thousand scored accounts and their neighbours.

**Bounds.** The caps are the sampler: deterministic, newest first, so an account's subgraph is
the same in every run. Leaves keep billers, ledgers, carrier lines and terminals from pulling
in thousands of strangers (the card-interest ledger takes about 100,000 postings a month at
300,000 people), and every time-ordered list reads at most 256 payments. B6 costs at most the
kept members times the nodes each shares times 200, plus 30 days of the members' payments.

**`mule bench`** runs on worlds of 30,000, 100,000 and 300,000 people and writes one table
per world to `results/bench/`:

- fetch time and response size per query and request, with work counters, for accounts drawn
  in training proportions, positives reported apart (a collector's subgraph is the largest,
  about 2,000 hop-2 accounts), at 1, 2 and 4 concurrent requests;
- B1's worst account and its steps;
- subgraph sizes by type, community sizes, cache bytes per account and in total;
- TigerGraph's memory in use and install time per query (the old full install took about 50
  minutes, and one install timed out after 5,402 seconds);
- parse and build time per batch, GPU step time and peak memory, examples per second;
- `profile_hubs`' histograms and the share of positives that are hubs, which must be 0;
- a projection of a whole-world scoring run, against its fallback;
- whether co-user rows are cheaper per Monday for every customer or on demand; the rule is
  the same either way.

Whole-world scoring is settled against its fallback on the 30,000-person bench before any
300,000-person world is generated. The README's recommended training architecture follows
from the bench tables and is updated from them: whether built batches stay on the GPU or in
pinned host memory, the request size and the parallel requests. PhantomLedger took about 63
minutes and 13.7 GB for 200,000 people over 366 days; 300,000 people over two years was
extrapolated at 3.2 hours and 215 GB.

## Package layout

| Path | Holds |
|---|---|
| `config.py`, `paths.py` | frozen, validated settings (worlds, roles, Mondays, caps, thresholds, seeds; the lookback must exceed the 30-day "new" window); output locations |
| `tigergraph/` | connection, client with retries, executor, installer, query names, versions and row parsers, the world identity |
| `data/` | the Monday calendar, the population, the labels reader, the keyed draws, the training examples and weights |
| `subgraph/` | fetch and the cache; rows to tensors, degree columns and ablation switches |
| `model/` | the Fourier time encoding, the heterogeneous attention model, the nnPU loss |
| `training/` | the trainer, selection, the prior grid and seeds, the contract lock, scoring with a saved model |
| `comparisons/` | the two XGBoost comparisons |
| `evaluation/` | the answer-key reader, test scoring and the fallback list, metrics and the ring bootstrap, groups, the missed-mule report, the feature audit, the clock probe, identification bias |
| `testing/` | the reference graph, reference worlds and fixtures, parity, the planted-slip harness and the version restamper |
| `bench.py`, `cli.py` | the benchmark; the commands |

GSQL: `gsql/schema/schema.gsql`; `gsql/queries/` (`list_accounts`, `read_labels`, `read_hubs`,
`read_world`, `profile_hubs`, `read_money_flow`, `read_counterparties`, `read_activity_change`,
`read_payment_events`, `read_shared_identity`, `read_community`); `gsql/evaluation/`
(`read_answer_key`).

The import contracts in `pyproject.toml` keep every layer exhaustive: no package but
`evaluation/` imports the answer-key reader, and none of `tigergraph/`, `data/`, `subgraph/`,
`model/`, `training/` or `comparisons/` imports `evaluation/`. The README says to run
`pip install -e` after a pull, and pytest filters the torch_geometric deprecation warnings
(about 14,700 a run) for that module only.

## Order of work

1. This design and [the features](features.md), for the owner to read before any code.
2. In parallel, each step implemented, reviewed and fixed:
   - PhantomLedger: the shortcut fixes, activity model, lifecycle, identification and burn-in,
     each in its citation ledger, with the research dossiers moved into its `docs/research/`;
   - `tf_gnn_loader_v2`: the new contract, every exported field loaded;
   - MulePatternLearner: first `git revert` of dc975ee, d8bcf02 and c6376eb, which implement the
     superseded October 17 design (history keeps them reachable), back to the rebuild of
     fc63ad9, whose executor, installer, reference graph, contract lock, nnPU fixes and
     metrics are kept; then module by module against the reference graph and the export
     contract.
3. The 5,000-person test on TigerGraph Cloud.
4. The owner reviews all three repositories; then the push to `main`.

## Limitations

- The worlds are synthetic, however well cited, and the features were designed knowing the
  generator.
- Much of the mule research is British, for a simulated US bank.
- Identification is not completely at random, so nnPU's assumption does not hold exactly;
  evaluation measures the effect against the full answer key.
- `pu_label` has no false positives and no identification through shared devices, both unlike
  a real bank's labels.
- A mule's onset and last activity come from PhantomLedger's truth, so they are sharper than an
  investigation's reconstruction. Event time equals knowledge time; a bank would need arrival
  times to match.
- Label noise: organizers' accounts that ring money reached are mules, and recruits no money
  reached are not.
- `mule_ring_id` is the ring of the first landing payment, and about 6% of mule persons serve a
  second ring.
- A ring device appears on only about 12% of mules, so shared identity is a strong signal for a
  minority, and money flow carries the rest.
- Early warning is conditional on PhantomLedger's declared lead law.
- Dwell near t is censored: an inflow less than w before t shows only part of its curve.
- Account newness read from behaviour is real, and only as strong as PhantomLedger's
  lifecycle makes it.
- Results stay development only until PhantomLedger's Clocks gates pass and the clock probe
  shows chance.
- Weekly scoring flags a short burst up to six days after it ends.
- Each world has few identified mules (about 750 a year at 300,000 people), so the intervals
  matter and selection is noisy.
- Token and Address are not read; device and IP are the identity signal.
