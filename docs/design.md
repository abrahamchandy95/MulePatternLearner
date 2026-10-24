# MulePatternLearner design

The specification the code builds. Its decisions were agreed with the owner in the grilling
session of 2026-10-17, and it replaces the October 17 revision of this file (commits 5637d9c
and ed62329), which was written before that session. Where three reviews of this design found
a choice the owner had not made, the design stated a safe interim rule and listed the choice
under [Pending the owner](#pending-the-owner). The owner settled most of them later the same
day, and those decisions are written in below; what is still open stays listed there. Nothing
here is built yet.

- [The features](features.md) is part of this design: every number that reaches the model in
  a batch, with its exact definition, encoding, width, leak guard and the generator gates it
  depends on. This file says why each query and field exists; that one says exactly what
  each is.
- [Future design work](future_design.md) holds what the owner has raised but not decided.
- The research behind the decisions lives in PhantomLedger's `docs/research/` folder: the
  generator shortcut dossier (`generator_shortcuts_2026-10-17.md`, 63 shortcuts with their
  fixes, its implementation plan and the citation checks), the realism and burst dossiers
  (`generator_realism_2026-10-15.md`, `mule_bursts_2026-10-15.md`), the mule activity research
  (`mule_activity_2026-10-17.md`) and the checked customer lifecycle research
  (`lifecycle_2026-10-17.md`).
- [The GSQL dwell research of 2026-10-17](research/gsql_dwell_2026-10-17.md) and its check,
  with the checker's corrections applied, are in this repository's `docs/research/`.

## Standing rules

| Rule | Where it comes from |
|---|---|
| Leaks are absolutely not allowed. | The owner |
| TigerGraph and GSQL are the main deliverable and Python the additional one. TigerGraph computes every value the model reads from the graph; Python joins rows by id, encodes them, builds tensors, and counts the edges of each built subgraph for the [degree columns](features.md#degree-columns), which exist only once a batch is assembled and an ablation switch has run. The owner agreed to that one exception on 2026-10-17. | The owner: "our whole deliverable is TigerGraph and GSQL with the python as an additional deliverable"; "we want as many things to be calculated in TigerGraph as possible and exported to the GNN" |
| Query names start with a verb. | The owner's standing rule, confirmed on 2026-10-17 |
| Commands take no options; dates and settings live in config. | The same |
| GSQL computes nothing that nothing reads. | The same |
| Every table and every column PhantomLedger exports is loaded, so the schema holds every exported column and the export holds nothing the schema lacks. | The owner: "when I look at the uploaded data, i dont want to see missing fields in the load data page" |
| PhantomLedger gets no new flags. It simulates the whole universe of behaviour, and a use case only decides what its label means. | The owner: "we are defining the behavior of the mule usecase now"; "we code in behavoir of the whole universe" |
| Every behaviour in PhantomLedger is backed by research, or declared as a choice, in its citation ledger. | The owner: "every line in phantomledger is research backed" |
| PhantomLedger keeps every researched behaviour, in every use case. A shortcut is fixed by making the behaviour's implementation realistic, never by deleting the behaviour. | The owner (2026-10-17): "The cover payments and all the other things that phantomledger makes are behaviors! we do not want to remove these"; "in the universe, it does not matter what we are trying to label, the behaviors of people would be always the same" |
| One branch, `main`, the same on both remotes; nothing is pushed without the owner's confirmation; `redesign` is merged into `main` at that push. | The owner's standing rule, confirmed on 2026-10-17 |

## What we are testing

A temporal heterogeneous graph neural network learns the behaviour of money-mule accounts
from what accounts do over time and how they are connected: their payments with their speed
and order, their owners, and the devices and IPs they share with other customers. On every
Monday it ranks a bank's accounts by how likely each is to be acting as a mule.

- **Behaviour, not detection.** The owner's framing (2026-10-16): a model trained today can
  learn from a period long past, whose identified mules are all known by now, so "the date
  of discovery adds unnecessary complexity". Training uses a period whose identified mules
  are all in, and nothing reads when a mule was found. "All in" needs identification to see
  past the last scored Monday (see [Identification horizon](#identification-horizon)).
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

- A mule is every account a ring recruited as a mule, whether or not ring money reached it,
  and every internal account ring money landed in, whatever started the money's chain:
  scam victims' money, an organizer's own funds (the bipartite, cycle and invoice
  playbooks) or drug and structured cash. Organizers' own accounts that ring money landed
  in are mules too. The label does not depend on victims, which belong to the
  transaction-fraud use case (the owner, 18 October 2026: "victims are for transaction fraud
  (a different use case). we are currently interested in mule detection usecase!!").
- A recruit that no ring money reached is still scored: it is linked to other mules, and
  that graph behaviour is what the model should learn (the owner: "even if they dont
  necessarily have many transactions, they will be connected to other mules, and we will get
  some graph behavior from them").
- Account-takeover drops and impostor payees become mules with no ring (`mule_ring_id` -1)
  once PhantomLedger's scam payees step lands.
- PhantomLedger also decides which mules the bank identifies (see
  [Identification](#identification-as-phantomledger-defines-it)).

### Worlds and calendar

| Item | Decision | Reason |
|---|---|---|
| Worlds | Three independent worlds from different seeds: one trains, one selects, one tests. Reported runs rotate the roles (A/B/C, B/C/A, C/A/B); development uses one rotation. | Every test account, ring, device and IP is new to the model, which is "mule accounts created after the GNN was trained", with no split machinery. Rotation gives three test numbers and a spread across worlds. |
| Sizes | Development worlds have 30,000 people, reported worlds 300,000. Development worlds use other seeds, never those of A, B or C. | Thresholds and caps are tuned on development worlds only, from label-free statistics or the selection score (see the Same rule for all guard), and frozen before the reported worlds are generated. A change made after a reported run, whatever prompted it, needs new reported worlds from new seeds before the next report, and the manifest records which run's report led to which change. |
| Burn-in, representative start and label tail | PhantomLedger simulates 730 days per world, defined in the use case, not a flag: 275 burn-in days, not exported, then the 455 exported days. After the export it simulates a 90-day label tail, also not exported, which only identification reads. Laundering runs throughout. Rings start mid-life at the burn-in's start: the rings alive on the first simulated day are drawn from the stationary law (a representative start). MulePatternLearner reads only the 455 exported days. | The burn-in fills the mule-level history: earlier ring payments in a lookback, open tenures, balances. It does not by itself make the ring count steady: rings born at a constant rate and living about 730 days on average, started cold, would reach about 0.39 of their steady-state count by the first scored Monday and about 0.63 by the last, a rise that would put more positives late in the year. With the representative start the weekly active-ring count is flat from the first simulated day, which PhantomLedger's stationarity gates check. The label tail lets identification see whole mule lives (see [Identification horizon](#identification-horizon)). |
| Export window | 455 days: 90 days of history only, then 365 scored days holding 52 Mondays. Laundering may run from day 0. There is no laundering-free warm-up. The window may grow if research shows a need. | Every scored Monday has a full 90-day lookback, for mules and ordinary accounts alike, and nobody loads a year nobody reads (the owner: "we would have our window of 365 days + 90 days"). |
| Scored Mondays | Config holds one export start date, 2024-01-01 (day 0, PhantomLedger's `--start`, a Monday), the same for every world, and each world's fixed seed and size. The burn-in then runs from 2023-04-01, the export to 2025-03-30 and the label tail to 2025-06-28. The scored Mondays are the 52 Mondays at 00:00 UTC from the first Monday at or after day 90, 2024-04-01 to 2025-03-24; `mule prepare` derives them and checks them against the loaded world. | The calendar is the same in every world. Which calendar does not matter, since no feature reads the clock, except that every exported time must fit the schema (see [The export contract](#the-export-contract)). |
| Scoring frequency | Weekly, on Mondays, everywhere. Daily scoring is a later option once the benchmark shows its cost (see [future design work](future_design.md)). | A mule active Tuesday to Thursday is still scored the next Monday with its whole burst in view. |

### What PhantomLedger must provide

Each item is research-backed or declared in PhantomLedger's citation ledger, follows the
shortcut dossier's implementation plan as amended by the owner's answers, and is checked by
a gate (see [Gates](#gates)).

| Area | What the world must do |
|---|---|
| Shortcuts | Every one of the dossier's 63 shortcuts that a model could exploit is fixed with research-backed behaviour or declared. The owner: "get rid of all the shortcuts". A shortcut is fixed by making the behaviour's implementation realistic, never by deleting the behaviour (see [Standing rules](#standing-rules)): cover payments stay in every use case and go to the holder's own contacts, billers and merchants on the holder's own clock, as a mule acting like a normal account would make them; an account its controller operates whole (controller-opened, or dormant then controlled) gets its cover from the controller, on the controller's clock and devices, its P2P paid to a P2P platform; ordinary people get second jobs at the BLS rate, and whether ring members keep the cover's second payroll at its own rate is open with the owner in PhantomLedger's plan; and the ring budget cut, a volume control of the generator rather than a behaviour, gives way to the fixed declared ring rate the owner chose, so every planned ring payment is made. Every residual shortcut gets a world check, and every ceiling no research sets is declared at the value PhantomLedger's plan suggests (see [Gates](#gates)). |
| Mule activity | One active period per account per ring; a second episode only through a second ring (0.06 of mule persons). Victims over the account's life are the sum of today's per-burst draws, so credits per mule stay about the same. The first victim arrives at onset plus the recruitment lag; later victims after a renewal gap LogNormal with median 2 days and sigma 1.3, kept to [10 minutes, 21 days], all declared choices. The active period belongs to the account and is shared by every playbook. Long quiet belongs before the first misuse. No source supports quiet gaps of weeks to months between bursts in one account (`mule_activity_2026-10-17.md`, committed as PhantomLedger 77e0c61). |
| Ring devices and IPs | They belong only to controller-operated accounts (online handover, organizers' own ring accounts and controller-opened accounts), about 18% of mules instead of 70% (12% before controller-opened accounts are counted). A tenure starts at the first attempted ring payment minus a lead (median 1 day, at most 14 days) and is built from attempted payments. A controller who logs in and never pays out is not generated, since no research gives a rate. |
| Usage edges | Every observed device and IP, carrier networks, public terminals and hotspots included, gets usage edges at the declared coverage; `read_hubs` marks the big ones as crowds. |
| Link tenures built from uses | Every `Party_Uses_Device` and `Party_Uses_IP` tenure that PhantomLedger builds from uses (observed payments, or a ring's attempted payments), ring and ordinary alike, starts a new tenure wherever two consecutive uses are more than 21 days apart and ends at most four days after the start of its last use's day (the end of that day plus a tail of 0 to 3 days). Tenures are derived from every simulated use, burn-in included, never from exported rows only. Every other tenure ends at a real event of its own (a new phone, a move, a closure). A world check proves it on every party link table. Without it a link's end is set by uses after t: a hotspot used on 1 January and 1 June would read current on 1 March. |
| Onward money | A declared 0.6 of mules' onward value goes to accounts at other banks, following RUSI's direction. The share of victims at the same bank is pinned in the dossier. |
| Identification | The mechanism of [Identification](#identification-as-phantomledger-defines-it), written as `pu_label` only. Identification closes nothing, and it sees past the last scored Monday (see [Identification horizon](#identification-horizon)). |
| Customer lifecycle | One research-backed change: relationship starts backdated years before the data by the customer's age (the owner: a world of 2030 to 2031 must not have its first account opened in 2030); each account's own opening date; small-business and other secondary accounts; device, IP and token starts; ordinary openings and closures all year; mule account origins; no exported row before its account opens. No date before 1970 (the owner: "we do not need data before 1970"): backdated relationship, account and link starts are drawn from their distributions truncated at 1970-01-02 by redrawing, never by clamping, so nothing piles up at 1970 and very few of a reported world's 300,000 people get starts that early. Small-business accounts are deposit accounts (`account_type` `deposit`), so a business account used as a mule is scored; no new account type is added. |
| Lifecycle numbers | New to the bank: at least 0.033 of customers a year, a floor (NAFBS 2024: 3.3% of US adults have held their primary account under a year; 0.014 at 60 and over), by age, so that about 0.8% or more of customers joined in the last 90 days on any date. Closures from ordinary churn and death only: charge-off about 0.06 of accounts a year, switch-outs 0.02 to 0.03 a year. Small-business accounts for 0.045 of adults (adults only), mostly owners without employees, with owner transfers in both directions. One mule account per customer at one bank, a second with chance 0.02 (UK 2025: 1.02 accounts per offboarded customer). |
| Mule origins | Three: the mule's own established account; an account the person opens for the purpose; and, new, accounts a controller opens under a stolen or synthetic identity (about 0.10 of mule accounts, declared, below Experian's bound of about 0.23), run from the controller's devices, with legitimate look-alikes (new, quiet, single-device customers) so that "new and quiet" is not a mule-only marker. |
| Device and IP link lives | The phone keeps its 40 months; a web device is keyed on a cookie or login token (its life uncited); home IPs are a two-class mixture of about 0.4 static lines and spells of one to four months, so 30% to 40% of home-IP links start inside a 90-day lookback. Legitimate new links must be common, or "new link" becomes mule-only. |
| Export | The contract under [The graph](#the-graph). Payment ids are exported as BLAKE2b pseudonyms like entity ids, not row numbers, and within one second their order is independent of the laundering flag and the ring. No self-payments. Links made at sign-up start exactly at `joined_ms` (PhantomLedger's own check; MulePatternLearner's [known-start rule](#time) no longer depends on it). A NaN amount fails the export. |
| Identity and gate records | For every world, PhantomLedger writes two tables in PostgreSQL schema `mule_temporal_gates`: `world_record` (the PhantomLedger commit, a clean-tree flag, the code-gate certificate, the world's seed, size and calendar, and the world fingerprint computed from the 22 exported tables) and `gate_record` (for every gate, its entry id, kind (code or world) and pass, never its value or bound). The fingerprint's fields are those `read_world` shares with it, under `read_world`'s key names (see [Cache](#cache)). The loader refuses a world without both, checks the fingerprint against the tables it loads, and after a passing verification writes them as `world_<seed>.identity.toml` and `world_<seed>.gates.toml` into the folder MulePatternLearner's `.env` names (`MULE_PL_RECORDS_DIR`). Values and bounds stay in PhantomLedger's run summary, which no MulePatternLearner code opens. See [Gates](#gates). |
| Clocks | The Clocks step (no rows stamped at second 0, no same-second couplings, victim, mule and organizer clocks like their holders' own, legitimate night ATM use, the account-takeover clock, legitimate rails on every day of the month) must pass before a reported run, because durations measured to t carry the clock of their event. |
| Rails | Card cash-outs on rail `card` and own-account moves on rail `internal`, so that the `unknown` rail is honest. |

Data for PhantomLedger's research: the Federal Reserve's 2025 Diary of Consumer Payment Choice
(`dcpc-2025-public-csv.zip`, 2.9 MB, downloaded with the owner's approval into the research
folder's `data/dcpc_2025/`) anchors the hour-of-day profiles. The RBIH mule data on Kaggle is
optional and the owner's to download. For the lifecycle tenure fit, the NAFBS 2024 public-use
file and its data dictionary, and the SCF 2025 full public data and its codebook, were
downloaded with the owner's approval into the research folder's `data/` (`nafbs_2024/`,
`nafbs_2024_dictionary/`, `scf_2025/`, `scf_2025_codebook/`), and are read only as data.

**Expected label counts.** Ring births are calibrated so identified mule customers come to
about 25 per 10,000 customers per scored year: about 75 in a 30,000-person world and 750 in a
300,000-person one. The identified share of true mules is then measured and reported, not set.
Development worlds test the pipeline; priors, epochs and features are not chosen from them
except where this design says so.

### Gates

A value is only as honest as PhantomLedger's behaviour. [Generator gates each feature depends
on](features.md#generator-gates-each-feature-depends-on) maps every dossier entry to the
features its cue reaches, and names the gates the features add. Every gate reads the truth
somewhere, so only evaluation code reads a gate's result, under the rule agreed with the
owner that "nothing reads evaluation's output back". `mule evaluate` writes a reported result
only when, for each of the rotation's three worlds:

- PhantomLedger's gate record for that world matches the world `mule audit` read from
  TigerGraph on the fields they share (see [Cache](#cache)), comes from a clean tree, and
  shows every code gate of its commit and
  every world check mapped to a feature as passed;
- `mule audit` passed on it: its label checks, the clock probe and the identified share by
  onset month;
- the [feature audit](#the-feature-audit) passed on the development rotation for the same
  query versions and settings.

Otherwise its results are marked development only, as on development worlds. The operator
runs `mule audit` on each world before `mule train`, and no training code reads its records.
The one stated exception to the rule above, agreed by the owner on 2026-10-17: a failed audit
may stop `mule train`. Before any training code runs, the `mule train` command reads the pass
or fail of each audit record of its rotation's worlds, and nothing else in them, and stops
when a failure is one that stops the run: on a reported world any failed check (see the Same
rule for all guard), and on a development world a failed label check or a failed feature
audit. Training code itself still imports nothing of evaluation.

**Ceilings.** A lift or distance gate has a ceiling declared before any development world is
generated: from research where it exists (the newness lift: 10, the top of the 5 to 10 times
band found in reality), and otherwise a declared bound, which the owner set on 2026-10-17 at
the value PhantomLedger's plan suggests; every such ceiling is gated. A development world must
meet a ceiling, never set it. Each gate belongs to one repository: the
lift gates on generator behaviour are PhantomLedger's world checks, defined once in
[the features](features.md#gates-the-features-add); the clock probe, the feature audit and the
label checks are MulePatternLearner's.

## The graph

### The export contract

PhantomLedger's mule use case exports exactly these 22 tables, and `tf_gnn_loader_v2` loads
every table and every column of them. One UTC millisecond clock, 0 for none. Each link holds
from `valid_from_ms` until `valid_to_ms` (0 while open), and `DISCRIMINATOR(valid_from_ms)`
keeps repeated links of one pair. Every event edge repeats its payment's `event_ts_ms`.

Every time stays UINT milliseconds in all three repositories (the owner: "unsigned integer is
fine"), so every exported date must be after 1970-01-01 00:00 UTC, and no data before 1970 is
needed. Untruncated tenure draws could reach before 1970, so PhantomLedger redraws every
backdated relationship, account and link start until it falls on or after 1970-01-02, at most
about 54 years before the export start of 2024-01-01: each start's distribution is truncated
there by redrawing, never by clamping (see
[What PhantomLedger must provide](#what-phantomledger-must-provide)). PhantomLedger's exporter
still stops on an earlier time, and the loader still refuses one, so such a world cannot load.

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
`payment_rail` is `ach`, `card`, `cash`, `check`, `internal` or `unknown`; `mule prepare`
refuses a world with any other value, from `read_world`'s counts per value. `is_mule` and
`pu_label` share one convention: 1, 0 for any other internal account, and -1 exactly on
external accounts. Every account has at most one owner and one ownership tenure; the loader
refuses otherwise, since a second tenure filled two device cap slots in the rebuild.

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

MulePatternLearner's `gsql/schema/schema.gsql` is the master copy of the DDL, for TigerGraph
4.2.5. Three copies exist and are kept identical by tests: the master; the loader's
`gsql/mule_temporal/schema.gsql`, a byte copy pinned to a MulePatternLearner commit; and
PhantomLedger's `schemas/mule_temporal.gsql`, also a byte copy, which its own tests compare
with the master at the same pin. In the master:

- each type's attributes are in the order of its contract table's columns;
- every attribute's default is written out: 0 for times and `amount`, false for
  `is_external`, the empty string for `account_type` and `payment_rail`, and -1 for
  `is_mule`, `pu_label` and `mule_ring_id`, though the loader leaves no attribute at its
  default;
- there is no loading job and no query that writes: MulePatternLearner writes nothing to the
  graph, and loading is the loader's job alone.

`mule-load push` creates the schema from its copy when the graph is empty and checks it exactly
otherwise. It writes each column into the attribute of the same name, from headerless PSV
shards, by a position computed from the column names.

Before loading, the loader refuses an export with: a repeated id; a dangling edge end; a
repeated link of one pair and start, which `DISCRIMINATOR(valid_from_ms)` would merge; a
payment without exactly one sender and one receiver; an event edge time unlike its payment's;
a payment outside the life of either of its accounts; a payment between two external
accounts; an account with two owners or two ownership tenures; label columns outside the
convention above; payment ids whose order follows event time. It reports, without refusing,
self-payments and links that start within a day after a join but not at it, since it cannot
tell a sign-up link from a later one; PhantomLedger's own world checks hold both to zero.

After loading, `mule-load push` verifies the graph against the export, and a world that fails
is not ready for MulePatternLearner:

- **Every attribute is filled from the export,** none left at a default: every column's counts
  and fingerprints in the graph, read in both edge directions, equal the export's, so no
  attribute in the schema was left unwritten and no exported column was left out.
- **The graph holds exactly the 455 exported days:** every payment's `event_ts_ms` lies in
  `[export start, export start + 455 days)`, with payments on every one of those days, and
  every link and account end is 0 or inside the window (PhantomLedger writes an end at or
  after the export's end as 0, and exports no payment with amount 0). The owner: "this is
  fine as long as we have 365 + 90 days worth of data in TigerGraph, ensure that this is the
  case with the loader".

### Who may read what

| Columns | Meaning | Read only by |
|---|---|---|
| `pu_label` | the bank identified the account as a mule (1), or not (0); -1 on external accounts | `read_labels` (training, selection, and evaluation's groups) |
| `mule_onset_ms`, `mule_last_ms` | the account's first and last ring payment: the activity period an investigation records. A recruit no ring money reached has its recruitment and its leaving the ring (the ring's end, the account's closure or its holder's death, censored at the export's end) | `read_labels`, which prints them only for `pu_label` 1; `read_answer_key` |
| `is_mule` | the truth: 1 mule, 0 not, -1 external (unknown) | `read_answer_key` (evaluation) |
| `mule_ring_id` | the ring of the account's first landing payment, else the ring that recruited it; -1 for none | `read_answer_key` (evaluation: ring groups and ring-resampled uncertainty) |
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
  against t, because PhantomLedger sets the end of every tenure it builds from uses by the
  uses that follow, a ring's planned payments included, so "still active at t" would say
  whether more uses are coming.
- The visible and current tests themselves are decided by the past only because no tenure
  built from uses is quiet for more than 21 days and none ends more than four days after the
  start of its last use's day, a ring tenure's lead counting as quiet, and 21 + 4 is under the
  30 days of the current test (see
  [What PhantomLedger must provide](#what-phantomledger-must-provide)). A tenure open across t
  then began, or had a use, within the 21 days before t, so it reads current from the past
  alone, and a tenure that ended before t is past.
- An account exists at t when `opened_ms < t`, and is open at t when it exists and
  (`closed_ms = 0` or `closed_ms >= t`). Openness is used only to choose the scored
  population.

**Known starts.** A date of record dates the customer or the data, never its behaviour:

- One rule decides every start s of a customer on a device or IP, whether a link's
  `valid_from_ms` or the first seen use in the lookback: s is **known** when
  `s > t - 90 d`, `s > joined_ms`, and an account the customer owns (visible ownership) sent
  a payment in `[t - 90 d, s)` that was not seen on that node.
- Why each part: a start before the lookback dates the data. A start at sign-up dates the
  customer. And since PhantomLedger builds links from observed uses, a joiner's first link
  starts at its first use, hours or days after `joined_ms`, so "after joining" alone would
  make every joiner's first device and IP read new, a direct function of the join date. An
  earlier own payment elsewhere shows the customer was already active, so the start is a
  change in behaviour rather than the start of the customer.
- An unknown start shows no time since start, never counts as new, and counts as `t - 90 d`
  wherever a sharing start is needed.

**Ties.** Payments in one second have no order, because PhantomLedger's row order sorts
by fraud flag and ring id within a second.

- Every count, window, burst, gap, B1 walk and community value is order-free: swapping the
  ids of same-second payments changes no B1, B2, B3 or B6 value.
- B1 walks the outflows of one second oldest first by the name of their rail (`ach`, `card`,
  `cash`, `check`, `internal`, `unknown`, `zelle`), then by payment id, so of two outflows of
  one second the one whose rail comes first spends the newest cents first. Only the split by
  rail depends on which outflow took which cents, and it must not follow an id; two outflows
  of one rail change no value whichever goes first, so the ids only break ties that change
  nothing.
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
reports the identified active share from `pu_label` alone. The identified share of true mules
needs the truth, so `mule audit` reports it, with the share by onset month that the
[identification horizon](#identification-horizon) gate holds flat.

### Identification horizon

Identification cannot see past the end of PhantomLedger's simulation, so without more days
mules active in the last weeks of the export would have fewer victim reports and traces in
view and be identified less often. Every world shares one calendar, so the positive rate would
fall at the same point of every training and selection year, and seasonal spending could tell
the model when that is. So PhantomLedger simulates a 90-day label tail after the export, not
exported, as it does the burn-in before it: cases rooted in exported payments can still be
reported and traced there, and identification sees whole mule lives. `mule audit` checks it:
the identified share of true mules by onset month stays flat across the scored year, each
month within its binomial band around the pooled share. A world that fails it gives
development-only results.

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

The checks split by what they read, so that data code never reads the truth (agreed with the
owner: `is_mule` is "the truth, read only by evaluation").

- **`mule prepare`** (data code, reading `read_labels` and `list_accounts` only) refuses a
  world where `read_labels` returns a row without `0 < mule_onset_ms <= mule_last_ms`, and
  reports every identified mule that is never in the scored population on a Monday of its
  activity.
- **`mule audit`** (evaluation code, reading `read_answer_key`) refuses a world where a
  `pu_label` 1 account has `is_mule` other than 1; an `is_mule` 1 account lacks its activity
  dates or a ring value (a ring id or -1); a scorable account lacks truth; or a truth join
  shares a column other than the account id. It counts `is_mule` 1 by `account_type` and
  ownership and reports every true mule outside the scored population on any of its active
  Mondays as a mule never scored there: the agreed population assumes every mule is an owned
  deposit account, and this checks it on every world.
- The loader and PhantomLedger already refuse the label columns outside their convention:
  `is_mule` and `pu_label` are -1 exactly on external accounts.

## Worlds, examples and weights

### The scored population

On each Monday t the **scorable accounts** are the internal deposit accounts open at t with a
visible ownership link: owned by a customer. Ring money moves between deposit accounts. Other
accounts (credit cards and brokerage accounts, which customers own too, the bank's ledgers,
other banks' accounts) appear only as neighbours; a payment to one's own card is a same-owner
move (the owner, 18 October 2026). `account_type` chooses the population and is not a feature: the rail on each
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
  as a clock; the [identification horizon](#identification-horizon) gate holds the cause
  flat), and the identified active share. It refuses a selection world with fewer than 5
  identified active mules.

### Selection world

Scored whole: every scorable account on every Monday. Only `pu_label` is read, as a bank
would: the positives are the identified mules active that Monday, and identified mules'
inactive Mondays are left out of the ranking. Hidden mules are ranked like anyone else.
The answer key never chooses the model: if every true mule were scored in selection, the
truth would pick the epoch and prior.

### Test world

Scored whole: every scorable account on every Monday. Every true mule is scored on every Monday
of its activity and on its early-warning Monday, beside every normal account.
The answer key is read only by evaluation code, once the world is cached. True mules on Mondays outside their activity
are left out of the rankings, being neither mule nor clean; their scores before onset count as
early warning and after activity are reported apart.

### If whole-world scoring is too slow

`mule bench` decides on the 30,000-person worlds, before any 300,000-person world is
generated. If a fallback is needed it is written per world:

- **Selection world:** every identified mule on its active Mondays, plus a uniform keyed sample
  of every other scorable account, hidden mules included at the same rate, each weighted by the
  inverse of its chance of inclusion, so that no sampled account stands for more than about 10.
  The answer key is never read.
- **Test world:** every true mule on its active and early-warning Mondays, plus a uniform
  weighted sample of the rest under the same bound. `mule audit` builds this list while the
  world is loaded, since the list depends on the truth and never on a model, and fetches it
  through the fetch code, which never imports evaluation; the cache stores rows only, never
  why an account was chosen. One account's subgraph and score never depend on its batch, so
  the list cannot change a feature or a score.
- Roles rotate, so a world cached for its test role later serves as a selection or training
  world. Training, selection and scoring therefore read only the keys they computed
  themselves (the training draw, the scored population), never the cache's index, and a test
  runs selection on a world after a test-role fallback fetch of it, with the answer key
  erased, and requires every selection input byte-identical.

Scoring one Monday in four was rejected: most mule lives are under 30 days, so it would catch a
short-lived mule's active Monday with chance about 1 in 4. A small sample of normal accounts,
each standing for many, makes PR AUC fragile, hence the bound of about 10.

## The queries

Each query has one job, and its reason is what it extracts that the model needs (the owner:
"each query must have a dedicated scope of what behavior we want to extract from the dataset
and pass into the model. you need justification for each query as to why we need it"). The
lookback is 90 days and no query reads clock time.

| Query | Job | Runs | Prints |
|---|---|---|---|
| `list_accounts` | the scorable accounts at t | once per (world, Monday) | account id and owner id |
| `read_labels` | the identified mules and their activity period | once per world | accounts with `pu_label` 1 and their `mule_onset_ms`, `mule_last_ms` |
| `read_answer_key` | the truth | evaluation only, once per world by `mule audit`; the fetch executor refuses it | `is_mule`, `mule_ring_id`, the activity period, `opened_ms`, `account_type`, whether the account has an owner |
| `read_hubs` | the accounts, devices and IPs too connected to expand | once per (world, Monday) | account hubs, device and IP crowds and hubs, each with its measure, and `observed_only` for devices and IPs |
| `read_world` | the loaded world's identity | once per world | counts and sums (see [Cache](#cache)) |
| `profile_hubs` | the distributions behind the hub and crowd thresholds | `mule bench` only | histograms |
| `read_money_flow` (B1) | velocity | per (world, Monday, account), cached | the velocity columns, and one row per tracked inflow, carrying its lot's values |
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
| Account hub | more than 200 distinct customers own the other accounts of its payments in `[t - 90 d, t)`, each of which exists at t; ownerless counterparties are not counted; an account with at most 200 payments in all time skips the test, since it cannot pass it | billers, merchants, ledgers |
| Crowd | a device or IP with more than 8 distinct customers in the lookback, registered or not: those with a visible enrolment link to it plus the owners (visible ownership) of deposit accounts that exist at t and sent a payment seen on it in the lookback | carrier networks, public terminals |
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
- One account rule serves the crowd measure, `customers_90d`, the sharing rule and the
  co-user candidates: a customer counts through an enrolment link or through a deposit
  account it owns (visible ownership) that exists at t. So a node that is not a leaf has at
  most 8 sharers, and `customers_30d` never exceeds `customers_90d`.
- Both thresholds are checked against the bench histograms on development worlds built after
  PhantomLedger's households and ring timing steps, counting customers, never accounts, and
  frozen before any reported run. On development worlds, read from the answer key in
  evaluation: no positive is an account hub, and the share of ring devices and IPs that are
  leaves on their ring's active Mondays is near 0.

**Sharing a device or IP.** A customer shares a device or IP that is not a leaf when it has a
current enrolment link to it, or a deposit account it owns (visible ownership) that exists at
t sent a payment seen on it in `[t - 30 d, t)`. Its sharing start there is the earliest of its
link starts and its first seen use in the lookback, each counted as `t - 90 d` when unknown
under the [known-start rule](#time). Two **co-users** began
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
`mule_onset_ms`, `mule_last_ms`, `opened_ms` (for the group of accounts opened within 90 days
of t), `account_type` and whether it has an ownership link (for the population check), paged
by id. Only the evaluation and testing packages import its reader, and the fetch executor
refuses to run it. Evaluation needs the activity dates of every mule, hidden ones included, to
build its groups and early warning. `mule audit` runs it once per world while the world is
loaded and keeps its rows in the evaluation store (see [The pipeline](#the-pipeline)), so no
world is reloaded for evaluation.

### `read_hubs`

Prints the Monday's account hubs, crowds and device and IP hubs with their measure (distinct
customers over 90 days), and each device's and IP's `observed_only`, under the rules above.

- **Account hubs** travel as a `SET<VERTEX>` parameter to B2, B4, B5 and B6: an account hub
  needs to be known before a subgraph is built, and its test reads a whole lookback of a busy
  account, so it runs once per Monday rather than once per request. B5 takes it because it
  finds B4's accounts again, and an expanded hop-1 account is one that is no hub. Billers,
  merchants and ledgers are few, and `mule bench` measures the request bytes.
- **Device and IP leaves** are never sent as a parameter: with carrier IPs renumbering daily
  there can be hundreds of thousands of crowds per Monday, and resending them with each of
  about 720,000 requests would dwarf the answers. B5 and B6 apply the same crowd rule inline
  to each device or IP they touch, stopping at the ninth customer, since they need only
  whether it is a leaf. A leaf's row (its measure, kind and `observed_only`) is joined by id
  from `read_hubs`' cached output when the batch is built, and the batch builder refuses a
  node whose B5 inline leaf flag disagrees with `read_hubs`; parity checks that B5's flags
  agree. B6 prints no per-node flag, so its hubs and crowds are checked by the parity of its
  whole community block with the reference graph. That check meets B6's inline crowd rule on
  TigerGraph only where a sampled community reaches a crowd. The reference graph's own test
  world, where a crowd sits between two members, pins the rule the reference applies and does
  not exercise B6 on TigerGraph.

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
Inflows of one second form one lot, so their order never matters, and outflows of one second
are walked by rail name, then id (see [Time](#time)), so no id decides a value. Only outflows
before t count: an inflow received less than w before t shows only what left by t, and the
payment's age on its edge tells the model how much of its curve it could see. On development
worlds, a check confirms that busy ordinary accounts' card spending does not fill the short
dwell shares; if it does, the dwell shares count transfer and cash outflows only.

**In GSQL.** The owner chose to write the walk in TigerGraph: "we need to try b". It is
plain GSQL, with no user-defined function and no subquery: local scalars, FOREACH and WHILE in
POST-ACCUM over ListAccums an earlier SELECT filled, heaps ordered (ts DESC, rail DESC, id DESC)
with explicit capacities, and a suffix-minimum early stop. Lot k occupies cents `[L, L + c)` of a
running level and still holds `max(0, min(lowest level since k, L + c) - L)` cents; each fall
of that minimum inside its band is a piece that left at that outflow's time. A global OrAccum
records a heap that did not iterate in order, and the query then stops with a status. A port
of this sketch matched full-history last-in, first-out matching in 20,000 random worlds with no
mismatch ([the GSQL dwell research and its check](research/gsql_dwell_2026-10-17.md)). The
Python reference is an independent explicit stack of lots written from the
specification. A request names its accounts by id, a list of strings, not as a vertex set,
from which an installed query drops an id the graph lacks without a word, so the query itself
refuses an unknown, repeated or invisible account. It prints one row per tracked inflow, a
payment id with its lot's `left_24h` and dwell, because a GSQL tuple cannot hold a list of
ids. The walk compiles on a live TigerGraph (see [Query plumbing](#query-plumbing)) and is
verified there on loaded data in [the 5,000-person
test](#the-5000-person-test-on-tigergraph-cloud); its worst account (an account later emptied
by a sweep, up to about 32,896 steps; a retaining mule about n + m) is measured by `mule
bench`.

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
payer's `is_external`. The share of victims at the same bank is a declared value until a
source is found, and pinning it is not enough: the split would still learn the generator's
choice. So PhantomLedger's world checks hold the lift of the external share of new payers, and
of onward P2P value, between mules and matched ordinary accounts under declared ceilings, and
until the share is sourced the split is listed as a declared cue in
[Limitations](#limitations). "New" means the first payment in the lookback: a payer older
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
kept to 10 minutes to 21 days); about 0.82 of mule lives are under 30 days under that model
(provisional, to be re-measured in the generator; 0.695 today).

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
stronger: a PhantomLedger world check holds the `is_mule` lift of "no payment in the first 60
days of the lookback, some after" at or under 10, the top of the 5 to 10 times band found in
reality (today it is a 37 to 85 times marker).

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

**Mule behaviour.** Controller-operated accounts, about 18% of mules with controller-opened
accounts counted, hold the ring's device
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

**Every account gets owner-enrolment edges.** Each account in a subgraph whose customer's
devices and IPs lie within two hops of the scored account gets edges to that customer's kept
devices and IPs: the scored account, the other accounts of its customer and every expanded
hop-1 account; a co-user's account gets its customer's links to the scored account's devices
and IPs only. So an account with no payments that shares a device with an active mule reads
the mule account's behaviour within two layers: the owner's Account Z (2026-10-14), which
must be scored differently from a payment-less account that shares nothing. The
alternative, a third layer, was rejected.

Hop-2 accounts and hub accounts get none, nor does a co-user's account for its other links.
Those devices and IPs would sit three hops from the scored account, beyond two layers: their
only trace on the score would be a count in those accounts' degree columns, at the cost of a
fetch per hop-2 account. Every subgraph is built this way, so the rule is the same for all.
The owner asked for every subgraph account, and on 2026-10-17 agreed to this rule:
owner-enrolment edges for the scored account and its direct neighbours only.

**Link ends.** `active`, `since_end` and `held`, agreed earlier in the session, are dropped,
and the "now" counts become 30-day counts: PhantomLedger sets the end of a tenure built from
uses by the uses that follow, so they read the future. `since_last_seen`, the time since the
customer's accounts last used the node on a payment, replaces them with observed evidence.

**Deposit-only co-user accounts.** Accounts reached through a shared device or IP are deposit
accounts only, one candidate per customer, deduplicated by customer before the caps, under the
one account rule of [Shared rules](#shared-rules).
The cap of 16 other customers per node was dropped: a node with more than 8 customers is a
crowd and never expanded, so a non-leaf node has at most 7 other customers.

| Where | Field | Why |
|---|---|---|
| enrolment edge (customer and owner-enrolment) | `since_start` (only with a known start) | ring tenures are recent; links made at sign-up date the customer, not behaviour |
| enrolment edge | `since_last_seen` | a controller's device on file is used in bursts; the observed stand-in for the dropped link end |
| seen edge | `uses_90d`, `since_first_use` (only with a known first use), `since_last_use` | a ring device nobody registered still shows on members' payments, used briefly and recently; a new phone looks alike but has no co-users |
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
leaving out account hubs. The payment round adds members, never links: `density` and
`new_link_share_30d` count sharing links only (the owner, 2026-10-17).

**The payment round** was added because the device and IP rounds see a ring only through its
endpoints, which about 82% of mules do not hold; for them the block would read like a customer
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
  record them; fetch refuses a version with no passing parity record (see
  [Commands](#commands)).
- Every per-account query refuses invalid parameters and duplicate, unknown or invisible
  (not existing at t) accounts with a status row, and `read_counterparties`,
  `read_payment_events`, `read_shared_identity` and `read_community` also refuse a member of
  their hub set that does not exist at t (`invisible_hub`). `list_accounts`, `read_labels` and
  `read_answer_key` page by strictly increasing id.
- Every PRINT of a vertex set is a projection: a bare `PRINT Accounts;` once leaked every
  attribute while all 160 tests passed.
- Per-request state is kept in accumulators keyed by the scored account, never in a vertex
  flag two scored accounts of one request could share.
- Work counters are printed only because `mule bench` reads them; the reference graph does not
  mirror them and parity ignores them.
- Each query is checked offline by `gsql-lsp check` against `gsql/schema/` as it is written,
  before the Python above it is built, and compiled on the live server as it is written. The
  owner allowed the MulePatternLearner workspace to be wiped early for this, once the new
  schema and the loader's wipe step land (see [Order of work](#order-of-work)), so no query
  waits for the end to meet the server. All twelve queries (`list_accounts`, `read_labels`,
  `read_hubs`, `read_world`, `profile_hubs`, `read_money_flow`, `read_counterparties`,
  `read_activity_change`, `read_payment_events`, `read_shared_identity`, `read_community` and
  `read_answer_key`) compile on the live server, B1's heap and list functions and B6's loop of
  rounds included, and their heaps take literal capacities, so no parameter is a capacity. A
  vertex set that B6's loop assigns from `to_vertex_set` is declared `(ANY)` where it is first
  assigned, since the server gives `to_vertex_set` no static type. A heap of payments ordered
  (ts DESC, rail DESC, id DESC), probed on the live server with 76 payments, kept its first 30
  and iterated in exactly that order, which B1's own order check read as sorted. What an empty graph cannot show is the order of the
  heap of one account's payments, so B1 still checks every heap's order at run time and stops
  with a status when one is out of order. Runs on loaded data, the live B1 verification and
  parity come in [the 5,000-person test](#the-5000-person-test-on-tigergraph-cloud).
- The fetch executor runs only an allowlist: `list_accounts`, `read_labels`, `read_hubs`,
  `read_world` and B1 to B6. It refuses `read_answer_key`, the loader's `verify_load` and
  anything else installed in the graph. `profile_hubs` runs only from `mule bench`,
  `read_answer_key` only from evaluation's executor, and parity's executor runs every query.
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
fixes make it honest, and a reported result is refused for a world that fails those gates.
No gain is credited to the encoding without the per-payment timing ablation.

## Leak rules and how tests enforce them

Leaks are absolutely not allowed. Each guard below is named once; [the
features](features.md) cites them by name for every feature.

| Guard | Rule | Enforced by |
|---|---|---|
| Labels | Only `read_labels` reads `pu_label`, and it prints `mule_onset_ms` and `mule_last_ms` only for `pu_label` 1. Only `read_answer_key` reads `is_mule` and `mule_ring_id`, and it also prints every mule's dates and `opened_ms`. No feature query reads any of them, and no feature says whether any account is known or caught. Labels and weights never join a node row. | erasing every label column leaves every query's answer identical; projection-only PRINTs; text tests fail any other reader of these columns; the fetch executor's allowlist; import contracts |
| Files that carry labels | The loader's manifest, its verification report and its `Account` shards hold label counts or label columns; PhantomLedger's run summary holds values computed from the truth, and its gate record pass flags computed from it; the evaluation store holds the answer key. Only `evaluation/` and `testing/` open them, and nothing in `tigergraph/`, `data/`, `subgraph/`, `model/`, `training/` or `comparisons/` imports `testing/`. | a path contract: each of these locations has one accessor in `paths.py`, and a test scans every module and fails any call to it outside `evaluation/` and `testing/` |
| Selection by labels only | Selection reads `pu_label`, never the answer key, in every scoring mode. | with the answer key erased, every selection input is byte-identical |
| Before t | Values come only from payments with `t - 90 d <= ts < t` and links with `valid_from_ms < t`; an account exists only when `opened_ms < t`. | fixtures with a payment, a link start and an account opening exactly at t, a link end exactly at `t - 90 d` and at `t - 30 d`, and a closure exactly at t; text tests that require one exact allowed form for every bound; an account alone equals itself among 31 others |
| No link ends as values | `valid_to_ms` appears only in the visible and current tests, both at least 30 days back; `closed_ms` only in the population test. These are decided by the past only while no tenure built from uses, ring or ordinary, is quiet for more than 21 days or ends more than four days after the start of its last use's day, which a PhantomLedger world check proves on every party link table. | fixtures of two holders with identical pasts, one with a later ring payment and one with a later ordinary use of a hotspot, giving identical rows at t; text tests allowing only the two forms; PhantomLedger's gate record |
| Lookback | Every window and duration lies inside `[t - 90 d, t)`, which with 90 days of history is always inside the export, so no value reaches the data start, a clamped date or the burn-in boundary. | `mule prepare` checks the first Monday against the first payment, never the first join |
| Known starts | A link start or first seen use is used only when known: after the lookback start, after joining, and after a payment of the customer's own not seen on that node. | fixtures: a joiner and a long-standing customer with the same behaviour read the same; a joiner whose first device and IP links start hours after joining reads no new link and no `since_start`; a pre-data customer reads the same on two Mondays far apart |
| No dates of record | `opened_ms`, `joined_ms` and `closed_ms` decide existence, the population and the known-start test, never a value or an order; no token is read; caps order by activity. | a perturbation test moves every date of record and link start before `t - 90 d` by keyed amounts that keep each on the same side of every bound it meets, on the reference graph and in live parity, and requires every answer byte-identical; a test that the batch builder never reads `opened_ms` |
| No identification closure | Identification closes nothing in PhantomLedger, so no end, silence or missing account is caused by a catch. | a caught neighbour reads exactly like an uncaught one |
| Clock | No column holds a clock value. Durations measured to t (payment `age`, `since_last_in`, `since_last_out`, enrolment `since_start` and `since_last_seen`, seen `since_first_use` and `since_last_use`) carry their event's clock time, since t is always Monday 00:00, and a monthly payment in the subgraph can fix the day of month. | a reported result is refused unless PhantomLedger's Clocks gates passed and the clock probe of `mule audit` passed: a classifier on the residues of every such duration modulo 60 seconds, an hour, a day and a week, and on each event's day of month, must predict `is_mule` so poorly that the upper end of its ring-resampled 95% ROC AUC interval is below 0.55 (set by the owner on 2026-10-17). A probe with too few mules has a wide interval and fails, so low power never passes. Until then results are development only |
| Order-free ties | Same-second payments have no order (see [Time](#time)). | a fixture swapping same-second ids changes no B1, B2, B3 or B6 value; the loader refuses payment ids that follow event time |
| Same rule for all | Caps, orders, hub handling and subgraph shape are identical for positives, unlabelled draws, neighbours and accounts scored in selection or test; one query with one set of settings builds all of them. Every setting is chosen on development worlds only, from label-free statistics (the bench histograms, cost) or the selection score, never the answer key; the pass or fail checks the owner approved (the feature audit, the hub and leaf checks, the gates) are the only reads of the truth there. On reported worlds a check only verifies and a failure stops the run. | the cache key holds query versions and settings; the contract lock; the manifest records what each setting was chosen from |
| Keys chosen by their reader | Training, selection and scoring read only the cache keys they computed themselves, never the cache's index, so rows fetched for a test-role list chosen by the truth never steer them. | selection on a world after a test-role fallback fetch of it, with the answer key erased, gives byte-identical inputs |
| Test worlds stay unseen | Nothing learned from a reported world's answer key changes the next reported result on the same worlds: missed mules are investigated on development worlds, a change prompted by a reported run needs new reported worlds from new seeds, and the number of seeds of the GNN and of each ablation is fixed at five, and each XGBoost comparison is trained once, before any result is seen. | the manifest records the worlds whose reports informed each change; the seed count is in the contract lock of the GNN and of each ablation (an XGBoost comparison has no seed to lock) |
| No label conditioning | Graph-structure features are computed identically for every account whatever its label. | erasing the labels (above) |
| Evaluation never feeds back | Time to first flag and every other evaluation output never become features (the owner: "dont sent this time to the model"). No query, data or training code imports evaluation, and nothing outside evaluation reads evaluation's output: `mule audit`'s pass records are read by `mule evaluate`, and, for pass or fail only, by the `mule train` command before any training code runs (the one exception, see [Gates](#gates)). | import contracts (the XGBoost package included); the path contract above |
| Gates | A value is only as honest as PhantomLedger's behaviour. | `mule evaluate` refuses a reported result unless every gate mapped to a feature passed (see [Gates](#gates)) |

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
| A first seen use dating the customer | a joiner's first payment makes a new co-user pair | Known starts |
| A link start dating the customer | a customer joins on 1 April and first pays from a phone that evening; on 13 April the link would read new with `since_start` about 12 days | Known starts (no earlier own payment, so unknown) |
| A link end set by later uses | two customers use a cafe hotspot on 2 March and next on 25 May; one tenure each runs to 26 May, so on 13 April both read current and become co-users | No link ends as values, by PhantomLedger's 21-day split of every tenure built from uses |
| Truth in data code | `mule prepare` reading `is_mule` to count mules, or a gate report with values computed from the truth | Checks on every world (truth checks in `mule audit`); gate records hold pass flags only; Files that carry labels |
| A test world reused after its answer key was read | features added for missed mules of world C, then re-tested on world C | Test worlds stay unseen |
| Seeds added after seeing test intervals | more seeds for an ablation until its interval clears zero | Test worlds stay unseen |
| Cached keys chosen by the truth | world C's test-role fallback rows read back when world C is the selection world | Keys chosen by their reader |
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
  Xu 2021), because with positives this rare the textbook weighting lets the loss settle at
  the prior and stop learning. The positive risk keeps each mule's weight
  of 1 divided by its active Mondays; the negative-risk correction uses unit weights over
  positives, because a hidden mule enters the unlabelled draw once per active Monday. The loss
  for a positive uses `1 - sigmoid(f)` for float32, and a test drives the negative risk below
  zero.
- **Prior.** The share of hidden active mules among the unlabelled is unknown, as in a bank.
  The grid is 1, 3 and 10 times the identified active share measured in the training world
  (identified active mules divided by scorable accounts, averaged over Mondays). The grid is
  run on the development rotation and judged by the selection score alone: if every pair of
  priors has a paired interval of the score difference that contains zero (resampled by
  identified mule account, since ring ids are evaluation only), the prior is fixed at 1 times
  for reported runs and that is stated, saving two thirds of selection training. It is never
  set from PhantomLedger's identified-share report or from the answer key.
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
| XGBoost, graph features | the same plus its customer row (8), the rows of the devices and IPs its customer has an enrolment edge to in the subgraph, aggregated by mean and maximum, and its community block (8) | what message passing adds beyond hand-made graph features |
| GNN | everything | the model |
| GNN, scored account alone | its own row and own payment edges; every other node zeroed, no other edge, no community | what neighbours add |
| GNN without devices and IPs | no device and IP nodes or relations; customer rows keep `accounts_owned` only; no community | what shared identity adds |
| GNN without community | head Linear(64, 32) | whether two hops already see the ring |
| GNN without per-payment timing | every payment-edge duration and dwell field zeroed | what the temporal encoding adds |

- XGBoost replaced gradient-boosted trees at the owner's request ("just train it against
  industry standard xgboost"). It learns identified against unlabelled with each label's
  weights rescaled to equal totals: when labels are selected completely at random, a classifier
  of labelled against unlabelled ranks like one of positive against negative (Elkan and Noto,
  2008). It uses XGBoost's default settings, with the number of rounds chosen on the selection
  world by the same selection score, as the GNN chooses its epochs. At those defaults a seed
  changes nothing (five seeds gave byte-identical boosters), so each of the two comparisons is
  trained once, with no seed: both have no seed spread, and the GNN's five seeds are compared
  with that one model. Every XGBoost fit and prediction, the clock probe's included, runs in a
  Python process of its own that never imports torch, because torch and XGBoost load different
  OpenMP runtimes and a process that holds both crashes in XGBoost's threads.
- Each ablation trains and selects its epoch like the full model, at the prior the full model
  chose for that seed, with the same five seeds.
- The GNN and each ablation have exactly five seeds, fixed before any result is seen, and each
  XGBoost comparison is one model with no seed. A difference is called shown only when its
  paired, ring-resampled interval excludes zero; an interval that straddles zero is reported as
  inconclusive, never answered with more seeds.
- What to expect: a model without message passing may come close to the GNN once it is given
  counts; plan for a close result. With a ring device
  on about 18% of mules, the device and IP ablation may move little; the share of
  positives with any ring-device link is reported beside it. Fix PhantomLedger's shortcuts
  first, since a shortcut favours the simple models.

## Evaluation

Evaluation is the only code that reads the answer key for a result (testing reads it only to
check queries). It scores the test world from the cache after selection is frozen, with the
answer key from the evaluation store.

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

Lift is never printed beside recall at the same budget. The top of the list leads because a
bank works its alert list from the top, and PR AUC can be carried by a few mules ranked above
every normal account.

### Groups

Each group is ranked against non-mules with the other groups removed, as an investigator
removes known cases:

- mules in fully hidden rings (no member identified): the headline;
- unidentified members of rings that surfaced;
- identified mules (`pu_label` 1 in the test world; the model never saw them either);
- mules with no ring (account-takeover drops and impostor payees);
- mules whose account opened within 90 days of t (from `read_answer_key`'s `opened_ms`):
  purpose-opened mules are the hard case;
- mules scored with no payment in the lookback (on an early-warning Monday, an account
  opened for a ring and not yet used, or a recruit no ring money reached), which only links
  to other mules, such as shared devices and IPs, can reveal: the
  owner's Account Z, reported beside the enrolment ceiling (72% of device and 61% of IP
  associations are on file);
- recall by hop distance from the nearest identified mule: on each Monday, the fewest edges,
  of any of the 16 relations and in either direction, between the mule's account and an
  account with `pu_label` 1 inside the mule's own fetched subgraph at that t, in bins 1, 2 and
  "not within two hops" (an account node counts; customer, device and IP nodes are steps on
  the way, so a shared device puts two mule accounts two edges apart). Evaluation computes it
  in Python from the cached edge lists: it is an evaluation output, never a model input.

### Uncertainty

A ring bootstrap within each world resamples whole rings with all their account-Mondays, each
ringless mule as its own cluster, and whole non-mule accounts with all their Mondays, since an
account's 52 Mondays are correlated. Comparisons use paired replicates. The spread over the
five seeds (of the GNN and its ablations only, since each XGBoost comparison is one model) and
the three rotations is reported beside it.

### The missed-mule report

The owner: "we need to score every one of the true mules that were hidden from the model. the
model needs to be able to identify all of them, or atleast score them high. if not, we need to
investigate why. maybe theres a bug, or we are missing some features that need to be added."

- **Who is in it.** Every true mule hidden from the model that did not score high: one never
  flagged in the top 5% on any of its active Mondays (**missed**), and one whose median rank
  over its active Mondays is outside the top 5% though it was flagged on some (**weak**). The
  owner set both thresholds on 2026-10-17.
- **What it holds.** For each, its decoded feature rows on each active Monday, its subgraph
  (nodes and edges with their fields), its group, its ring, and how each ring-mate scored and
  whether it was identified.
- **What follows.** Each is explained as a bug or a missing feature. The investigation and any
  change it leads to happen on development worlds; a reported run's report may prompt a
  change only under the Test worlds stay unseen guard, with new reported worlds before the
  next report. The report is written to the run's results and read by nothing else.

### The feature audit

On development worlds, `mule audit` (evaluation code, reading the cache and the answer key)
measures each feature's univariate separation of active mules from non-mules over scored
account-Mondays: its ROC AUC against `is_mule`, with a ring-resampled interval. Every column
of the scored account's row, its customer row, its own device and IP rows (aggregated by mean
and maximum), its own edge fields (likewise), its degree columns and its community block is
audited.

- A feature with no separation (its interval contains 0.5) is researched or dropped.
- A feature that separates almost perfectly (ROC AUC at least 0.95, or at most 0.05) is a
  suspected leak or shortcut and stops the run.
- A near-pure region also stops the run, since a marker held by a minority of mules and by no
  ordinary account keeps its ROC AUC near 0.5 to 0.6: for each feature, at each flag value
  and in its top and bottom 0.1% of scored account-Mondays, a region that holds at least 1% of
  active true-mule account-Mondays and in which true mules are more than half (set by the
  owner on 2026-10-17).
- A stopped run resumes only once the cause is fixed, or the owner accepts it on the evidence
  as real behaviour with real look-alikes. `mule evaluate` refuses a reported result until a
  development audit of the same query versions and settings passes. The owner set the 0.95 and
  near-pure thresholds on 2026-10-17, and a failed audit stops `mule train` (see
  [Gates](#gates)).

On development worlds the same command also checks that no positive is an account hub, that
the share of ring devices and IPs that are leaves on their ring's active Mondays is near 0,
and that busy ordinary accounts' card spending does not fill B1's short dwell shares.

### Identification bias

Victim reports may identify more often the mules whose victims bank here, so
`new_internal_payers_*` could learn who gets caught rather than what mules do.
Evaluation reports identified against hidden mules by victim origin: the internal share of
their new payers over their active Mondays, from their own B2 rows, and recall of hidden mules
by that share. It also reports how well the features separate identified from hidden mules.
On the development rotation only, for seed 0, training keeps every epoch's checkpoint, and
evaluation rescores the selection world with each to report the correlation of the selection
score with recall of all mules per prior and epoch, never used to select.

## The pipeline

**Where each step runs.** PhantomLedger, then `tf_gnn_loader_v2`, then
MulePatternLearner, one command per repository for generation and loading. TigerGraph is the
MulePatternLearner workspace on TigerGraph Cloud that the repository's `.env` points to
(host, graph name `Mule_Pattern_Learner`, secret); `TransactionFraud_GNN` lives in a different
workspace and is never touched. The 5,000-person pipeline runs from this laptop, all of it:
PhantomLedger, the loader and MulePatternLearner. The 300,000-person worlds are generated and
loaded from the NVIDIA training machine, which also fetches and trains.

1. **Code** reaches the training machine by git: pushed to `main` after the owner's code
   review of all three repositories, then pulled. It carries the parity record of the
   5,000-person test (`gsql/parity.toml`, committed with the code), so the training machine
   needs no 5,000-person world of its own and runs no parity; its fetch refuses any query
   version the record does not cover.
2. **`mule install`** installs the queries once; `mule-load clear` keeps the schema and the
   queries between worlds.
3. **For each world, one at a time,** in the one graph:
   1. PhantomLedger generates it, writing its `world_record` and `gate_record` in schema
      `mule_temporal_gates`;
   2. `mule-load push` loads and verifies it (the loader pins pyTigerGraph below 2 while
      MulePatternLearner runs 2.0.4, so the two keep separate environments), and after a
      passing verification writes the two records as `world_<seed>.identity.toml` and
      `world_<seed>.gates.toml` into `MULE_PL_RECORDS_DIR`;
   3. `mule prepare` reads `read_world`, finds the configured world whose identity file
      matches on the fields the two share, checks the lookback and the labels, and writes
      the training examples;
   4. `mule fetch` fills the cache for every role the world plays in the configured rotations;
   5. `mule audit` reads the answer key into the evaluation store, checks the gate record,
      builds and fetches a test-role fallback list if one is configured, runs its checks and
      writes pass records;
   6. `mule-load clear` empties the graph.
4. **`mule train`** trains from the cache and never queries TigerGraph. It first reads the pass
   or fail of the audit records, and stops on a failure that stops the run (see
   [Gates](#gates)).
5. **`mule evaluate`** scores the test world from the cache, reads the evaluation store and the
   pass records, and writes the results; it marks them development only unless every gate
   passed (see [Gates](#gates)).

No world is reloaded: everything evaluation needs from the graph, the answer key included, is
taken while the world is loaded.

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
- Fetch consults the index only to skip rows it already has. Training, selection and scoring
  read the rows of the keys they computed themselves (see the Keys chosen by their reader
  guard).
- **The evaluation store** sits beside the cache, keyed by world identity: the answer key's
  rows and `mule audit`'s pass records. Only `evaluation/` and `testing/` open it (see the Files
  that carry labels guard).
- **World identity** (`read_world`): counts per vertex type and per forward edge type (a
  reverse edge type is the forward edge read from its other end, so its count is the
  forward count and is not printed), and per value of
  `account_type`, `payment_rail` and `is_external`; second sums of `joined_ms`, `opened_ms`,
  `closed_ms`, each link type's starts and ends and each payment type's event times; cent
  sums per payment type; per edge type the sum of squared source degrees and the sum of the
  product of both ends' degrees (one link moving between vertices of degrees 2 and 1 leaves
  the squared sums unchanged); the first join and the first and last payment. All are
  integers. A second sum is the sum over rows of floor(ms / 1,000), each row's milliseconds
  rounded down to whole seconds before summing, so it fits a 64-bit integer; a cent sum is
  the sum over rows of floor(`amount` x 100 + 0.5). A world regenerated or changed has
  another identity, so stale rows are never read. The lookback check uses the first payment,
  since backdated joins make the first join years earlier than the data.
- **The fields PhantomLedger's record shares.** PhantomLedger's `world_record` carries a
  fingerprint computed from its exported tables under `read_world`'s key names:
  `vertices.<type>`, `edges.<type>` for the 15 forward edge types, `account_types.<value>`,
  `payment_rails.<value>`, `external_accounts`, `seconds.<type>.<column>`, `cents.<type>`,
  `first_joined_ms`, `first_event_ms` and `last_event_ms`. It does not carry the degree sums,
  which only `read_world` computes. PhantomLedger writes every one of these keys, zero counts
  included, so the comparison is exact and a missing key is an error (the owner, 18 October
  2026). `mule prepare` compares
  these shared fields, and only these, with `read_world`'s; the identity file's other keys
  (commit, clean tree, seed, size, calendar) are read as they are. A golden test in both
  repositories pins the shared fields on one small export.
- The cache, the evaluation store and the training examples live under the repository's
  `/data/` folder (`data/cache/`, `data/evaluation/`, `data/examples/`), which `.gitignore`
  already keeps out of git, as it does PhantomLedger's exports and `.env`.

### Results

`results/<run>/` holds the result tables, the model files and a manifest, and is never
committed: the owner zips it on the training machine and copies it to Downloads on this laptop.
`.gitignore` stays as it is.

The manifest records the code commit and whether the tree was dirty (a reported run refuses a
dirty tree: the NVIDIA results came from a commit with 71 files differing), the query versions,
the world identities, the GPU name and count, the CUDA, torch, PyG and XGBoost versions, wall
time per stage, TigerGraph calls and cache hits, what each tuned setting was chosen from, and
which earlier runs' reports informed a change. Every result is written through a temporary
file and a rename, and each command appends to an `events.jsonl` with the error that stopped
it. Results are tables and files, no charts. There is no baseline paragraph from the old
NVIDIA numbers, which came from a pipeline with known leaks.

### The demo

The TigerGraph workspace is wiped early, once the new schema and the loader's step "The
schema on an empty graph, and the early wipe" land (see [Order of work](#order-of-work)), and
the old demo schema and queries go with it. Only the
two demo queries that show an
account's network and its timeline (`gsql/demo/demo_account_web.gsql` and
`demo_account_timeline.gsql`, the owner's untracked files) are ported to the new schema, and
the demos are redone. They read the answer key and are never a training input.

## The 5,000-person test on TigerGraph Cloud

The MulePatternLearner workspace is its own TigerGraph Cloud workspace, and the repository's
`.env` points only to it. The whole pipeline of this test runs from this laptop. Once
PhantomLedger, the loader and MulePatternLearner are built:

1. Run `CLEAR GRAPH STORE -HARD` and drop every query, loading job and schema object in that
   workspace (`mule-load wipe`). The workspace was already wiped once, early, so that each
   query could be compiled on the live server as it was written; this wipe starts the test
   from an empty workspace.
2. Generate the 5,000-person world with PhantomLedger and load it with `tf_gnn_loader_v2`
   (one command per repository), using the same graph name and secret; the loader creates
   the schema and verifies the load, the 455 exported days and every attribute included.
3. Install every query (each already compiled on the live server at its own step), run every
   query, the live B1 verification and `mule parity` against the reference graph, and commit
   the parity record `gsql/parity.toml` it writes.
4. Run `CLEAR GRAPH STORE -HARD` again and tell the owner.

If after either wipe the `.env` secret no longer authenticates, stop and ask the owner. The owner
then generates and loads the 300,000-person worlds on the NVIDIA machine and trains there,
after the code review.

## Commands

Commands take no options: worlds, roles, dates and settings live in config.

| Command | What it does |
|---|---|
| `mule install` | installs the changed queries of `gsql/queries/` and `gsql/evaluation/` |
| `mule parity` | testing code: runs every installed query on the loaded world, and the reference graph on the same export (the loader's shards whose `read_world` identity equals the live graph's, found under the configured export folder), on a seeded sample of Mondays and accounts with mules oversampled; compares the canonical answers and refusals (B6's hubs and crowds are checked by the parity of its whole community block, since it prints no leaf flag; its inline crowd rule meets the live server only where a sampled community reaches a crowd), that an account fetched alone equals itself among 31 others, and that B5's inline leaf flags equal `read_hubs`; writes the versions that pass, the TigerGraph version and the world identity to `gsql/parity.toml`, committed with the code |
| `mule prepare` | data code: reads `read_world` and matches the configured world by its identity file, on the fields the two share; checks the scored Mondays and the lookback, the value sets and the labels from `read_labels`; reads the populations; writes the training examples and weights; reports counts, the identified active share and the weight balance per Monday |
| `mule fetch` | fills the cache for every example the configured roles of the loaded world need: the training draw, and every scorable account of every Monday of a selection or test role |
| `mule audit` | evaluation code, on the loaded world: reads the answer key into the evaluation store; checks PhantomLedger's gate record against the world's identity; builds and fetches a test-role fallback list if configured; runs the label checks, the clock probe and the identified share by onset month; on development worlds also the feature audit and the hub, leaf and card-spending checks; writes pass records to the evaluation store |
| `mule train` | first reads the pass or fail of the audit records of its rotation's worlds, and nothing else of them, and stops on a failure that stops the run (see [Gates](#gates)); then trains the GNN over the prior grid and five seeds with selection after each epoch, the ablations and the two XGBoost comparisons, once each; training code reads no gate or audit record |
| `mule evaluate` | scores the test world and writes the metrics, groups, uncertainty, missed-mule report and identification bias; marks the results development only unless every gate passed |
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

- fetch time, request size (with the account-hub parameter) and response size per query and
  request, with work counters, for accounts drawn
  in training proportions, positives reported apart (a collector's subgraph is the largest,
  about 2,000 hop-2 accounts), at 1, 2 and 4 concurrent requests;
- B1's worst account and its steps;
- subgraph sizes by type, community sizes, cache bytes per account and in total;
- TigerGraph's memory in use and install time per query (the old full install took about 50
  minutes, and one install timed out after 5,402 seconds);
- parse and build time per batch, GPU step time and peak memory, examples per second;
- the cost of the XGBoost file hand-off (`comparisons.process`): the bytes the parent writes
  under `TMPDIR` and the seconds to write, read and score each selection or test matrix, which
  at 300,000 people is about 9 GB of `.npy` per matrix (about 23 million rows of 102 `float32`
  numbers for the graph features);
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

**The carrier-IP edge bound** (the owner: set from the benchmark) is not set by `mule bench`,
which cannot tell a carrier IP from any other. PhantomLedger's cost measurement on its
30,000-person development world sets it, before any reported world is generated: it reports
usage edges per carrier IP and per person-year, the owner sets the bound on carrier-IP
usage edges per person-year from them, and a PhantomLedger world check enforces it on every
world (PhantomLedger's plan, "Households, devices and IPs"). `mule bench`'s IP crowd counts,
subgraph sizes and memory then show what those edges cost here.

## Package layout

| Path | Holds |
|---|---|
| `config.py`, `paths.py` | frozen, validated settings (the export start date; worlds with their seeds, sizes and rotations; caps, thresholds and seeds; the lookback must exceed the 30-day "new" window); every file location, each with one accessor, including the export folder, PhantomLedger's identity and gate record folder and the evaluation store |
| `tigergraph/` | connection, client with retries, executor, installer, query names, versions and row parsers, the world identity |
| `data/` | the Monday calendar, the population, the labels reader, the keyed draws, the training examples and weights |
| `subgraph/` | fetch and the cache; rows to tensors, degree columns and ablation switches |
| `model/` | the Fourier time encoding, the heterogeneous attention model, the nnPU loss |
| `training/` | the trainer, selection, the prior grid and seeds, the contract lock, scoring with a saved model |
| `comparisons/` | the two XGBoost comparisons |
| `evaluation/` | the answer-key reader, test scoring and the fallback list, metrics and the ring bootstrap, groups, the missed-mule report, the feature audit, the clock probe, identification bias |
| `testing/` | the reference graph, reference worlds and fixtures, the loader shard reader, parity, the planted-slip harness and the version restamper |
| `bench.py`, `cli.py` | the benchmark; the commands |

GSQL: `gsql/schema/schema.gsql` (no loading job); `gsql/queries/` (`list_accounts`,
`read_labels`, `read_hubs`, `read_world`, `profile_hubs`, `read_money_flow`,
`read_counterparties`, `read_activity_change`, `read_payment_events`, `read_shared_identity`,
`read_community`); `gsql/evaluation/` (`read_answer_key`); `gsql/parity.toml`, the parity
record (TOML, since `.gitignore` keeps every JSON file out of git).

The import contracts in `pyproject.toml` keep every layer exhaustive: no package but
`evaluation/` and `testing/` imports the answer-key reader; none of `tigergraph/`, `data/`,
`subgraph/`, `model/`, `training/` or `comparisons/` imports `evaluation/` or `testing/`; and
the path contract of the Files that carry labels guard holds. `testing/` may read every
column because it checks queries and builds no example, batch or model. The README says to run
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
     metrics are kept. The revert brings back `write_split.gsql` and the 27-table
     `loading.gsql`, so the first commit after it is the new schema, which removes both; then
     module by module against the reference graph and the export contract.
   - The early wipe, once MulePatternLearner's schema commit and the loader's step "The schema
     on an empty graph, and the early wipe" land (it follows the loader's "Wipe and clear" and
     "The new contract" steps, and runs the wipe itself): `mule-load wipe` in the
     MulePatternLearner workspace, then `mule-load install`, which creates the new schema on
     the empty graph. From then on each query is compiled on the
     live server as it is written. If after the wipe the `.env` secret no longer
     authenticates, stop and ask the owner.
3. The 5,000-person test on TigerGraph Cloud, run from this laptop.
4. The owner reviews all three repositories; then the push to `main`.

## Pending the owner

Choices the reviews found unmade. Each has the interim rule this design states, and none is
built past that rule until the owner answers. The owner settled every other choice listed here
on 2026-10-17, and each is written into its section above.

| Choice | Interim rule in this design |
|---|---|
| The internal and external new-payer split as a declared cue until the on-us share is sourced | kept, with its lift gates |

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
- A ring device appears on only about 18% of mules, so shared identity is a strong signal for a
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
- The internal and external split of new payers rests on a declared share of victims at the
  mule's own bank until a source is found; the lift gates bound what the model can learn from
  it, but the split is a declared cue.
- A link or first use with no earlier payment of the customer's own in the lookback is never
  a known start, so a new customer's devices and IPs never read new; the rule costs the model
  that signal for everyone alike.
- Shortcuts the dossier noted but has not researched (listed in
  [the features](features.md#generator-gates-each-feature-depends-on)) each get a world check,
  but its ceiling is declared rather than researched, so it bounds a cue without showing that
  the cue is as strong as in reality. No reported result passes until every such check
  passes.
