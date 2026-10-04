# The mule profile: what separates mules from other accounts

Part of [the diagnostic study of September 2026](diagnostic-study.md): its 8,233 sampled
accounts (all 233 mules, 8,000 uniform non-mules, each split at its own cutoff) and their
candidate pools, held against the generator's mule typology to find what a mule leaves behind
and what limits detecting it. Ground truth for analysis only. Non-mule shares, of the uniform
sample, estimate the population; weighted AP and ROC AUC weight each account to its population
as the audits do. Run 2's test audit for comparison: AP 0.0024, ROC AUC 0.782, prevalence
0.00084.

**Caveat.** The trace features were designed after reading the typology and test mules, so
their test numbers are optimistic. The $200 threshold is not sharply tuned (first-time inflows
of $100 to $500 from any peer: test AP 0.18 to 0.23, validation 0.21 to 0.31), and the
internal-peer version holds on train (ROC AUC 0.747, AP 0.116) and validation (0.801, 0.273).
This profile shaped the `pool_activity` and `pool_internal_inflows` groups, so the test audit
of every run that keeps them is for reporting, not selection.

## The mule typology and its traces

The generator (PhantomLedger's mule typology) gives each mule 8 to 26 inbound payments from
victim accounts over a 3 to 11 day burst, each forwarded 1 to 14 hours later, less a 5 to 10%
haircut, to one pinned destination. A "trace" is either sign below, read from the account's
candidate pool ("pool") or from the subset the evaluation sampler attends to ("eval draw"):

- **Internal first-time inflow of $200 or more**: an incoming payment (Zelle or other) of at
  least $200 from an internal peer with no earlier event of the pair.
- **Pass-through signature**: an incoming payment whose next outgoing payment follows within
  0.5 to 15.1 hours at an amount ratio of 0.88 to 0.96, or the mirror case on an outgoing one.

## Findings

1. **A strong, specific signal that run 2 did not use.** 88% of test mules have an internal
   first-time inflow of $200 or more in their pool (12.1% of test non-mules); counting them
   gives about 100 times run 2's AP ([Data or model](#data-or-model)).
2. **Weak at the train cutoff.** On 2024-07-01 only 42% of train mules show the inflow trace,
   and 52% have no trace at all ([The detection ceiling](#the-detection-ceiling)).
3. **Hidden test mules match revealed ones on volume, not on Zelle or pass-through.** On
   distinct peers, incoming payments and history length they sit at the same percentile of
   non-mules ([Revealed and hidden mules](#revealed-and-hidden-mules)).
4. **Almost no mule lacks history** (1 of 40 test mules has under 100 visible events); the
   practical ceiling is the missing trace.
5. **Two kinds of non-mule crowd the top**: busy legitimate Zelle receivers, and ring-side
   accounts labelled 0 (4 of the 40 top-ranked sampled non-mules received money from a true
   mule, against 0.6% of all non-mules).
6. **The limit was mainly the model, not the generator** ([Data or model](#data-or-model)).

## Revealed and hidden mules

Label sources: train 12 by victim report, 8 by monitoring, 140 hidden; validation 8 by victim
report, 1 by monitoring, 2 by network trace, 22 hidden; test 17 by victim report, 2 by
monitoring, 1 by network trace, 20 hidden.

Medians, non-mules / revealed / hidden mules:

| Feature | Train | Validation | Test |
|---|---|---|---|
| Visible events | 145 / 229.5 / 202 | 229 / 306 / 309 | 320 / 474 / 447.5 |
| Candidate events (pool) | 28 / 34.5 / 32 | 29 / 45 / 32.5 | 30 / 39 / 44 |
| Distinct-peer stratum events | 4 / 8 / 5 | 4 / 8 / 6.5 | 5 / 8 / 8 |
| Incoming payments in the pool (capped at 16) | 12 / 16 / 13 | 12 / 16 / 14 | 13 / 16 / 16 |
| Zelle inflows in the pool | 0 / 2 / 0 | 0 / 7 / 2 | 0 / 7 / 3.5 |
| 30-day distinct payers | 1 / 4 / 3 | 1 / 4 / 2.5 | 1 / 3 / 3 |
| 90-day decayed inflow ($) | 11,226 / 30,701 / 26,284 | 13,695 / 36,207 / 29,737 | 14,719 / 34,606 / 27,916 |

Share of accounts with a trace:

| Split | Group | Internal first-time inflow of $200+ (pool) | Same (eval draw) | Pass-through (pool) | Any trace (pool) | Any trace (eval draw) | Any Zelle in the pool |
|---|---|---|---|---|---|---|---|
| Train | non-mule | 16.2% | 11.9% | 6.5% | 21.0% | 14.6% | 35.2% |
| Train | revealed | 80% | 55% | 30% | 80% | 70% | 80% |
| Train | hidden | 36% | 29% | 23% | 44% | 32% | 57% |
| Validation | non-mule | 12.6% | 8.8% | 7.2% | 18.6% | 12.2% | 39.4% |
| Validation | revealed | 100% | 91% | 82% | 100% | 91% | 100% |
| Validation | hidden | 41% | 23% | 32% | 45% | 23% | 77% |
| Test | non-mule | 12.1% | 7.6% | 8.3% | 18.8% | 11.2% | 42.6% |
| Test | revealed | 95% | 85% | 95% | 100% | 95% | 100% |
| Test | hidden | 80% | 50% | 55% | 80% | 55% | 75% |

Test ROC AUC against the non-mules, revealed / hidden: distinct-peer stratum 0.873 / 0.886,
incoming payments in the pool 0.866 / 0.837, visible events 0.719 / 0.725, Zelle inflows in
the pool 0.852 / 0.646, mean flow observation seconds 0.803 / 0.645.

Zelle sets hidden mules apart: revealed mules come mainly from victim reports, and only Zelle
inflows carry a fraud verdict in this graph, so a mule without Zelle is revealed only by
monitoring or network tracing. 5 of the 20 hidden test mules and 60 of the 140 hidden train
mules have no Zelle by their cutoff; every revealed test and validation mule has it.

The two are separable: cross-validated LR on within-split percentiles tells them apart at ROC
AUC 0.75 over all 233 mules (0.70 to 0.79 over 20 seeds) and 0.70 within test. LR on pool
features trained on the 20 revealed train mules gives test ROC AUC 0.933 for revealed and
0.848 for hidden mules (AP against the non-mules 0.111 and 0.003); on all 160 train mules,
0.969 and 0.909. The label bias is real, works through Zelle and victim reports, and costs
most at the head of the ranking.

## The detection ceiling

| Split | Mules | No trace in the pool | No trace in the eval draw | No trace and no Zelle |
|---|---|---|---|---|
| Train | 160 | 83 (52%) | 101 (63%) | 45 |
| Validation | 33 | 12 (36%) | 18 (55%) | 4 |
| Test | 40 | 4 (10%) | 10 (25%) | 2 |

- **History.** Fewest visible events: train 25, validation 109, test 46. The 233 sampled
  accounts with no history are all non-mules (3.1%, 2.3%, 3.2% of the sampled non-mules), so a
  no-history ceiling barely exists.
- **Training positives without a trace.** 4 of the 20 revealed train mules have none in the
  pool on 2024-07-01 (3 of the 8 revealed by monitoring, which ignores behaviour, and 1 of the
  12 by victim report); 6 of 20 none in the eval draw.
- **Why.** Each burst starts at a uniform offset inside its ring's window, so by mid-year about
  half have not happened: the earliest pass-through event in the pool is a median 57 days
  before the cutoff for train mules, 126 for test mules. Phase-1 visibility also hides about
  30% of internal counterparties at the train cutoff. Test mules are scored with the whole
  year visible, and 90% carry a trace.

## The non-mules that look most like mules

Ranked by the count of internal first-time inflows of $200 or more, 20 test mules lie in the
top 68 population accounts (precision 0.30) and 30 in the top 380; the other 4 traced mules
come only around 7,800, since 12.1% of non-mules (about 5,800 test accounts) have one such
inflow or more. An LR on pool aggregates trained on the train ground truth needs the top 895
accounts for 20 mules and 3,036 for 30.

The count's top 40 sampled non-mules, medians:

| Feature | Top 40 non-mules | Mules | All non-mules |
|---|---|---|---|
| Internal first-time inflows of $200+ | 2 | 3 | 0 |
| Zelle inflows | 6 | 5.5 | 0 |
| Distinct-peer stratum | 8 | 8 | 5 |
| Visible events | 368 | 459 | 320 |
| 30-day distinct payers | 3 | 3 | 1 |

12 of the 40 also send Zelle and 3 show pass-through: busy legitimate Zelle users receiving
from new internal peers. Without the internal-peer condition, first-time inflows of $200 or
more mislead: 79% of the non-mules' come from external peers (median $2,097, mostly ACH or
unknown rail, like payroll) against 11% of the mules' (median $968, a third Zelle), and the
external-peer version alone ranks at ROC AUC 0.68.

The closed-world label makes only the designated mule positive. The ring's other members,
which receive the mules' forwards as layering hops, are labelled 0: about two per mule under
the generator's default ring settings (not checked against the graph). The 3 sampled test
non-mules that received at least $200 from a mule rank at the 100th, 99.9th and 81st
percentile of the count, and the top two look exactly like mules (7 first-time inflows of $200
or more, 4 pass-through events, 9 to 12 Zelle inflows, 14 to 16 Zelle outflows).

## Data or model

The model, mostly. The typology leaves a clear footprint, with no round-amount artefact and no
sign that mule behaviour is indistinguishable, and simple models on the same inputs beat run 2
on the test population:

| Ranking | Test AP | Test ROC AUC |
|---|---|---|
| Count of internal first-time inflows of $200+, pool (distinct-peer tie-break; raw 0.233 and 0.921) | 0.276 | 0.968 |
| The same count, eval draw only | 0.204 | 0.931 |
| Count of first-time inflows of $200+ from any peer, pool | 0.226 (0.14 to 0.42) | 0.94 |
| Logistic regression on pool aggregates, all 160 train mules | 0.103 (0.04 to 0.20) | 0.939 |
| The same, only the 20 revealed train mules (PU) | 0.062 | 0.891 |
| Logistic regression on the mean and maximum of the eval draw | 0.006 | 0.83 |
| Logistic regression on the summary features the model did not get | 0.004 | 0.76 |
| Run 2's audit | 0.0024 | 0.782 |

The eval draw's mean and maximum do as poorly as run 2, while one count over the same slots
does 80 times better. The signal is a conjunction to count (incoming, first-time peer, $200 or
more, internal peer); averaging the edge inputs dilutes it, and run 2's constant root and
softmax-weighted average cannot count. That fits the numbers but was not proven; run 3, with
the pool counts and the slot sum, reached audit AP 0.134.

Some features are weak by design: the flow delay is the time to the next outgoing payment of
any kind (for busy accounts usually a card payment within hours), and windows ending at the
cutoff rarely catch a 3 to 11 day burst at a random time (7-day out-to-in ratio: ROC AUC 0.51
to 0.55).

The data's own limits, real but secondary: bursts at random times, so one mid-year training
cutoff learns from weak or traceless positives; closed-world labels; first-time peers are
commoner at the train cutoff (16.2% of train non-mules against 12.1% at test have an internal
first-time inflow of $200 or more), since pair history is relative to the visible history and
phase-1 visibility hides internal peers; and, in this load, no ring structure in the labels
(`mule_ring_id` was -1 for every mule).

## Not carried over

The profile scripts (`profile/p1_groups.py` to `p11_misc.py`, `load_messages.py`) read the
study's saved candidate pools from local files outside the repository and answered these
questions once. Their trace definitions became `pool_activity` and `pool_internal_inflows`
(the $200 band as bands of 100 and 1,000); the analyses worth rerunning are
`mule diagnose univariate`, `baselines` and `subgroups`.
