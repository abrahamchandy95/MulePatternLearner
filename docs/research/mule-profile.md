# The mule profile: what separates mules from other accounts

Part of [the diagnostic study of September 2026](diagnostic-study.md), on the same
8,233 sampled accounts (all 233 mules and 8,000 uniform non-mules, each split read at its
own cutoff) and the candidate pools TigerGraph returned for them. It compared the accounts
with the data generator's mule typology to find what a mule leaves behind, and what limits
detecting it. Ground truth was read for this analysis only. Non-mule shares are shares of
the uniform sample, so they estimate the split's population; weighted AP and ROC AUC weight
each account to its population, as the audits do. Run 2's audit, for comparison: test AP
0.0024 and ROC AUC 0.782 at a prevalence of 0.00084.

**Caveat: the trace features were designed after reading the generator's mule typology and
looking at test mules,** so their test numbers are optimistic. The $200 threshold is not
sharply tuned (first-time inflows of $100 to $500 from any peer give test AP 0.18 to 0.23
and validation 0.21 to 0.31), and the internal-peer version holds on train (ROC AUC 0.747,
AP 0.116) and validation (0.801, 0.273). This is the profile that shaped the
`pool_activity` and `pool_internal_inflows` groups, and the reason the test audit of every
run that keeps them is for reporting, not selection.

## The mule typology and its traces

The generator (PhantomLedger's mule typology) gives each mule 8 to 26 inbound payments from
victim accounts over a burst of 3 to 11 days, each forwarded 1 to 14 hours later with a 5 to
10% haircut to one pinned destination. Two traces were derived from the messages of an
account's candidate pool (the pool) or from the subset the model's evaluation sampler
attends to (the eval draw):

- **An internal first-time inflow of $200 or more**: an incoming payment (Zelle or other)
  with no earlier event of the pair, of at least $200, from an internal peer.
- **A pass-through signature**: an incoming payment whose next outgoing payment follows
  within 0.5 to 15.1 hours at an amount ratio of 0.88 to 0.96, or the mirror case on an
  outgoing payment.

"Trace" below means at least one of the two.

## Findings

1. **The data carries a strong, specific mule signal, and run 2 did not use it.** 88% of
   test mules have an internal first-time inflow of $200 or more in their pool, against
   12.1% of test non-mules. Counting those inflows alone gives test AP 0.276 and ROC AUC
   0.968 (with a tie-break on the distinct-peer count; 0.233 and 0.921 raw); counting them in
   the eval draw, the slots the model attended to, gives AP 0.20 and ROC AUC 0.93. Both are
   about 100 times run 2's AP.
2. **The signal is weak at the train cutoff.** Only 42% of train mules (36% of the hidden
   ones) show the inflow trace on 2024-07-01, against 88% of test mules on 2025-01-01. 52%
   of train mules have no trace in their pool, and 4 of the 20 revealed training positives
   have none.
3. **Hidden test mules look like revealed ones on volume, less so on Zelle.** On distinct
   peers, incoming payments and history length, hidden and revealed test mules sit at the
   same percentile of non-mules. But 25% of hidden test mules have no Zelle at all, against
   none of the revealed ones, and the pass-through signature appears in 55% of hidden test
   mules against 95% of revealed ones. At the train and validation cutoffs hidden mules are
   much weaker than revealed ones.
4. **Almost no mule lacks history.** The fewest visible events of any mule is 25, and only
   1 of the 40 test mules has fewer than 100. The practical ceiling is the missing trace: 4
   of the 40 test mules (10%) have none in their pool, and 10 of 40 (25%) none in the eval
   draw.
5. **Two kinds of account crowd the top of a ranking.** Busy legitimate Zelle receivers with
   a few first-time inflows, and ring-side accounts the generator labels 0: 4 of the 40
   top-ranked sampled non-mules received money from a true mule, against 0.6% of all
   non-mules.
6. **The limit was mainly the model, not the generator.** There is no sign that mule
   behaviour is indistinguishable, and no round-amount artefact. The data's own limits are
   real but secondary: bursts at random times of the year, and closed-world labels that make
   the ring-side accounts negatives.

## Revealed and hidden mules

Label sources: train 12 revealed by victim report and 8 by monitoring, 140 hidden;
validation 8 by victim report, 1 by monitoring and 2 by network trace, 22 hidden; test 17 by
victim report, 2 by monitoring and 1 by network trace, 20 hidden.

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

ROC AUC of each kind of test mule against the non-mules, revealed / hidden: distinct-peer
stratum 0.873 / 0.886, incoming payments in the pool 0.866 / 0.837, visible events 0.719 /
0.725, Zelle inflows in the pool 0.852 / 0.646, mean flow observation seconds 0.803 / 0.645.

What sets hidden test mules apart is Zelle. Revealed mules are selected mainly through
victim reports, and in this graph only Zelle inflows carry a fraud verdict, so a mule
without Zelle can be revealed only by monitoring or network tracing. 5 of the 20 hidden test
mules, and 60 of the 140 hidden train mules, have no Zelle by their cutoff; no revealed test
or validation mule lacks it.

Revealed and hidden mules are separable: a cross-validated logistic regression on
within-split percentiles tells them apart with ROC AUC 0.75 over all 233 mules (0.70 to 0.79
over 20 seeds) and 0.70 within test. And a model trained on revealed labels ranks hidden
mules lower: logistic regression on pool features trained on the 20 revealed train mules
scores test ROC AUC 0.933 for revealed mules and 0.848 for hidden ones (AP against the
non-mules 0.111 against 0.003); trained on all 160 train mules, 0.969 against 0.909. The
label bias is real, works through Zelle and victim reports, and costs the most at the head
of the ranking.

## The detection ceiling

| Split | Mules | No trace in the pool | No trace in the eval draw | No trace and no Zelle |
|---|---|---|---|---|
| Train | 160 | 83 (52%) | 101 (63%) | 45 |
| Validation | 33 | 12 (36%) | 18 (55%) | 4 |
| Test | 40 | 4 (10%) | 10 (25%) | 2 |

- **History.** The fewest visible events: train 25, validation 109, test 46. The 233 sampled
  accounts with no history at all are all non-mules (3.1%, 2.3% and 3.2% of the sampled
  non-mules), so a no-history ceiling barely exists.
- **Training positives without a trace.** 4 of the 20 revealed train mules have no trace in
  their pool on 2024-07-01: 3 of the 8 revealed by monitoring, which does not depend on
  behaviour, and 1 of the 12 revealed by a victim report. In the eval draw, 6 of 20 have
  none.
- **Why.** The generator draws each burst at a uniform offset inside its ring's window, so
  by mid-year about half of the bursts have not happened. The earliest pass-through event in
  the pool is a median 57 days before the cutoff for train mules and 126 days for test
  mules. Phase-1 visibility also hides about 30% of internal counterparties at the train
  cutoff. Test mules are scored with the whole year visible, and 90% of them carry a trace.

## The non-mules that look most like mules

Ranked by the single count of internal first-time inflows of $200 or more, 20 test mules
are in the top 68 population accounts (precision 0.30) and 30 in the top 380; the remaining
4 traced mules come only around 7,800, because 12.1% of the non-mules (about 5,800 test
accounts) have at least one such inflow. A logistic regression on pool aggregates, trained
on the train ground truth, needs the top 895 accounts for 20 mules and 3,036 for 30.

The top 40 sampled non-mules of the single count, medians against mules and against all
non-mules:

| Feature | Top 40 non-mules | Mules | All non-mules |
|---|---|---|---|
| Internal first-time inflows of $200+ | 2 | 3 | 0 |
| Zelle inflows | 6 | 5.5 | 0 |
| Distinct-peer stratum | 8 | 8 | 5 |
| Visible events | 368 | 459 | 320 |
| 30-day distinct payers | 3 | 3 | 1 |

12 of the 40 also send Zelle and 3 have a pass-through signature: busy legitimate Zelle
users receiving money from new internal peers. Without the internal-peer condition,
first-time inflows of $200 or more mislead: the non-mules' come 79% from external peers
(median $2,097, mostly ACH or unknown rail, like payroll), the mules' 11% (median $968, a
third of them Zelle), and the external-peer version alone ranks at ROC AUC 0.68.

The generator's closed-world label makes only the designated mule account positive. The
accounts of the ring's other members, which receive the mules' forwards and act as layering
hops, are labelled 0, and its default ring settings imply about two of them per mule (a
count the study did not check against the graph). The 3
sampled test non-mules that received at least $200 from a mule rank at the 100th, 99.9th and
81st percentile of the single count, and the top two look exactly like mules (7 first-time
inflows of $200 or more, 4 pass-through events, 9 to 12 Zelle inflows and 14 to 16 Zelle
outflows).

## Data or model

The model, mostly. The typology leaves a clear footprint, and simple models on the same
inputs beat run 2 on the test population:

| Ranking | Test AP | Test ROC AUC |
|---|---|---|
| Count of internal first-time inflows of $200+, pool | 0.276 | 0.968 |
| The same count, eval draw only | 0.204 | 0.931 |
| Count of first-time inflows of $200+ from any peer, pool | 0.226 (0.14 to 0.42) | 0.94 |
| Logistic regression on pool aggregates, all 160 train mules | 0.103 (0.04 to 0.20) | 0.939 |
| The same, only the 20 revealed train mules (PU) | 0.062 | 0.891 |
| Logistic regression on the mean and maximum of the eval draw | 0.006 | 0.83 |
| Logistic regression on the summary features the model did not get | 0.004 | 0.76 |
| Run 2's audit | 0.0024 | 0.782 |

The mean and maximum of the eval draw do as poorly as run 2, while a single count over the
same slots does 80 times better. The signal is a conjunction to be counted (incoming, a
first-time peer, $200 or more, an internal peer), and averaging the edge inputs dilutes it;
run 2's root query was one constant vector and its aggregation a softmax-weighted average,
neither of which can count. That explanation fits the numbers but was not proven; run 3,
with the pool counts and the slot sum, reached an audit AP of 0.134.

Some features are weak by design, not because mules behave normally: the flow delay measures
the delay to the account's next outgoing payment of any kind (for busy accounts, usually a
card payment within hours), and windows that end at the cutoff rarely catch a 3 to 11 day
burst at a random time of the year (the 7-day out-to-in ratio has ROC AUC 0.51 to 0.55).

The data's own limits: bursts at random times, so a single mid-year training cutoff learns
from weak or traceless positives; closed-world labels; first-time peers are more common at
the train cutoff (16.2% of train non-mules have an internal first-time inflow of $200 or
more, against 12.1% at test) because the pair history is relative to the visible history,
and phase-1 visibility hides internal peers; and, in this load, no ring structure in the
labels (`mule_ring_id` was -1 for every mule).

## Not carried over

The profile scripts (`profile/p1_groups.py` to `p11_misc.py` and `load_messages.py`) read
the study's saved candidate pools from local files, which stay outside the repository, and
answered these questions once. Their trace definitions became the `pool_activity` and
`pool_internal_inflows` groups (the $200 band as the bands of 100 and 1,000), and the
analyses that should run again (each feature alone, the baselines, the revealed and hidden
mules) are `mule diagnose univariate`, `baselines` and `subgroups`.
