# Label reveal

Which mules the model may know, and since when. A fresh PhantomLedger load masks every
mule, leaving training no positives, so the first preparation runs
[`reveal_mule_labels`](../reference/queries.md#reveal_mule_labels)
(`gsql/queries/label_reveal.gsql`) once: it decides which mules a bank would
realistically have confirmed, and when, and writes that into the graph's [Account label
contract](../reference/labels.md), with no file involved. Training reads only the
revealed positives and their discovery clocks; the ground truth stays for the audit.

## Why not reveal 20 at random

Banks mostly find mules reactively: a victim reports a scam to their own bank, the report
reaches the bank holding the receiving account, which investigates, and investigators
follow the money to linked accounts; proactive monitoring is real but smaller. So
labelled mules are a biased sample ("selected at random" in positive-unlabelled terms,
not "selected completely at random"), known only after a delay, and a uniform reveal at
the cutoff would hide both effects and flatter the model.

The graph's per-payment fraud verdict on Zelle transfers (`fraud_label`,
`label_available_ts_ms`) arrives at the payment instant, as the PhantomLedger generator
documents. The reveal uses it only as "this payment was a scam" and simulates the
reporting, notification and investigation delays itself.

## The discovery model

Each internal mule gets one discovery time from three channels, with deterministic draws
(a modular hash of event sequences and the salt, so the same graph always reveals the
same mules):

| Channel | Mechanism | Parameters | Basis |
|---|---|---|---|
| Victim report | Each fraud-labelled Zelle inflow is reported with probability 0.65 | `p_report = 0.65` | 56% of UK APP victims reported to their bank, 26% nowhere ([PSR victim survey 2025](https://www.psr.org.uk/media/2oflsqit/psr_app-victim-fraud-2025_report_web_version.pdf)); range 0.55 to 0.75 |
| | Reporting delay, by scam type | A lognormal mixture: 69% fast (median 1 day, sigma 1.3), 10% medium (14 days, 1.0), 21% slow (60 days, 1.0) | Case mix from [UK Finance 2025](https://www.ukfinance.org.uk/system/files/2025-05/UK%20Finance%20Annual%20Fraud%20report%202025.pdf); more than half realise within a day ([BioCatch 2024](https://www.biocatch.com/press-release/nearly-half-of-scam-victims-lost-money-more-than-once-in-last-two-years)); investment and romance victims take months ([Lloyds 2023](https://www.lloydsbankinggroup.com/media/press-releases/2023/lloyds-bank-2023/lloyds-bank-issues-warning-over-crypto-scams.html), [UK Finance 2026](https://www.ukfinance.org.uk/system/files/2026-06/UK%20Finance%20Fraud%20Report%202026.pdf)). The medium component is an assumption |
| | Notification to the mule's bank | 85% within a day, 15% lognormal (7 days, 1.0) | Zelle rule: report to Early Warning within one business day ([EWS 2025](https://www.earlywarning.com/sites/default/files/2025-09/EWS%20Actions%20to%20Address%20Payments%20Fraud%20RFI%20(September%2018%202025).pdf)); the tail reflects late reporting alleged by the [CFPB 2024](https://files.consumerfinance.gov/f/documents/cfpb_Zelle-Complaint_2024-12.pdf) |
| | A report leads to action | First report 0.5, later reports 0.7 | Early Warning restricts after independent reports; many accounts with five or more complaints were not restricted (CFPB 2024). Not measured anywhere: a declared judgement |
| | Investigation | Lognormal (5 days, 1.0) | Zelle's 5-business-day response rule (EWS 2025); 84% of UK claims closed within 5 business days ([PSR 2025](https://www.psr.org.uk/news-and-updates/thought-pieces/thought-pieces/the-story-so-far-a-snapshot-of-what-we-ve-seen-since-our-app-scams-reimbursement-requirement-went-live/)) |
| Network trace | A discovered mule exposes each mule it exchanged money with before its discovery | 0.25 per counterpart, lognormal (30 days, 1.0), three rounds | Firms act on first-generation mules and trace funds through several accounts ([FCA 2025](https://www.fca.org.uk/publications/multi-firm-reviews/firms-use-national-fraud-database-money-mule-account-detection-tools), [FCA 2026](https://www.fca.org.uk/publications/multi-firm-reviews/money-mules-activity-cashing-out-findings)). The probability is an assumption |
| Monitoring | A constant proactive hazard from account opening | 0.00045 per day (about 8% by July, 15% by year end) | No public rate exists; a declared assumption, range 7% to 30% a year |

The discovery time is the earliest channel's. The label becomes available at the end of
that UTC day, with the last sequence at or before it.

## Which mules are revealed

A mule is eligible for its scope partition's split only if discovered before that
split's cutoff (train 2024-07-01, validation 2024-10-01, test 2025-01-01). Up to
`scope.reveal_per_split` (20) eligible mules per split are revealed by stratified Pareto
pi-ps sampling (Rosen 1997), which gives each its intended inclusion probability
exactly. The propensity is `0.05 + 0.95 * sigmoid(z)`, with `z` the standardised
`log(1 + reports received by the cutoff)`: mules with more victim reports are likelier
to be confirmed, but every eligible mule keeps a positive chance (the positivity
condition of SAR-aware positive-unlabelled learning; [Bekker, Robberechts and Davis
2019](https://arxiv.org/abs/1809.03207)).

A shortfall is reported, never filled: a mule no channel found by the cutoff is not
revealed. On the reference graph, with the configured salt 42, the job found 27, 11 and
23 mules (train, validation, test) discovered before the cutoffs and revealed 20, 11 and
20; validation cannot reach 20 without revealing mules no bank would yet have confirmed.

An earlier offline simulation (written before the job; its script and exports were not
kept) gives the spread over 1,000 runs with independent random draws: mules discovered
before each cutoff, median with the 5th to 95th percentile range.

| Split | Mules | Discovered before the cutoff |
|---|---:|---|
| Train (by 1 July 2024) | 160 | 36 (29 to 44) |
| Validation (by 1 October 2024) | 33 | 14 (10 to 17) |
| Test (by 1 January 2025) | 40 | 23 (18 to 28) |

It was simpler than the job: monitoring started on 1 January 2024 for every mule rather
than at the account's first observation; only the first acted-on report counted; tracing
drew a fresh chance for every link in every round (the job gives each pair of mules one
fixed chance); and a mule found by tracing could expose more mules in the same round (the
job updates every mule once per round).

`mule diagnose reveal-spread` replays the method with the job's own hash: it reads the
job's inputs once (read-only) and runs `reference.label_reveal.plan`, the Python mirror of
`reveal_mule_labels`, for the salts 0 to 999 at the built-in budget. Per salt and split
it writes the mules, those discovered before each cutoff and those revealed to the
study's `reveal_spread.csv`, and its `report.md` prints each one's median and 5th to 95th
percentile beside the configured salt's outcome (drawn in `reveal_spread.png`).
`tests/integration/test_label_reveal.py` (marker `graph`) checks the installed job against
the mirror in a dry run (`apply = FALSE`, `force = TRUE`, so nothing is written even on a
revealed graph): the revealed set, each revealed mule's channel and availability clock,
and the eligible count per split.

## What is written

With `apply = TRUE` the job writes every internal Account's label fields
([Labels](../reference/labels.md#who-writes-the-labels) lists them): discovery clocks as
a mule's availability, and `is_mule_masked = FALSE` and `pu_label = 1` for a revealed
mule. External accounts stay unknown, since PhantomLedger does not calibrate external
mule roles. The label-contract check (`validate_label_contract`) must then report zero
violations.

The job runs once: on a graph with known labels it changes nothing, and later
preparations find them; `force = TRUE` reveals again with other parameters. A prepared
dataset keeps its own copy of the labels, never checked against the graph's (the dataset
id omits the reveal's settings, since they act once), so move the dataset aside to
prepare it again on new labels.

## Limitations

- The action probability per report, the trace probability and the monitoring hazard
  are declared assumptions; no regulator or paper publishes them.
- Only Zelle inflows carry a fraud verdict in the graph, so victim reports on other rails
  are not simulated.
- Ring take-downs by law enforcement are not simulated: the reveal reads no ring id. The
  reference load left `mule_ring_id` at -1 for every mule; PhantomLedger's mule-temporal
  export records each mule's ring, which the audits resample by.
- nnPU assumes positives are selected completely at random; this reveal deliberately
  selects them as a bank would, the setting a production model faces. The audit
  (`mule evaluate`) scores every mule of a split, revealed or hidden, and leads with the
  hidden ones, ranked with the revealed mules removed.
