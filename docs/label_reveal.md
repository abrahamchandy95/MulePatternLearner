# Label reveal: which mules the model may know, and since when

A fresh PhantomLedger load masks every mule, so training would have no positives.
The first `mule-temporal train` therefore runs `temporal_reveal_mule_labels`
([`gsql/temporal/label_reveal.gsql`](../gsql/temporal/label_reveal.gsql)) once. It
decides which mules a bank would realistically have confirmed, and when, and writes
that into the existing [Account label contract](account_mule_labels.md). Training
reads only the revealed positives and their discovery clocks
(`label_policy = "graph_observed"`); ground truth stays in the graph for the oracle
audit. No file is involved.

## Why not reveal 20 at random

Banks do not find mules at random. They mostly find them reactively: a victim
reports a scam to their own bank, the report reaches the bank holding the receiving
account, and that bank investigates. Investigators then follow the money to linked
accounts. Proactive monitoring is real but smaller. Labelled mules are therefore a
biased sample of all mules (the "selected at random" case of positive-unlabeled
learning, not "selected completely at random"), and they become known only after a
delay. A uniform reveal at the cutoff would hide both effects and flatter the model.

The graph's per-payment fraud verdict on Zelle transfers (`fraud_label`,
`label_available_ts_ms`) is an oracle that arrives at the payment instant (the
PhantomLedger generator documents this in its `docs/mule_temporal.md`). The reveal
uses it only as "this payment was a scam" and simulates the reporting,
notification and investigation delays itself.

## The model

For each internal mule, one discovery time is simulated from three channels, with
deterministic draws (a modular hash of event sequences and the salt, so the same
graph always reveals the same mules):

| Channel | Mechanism | Parameters | Basis |
|---|---|---|---|
| Victim report | Each fraud-labelled Zelle inflow is reported with probability 0.65 | `p_report = 0.65` | 56% of UK APP victims reported to their bank, 26% nowhere ([PSR victim survey 2025](https://www.psr.org.uk/media/2oflsqit/psr_app-victim-fraud-2025_report_web_version.pdf)); range 0.55 to 0.75 |
| | Reporting delay, by scam type | lognormal mixture: 69% fast (median 1 day, sigma 1.3), 10% medium (14 days, 1.0), 21% slow (60 days, 1.0) | Case mix from [UK Finance 2025](https://www.ukfinance.org.uk/system/files/2025-05/UK%20Finance%20Annual%20Fraud%20report%202025.pdf); more than half realise within a day ([BioCatch 2024](https://www.biocatch.com/press-release/nearly-half-of-scam-victims-lost-money-more-than-once-in-last-two-years)); investment and romance victims take months ([Lloyds 2023](https://www.lloydsbankinggroup.com/media/press-releases/2023/lloyds-bank-2023/lloyds-bank-issues-warning-over-crypto-scams.html), [UK Finance 2026](https://www.ukfinance.org.uk/system/files/2026-06/UK%20Finance%20Fraud%20Report%202026.pdf)). The medium component is an assumption |
| | Notification to the mule's bank | 85% within a day, 15% lognormal (7 days, 1.0) | Zelle rule: report to Early Warning within one business day ([EWS 2025](https://www.earlywarning.com/sites/default/files/2025-09/EWS%20Actions%20to%20Address%20Payments%20Fraud%20RFI%20(September%2018%202025).pdf)); the tail reflects late reporting alleged by the [CFPB 2024](https://files.consumerfinance.gov/f/documents/cfpb_Zelle-Complaint_2024-12.pdf) |
| | A report leads to action | first report 0.5, later reports 0.7 | Early Warning restricts after independent reports; many accounts with five or more complaints were not restricted (CFPB 2024). Not measured anywhere: a declared judgement |
| | Investigation | lognormal (5 days, 1.0) | Zelle's 5-business-day response rule (EWS 2025); 84% of UK claims closed within 5 business days ([PSR 2025](https://www.psr.org.uk/news-and-updates/thought-pieces/thought-pieces/the-story-so-far-a-snapshot-of-what-we-ve-seen-since-our-app-scams-reimbursement-requirement-went-live/)) |
| Network trace | A discovered mule exposes each mule it exchanged money with before its discovery | 0.25 per counterpart, lognormal (30 days, 1.0), three rounds | Firms act on first-generation mules and trace funds through several accounts ([FCA 2025](https://www.fca.org.uk/publications/multi-firm-reviews/firms-use-national-fraud-database-money-mule-account-detection-tools), [FCA 2026](https://www.fca.org.uk/publications/multi-firm-reviews/money-mules-activity-cashing-out-findings)). The probability is an assumption |
| Monitoring | Constant proactive hazard from account opening | 0.00045 per day (about 8% by July, 15% by year end) | No public rate exists; a declared assumption, range 7% to 30% a year |

A mule's discovery time is the earliest of its channels. Its label becomes
available at the end of that UTC day, with the last sequence at or before it.

## Which mules are revealed

A mule is eligible for its scope partition's split only if it was discovered before
that split's cutoff (train 2024-07-01, validation 2024-10-01, test 2025-01-01).
Among the eligible, up to `reveal_per_split` (default 20) are revealed per split by
stratified Pareto pi-ps sampling (Rosen 1997), which gives each mule its intended
inclusion probability exactly. The propensity is
`0.05 + 0.95 * sigmoid(z)`, with `z` the standardized `log(1 + reports received by
the cutoff)`: mules with more victim reports are likelier to be confirmed, but every
eligible mule keeps a positive chance (the positivity condition of SAR-aware PU
learning; [Bekker, Robberechts and Davis 2019](https://arxiv.org/abs/1809.03207)).

A shortfall is reported, never filled: a mule no channel had found by the cutoff is
not revealed. On the 2024 PhantomLedger snapshot, the table below gives the mules
discovered before each cutoff (median, with the 5th to 95th percentile range).
Validation cannot reach 20 without revealing mules no bank would yet have confirmed.

| Split | Mules | Discovered before the cutoff |
|---|---:|---|
| Train (by 1 July 2024) | 160 | 36 (29 to 44) |
| Validation (by 1 October 2024) | 33 | 14 (10 to 17) |
| Test (by 1 January 2025) | 40 | 23 (18 to 28) |

The table comes from an earlier offline simulation, made before the job was
written: 1,000 runs with independent random draws (numpy) instead of the job's
hash, over local exports of the mules' fraud-labelled Zelle inflows, the events
between mules and the mules' splits. That script and its exports were not kept. It
also simplified the model: monitoring started on 1 January 2024 for
every mule rather than at the account's first observation, only the first acted-on
report (in report order) counted, and network tracing drew a fresh chance for every
link in every round, where the job gives each pair of mules one fixed chance; and a mule found by
tracing could expose further mules within the same round, where the job updates
every mule once per round. The job
itself, with the configured salt 42, found 27, 11 and 23 mules (train, validation,
test) discovered before the cutoffs and revealed 20, 11 and 20.

`scripts/temporal/simulate_label_reveal.py` reproduces the method with the job's own
hash. It reads the job's inputs once (read-only, the same query as the check below),
then runs `plan` from
[`reveal_model.py`](../src/mule_pattern_learner/temporal/live/reveal_model.py), the
Python mirror of `temporal_reveal_mule_labels`, for many salts and prints the median and 5th to 95th percentile of the mules
discovered before each cutoff and of those revealed:

```bash
python scripts/temporal/simulate_label_reveal.py --runs 1000
```

`scripts/temporal/verify_label_reveal.py` checks the installed job against the same
mirror: it runs the job with `apply = FALSE` and compares the revealed set, the
channel and availability clock of every revealed mule, and the eligible count per
split. It passes `force = TRUE` so the check also
runs on a graph whose labels were already revealed; with `apply = FALSE` the job
writes nothing.

## What is written

With `apply = TRUE` the job writes every internal Account:
`mule_label_known = TRUE` and effective clocks at first observation. Mules get their
discovery clocks as availability; revealed mules get `is_mule_masked = FALSE` and
`pu_label = 1`, the others stay masked. `mule_label_source` records the version, the
salt and, for revealed mules, the channel. External accounts stay unknown, because
PhantomLedger does not calibrate external mule roles. The contract check
(`temporal_validate_account_supervision`) must report zero violations afterwards.

The job runs once: a graph that already has known labels is left alone. To reveal
again with other parameters, call the query with `force = TRUE`; prepared runs keep
their own copy of the labels they were trained on.

## Limitations

- The per-report action probability, the network-trace probability and the
  monitoring hazard are declared assumptions; no regulator or paper publishes them.
- Only Zelle inflows carry a fraud verdict in the graph, so victim reports on other
  rails are not simulated.
- Ring take-downs by law enforcement are not simulated: the ring id is ground truth
  that the graph does not populate.
- nnPU assumes positives are selected completely at random; this reveal is
  deliberately not, which is the realistic setting a production model faces.
  Evaluate with the oracle audit (`mule-temporal evaluate-final`), which scores all
  test mules, not just the revealed ones.
