# Reference runs

Three full training runs on the CUDA host, on the same dataset and the same revealed labels.
They record where the model stood before the restructuring and why the built-in settings are
what they are. The checkpoints of runs 1 and 2 load only with the code at the `pre-restructure`
tag. The diagnostic study that sits between runs 2 and 3 is recorded in
[the diagnostic study](diagnostic-study.md), [the mule profile](mule-profile.md) and
[the nnPU positive weight](nnpu-positive-weight.md).

## How the numbers are measured

- **Proxy metrics** are what training records: a split's revealed mules against a sample of up
  to 2,000 unlabelled accounts, which count as negatives. Validation has 11 revealed mules, so
  its proxy prevalence is 0.0055 and its AP moves in large steps. The selected ("best") epoch
  is the one with the highest validation proxy AP, and training stops after 6 epochs without
  improvement.
- **The ground-truth audit** (`mule-temporal evaluate-final`) scores all 40 test mules plus
  2,000 uniform non-mules, weighted by inverse inclusion probability to the 47,749 test accounts
  (prevalence 0.00084). The sample depends only on the test population, the truth and the split
  seed, so runs 2 and 3 were audited on the same accounts. Recall in the top k% is the share of
  the 40 mules ranked within the top k% of accounts by weight.

## Results

| | Run 1: textbook nnPU | Run 2: imbalanced nnPU | Run 3: built-in run |
|---|---|---|---|
| Positive weight | `prior`, 0.001 | `balanced` (1 - prior), 0.999 | `balanced` (1 - prior), 0.999 |
| Pool counts and slot sum | no | no | yes |
| Validates a moving average of the weights | no | no | yes, decay 0.99 |
| Epochs run (selected) | 30, in 3.3 hours | 13 (7) | 11 (5) |
| Validation proxy AP / ROC AUC | 0.017 / 0.76 | 0.096 / 0.894 | 0.566 / 0.959 |
| Test proxy AP / ROC AUC | 0.020 / 0.73 | 0.049 / 0.875 | 0.595 / 0.980 |
| Audit AP / ROC AUC | not recorded | 0.00244 / 0.782 | 0.134 / 0.931 |

Audit: mules ranked in the top share of the 47,749 test accounts (recall in brackets).

| Top share of accounts | Run 2 | Run 3 |
|---|---|---|
| 1% | 1 of 40 (0.025) | 19 of 40 (0.475), precision 0.040 |
| 5% | 7 of 40 (0.175) | 26 of 40 (0.65) |
| 10% | 15 of 40 (0.375) | 31 of 40 (0.775) |
| 20% | 22 of 40 (0.55) | not recorded |
| 50% | 35 of 40 (0.875) | not recorded |

**Run 1, textbook nnPU.** The loss collapsed to 0.0010 within one epoch, with every score near
1.7e-7: with the positive weight equal to the prior, scoring every account near zero costs only
the prior.

**Run 2, imbalanced nnPU, no pool counts.** The model learned in its first epoch and then
drifted: the validation proxy AP swung between 0.011 and 0.096 from epoch to epoch, and early
stopping kept the peak at epoch 7. The scores were bimodal, most near 0.00007 or 0.99999. The
swing is why validation now scores a moving average of the weights.

**Run 3, the current built-in run.** The `pool_activity` and `pool_internal_inflows` groups, the
slot sum and weight averaging (decay 0.99); 101,121 parameters. Early stopping ended it at epoch
11 with epoch 5 selected. The F1 threshold picked on the 11 validation mules (0.99999) gives an
audit precision of 0.077 and recall of 0.15. With 11 positives that threshold is not meaningful;
read the ranking metrics and the budgets instead.

## The diagnostic study between runs 2 and 3

The study asked why run 2 could not detect mules. It used TigerGraph read-only and ground truth
for analysis only.

- In run 2 every account had the same root input vector, so a mule could only show through
  attention over at most 16 sampled payments.
- The strongest mule signals are counts over the account's candidate pool, the payments
  TigerGraph already returns for it: distinct payers, incoming payments, and inflows from
  first-time payers.
- Logistic regression on the study's 16 pool features, trained on the 20 revealed training
  mules, reached test ROC AUC 0.946 to 0.947, AP 0.20 to 0.26 and recall 0.675 to 0.70 in the
  top 1%, depending on how the unlabelled training accounts were treated. The study scored all
  40 test mules against 3,000 uniform non-mules drawn with seed 7, not the audit's 2,000 drawn
  with seed 42, so these numbers are not paired with the runs' audits.
- Logistic regression on the 165 account-level features of the baselines (not the pool
  features), trained on k random training mules (revealed or hidden) against 3,000 training
  non-mules, gives this label-count curve on the same test sample (the mean of 5 draws, and a
  single run at k = 160):

  | k | 10 | 20 | 40 | 80 | 160 |
  |---|---|---|---|---|---|
  | ROC AUC | 0.68 | 0.78 | 0.83 | 0.87 | 0.90 |

- Revealed mules are louder than hidden ones: they show the typology's traces more often.

**Caveat.** The first-time and internal inflow counts were chosen after reading the generator's
mule typology and test-split mules, so test results for them are optimistic. Decisions use the
validation audit; the test audit is for reporting.

## Where the details are

| Note | What it holds |
|---|---|
| [The diagnostic study](diagnostic-study.md) | the constant root input, the baselines on the root features and the pool counts, the label-count curve, the cutoff shift, and where each script of the study went |
| [The mule profile](mule-profile.md) | what separates mules from other accounts; revealed against hidden mules; the detection ceiling |
| [The nnPU positive weight](nnpu-positive-weight.md) | the collapse of run 1 and the simulation of the positive weight |
