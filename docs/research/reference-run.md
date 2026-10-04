# Reference runs

Three full training runs on the CUDA host, on the same dataset and revealed labels. They
record where the model stood before the restructuring and why the built-in settings are what
they are. Their checkpoints load only with the code before the layered restructure, such as
its last commit, f203226. The last section is the built-in run on the [300,000-person
graph](#the-300000-person-graph-2026-10-04), which led the reveal to make every discovered
mule known.

## How the numbers are measured

- **Proxy metrics**, recorded in training: a split's revealed mules against up to 2,000
  unlabelled accounts counted as negatives. Validation has 11 revealed mules, so its proxy
  prevalence is 0.0055 and its AP moves in large steps. The selected ("best") epoch has the
  highest validation proxy AP; training stops after 6 epochs without improvement.
- **The ground-truth audit** (that code's `evaluate-final`): all 40 test mules plus 2,000
  uniform non-mules, weighted by inverse inclusion probability to the 47,749 test accounts
  (prevalence 0.00084). The sample depends only on the test population, the truth and the
  split seed, so runs 2 and 3 were audited on the same accounts. Recall in the top k% is the
  share of the 40 mules within the top k% of accounts by weight.

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

- **Run 1, textbook nnPU.** The loss collapsed to 0.0010 within one epoch, every score near
  1.7e-7: with the positive weight equal to the prior, scoring every account near zero costs
  only the prior ([the nnPU positive weight](nnpu-positive-weight.md)).
- **Run 2, imbalanced nnPU, no pool counts.** It learned in its first epoch, then drifted: the
  validation proxy AP swung between 0.011 and 0.096 from epoch to epoch, and early stopping
  kept the peak at epoch 7. Scores were bimodal, most near 0.00007 or 0.99999. The swing is why
  validation now scores a moving average of the weights.
- **Run 3, the current built-in run.** The `pool_activity` and `pool_internal_inflows` groups,
  the slot sum and weight averaging (decay 0.99); 101,121 parameters. Early stopping ended it
  at epoch 11, epoch 5 selected. The F1 threshold from the 11 validation mules (0.99999) gives
  audit precision 0.077 and recall 0.15; with 11 positives it is not meaningful, so read the
  ranking metrics and budgets instead.

## The diagnostic study between runs 2 and 3

Why could run 2 not detect mules? Every account had the same root input, while the strongest
signals are counts over the candidate pool TigerGraph already returns (distinct payers,
incoming payments, inflows from first-time payers); that led to run 3's pool counts and slot
sum. The study's numbers, its caveat on test results and what became of its scripts are in
its notes:

| Note | What it holds |
|---|---|
| [The diagnostic study](diagnostic-study.md) | the constant root input, the baselines on the root features and the pool counts, the learning curve, the cutoff shift, and where each script of the study went |
| [The mule profile](mule-profile.md) | what separates mules from other accounts; revealed against hidden mules; the detection ceiling |
| [The nnPU positive weight](nnpu-positive-weight.md) | the collapse of run 1 and the simulation of the positive weight |

## The 300,000-person graph (2026-10-04)

The built-in run, retrained on PhantomLedger data of 300,000 people with ring ids, split
50, 25 and 25% (commit 56f4b30, dataset `c6e1af6004a5`). Each held-out split has about
119,000 accounts and 101 mules; the reveal capped every split at 20 known mules.

| Audit | Validation | Test |
|---|---|---|
| Hidden mules / revealed | 81 / 20 | 81 / 20 |
| Hidden-mule AP (90% interval) | 0.209 (0.132 to 0.310) | 0.168 (0.100 to 0.276) |
| Hidden-mule ROC AUC | 0.764 | 0.800 |
| Hidden mules in the top 1 / 5 / 10% | 0.333 / 0.370 / 0.432 | 0.309 / 0.481 / 0.519 |
| Every mule's AP / ROC AUC | 0.272 / 0.783 | 0.264 / 0.832 |

- **A third found, the rest not.** A third of the hidden mules rank in the top 1%; from
  there to the top 10% the recall barely moves. The model finds one kind of mule sharply
  and ranks the others like everyone else.
- **Twenty training mules.** Training learned from the 20 revealed train mules. The loss
  fell to 0.02, and the validation proxy ROC AUC peaked at epoch 2 (0.896) and slid to
  0.857 while its AP on the 20 revealed validation mules rose (epoch 7 selected): the model
  narrowed to the kind of mule it was shown. On the 200,000-person graph a logistic
  regression went from AP 0.14 with 20 train mules to 0.24 with 160
  ([the diagnostic study](diagnostic-study.md)).
- **What changed.** The reveal now makes every mule a bank would have discovered by the
  cutoff known, as a bank trains on every mule it has confirmed
  ([Label reveal](../explanation/label-reveal.md)). The hidden mules left are then the
  ones no channel had found, so hidden-mule numbers are not comparable with this run's.
- **What to test next.** PhantomLedger's rings act in a 7 to 14 day burst and behave
  normally otherwise, and the model reads a context of each account's recent and a few
  older payments. `mule diagnose activity-timing` sorts each audited mule by when its last
  fraud-labelled inflow came before the cutoff: a run that finds the recent ones and
  misses those active months earlier needs a longer view of history.
