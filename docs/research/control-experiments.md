# The control experiments: the first three seeds

The `controls` suite trained its seven variants over the seeds 42, 43 and 44 on the CUDA
host: 21 runs on one dataset, each audited against the ground truth on validation and
test ([Run the control experiments](../how-to/run-control-experiments.md)). This note
reads the validation audit AP of every mule, which decisions used when the suite ran; the
test audit is for reporting and is not read here. Decisions now use the validation AP of
the hidden mules, ranked with the revealed mules removed (see the follow-up), which these
runs did not report. It reads the audit in the spirit the project holds to: the data are
synthetic, so the experiments study the method (the loss, how a model is selected, how
it is evaluated, whether seeds are worth combining), never which inputs to delete.

## The results

Validation audit AP, the mean over the three seeds with the standard deviation between
them, and the mean epoch training selected. An epoch is 100 steps.

| Variant | Change | AP, mean over seeds | Standard deviation | Mean selected epoch |
|---|---|---|---|---|
| `drop_time_encoding` | without the time encoding | 0.167 | 0.154 | 1.3 |
| `no_weight_average` | validates and keeps the raw weights | 0.135 | 0.150 | not recorded here |
| `baseline` | the built-in run | 0.106 | 0.081 | 2.7 |
| `no_attention` | the summary model: the root's own inputs, no attention | 0.072 | 0.0034 | 2.3 |
| `no_slot_sum` | attention without the slot sum | 0.063 | 0.054 | 8.3 |
| `no_pool_counts` | without both pool groups | 0.022 | 0.022 | 9.0 |
| `prior_weight` | textbook nnPU: the revealed mules weighted by the prior | 0.0011 | 0.0001 | not recorded here |

`prior_weight`'s was the only delta from the baseline that was consistent, by the rule of
the time: one sign in every seed and a paired interval that excluded zero. With the
context cache serving most contexts, a run took from 0.00 to 0.25 hours on average for
each variant, so many more runs are affordable.

## What is established

- **Textbook nnPU collapses at this mule rate.** With the revealed mules weighted by the
  prior, 0.001, every seed ranked mules at an AP of about 0.001, near what a random
  ranking gets at a prevalence of that order, and the seeds hardly differed. It repeats
  the collapse of the first reference run and of the simulation in [the nnPU positive
  weight](nnpu-positive-weight.md): scoring every account near zero costs the textbook
  objective only the prior.
- **The balanced positive weight is needed.** The baseline's mean AP is about a hundred
  times `prior_weight`'s. This is a finding about the loss under extreme imbalance, not
  about this generator: with so few positives among so many accounts the textbook
  objective is cheapest at the constant scorer whatever the data, as the simulation, a
  problem of the dataset's proportions with no graph, showed.

## What is strong but specific to this generator

- **The pool counts.** Without both pool groups the mean AP fell to 0.022, a fifth of the
  baseline's, and its runs selected late epochs (9.0 on average); over three seeds the
  delta was not consistent. The groups count, over the root's candidate pool, distinct
  payers and first-time inflows from inside the bank, and they were designed after
  reading this generator's mule typology and test-split mules ([the diagnostic
  study](diagnostic-study.md)). A large effect is what that design predicts here, so it
  says what this generator's mules look like more than what a bank's data would reward.

## What is noise so far

- `drop_time_encoding` (0.167) and `no_weight_average` (0.135) above the baseline (0.106),
  and `no_slot_sum` (0.063) below it: each difference is smaller than the standard
  deviation between seeds on either side (0.054 to 0.154), and none was consistent. Three
  seeds cannot tell them from zero, and nothing here is a reason to change the built-in
  run. Were a drop to come out ahead with more seeds, it would say what this generator's
  mules carry, not which input to delete.

## The method problems the suite found

1. **The spread between seeds is as large as the differences between variants.** The
   baseline's three seeds have a standard deviation of 0.081 around a mean of 0.106, and
   every difference but `prior_weight`'s and the pool counts' is smaller. With three
   seeds a variant, a comparison mostly measures which seeds were drawn, and the paired
   interval of the time, which resampled the audit sample alone, could not show it.
2. **Selection on 11 revealed validation mules picked very early epochs for the
   baseline, `no_attention` and `drop_time_encoding`.** Their selected epochs were 2.7,
   2.3 and 1.3 on average, while `no_slot_sum` and `no_pool_counts`, of lower mean APs,
   selected late ones (8.3 and 9.0). At 100 steps an epoch, the weights the three early
   variants kept had trained about 230 to 270 steps, and 130 for the drop. With 16 of
   each step's 64 roots drawn from the 20 revealed training mules, each had been seen
   only about 100 to 220 times by then, and the moving average of the weights, whose
   span grows with the steps until step 890, averaged only about the last 13 to 27
   steps. The proxy AP of 11 mules hangs on where the top few rank, so an early epoch
   that ranks them well wins, and training stops six epochs later. That adds variance to
   the variants that stop early, and since the baseline is one of them it blurs
   `no_weight_average`'s question: so early, the average and the raw weights are close.
3. **The audit has 33 validation mules.** Its intervals are wide, so differences of a few
   hundredths of AP need many seeds, or more mules, to be told apart.

## Why no_attention points to ensembles

The summary model of `no_attention` reads only the root's own inputs, and its three seeds
agreed to a standard deviation of 0.0034, against the graph model's 0.081; its mean, 0.072,
lies below the baseline's but well within the baseline's spread. That is not a reason to
delete the neighbour inputs. It is a question of method: the graph model's ranking
depends on its seed (its initial weights, the batches it draws and the neighbours it
samples), and if the scores of several seeds, averaged, keep the graph model's mean and
gain the summary model's steadiness, a seed ensemble is the way to use it.

## The follow-up

- **Ten seeds**, 42 to 51. Complete runs of the same settings are kept, but the built-in
  scope is now `strict_mule_v3`, of other split shares, and the scope id is a setting of
  every run, so every run of `strict_mule_v2` is trained again, not only the seven new
  seeds: the `controls` suite trains all 90 runs.
- **The `methods` suite** (`python scripts/run_experiments.py methods`) asks how to choose a
  model when a handful of mules are known: by the proxy ROC AUC, which counts where every
  known mule ranks, by the run's own nnPU risk on the validation sample, or not at all
  (ten epochs, the last weights kept). It also asks how much a mule rate known only
  roughly matters, with the prior at a tenth and at ten times the built-in one; under the
  balanced weight a near-null result is expected, and is itself a finding about it.
- **Seed ensembles**: every suite's report audits each variant's seeds combined, their
  scores averaged on the log-odds scale, beside the mean of its seeds.
- **Intervals over both sources**: a delta's interval now resamples the seeds, paired by
  seed, as well as the audit sample, the audit-only interval stays beside it, and the
  report says how many seeds agree on the sign and how many of its comparisons would
  exclude zero by chance. A delta is consistent when its interval over both sources
  excludes zero on the side of the mean: that interval already widens with seeds that
  disagree, so the seeds that agree are shown beside it and do not decide, and with ten
  seeds a rule that every seed agree would refuse a clear difference for one noisy
  seed.
- **Proxy reliability**: how far the validation proxy, the only thing a bank has, ranks
  the runs and the variants as the audit does, within each selection rule, and on the
  hidden mules alone.
- **Two linear controls** in the `controls` suite. A logistic regression of the root's
  own model inputs, fitted on the same 20 revealed train mules (the diagnostics' "model,
  LR"), reached a validation audit AP of 0.232 against the baseline run's 0.081, with the
  same ROC AUC. `linear` trains one linear layer of those inputs with the run's own loss,
  selection and pipeline, and `wide_and_deep` adds that layer's output to the graph
  model's logit, so the suite can tell whether the graph model loses what a linear score
  of the root's inputs finds, and whether it gains from having both.
- **The hidden mules first.** The model exists to find the mules nobody knows on the
  scoring date, and the diagnostics of the reference graph showed the built-in run
  ranking the revealed mules far better than the hidden ones: validation's top 1% held 7
  of its 11 revealed mules and 5 of its 22 hidden ones. Every audit now ranks the hidden
  mules against the non-mules with the revealed ones removed, as an investigator would
  remove the cases already known, and a suite ranks, compares and ensembles its variants
  by their validation AP of the hidden mules; the AP of every mule stays beside it.
- **A larger synthetic dataset with more mules at the same rate.** 33 validation mules
  make wide audit intervals and 11 revealed ones a noisy selection; a larger population
  at the same mule rate keeps the problem's imbalance and gives both more mules. It needs
  a new load of the generator's data into the graph, a step for the owner.
