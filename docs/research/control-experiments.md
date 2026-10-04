# The control experiments: the first three seeds

The `controls` suite's seven variants over seeds 42, 43 and 44 on the CUDA host: 21 runs on
one dataset, each audited on validation and test ([Run the control
experiments](../how-to/run-control-experiments.md)). Read here: the validation audit AP of
every mule, the decision metric then (the test audit is for reporting). Decisions now use the
hidden mules' validation AP ([The follow-up](#the-follow-up)), which these runs did not
report. The data are synthetic, so the suite studies the method (loss, selection, evaluation,
combining seeds), never which inputs to delete.

## The results

Validation audit AP over the three seeds, and the mean selected epoch (an epoch is 100 steps):

| Variant | Change | AP, mean over seeds | Standard deviation | Mean selected epoch |
|---|---|---|---|---|
| `drop_time_encoding` | without the time encoding | 0.167 | 0.154 | 1.3 |
| `no_weight_average` | validates and keeps the raw weights | 0.135 | 0.150 | not recorded here |
| `baseline` | the built-in run | 0.106 | 0.081 | 2.7 |
| `no_attention` | the summary model: the root's own inputs, no attention | 0.072 | 0.0034 | 2.3 |
| `no_slot_sum` | attention without the slot sum | 0.063 | 0.054 | 8.3 |
| `no_pool_counts` | without both pool groups | 0.022 | 0.022 | 9.0 |
| `prior_weight` | textbook nnPU: the revealed mules weighted by the prior | 0.0011 | 0.0001 | not recorded here |

Only `prior_weight`'s delta was consistent by the rule of the time (one sign in every seed, a
paired interval excluding zero). With the context cache serving most contexts, a run averaged
0.00 to 0.25 hours per variant, so many more runs are affordable.

## What is established

- **Textbook nnPU collapses at this mule rate.** The prior weight, 0.001, gave AP about 0.001
  in every seed (near random, seeds nearly equal), as in run 1 and the simulation of [the nnPU
  positive weight](nnpu-positive-weight.md).
- **The balanced positive weight is needed.** The baseline's mean AP is about 100 times
  `prior_weight`'s. This is about the loss under extreme imbalance, not this generator: the
  graph-free simulation of the dataset's proportions also made the constant scorer cheapest.

## What is strong but specific to this generator

- **The pool counts.** Without both groups mean AP fell to 0.022, a fifth of the baseline's,
  and runs selected late epochs (9.0); over three seeds not consistent. The groups (distinct
  payers and first-time inflows from inside the bank, over the root's candidate pool) were
  designed after reading this generator's typology and test mules ([the diagnostic
  study](diagnostic-study.md)), so a large effect describes this generator's mules more than
  what a bank's data would reward.

## What is noise so far

- `drop_time_encoding` (0.167) and `no_weight_average` (0.135) above the baseline (0.106), and
  `no_slot_sum` (0.063) below: each gap is under the seed standard deviation on either side
  (0.054 to 0.154) and none was consistent, so nothing changes the built-in run. A drop ahead
  with more seeds would describe this generator's mules, not argue for deleting an input.

## The method problems the suite found

1. **The seed spread rivals the variant differences.** The baseline's standard deviation,
   0.081 around 0.106, exceeds every difference but `prior_weight`'s and the pool counts'. With
   three seeds a comparison mostly measures the seeds drawn, which the paired interval of the
   time, resampling only the audit sample, could not show.
2. **Selection on 11 revealed validation mules stopped some variants very early.** Mean
   selected epochs: baseline 2.7, `no_attention` 2.3, `drop_time_encoding` 1.3, against 8.3
   and 9.0 for the lower-AP `no_slot_sum` and `no_pool_counts`. The early three kept weights
   after about 230 to 270 steps (130 for the drop): with 16 of each step's 64 roots drawn from
   the 20 revealed train mules, each mule had been seen only about 100 to 220 times, and the
   weights' moving average (its span grows with the steps until step 890) covered only the
   last 13 to 27 steps. An 11-mule proxy AP hinges on the top few ranks, so an early epoch
   that ranks them well wins, and training stops six epochs later. That adds variance to the
   early stoppers, the baseline among them, and blurs `no_weight_average`'s question: so early,
   the average and the raw weights are close.
3. **The audit has 33 validation mules.** Wide intervals, so gaps of a few hundredths of AP
   need many seeds or more mules.

## Why no_attention points to ensembles

The summary model reads only the root's own inputs. Its seeds agreed to a standard deviation
of 0.0034 against the graph model's 0.081, and its mean, 0.072, is below the baseline's but
well within its spread. That is no reason to delete the neighbour inputs, but a question of
method: the graph model's ranking depends on its seed (initial weights, batches, sampled
neighbours), so if averaging several seeds' scores keeps its mean and gains the summary
model's steadiness, a seed ensemble is the way to use it.

## The follow-up

- **Ten seeds**, 42 to 51. Complete runs of the same settings are kept, but the scope id is a
  setting of every run and the built-in scope is now `strict_mule_v3` (other split shares), so
  every `strict_mule_v2` run trains again: `controls` trains all 90 runs.
- **The `methods` suite** (`python scripts/run_experiments.py methods`): how to choose a
  model when a handful of mules are known (the proxy ROC AUC, which counts where every known
  mule ranks; the run's own nnPU risk on the validation sample; or not at all: ten epochs, the
  last weights kept), and how much a roughly known mule rate matters (the prior at a tenth and
  ten times the built-in one; under the balanced weight a near-null result is expected and is
  itself a finding).
- **Seed ensembles.** Every suite's report audits each variant's seeds combined (scores
  averaged on the log-odds scale) beside the mean of its seeds.
- **Intervals over both sources.** A delta's interval resamples the seeds (paired by seed) and
  the audit sample, beside the audit-only interval; the report gives how many seeds agree on
  the sign and how many comparisons would exclude zero by chance. A delta is consistent when
  the two-source interval excludes zero on the side of the mean. Seed agreement is shown but
  does not decide: the interval already widens when seeds disagree, and with ten seeds
  requiring them all to agree would refuse a clear difference for one noisy seed.
- **Proxy reliability.** How far the validation proxy, the only thing a bank has, ranks runs
  and variants as the audit does, within each selection rule and on the hidden mules alone.
- **Two linear controls** in `controls`. A logistic regression of the root's own model inputs,
  fitted on the same 20 revealed train mules (the diagnostics' "model, LR"), reached
  validation audit AP 0.232 against the baseline run's 0.081, at the same ROC AUC. `linear`
  trains one linear layer of those inputs with the run's own loss, selection and pipeline;
  `wide_and_deep` adds that layer's output to the graph model's logit. Does the graph model
  lose what a linear score of the root's inputs finds, and gain from both?
- **The hidden mules first.** The model exists to find mules nobody knows on the scoring date,
  yet the reference graph's diagnostics showed the built-in run ranking revealed mules far
  above hidden ones (validation's top 1%: 7 of 11 revealed, 5 of 22 hidden). Every audit now
  ranks the hidden mules against the non-mules with the revealed ones removed, as an
  investigator removes known cases, and a suite ranks, compares and ensembles its variants by
  their validation AP of the hidden mules, every mule's AP beside it.
- **A larger synthetic dataset at the same mule rate.** 33 validation mules make wide audit
  intervals and 11 revealed ones a noisy selection; more mules at the same rate keep the
  imbalance. It needs a new load of the generator's data into the graph, a step for the owner.
