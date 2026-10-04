# Run the control experiments

Measure what each part of the built-in run contributes: train variants of it over the
ten seeds 42 to 51, audit every run, and compare each variant with the baseline on the
same accounts. The script takes suite or variant names only. [The control
experiments](../research/control-experiments.md) reads the `controls` suite's first three
seeds: what they established, what is noise so far, and the method problems that led to
the ten seeds, the `methods` suite, the seed ensembles, the intervals over both seeds and
accounts, and the proxy's reliability.

## Run a suite

On the machine that trains (the CUDA host), after `mule train` and `mule evaluate` have
made the baseline's seed 42:

```bash
python scripts/run_experiments.py                            # the controls suite
python scripts/run_experiments.py methods                    # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                     # suites, variants, questions, changes
```

The baseline is always included, and its seed 42 is the run `mule train` makes. The
script, in order:

1. **Validates every variant offline**: each seed's configuration, feature plan and model
   on the CPU, one dataset for all and a distinct fingerprint for each. A failure names
   the variant, before anything connects.
2. **Prepares the dataset** once, as `mule train` does, on the suite's one connection.
3. **Plans every run** and shows the plan as a matrix of variants and seeds, each run's
   action (`keep`, `train`, `resume` or `archive`) in its cell, with the hours the runs
   to train should take (see [Cost](#cost)): a range estimated from the hours the suite's
   finished runs took (the `suite` event's `estimate_hours` and `estimated_from`) and an
   upper bound from the median seconds per step of the latest graph run's `history.csv`
   (`bound_hours` and `timed_from`). A
   complete run of the same settings and dataset is kept, an interrupted one resumes, and
   one whose settings differ, whose `config.json` cannot be read or that was trained on
   another dataset is moved whole to `results/archive/<variant>/seed-<n>/<UTC time>/` and
   trained again. Nothing is deleted.
4. **Trains**, seeds outer and variants inner, each run into `results/<variant>/seed-<n>/`
   with its own figures.
5. **Audits** validation and test for every complete run that lacks them, reading the
   ground truth once for the whole suite.
6. **Compares** the runs and writes `summary.csv`, `comparison.csv`, six figures and
   `report.md` under `results/experiments/<suite>/`, always, even after a failure.

Each run shows its own lines as it trains and audits, then one line as each of its steps
finishes ("baseline seed 43 trained: best epoch 5, validation proxy AP 0.566, 58 min", a
`run_finished` event). A run's own error is a `run_failed` event and the suite goes on; a
TigerGraph outage (the retry budget ran out while the graph was unavailable) is a
`suite_stopped` event and stops the training and audits, since every later run would fail
the same way. The suite's own events are recorded in
`results/experiments/<suite>/events.jsonl`, with the install and connecting that come
before preparation; preparation's are in the dataset's, and each run's in its own. The
script ends with how the suite did, the runs' errors, the top ten variants ranked by
their validation audit AP as `report.md` ranks them (with the delta from the baseline
and the test AP) and where the report is, and exits 1 unless every run is trained and
audited without an error. A run whose figures failed after its numbers were saved stays complete in the
tables, with its error beside it; `mule report results/<variant>/seed-<n>` redraws them.
Run the script again to finish: complete runs are kept. So are the runs of a suite
trained with fewer seeds than `experiments.variants.SEEDS` holds now, such as the
controls suite's first three: running it again trains the new seeds only.

A suite of chosen variants is named by their names joined with hyphens, as in
`results/experiments/no_attention-prior_weight/`.

The data are synthetic, so the suites study the method: the loss, how a run chooses its
weights, how it is evaluated and whether seeds are worth combining, not which inputs to
delete. Run `controls` first: it settles the positive weight (`prior_weight`) and the
model's mechanisms, and every run of every suite holds the source, the revealed labels
and the cutoffs fixed. Then run `methods`, which asks how to choose a model when a handful
of mules are known and how much an only roughly known mule rate matters; the baseline's
runs are shared. `feature_drops` says what each input group carries for this generator's
mules, which is a fact about the generator rather than advice on what to drop.

## The suites and variants

Variants are declared in `src/mule_pattern_learner/experiments/variants.py`. A variant is
a question and a change to the built-in run; it never sets a seed, and none touches
`dataset.seed`, `dataset.split_seed` or `scope.reveal_salt`, so every variant trains on
the baseline's dataset.

| Suite | Variants |
|---|---|
| `controls` (the default) | `baseline`, `no_attention`, `no_slot_sum`, `no_pool_counts`, `prior_weight`, `no_weight_average`, `drop_time_encoding` |
| `feature_drops` | `baseline` and one drop per built-in group but `message_core`: `drop_entity_meta`, `drop_hub_indicator`, `drop_time_encoding`, `drop_pair_history`, `drop_flow_timing`, `drop_pool_activity`, `drop_pool_internal_inflows` |
| `methods` | `baseline`, `select_on_roc_auc`, `select_on_pu_risk`, `fixed_10_epochs`, `prior_tenth`, `prior_tenfold` |
| `all` | Every variant of the three, each once |

| Variant | Question | Change |
|---|---|---|
| `no_attention` | Does attention over sampled neighbours add anything beyond the root's own inputs, pool counts included? | `model.architecture = "summary"`, no slot sum |
| `no_slot_sum` | Does the per-slot MLP sum help beyond attention? | `model.slot_sum = false` |
| `no_pool_counts` | How much of the ranking comes from the candidate-pool counts? | without both pool groups |
| `prior_weight` | Does the balanced positive weight beat textbook nnPU across seeds? | `loss.positive_weight = "prior"` |
| `no_weight_average` | Does selecting on the moving average of the weights help? | `training.weight_average_decay = 0.0` |
| `drop_<group>` | What does the model lose without the group? | without the group, and the groups that read it (`drop_pair_history` also drops both pool groups, `drop_flow_timing` also `pool_activity`) |
| `select_on_roc_auc` | When only a handful of mules are known, does choosing the epoch by their ROC AUC, which counts where every known mule ranks, pick better models than their AP, which hangs on the top few? | `training.selection = "validation_roc_auc"` |
| `select_on_pu_risk` | Can the training objective itself, the nnPU risk on the validation sample, choose the epoch as well as a ranking metric of a handful of known mules? | `training.selection = "validation_pu_risk"` |
| `fixed_10_epochs` | If a handful of known mules is too few to choose an epoch on, is it better not to choose: train ten epochs and keep the last weights, averaged? | `training.selection = "none"`, `training.epochs = 10` |
| `prior_tenth` | A bank knows its mule rate only roughly: how much does the ranking change if the assumed rate is a tenth of the built-in prior? | `loss.class_prior = 0.0001` |
| `prior_tenfold` | The same, if the assumed rate is ten times the built-in prior? | `loss.class_prior = 0.01` |

Under the balanced positive weight the prior acts mainly through the negative-risk
correction, and through how often the non-negative clamp fires (`corrected_steps` in each
run's `history.csv`), so `prior_tenth` and `prior_tenfold` are expected to differ little
from the baseline. A near-null result is itself a finding: the balanced weight makes the
ranking robust to a mule rate known only to a factor of ten. The three selection variants
change only which epoch is kept and when training stops. On the same host settings they
follow the baseline's schedule, so the epochs both train request the contexts the
baseline cached and cost little. The epochs past the baseline's early stop are new:
`select_on_roc_auc` and `select_on_pu_risk` may stop later than it, and
`fixed_10_epochs` always trains ten, more whenever the baseline stopped sooner.

Variants only drop groups or change the model, the loss or the training: training reads
only the built-in run's groups, so no variant adds one. How well a table of the account's
own activity ranks mules, with no neighbour input, is a question for the diagnostics
baselines ([Run the diagnostics](run-diagnostics.md)).

## Cost

The `controls` suite is 70 runs (60 graph runs and the 10 cheap `no_attention` runs,
which fetch no children). A graph run takes about 3 seconds per step on the CUDA host
while it requests its contexts from TigerGraph, about an hour with early stopping; once
the context cache holds them it is far faster. Over the controls suite's first three
seeds each variant's runs took from 0.00 to 0.25 hours on average. A suite trained with
fewer seeds keeps its complete runs, so running `controls` again after the seeds became
ten trains only the seven new ones: 49 runs.

The plan gives two numbers for the runs it trains:

- **The estimate** is a range from the hours the suite's finished runs took (each run's
  `metrics.json`). The suite trains the seeds in turn and the baseline first within
  each, so a new seed's baseline is a cold first run that fills the context cache for
  its seed, and the variants after it read much of it. Each run to train is taken to
  last from the fewest to the most hours a finished run of its variant took; a variant
  with no finished run takes the range of the finished runs of the variants other than
  the baseline. A run that resumes is counted as a whole run, so the estimate is high
  for it. A suite with no finished run has no estimate.
- **The bound** takes every run to train all its epochs at the median seconds per step
  of the latest graph run's `history.csv`. Summary runs are faster, and early stopping
  and the cache end most runs far sooner, so it is an upper bound.

Seeds, and variants that request the same groups, share the cache's entries, and drops
of the client-computed groups (the hub indicator and both pool groups) request the same
contexts as the baseline. Run variants one after another rather than as parallel
processes, which may request the same contexts twice. The paired intervals take about 15
seconds for the controls suite of three seeds and grow with the runs.

## Read the comparison

Open `results/experiments/<suite>/report.md`. It ranks the variants by the seed-mean
validation audit AP and marks the test audit "for reporting, not selection".

- **The paired delta** (`validation_ap_delta`) is the variant's seed-mean validation AP
  minus the baseline's, over the seeds both completed, each seed paired with the
  baseline's run of the same seed. Every audit of a dataset scores the same accounts, so
  each bootstrap replicate resamples those accounts and their rings once and applies the
  resample to every run. Its interval (`validation_ap_delta_low` and `_high`) covers both
  sources of uncertainty: each replicate also resamples the seeds, so it widens with the
  spread between them. The audit-only interval beside it (`validation_ap_delta_audit_low`
  and `_high`) resamples the accounts alone, for these seeds. Over the controls suite's
  first three seeds the spread between seeds was as large as the differences between
  variants, which only the two-source interval shows (`comparison_delta.png` draws both,
  with the per-seed deltas). Its seed half is a percentile bootstrap over the seeds,
  which is too narrow with few of them: resampling n seeds understates their variance
  by about (n - 1) / n and draws few distinct sets (three seeds give ten), so the
  interval is reliable only with many seeds, such as the ten every suite now trains.
- **The seeds that agree** (`validation_ap_delta_agreeing` of `validation_ap_delta_seeds`,
  "8 of 10" in report.md) are those whose own delta has the sign of the mean.
- **`consistent`** is true when the two-source interval excludes zero and every seed
  compared agrees on the sign. report.md states how many comparisons the suite makes
  and how many would exclude zero by chance: the `all` suite compares 17 variants with
  the baseline, so at 90% about 1.7 would even if no variant differed. Treat a single
  consistent delta as a lead to repeat.
- **Seed ensembles** combine each variant's seeds (two or more) into one model: their
  scores of the accounts every audit scored, averaged on the log-odds scale, audited on
  validation and test with a run's ranking metrics. Like the seed means, an ensemble has
  an interval for its AP alone, over the same replicates; a run's audit report has one
  for every ranking metric. On the log-odds scale a seed that is confident about an
  account weighs more than a hesitant one, where averaging ranks would give every seed
  the same say. report.md
  ranks them in a section of their own beside the mean of each variant's seeds, and
  `comparison_ap.png` draws each as a hollow diamond below its variant's mean. An
  ensemble above the mean of its seeds gains from their disagreement: a method question
  for a ranking that varies from seed to seed.
- **The proxy's reliability** says how far a bank, which has only the proxy, could trust
  it to choose among these models: Spearman's rank correlation of each run's selected
  validation proxy AP with its validation audit AP, with n beside it, over all runs,
  within each selection rule (selecting on a criterion biases the selected proxy AP), over
  the variants' rankings by their seed means, and against the audit on the hidden mules
  alone, which the proxy never sees. `comparison_proxy_vs_audit.png` draws the runs and
  gives the same numbers.
- **`unpaired_accounts`** counts validation accounts some run's audit rejected; they are
  left out of the pairing.
- **`differs`** lists runs whose commit, dirty state, device or sampler backend differ
  from the suite's usual value; their differences are not the variant's alone.
- **The pool groups' test numbers are optimistic**: the groups were designed after
  reading test-split mules, which the report repeats.

The reference graph holds 233 mules across its three splits, 33 in validation and 40 in
test, so an audit has few mules and its intervals are wide. [Outputs](../reference/outputs.md#a-suite-resultsexperimentssuite)
lists every column and figure, and `mule report results/experiments/<suite>` redraws them
offline.

## Declare a new variant

Add a `Variant(name, question, change)` to `experiments/variants.py`, where `change` maps
a `RunConfig` to another with `dataclasses.replace` or `RunConfig.with_changes`, and add
it to a suite. The tests build every variant's configuration, plan and model offline,
check that it shares the baseline's dataset and has a fingerprint of its own, and the
script lists it under `--help`. A variant that changed a dataset setting would need a
dataset of its own, and the suite refuses it.
