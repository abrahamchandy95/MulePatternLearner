# Run the control experiments

Measure what each part of the built-in run contributes: train variants over the ten seeds
42 to 51, audit every run, and compare each variant with the baseline on the same
accounts. [The control experiments](../research/control-experiments.md) reads the
`controls` suite's first three seeds: what they established, what is noise so far, and the
method problems that led to the ten seeds, the `methods` suite, the seed ensembles,
intervals over both seeds and accounts, and the proxy's reliability.

## Run a suite

On the CUDA host, after `mule train` and `mule evaluate` made the baseline's seed 42. The
script takes suite or variant names only:

```bash
python scripts/run_experiments.py                            # the controls suite
python scripts/run_experiments.py methods                    # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                     # suites, variants, questions, changes
```

The baseline is always included; its seed 42 is the `mule train` run. Chosen variants make
a suite named by their names joined with hyphens. The script:

1. **validates every variant offline** before connecting (each seed's configuration,
   feature plan and model on the CPU, one shared dataset, a distinct fingerprint each); a
   failure names the variant;
2. **prepares the dataset** once, as `mule train` does, on the suite's one connection;
3. **plans every run**, shown as a matrix of variants and seeds with each run's action
   (`keep`, `train`, `resume` or `archive`) and the hours to train ([Cost](#cost)). A
   complete run of the same settings and dataset is kept and an interrupted one resumes;
   any other is archived and trained again ([The
   archive](../reference/outputs.md#the-archive-resultsarchive)). Nothing is deleted;
4. **trains**, seeds outer and variants inner, each into `results/<variant>/seed-<n>/`
   with its figures;
5. **audits** validation and test of every complete run lacking them, reading the ground
   truth once per suite;
6. **compares** into `results/experiments/<suite>/`, even after a failure
   ([Outputs](../reference/outputs.md#a-suite-resultsexperimentssuite)).

Each run prints its own lines, then one per finished step ("baseline seed 43 trained: best
epoch 5, validation proxy AP 0.566, 58 min", `run_finished`). A run's error is
`run_failed` and the suite goes on; a TigerGraph outage (retries ran out) is
`suite_stopped` and ends training and audits, since every later run would fail too. The
script ends with the outcome, the runs' errors, the top ten variants as `report.md` ranks
them (with the delta, the hidden mules' test AP and every mule's validation AP) and the
report's path, and exits 1 unless every run trained and audited without error. A run whose
figures failed after its numbers were saved stays in the tables with its error;
`mule report results/<variant>/seed-<n>` redraws them.

Run the script again to finish: complete runs of the same settings are kept, so a suite
trained with fewer seeds than `experiments.variants.SEEDS` holds trains only the new ones.
A run of another scope id differs in its settings, though ([Cost](#cost)).

The data are synthetic, so the suites study the method (the loss, how a run chooses its
weights, how it is evaluated, whether seeds are worth combining), not which inputs to
delete; every run holds the source, revealed labels and cutoffs fixed. Run `controls`
first: it settles the positive weight (`prior_weight`) and the model's mechanisms. Then
`methods`: how to choose a model when a handful of mules are known, and how much a roughly
known mule rate matters; it shares the baseline's runs. `feature_drops` says what each
input group carries for this generator's mules, a fact about the generator, not advice on
what to drop.

## The suites and variants

Variants are declared in `src/mule_pattern_learner/experiments/variants.py`, each a
question (in full under `--help`) and a change to the built-in run. None sets a seed or
touches `dataset.seed`, `dataset.split_seed` or `scope.reveal_salt`, so all share the
baseline's dataset.

| Suite | Variants |
|---|---|
| `controls` (the default) | `baseline`, `no_attention`, `no_slot_sum`, `no_pool_counts`, `linear`, `wide_and_deep`, `prior_weight`, `no_weight_average`, `drop_time_encoding` |
| `feature_drops` | `baseline` and one drop per built-in group but `message_core`: `drop_entity_meta`, `drop_hub_indicator`, `drop_time_encoding`, `drop_pair_history`, `drop_flow_timing`, `drop_pool_activity`, `drop_pool_internal_inflows` |
| `methods` | `baseline`, `select_on_roc_auc`, `select_on_pu_risk`, `fixed_10_epochs`, `prior_tenth`, `prior_tenfold` |
| `all` | Every variant of the three, each once |

| Variant | Question | Change |
|---|---|---|
| `no_attention` | Does attention over sampled neighbours add anything beyond the root's own inputs, pool counts included? | `model.architecture = "summary"`, no slot sum |
| `no_slot_sum` | Does the per-slot MLP sum help beyond attention? | `model.slot_sum = false` |
| `no_pool_counts` | How much of the ranking comes from the candidate-pool counts? | without both pool groups |
| `linear` | Can one linear layer of the account's own inputs, which a bank could explain score by score, find hidden mules as well as the graph model, with the same loss and selection? | `model.architecture = "linear"`, no slot sum |
| `wide_and_deep` | A logistic regression of the account's own inputs, fitted on the same revealed mules, ranked the reference graph's validation mules better than the graph model: does such a linear score added inside the graph model find more hidden mules than either alone? | `model.architecture = "wide_and_deep"` |
| `prior_weight` | Does the balanced positive weight beat textbook nnPU across seeds? | `loss.positive_weight = "prior"` |
| `no_weight_average` | Does selecting on the moving average of the weights help? | `training.weight_average_decay = 0.0` |
| `drop_<group>` | What does the model lose without the group? | without the group and the groups that read it (`drop_pair_history` also drops both pool groups, `drop_flow_timing` also `pool_activity`) |
| `select_on_roc_auc` | With a handful of known mules, is their ROC AUC (every rank counts) a better epoch choice than their AP (the top few decide)? | `training.selection = "validation_roc_auc"` |
| `select_on_pu_risk` | Can the training objective, the nnPU risk on the validation sample, choose the epoch as well as a ranking metric? | `training.selection = "validation_pu_risk"` |
| `fixed_10_epochs` | If a handful of mules is too few to choose an epoch on, is it better not to choose: ten epochs, last weights averaged? | `training.selection = "none"`, `training.epochs = 10` |
| `prior_tenth` | A bank knows its mule rate only roughly: how much does the ranking change at a tenth of the built-in prior? | `loss.class_prior = 0.0001` |
| `prior_tenfold` | The same at ten times the prior | `loss.class_prior = 0.01` |

- **The priors** act mainly through the negative-risk correction and how often the
  non-negative clamp fires (`corrected_steps` in `history.csv`) under the balanced weight,
  so `prior_tenth` and `prior_tenfold` should differ little from the baseline. A near-null
  result is a finding: the ranking is robust to a mule rate known only to a factor of ten.
- **The selection variants** change only which epoch is kept and when training stops. On
  the same host settings they follow the baseline's schedule and reuse its cached
  contexts, costing little. Epochs past its early stop are new: `select_on_roc_auc` and
  `select_on_pu_risk` may stop later, and `fixed_10_epochs` always trains ten, more
  whenever the baseline stopped sooner.
- **No variant adds a group**: variants only drop groups or change the model, loss or
  training, since training reads only the built-in groups. How well a table of the
  account's own activity ranks mules, with no neighbour input, is for the diagnostics
  baselines ([Run the diagnostics](run-diagnostics.md)).

## Cost

The `controls` suite is 90 runs: 70 graph runs and 20 cheap `no_attention` and `linear`
runs, which fetch no children. A graph run takes about 3 seconds a step on the CUDA host
while it requests contexts from TigerGraph, about an hour with early stopping, and far
less once the context cache holds them. Over the first three seeds each variant's runs
averaged 0.00 to 0.25 hours. Since the built-in scope became `strict_mule_v3`, of other
split shares, every `strict_mule_v2` run is archived and trained again: the next
`controls` suite trains all 90.

The plan gives two numbers for the runs it trains:

- **The estimate** (`estimate_hours`, `estimated_from` in the `suite` event): each run
  takes from the fewest to the most hours a finished run of its variant took (their
  `metrics.json`). Seeds train in turn, baseline first, so a new seed's baseline is a cold
  run that fills the cache for the variants after it. A variant with no finished run is
  costed from the baseline's, named in the plan ("linear and wide_and_deep, which have
  none, costed from the baseline's"; `costed_from_baseline`): high, since those are cold
  runs, more so for a variant that fetches no children. With no baseline run either it
  takes the range of every finished run. A resuming run counts whole (high). No finished
  run, no estimate.
- **The bound** (`bound_hours`, `timed_from`): every run trains all its epochs at the
  median seconds per step of the latest graph run's `history.csv`: an upper bound, since
  summary runs are faster and early stopping and the cache end most runs far sooner.

Seeds, and variants requesting the same groups, share cache entries; dropping a
client-computed group (the hub indicator, either pool group) requests the baseline's
contexts. Run variants one after another: parallel processes may request the same contexts
twice. The paired intervals take about 15 seconds for three seeds of `controls` and grow
with the runs.

## Read the comparison

Open `results/experiments/<suite>/report.md`; `mule report results/experiments/<suite>`
redraws it offline, and [Outputs](../reference/outputs.md#a-suite-resultsexperimentssuite)
defines every column and figure. It ranks variants by the seed-mean validation AP of their
hidden mules ([the audit](train-and-evaluate.md#audit)) and marks the test audit "for
reporting, not selection". Every comparison is of the hidden mules except the every-mule
columns and `comparison_ap_every_mule.png`.

- **The paired delta** (`validation_hidden_ap_delta`) pairs each seed with the baseline's.
  Its interval covers both the audit sample and the seeds; the audit-only interval covers
  the sample alone. Over the first three seeds the seed spread was as large as the
  differences between variants, which only the two-source interval shows
  (`comparison_delta.png` draws both). Its seed half is a percentile bootstrap, too narrow
  with few seeds: resampling n seeds understates their variance by about (n - 1) / n and
  draws few distinct sets (three seeds give ten), so it is reliable only with many seeds,
  like the ten every suite now trains.
- **`consistent`** means the two-source interval excludes zero on the mean's side.
  report.md gives the number of comparisons and how many would be consistent by chance:
  the `all` suite compares 19 variants, so at 90% about 1.9 would with no real difference.
  Treat a single consistent delta as a lead to repeat. The seeds that agree ("8 of 10")
  are shown beside it and do not decide it.
- **Seed ensembles** average a variant's seeds on the log-odds scale, where a confident
  seed weighs more than a hesitant one (averaging ranks would give each the same say).
  report.md ranks them in their own section beside the seed means; `comparison_ap.png`
  draws each as a hollow diamond. An ensemble above its seeds' mean gains from their
  disagreement: a method question for a ranking that varies by seed.
- **The proxy's reliability** says how far a bank, which has only the proxy, could trust
  it to choose among these models: the rank correlation of each run's selected proxy AP
  with its hidden audit AP, which the proxy never sees. Selecting on a criterion biases
  the selected proxy AP, hence the correlation within each selection rule.
  `comparison_proxy_vs_audit.png` draws the runs.
- **`differs`** names runs whose commit, dirty state, device or sampler backend differ:
  their difference is not the variant's alone. **`unpaired_accounts`** counts validation
  accounts some audit rejected, left out of the pairing.
- **The pool groups' test numbers are optimistic**: the groups were designed after reading
  test-split mules, as the report repeats.

The reference graph holds 233 mules across its three splits, 33 in validation and 40 in
test, so audits have few mules and wide intervals.

## Declare a new variant

Add a `Variant(name, question, change)` to `experiments/variants.py`, where `change` maps
a `RunConfig` to another with `dataclasses.replace` or `RunConfig.with_changes`, and add
it to a suite. The tests build every variant's configuration, plan and model offline and
check it shares the baseline's dataset with a fingerprint of its own; `--help` lists it. A
variant changing a dataset setting would need its own dataset, so the suite refuses it.
