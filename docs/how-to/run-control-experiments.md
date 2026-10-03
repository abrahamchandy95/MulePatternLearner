# Run the control experiments

Measure what each part of the built-in run contributes: train variants of it over the
seeds 42, 43 and 44, audit every run, and compare each variant with the baseline on the
same accounts. The script takes suite or variant names only.

## Run a suite

On the machine that trains (the CUDA host), after `mule train` and `mule evaluate` have
made the baseline's seed 42:

```bash
python scripts/run_experiments.py                            # the controls suite
python scripts/run_experiments.py feature_drops              # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                     # suites, variants, questions, changes
```

The baseline is always included, and its seed 42 is the run `mule train` makes. The
script, in order:

1. **Validates every variant offline**: each seed's configuration, feature plan and model
   on the CPU, one dataset for all and a distinct fingerprint for each. A failure names
   the variant, before anything connects.
2. **Prepares the dataset** once, as `mule train` does, on the suite's one connection.
3. **Plans every run** and prints the plan as a `suite` event with `time_bound`: an upper
   bound from the median seconds per step of the latest graph run's `history.csv`. A
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

A run's own error is a `run_failed` event and the suite goes on; a TigerGraph outage (the
retry budget ran out while the graph was unavailable) is a `suite_stopped` event and
stops the training and audits, since every later run would fail the same way. The script
prints one JSON result and exits 1 unless every run is trained and audited without an
error. A run whose figures failed after its numbers were saved stays complete in the
tables, with its error beside it; `mule report results/<variant>/seed-<n>` redraws them.
Run the script again to finish: complete runs are kept.

A suite of chosen variants is named by their names joined with hyphens, as in
`results/experiments/no_attention-prior_weight/`.

Run `controls` first: it settles the positive weight (`prior_weight`) and the model's
mechanisms before the feature drops are compared, and every run of both suites holds the
source, the revealed labels and the cutoffs fixed. Then run `feature_drops`; the
baseline's runs are shared.

## The suites and variants

Variants are declared in `src/mule_pattern_learner/experiments/variants.py`. A variant is
a question and a change to the built-in run; it never sets a seed, and none touches
`dataset.seed`, `dataset.split_seed` or `scope.reveal_salt`, so every variant trains on
the baseline's dataset.

| Suite | Variants |
|---|---|
| `controls` (the default) | `baseline`, `no_attention`, `no_slot_sum`, `no_pool_counts`, `prior_weight`, `no_weight_average`, `drop_time_encoding` |
| `feature_drops` | `baseline` and one drop per built-in group but `message_core`: `drop_entity_meta`, `drop_hub_indicator`, `drop_time_encoding`, `drop_pair_history`, `drop_flow_timing`, `drop_pool_activity`, `drop_pool_internal_inflows` |
| `all` | Every variant of both, each once |

| Variant | Question | Change |
|---|---|---|
| `no_attention` | Does attention over sampled neighbours add anything beyond the root's own inputs, pool counts included? | `model.architecture = "summary"`, no slot sum |
| `no_slot_sum` | Does the per-slot MLP sum help beyond attention? | `model.slot_sum = false` |
| `no_pool_counts` | How much of the ranking comes from the candidate-pool counts? | without both pool groups |
| `prior_weight` | Does the balanced positive weight beat textbook nnPU across seeds? | `loss.positive_weight = "prior"` |
| `no_weight_average` | Does selecting on the moving average of the weights help? | `training.weight_average_decay = 0.0` |
| `drop_<group>` | What does the model lose without the group? | without the group, and the groups that read it (`drop_pair_history` also drops both pool groups, `drop_flow_timing` also `pool_activity`) |

Variants only drop groups or change the model, the loss or the training: training reads
only the built-in run's groups, so no variant adds one. How well a table of the account's
own activity ranks mules, with no neighbour input, is a question for the diagnostics
baselines ([Run the diagnostics](run-diagnostics.md)).

## Cost

About 3 seconds per step and about an hour per graph run with early stopping on the CUDA
host. The `controls` suite is 21 runs (18 graph runs and the 3 cheap `no_attention`
runs, which fetch no children); 20 train, since the baseline's seed 42 is `mule train`'s.
That is roughly a day back to back, less as the context cache fills: seeds, and variants
that request the same groups, share its entries, and drops of the client-computed groups
(the hub indicator and both pool groups) request the same contexts as the baseline. Run
variants one after another rather than as parallel processes, which may request the same
contexts twice. The printed bound is an upper bound: summary runs are faster and early
stopping ends most runs sooner. The paired intervals take about 15 seconds for the
controls suite and 30 for `all`.

## Read the comparison

Open `results/experiments/<suite>/report.md`. It ranks the variants by the seed-mean
validation audit AP and marks the test audit "for reporting, not selection".

- **The paired delta** (`validation_ap_delta` with its interval) is the variant's
  seed-mean validation AP minus the baseline's. Every audit of a dataset scores the same
  accounts, so each bootstrap replicate resamples those accounts and their rings once and
  applies the resample to every run. The interval covers the audit sample's uncertainty
  for these seeds, not the spread between seeds, which the per-seed deltas beside it show
  (`comparison_delta.png`).
- **`consistent`** is true only when every seed's delta has the same sign and the interval
  excludes zero. The `all` suite compares 12 variants with the baseline, so at 90% about
  one will exclude zero by chance: treat results as exploratory until repeated with more
  seeds.
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
