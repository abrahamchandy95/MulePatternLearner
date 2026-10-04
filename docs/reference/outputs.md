# Outputs

Every file the commands write, under two gitignored roots: `data/<dataset id>/` for
prepared datasets, `results/` for the rest. `paths.DatasetPaths`, `paths.RunPaths`,
`paths.SuitePaths` and `paths.DiagnosticsPaths` name the files; `artifacts` defines their
contents. A replaced file is written to a pending file, then renamed
(`artifacts.atomic_write`), so a crash never truncates a table, model or manifest;
`history.csv` and `events.jsonl` are appended to. The console shows progress and a
summary ([Command line](cli.md)); the full records are here.

## A prepared dataset: `data/<dataset id>/`

The dataset id fingerprints the dataset settings, the source id (identity of the loaded
data, recorded on the scope) among them
([Configuration](configuration.md#which-settings-matter-where)); runs and audits with
those settings share the directory. Each preparation stage is recorded in the manifest
when done, so an interrupted preparation resumes.

| File | Content |
|---|---|
| `manifest.json` | `source` (source id, vertex counts, hashes of the query files preparation ran, dataset id, dataset settings, scope id), `status` (`preparing`, then `ready`), `created_at_utc`, each parquet file's sha256, each split's population, observed positives per split (`known_mules`), each date's cutoff sequence (`cutoff_seqs`), the hub registry's threshold, scope and counts |
| `accounts.parquet` | Seed reservoirs and observed positives: `account_id`, `first_seen_seq`, `first_seen_ts_ms`, `group_id`, `observed_positive`, `known_from_ms`, `split`, `in_marginal` (in its split's label-blind reservoir; positives kept outside it are not) |
| `observed_labels.parquet` | `account_id`, `known_positive`, `known_from_ms`: revealed positives and discovery times ([Labels](labels.md#what-training-reads)) |
| `hubs.parquet` | Hub registry: `account_id`, `cutoff_seq`, `visibility_phase`, `max_visible`, `max_degree`, `reason` |
| `events.jsonl` | Each connected preparation's events: scope found or created, label reveal, hub counts, and the ready dataset's `dataset` event |
| `contexts/` | The context cache on disk: one gzip JSON file per context, `<first two hex digits>/<name>.json.gz`, holding TigerGraph's row |

At the built-in settings: at most 24,000 reservoir accounts plus the observed positives
(at most 40,000); accounts and labels are each capped at 100,000 rows. Training,
`mule check` and the audits check the files against the manifest's hashes and the query
files against the repository's. Every reader of the dataset's graph (training, audits,
diagnostics) first checks its vertex counts and scope against the recorded ones (the
frozen-source check).

**The context cache:**

- Entry name: fingerprint of the hop, context key, feature flags and candidate pool
  requested at that hop, `CONTEXT_CONTRACT`, dataset id and frozen source.
- Lookup is memory, cache, then TigerGraph; every returned row is kept, so later runs of
  the dataset (another seed, a variant requesting the same groups, the audits) request no
  context an earlier one read.
- Opened only after the frozen-source check (training, audits, diagnostics); `mule score`
  and `mule check` request from TigerGraph.
- An unreadable entry, or one naming another context, gives a `context_cache_refused`
  warning and is requested again. A readable entry is trusted as validated when written,
  so a run served wholly from the cache makes no Fourier spot check (`mule check` always
  requests).
- Cap: 1,500,000 entries (`contract.bounds.CONTEXT_CACHE_ENTRIES`, roughly 10 GB, an
  estimate until a baseline run measures its distinct contexts); beyond it the least
  recently used are removed until 90% remain. Each source counts the directory at its
  first write, then adds its own writes, so two processes filling one cache can overshoot
  until one evicts; eviction scans the directory while that source's request workers wait.
- An unwritable directory gives one `context_cache_unwritable` warning and is only read.
  Nothing deletes the cache; deleting `contexts/` by hand only costs the requests.

## A run: `results/<variant>/seed-<n>/`

`mule train` writes the built-in run, `results/baseline/seed-42/`.

| File | Written by | Content |
|---|---|---|
| `config.json` | train | `config` (every setting), `fingerprint`, `provenance`: `git_commit`, `git_dirty`, `versions` (package, torch, numpy, scikit-learn, pyTigerGraph), starting `device`, `threads` and `deterministic`, `sampler_backend`, `dataset_id`, `started` |
| `model.pt` | train | Selected weights (`SavedModel.FORMAT` 1): configuration, contract and input fingerprints, sampler and its fingerprint, threshold, `selected_on` (per `training.selection`: `validation_observed_label_proxy_ap`, `validation_observed_label_proxy_roc_auc`, `validation_observed_label_proxy_pu_risk` or `last_epoch`), dataset id, observed positives per split, training device, sampler backend |
| `resume.pt` | train | Resume state (`ResumeState.FORMAT` 1): model, optimizer, weight average, random generators, schedule position, selection and epochs so far, sampler backend, dataset id and its manifest's sha256, every segment's totals |
| `history.csv` | train | One row per log interval (`runtime.log_every_steps`) |
| `epochs.csv` | train | One row per epoch |
| `events.jsonl` | train, evaluate, score | Every event of the commands that worked on the run ([events.jsonl and the console](#eventsjsonl-and-the-console)) |
| `predictions/validation.parquet`, `predictions/test.parquet` | train | Proxy scores: `account_id`, `group_id`, `date`, `observed_label`, `score` |
| `metrics.json` | train | The record of a complete run |
| `audit/<split>.json`, `audit/<split>.parquet`, `audit/<split>_rejected.txt` | evaluate | Ground-truth audit of `validation` and `test` |
| `scores/<accounts stem>_<date>.parquet`, `scores/<accounts stem>_<date>_rejected.txt` | score | Scores of a file's accounts; ids TigerGraph rejected |
| `plots/<topic>_<figure>.png` | train, evaluate, report | Figures, PNG at 150 dpi |
| `report.md` | train, evaluate, report | Tables, linking the figures |

Every score is the logit's sigmoid in float64 (float32 scored every logit above about 17
as exactly 1, tying the top accounts). Scores rank accounts; under the balanced nnPU weight
they are not calibrated probabilities.

### history.csv

Loss, objective and timing are means over the interval; counters from `database_calls` on
are run totals, over every segment of a resumed run. A resumed run drops rows logged after
its resume position before logging them again.

| Column | Meaning |
|---|---|
| `epoch`, `step` | Epoch (from 1) and its steps done at the interval's end |
| `date` | Train cutoff of the interval's last step |
| `loss`, `objective` | Clamped nnPU loss minimised; unclamped nnPU risk |
| `corrected_steps`, `steps` | Steps whose non-negative correction fired; steps in the epoch's schedule |
| `seconds_per_step`, `batch_wait_seconds` | Timing |
| `database_calls` | Successful REST calls |
| `contexts_requested`, `contexts_distinct` | Contexts the batches asked for; distinct ones |
| `memory_hits`, `disk_hits` | Contexts served from memory; read from the cache instead of requested |
| `rejected_roots`, `stub_children` | Training roots TigerGraph rejected; hub children built as stubs |

### epochs.csv

| Column | Meaning |
|---|---|
| `epoch`, `loss`, `steps` | Per epoch |
| `validation_ap`, `validation_roc_auc`, `validation_pu_risk` | Validation proxy on observed labels; the risk is the run's own non-negative nnPU risk on the proxy sample, lower is better (missing in files written before it was recorded) |
| `weights` | Weights validated: `averaged` or `raw` |
| `selected` | Epoch whose weights `model.pt` holds, per `training.selection` (one of those three, or the last epoch) |
| `stopped` | Epoch after which early stopping ended the run |

### metrics.json

`status`, `dataset_id`, `seed`, `known_mules` (observed positives per split), `device`,
the loss (`loss`, `class_prior`, `positive_weight`, `objective`), `input_fingerprint`,
`parameter_count`, `revealed_training_accounts`, `best_epoch`, `observed_label_proxy`
(validation and test proxy metrics at the selected threshold), `validation_proxy`,
`evaluation_protocol` (`strict_inductive`), `performance_claim`,
`database_calls_during_training`, `elapsed_seconds` (wall clock, all segments),
`contexts` (requested, distinct, `memory_hits`, `disk_hits`, `disk_hit_rate`: share of
contexts not served by memory that the cache served, null when memory served all),
`rejections`, `sampler_backend`, `sampler_totals`, `rejected_evaluation_rows`,
`rejected_roots` per split, `max_rejected_root_fraction`, `proxy_unlabeled_limit`.

Proxy metrics score each split's revealed positives against up to 2,000 unlabelled
accounts counted as negatives: AP, ROC AUC, precision, recall and F1 at the threshold, and
precision and recall at the review budgets with unit weights. They describe the observed
labels, not the ground truth.

### The audit files

- `audit/<split>.parquet`, the scored sample: `account_id`, `is_mule`,
  `inclusion_probability` (chance the sample includes the account), `score`, `revealed`
  (label revealed before the split's cutoff), `ring_id`, `label_source`.
- `audit/<split>_rejected.txt`: sampled accounts TigerGraph rejected, one per line.
- `audit/<split>.json`: `split`, `purpose` (`decisions` for validation, `reporting` for
  test), `date`, `selection` (`model.pt`'s `selected_on`), `population_accounts`,
  `hidden_metrics`, `hidden_intervals`, `metrics`, `intervals`, `constants`,
  `revealed_positives`, `hidden_positives`, rejection counts, `scope`, `model_changed`.

| Key | Content |
|---|---|
| `hidden_metrics` | Lead; decisions use validation's `average_precision` here. Mules revealed by the cutoff are removed from the ranking, as an investigator removes known cases, and the hidden mules ranked against the non-mules left: `sample_accounts`, `sample_positives` (hidden mules), `estimated_population`, `weighted_prevalence`, `average_precision`, `roc_auc`, `precision_at_1pct`, `recall_at_1pct` and the same at `5pct` and `10pct`, all of that population, and `evaluation_sample` (begins `hidden_`). AP and ROC AUC are null without a hidden mule |
| `hidden_intervals` | The interval of each of those ranking metrics |
| `metrics` | Every mule, revealed included, estimating the whole split, each sampled account standing for 1 / `inclusion_probability`: `estimated_population`, `weighted_prevalence`, `average_precision`, `roc_auc`, `precision`, `recall` and `f1` at the frozen threshold, `precision_at_1pct`, `recall_at_1pct` and the same at `5pct` and `10pct` (reviewing the top 1, 5 or 10% of the estimated population). `evaluation_sample` ends in `_minus_rejected_negatives` when rejected non-mules were left out |
| `intervals` | Each ranking metric's 90% bootstrap interval: 1,000 replicates, seed 0, mules resampled by ring (a ringless mule alone), non-mules within their class, inclusion weights kept |
| `constants` | Sample size and seed, review budgets, bootstrap settings |

A budget ending inside a block of tied scores takes an equal share of each tied account,
so account ids and row order never matter. Readers refuse an `audit/<split>.json` without
`hidden_metrics` (earlier code): move that split's `audit/<split>.*` aside and run
`mule evaluate` again.

### The run's figures

The first six are drawn once training has saved every other file, the audit figures after
the audits; `mule report` redraws all.

| File | What it shows |
|---|---|
| `training_objective.png` | Loss and unclamped nnPU objective per interval, rolling mean, epoch boundaries |
| `training_corrections.png` | Share of steps whose non-negative correction fired |
| `validation_ranking.png` | Proxy AP and ROC AUC per epoch, nnPU risk on its own axis where `epochs.csv` has it, selected epoch marked on the selection rule's criterion with the rule named, prevalence line, which weights were validated |
| `training_throughput.png` | Seconds per step and batch wait; below, contexts requested, distinct, from memory, from the cache |
| `proxy_precision_recall.png` | Validation and test precision and recall on observed labels, titled as a proxy |
| `run_health.png` | Rejections by split and status, stub children, sampler totals, database calls |
| `audit_hidden_precision_recall.png` | Weighted precision and recall of the hidden mules per audited split, chance line, AP with its interval |
| `audit_hidden_roc.png` | Weighted ROC of the hidden mules per split, with AUC |
| `audit_hidden_capture.png` | Share of hidden mules found against the top share of remaining accounts reviewed (log axis), random and perfect lines, budgets labelled with recall and precision |
| `audit_precision_recall.png`, `audit_roc.png`, `audit_capture.png` | The same three of every mule, revealed included |
| `audit_threshold.png` | Weighted precision, recall and F1 of the test audit against the threshold (log10 odds), selected threshold marked |
| `audit_score_distribution.png` | Weighted densities of the test audit's log10 odds, mules against non-mules, threshold |
| `audit_revealed_hidden.png` | Where the test audit ranks revealed and hidden mules: share of accounts ranked above each, log axis, medians |

## A suite: `results/experiments/<suite>/`

A suite's runs are ordinary run directories in `results/`, shared with `mule train` and
other suites. A suite named by variants is named by them joined with hyphens.

| File | Content |
|---|---|
| `summary.csv` | One row per run, split and metric: `variant`, `seed`, `split`, `metric`, `value`, `status` (`complete`, `failed` or `stopped`), `commit`; then seed-ensemble rows (status `ensemble`) |
| `comparison.csv` | One row per variant against the baseline, then one per seed ensemble |
| `events.jsonl` | The suite's own events: the plan (`suite`, with the hours its runs should take to train: `estimate_hours` from the suite's finished runs, `costed_from_baseline` for new variants costed from the baseline's, `bound_hours`), `run_finished`, `run_failed`, `run_archived`, `suite_stopped` (an outage), and what precedes the dataset's own records (install, connecting) or a ready dataset (`dataset`) |
| `plots/comparison_*.png` | Seven figures |
| `report.md` | Variants and seed ensembles ranked by validation audit AP of hidden mules (every mule's AP and delta last), the proxy's reliability, tables and figures |

Every audit number comes twice: hidden mules first, named `hidden_` plus the metric
(`hidden_average_precision`, revealed removed as in `hidden_metrics`), then every mule.
Decisions use validation `hidden_average_precision`.

`summary.csv` metrics: each audit report's ranking metrics per split, both kinds; the
run's `best_epoch`, `parameter_count` and `training_hours` (no split); validation
`proxy_average_precision`; validation `hidden_paired_average_precision` and
`paired_average_precision` (on the accounts every audit of the suite scored); and, except
for the baseline, `hidden_average_precision_delta` and `average_precision_delta` against
the baseline's run of the same seed. A run that left no numbers keeps one metricless row.
A variant with at least two complete runs gets a seed ensemble: its seeds' scores of the
accounts every audit scored, averaged in log odds (each clipped 2^-50 from 0 and 1), then
audited on a run's ranking metrics, both kinds. Its rows: status `ensemble`, no seed or
commit, the ranking metrics per split, and `ensemble_seeds` (seeds combined, no split).

`comparison.csv`, hidden-mule columns first (decisions use them):

| Columns | Meaning |
|---|---|
| `variant`, `estimate`, `question`, `changes`, `seeds` | `estimate`: `seed_mean` for a variant's seed means, `ensemble` for its seed ensemble |
| `validation_hidden_ap`, `validation_hidden_ap_spread`, `validation_hidden_ap_low`, `validation_hidden_ap_high`, and the same for `test_hidden_ap` | Seed-mean hidden AP, spread over seeds, 90% interval of the seed mean |
| `validation_hidden_ap_delta`, `validation_hidden_ap_delta_low`, `validation_hidden_ap_delta_high` | Paired validation delta against the baseline; interval over seeds and audit sample (each replicate resamples accounts and rings once, and seeds paired by seed) |
| `validation_hidden_ap_delta_audit_low`, `validation_hidden_ap_delta_audit_high` | Delta interval over the audit sample alone |
| `validation_hidden_ap_delta_seeds`, `validation_hidden_ap_delta_agreeing` | Seeds both completed; seeds whose delta has the mean's sign (exactly zero agrees with neither) |
| `consistent` | Two-source interval excludes zero on the mean's side; agreeing seeds are shown, not deciding. Hidden mules only |
| `validation_hidden_roc_auc`, `test_hidden_roc_auc`, `validation_hidden_recall_at_1pct`, `validation_hidden_precision_at_1pct`, and so on for `5pct`, `10pct` and test | Seed means |
| `validation_ap`, `validation_ap_spread`, `validation_ap_low`, `validation_ap_high`, `test_ap`, `validation_ap_delta` with its intervals, seeds and agreeing seeds, `validation_roc_auc`, `validation_recall_at_1pct` and the rest | The same of every mule, without `hidden_` |
| `best_epoch`, `parameter_count`, `training_hours` | Run values |
| `unpaired_accounts` | Validation accounts left out of pairing because some audit rejected them |
| `differs` | Runs whose commit, dirty state, device or sampler backend differ from the suite's usual value |

- Seed-mean AP averages each run's own audit; its interval and the delta use the accounts
  every audit scored, so with `unpaired_accounts` above 0 the mean can fall outside its
  interval. The built-in rejection limit of 0 fails any audit that rejects an account, so
  built-in variants have 0.
- A seed-ensemble row holds per split, both kinds, its AP with the 90% interval over the
  same paired replicates, and ROC AUC, recall and precision without intervals (as seed
  means; a run's audit has an interval per ranking metric). No spread, delta, consistency
  or run values.
- A `comparison.csv` of earlier code (no `estimate` or no `validation_hidden_ap` column)
  is refused: rerun `scripts/run_experiments.py`, which rewrites the tables from the runs.

The report's proxy reliability: Spearman's rank correlation of the selected epoch's
validation proxy AP with validation audit hidden AP, each with its n, over complete runs;
within each selection rule's runs (selecting on proxy AP reports the best epoch value,
other rules do not); over variants by seed means; and last over runs against every mule's
audit AP.

| Figure | What it shows |
|---|---|
| `comparison_ap.png` | Validation and test audit hidden AP per variant: dot per seed, seed mean and interval, baseline line, and below each mean the seed ensemble as a hollow diamond on its interval |
| `comparison_delta.png` | Validation audit hidden AP minus the baseline's: seeds-and-accounts interval, audit-only interval as a thin line above, per-seed deltas, zero line; filled when consistent |
| `comparison_budget.png` | Validation audit hidden recall at 1%, 5% and 10% per variant |
| `comparison_capture.png` | Seed-mean validation hidden capture curves, a panel per variant with the baseline in each |
| `comparison_validation.png` | Seed-mean proxy AP per epoch, up to the last epoch every seed trained, a panel per variant with the baseline in each |
| `comparison_proxy_vs_audit.png` | Selected proxy AP against validation audit hidden AP per run, with the proxy reliability's correlations and n |
| `comparison_ap_every_mule.png` | As `comparison_ap.png`, of every mule, same order |

## A diagnostic study: `results/diagnostics/<dataset id>/`

| File | Content |
|---|---|
| `features.parquet` | Feature table, one row per sampled account per split: `account_id`, `split`, `date`, `is_mule`, `revealed`, `ring_id`, `label_source`, `inclusion_probability`, `weight` (1 / `inclusion_probability`), `rejected`, `context_contract`, `analytics_contract`, then a column per feature, `<family>__<name>` |
| `<analysis>.csv` | One long table per analysis, named with underscores (`learning_curve.csv`) |
| `study.json` | Dataset, run compared, reveal salt and budget, each analysis' last outcome (`written`, `kept`, or `skipped` with reason) |
| `events.jsonl` | Every `mule diagnose` event: each analysis' outcome (`diagnose`), feature table splits (`feature_table`), retries, warnings |
| `plots/<figure>.png` | Figures |
| `report.md` | Tables, linking the figures |

Feature families: `model` (root's model inputs), `messages` (hop-1 pool summaries),
`account` (analytics query's account history), `message_context`. Each table is long: key
columns, then `metric` and `value`; a key not applying to a row is empty.

| Table | Columns | Figures |
|---|---|---|
| `univariate.csv` | `feature`, `family`, `split`, `metric`, `value` | `univariate_auc.png` |
| `drift.csv` | `feature`, `family`, `model`, `setup`, `split`, `metric`, `value` | `drift.png` |
| `baselines.csv` | `baseline`, `features`, `model`, `split`, `metric`, `value`, `low`, `high` (bootstrap interval) | `baselines.png` |
| `learning_curve.csv` | `model`, `labels`, `mules`, `repeat`, `split`, `metric`, `value` | `learning_curve.png` |
| `subgroups.csv` | `split`, `subset`, `rank`, `metric`, `value` | `ap_concentration.png`, `ring_coverage.png` |
| `proxy_validity.csv` | `split`, `subset`, `metric`, `value` | `proxy_validity.png` |
| `reveal_spread.csv` | `salt`, `split`, `metric`, `value` | `reveal_spread.png` |
| `nnpu_simulation.csv` | `positive_weight`, `seed`, `metric`, `value` | `nnpu_simulation.png` |

- Tables ranking mules measure twice, as the audits do: hidden mules first (metrics named
  `hidden_` plus the metric, as `hidden_average_precision`, `hidden_roc_auc`), then every
  mule. Figures draw the hidden mules', except ring coverage.
- `subgroups.csv` subsets: `hidden` and `revealed` (each kind against the non-mules),
  `hidden` and `mules` for the running AP sum (`cumulative_average_precision`, by `rank`),
  and `rings`.
- `proxy_validity.csv` subsets: `hidden` and `revealed` from the run's audit samples
  (which hold every hidden mule), weighted to the population; `all`, the proxy
  predictions, unweighted.

## The archive: `results/archive/`

Before retraining, the experiments script moves a run whose settings differ from its
variant's, whose `config.json` is unreadable, or which used another dataset, whole to
`results/archive/<variant>/seed-<n>/<UTC time>/`, with a `run_archived` event naming the
difference. Nothing there is deleted.

## The report of `mule check`: `results/check.json`

Replaced by each `mule check`; nothing else reads it.

- `graph`; `scope_schema` (`present`, `missing` or `outdated`; outdated adds
  `scope_outdated`: `differences` from `gsql/schema/scope_vertex.gsql` in words, and the
  `scopes` the graph holds); `queries` (`up_to_date`, `stale` with each query's issues,
  `retired`); `cugraph` (probe `status`, device, reason); `dataset`; `problems`; `status`
  (`ready` or `not_ready`); `peak_process_rss_bytes`; `graph_writes` (always 0).
- Once the graph is ready, `source_open_seconds` and `first_step`: the step's device,
  determinism and seed, roots and accepted roots, batch statistics (stub and rejected
  counts, sampler backend), rejections, context requests, REST calls and retries,
  seconds, input and sampler fingerprints, `tensor_bytes`, `tensor_digests` (each
  tensor's dtype, shape and sha256, summaries of the floating ones), their one
  `batch_digest`, `loss`, `objective`, `train_step_seconds`, parameter count, the
  accelerator's peak memory.

## events.jsonl and the console

Each event is a JSON object with an `event` name, led in the file by its UTC `time` (ISO
8601 to the second, as `"2026-10-03T13:16:17+00:00"`). Events cover preparation stages,
installs, retries (a short `reason` beside the error), a run's start or resume (device,
threads, determinism, plan), training intervals, scoring of each validation and test
chunk, epochs, completion, audits, scoring, analyses, a suite's plan and runs, the
sampler backend, and warnings (such as `cugraph_probe`, `context_cache_refused`,
`host_settings`).

| The whole record goes to | For |
|---|---|
| The run's `events.jsonl` | Training, audits, scoring |
| The dataset's | Its preparation |
| The study's | `mule diagnose` |
| The suite's | The experiments script's own events, and those a command would send to `results/events.jsonl`, without `command` |
| `results/events.jsonl` | Anything emitted while none of the above records (below) |

The console shows a few of these events in short lines ([The console](cli.md#the-console)).

### A command's own records: `results/events.jsonl`

Each `mule` command appends what it emits while no run, dataset or study is recording,
naming itself (`command`, such as `"train"`) after `time`: the queries an install found
stale and up to date (`install`); an outdated scope vertex type's replacement before and
after (`scope_types`: differences, queries dropped); TigerGraph's output for each GSQL
write (`gsql`: each scope schema change, `CREATE`, `DROP`); an unanswered install request
with its ending error (`install_unanswered`); the compilation wait and end
(`install_wait`, `installed`); connection retries with the whole error (`retry`); a
dataset found ready (`dataset`); a complete run reported (`already_complete`); everything
`mule install` and `mule check` emit; and the error that stopped a command when nothing
else was recording ([The error that stops a command](#the-error-that-stops-a-command)).

Time and command tell records apart: the install of `mule install` from the one
`mule train` began with, or one session's retries from the next's. The file only grows;
nothing reads it.

## The error that stops a command

A command stopping on an error records it before exiting, in the `events.jsonl` recording
when it was raised (the run's, dataset's or study's), else in `results/events.jsonl`, as
for a single-attempt write (the scope schema change, a `CREATE QUERY`, a `DROP`). So the
cause sits beside the records that led to it, not only on stderr. The record names the
command (`command`) and gives the error's type and message on one line (`error`):

- **`command_stopped`**: TigerGraph's failures outlasted the retries; stderr has one line
  saying so, and `error` keeps that line's whole error.
- **`command_failed`**: any other error (a bug), shown by Python with its traceback;
  `error` is cut to about 200 characters, and the record adds `type` (as the traceback
  names it) and the whole `message`.

The experiments script records its own the same way, as `run_experiments.py`, in the
dataset's or suite's `events.jsonl`. An interruption (Ctrl-C) records nothing.
