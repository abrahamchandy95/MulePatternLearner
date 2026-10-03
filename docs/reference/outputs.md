# Outputs

Every file the commands write, and where. There are two roots, both gitignored:
prepared datasets, the inputs to training, go in `data/<dataset id>/`, and everything
the commands write lives under `results/`. `paths.DatasetPaths`, `paths.RunPaths`,
`paths.SuitePaths` and `paths.DiagnosticsPaths` name each file, and `artifacts` defines
what the tables and JSON files hold. A file a command replaces is written to a pending
file first and then renamed (`artifacts.atomic_write`), so a crash never leaves a
truncated table, model or manifest; `history.csv` and `events.jsonl` are appended to.
The console shows each command's progress and a short summary
([Command line](cli.md)); the records in full are in these files.

```
data/
└── <dataset id>/                 a prepared dataset
    ├── manifest.json
    ├── accounts.parquet
    ├── observed_labels.parquet
    ├── hubs.parquet
    ├── events.jsonl              the events of its preparation
    └── contexts/                 the context cache
results/
├── baseline/seed-42/             the built-in run, which `mule train` writes
├── <variant>/seed-<n>/           one run of a control experiment
├── experiments/<suite>/          a suite's comparison
├── diagnostics/<dataset id>/     the diagnostic study of a dataset
├── archive/                      runs moved aside because their settings changed
├── check.json                    the full report of the last `mule check`
└── events.jsonl                  what each command did before a run or dataset recorded it
```

## A prepared dataset: `data/<dataset id>/`

The dataset id is the fingerprint of the dataset settings: the source id (the identity of
the data loaded into the graph, recorded on the scope), `scope.id`, `scope.unowned`, the
`dataset` section and the sampler's candidate pools ([Configuration](configuration.md#which-settings-matter-where)).
Every run and audit with those settings shares the directory. It is prepared in stages,
each recorded in the manifest when it is done, so an interrupted preparation resumes.

| File | Content |
|---|---|
| `manifest.json` | `source` (the source id, the vertex counts, the hashes of the query files preparation ran, the dataset id, the dataset settings and the scope id), `status` (`preparing`, then `ready`), `created_at_utc`, the sha256 of each parquet file, the population of each split, the observed positives per split (`known_mules`), the cutoff sequence of each date (`cutoff_seqs`) and the hub registry's threshold, scope and counts |
| `accounts.parquet` | The seed reservoirs and the observed positives: `account_id`, `first_seen_seq`, `first_seen_ts_ms`, `group_id`, `observed_positive`, `known_from_ms`, `split` and `in_marginal` (whether the account is in its split's label-blind reservoir; positives kept outside it are not) |
| `observed_labels.parquet` | `account_id`, `known_positive`, `known_from_ms`: the revealed positives and their discovery times ([Labels](labels.md#what-training-reads)) |
| `hubs.parquet` | The hub registry: `account_id`, `cutoff_seq`, `visibility_phase`, `max_visible`, `max_degree`, `reason` |
| `events.jsonl` | The events of each preparation that connected: the scope found or created, the label reveal, the hub registry's counts and the `dataset` event of the ready dataset, one JSON object each |
| `contexts/` | The disk tier of the context cache: one gzip-compressed JSON file per context, `<first two hex digits>/<name>.json.gz`, holding the row TigerGraph returned |

At the built-in settings a dataset holds at most 24,000 reservoir accounts plus the
observed positives (at most 40,000), and neither the accounts nor the labels may exceed
100,000 rows. Training, `mule check` and the audits check the files against the
manifest's hashes and the query files against the repository's, and every reader of a
dataset's graph (training, the audits, the diagnostics) first checks its vertex counts
and scope against the recorded ones (the frozen-source check).

**The context cache.** An entry's name is the fingerprint of the hop, the context key,
the feature flags and candidate pool requested at that hop, `CONTEXT_CONTRACT`, the
dataset id and the frozen source. A run reads memory, then the cache, then TigerGraph,
and keeps every row TigerGraph returns, so the next run of the dataset (another seed, a
variant requesting the same groups, the audits) requests none of the contexts an earlier
one read. Only a connection that passed the frozen-source check opens it: training, the
audits and the diagnostics; `mule score` and `mule check` request from TigerGraph. An
entry that cannot be read, or that names another context, is refused with a
`context_cache_refused` warning and requested again. An entry that reads is trusted as
validated when its request wrote it: a run that reads every context from the cache
requests none, so it makes no Fourier spot check (`mule check` always requests). Beyond
1,500,000 entries (`contract.bounds.CONTEXT_CACHE_ENTRIES`, roughly 10 GB, an estimate
until a baseline run measures its distinct contexts) the least recently used entries are
removed until 90% remain. Each source counts the directory on its first write and then
adds its own writes, so two processes writing one dataset's cache at the same time can
fill it past the cap until one of them evicts; and an eviction scans the directory while
that source's request workers wait. A directory that cannot be written gives one
`context_cache_unwritable` warning and is then only read. Nothing deletes the cache;
removing `contexts/` by hand only costs the requests again.

## A run: `results/<variant>/seed-<n>/`

| File | Written by | Content |
|---|---|---|
| `config.json` | train | `config` (every setting), `fingerprint` and `provenance`: `git_commit`, `git_dirty`, the `versions` of the package, torch, numpy, scikit-learn and pyTigerGraph, the `device`, `threads` and `deterministic` the run started with, `sampler_backend`, `dataset_id` and `started` |
| `model.pt` | train | The selected weights (`SavedModel.FORMAT` 1): the configuration, the contract and input fingerprints, the sampler and its fingerprint, the threshold, the dataset id, the observed positives per split, the training device and the sampler backend |
| `resume.pt` | train | What an interrupted run continues from (`ResumeState.FORMAT` 1): the model, optimizer, weight average, random generators, schedule position, the selection so far, the epochs so far, the sampler backend, the dataset id and its manifest's sha256, and the totals of every segment |
| `history.csv` | train | One row per log interval (`runtime.log_every_steps`) |
| `epochs.csv` | train | One row per epoch |
| `events.jsonl` | train, evaluate, score | The full record of every event of the commands that worked on the run, one JSON object each ([events.jsonl and the console](#eventsjsonl-and-the-console)) |
| `predictions/validation.parquet`, `predictions/test.parquet` | train | The proxy scores: `account_id`, `group_id`, `date`, `observed_label`, `score` |
| `metrics.json` | train | The record of a complete run |
| `audit/<split>.json`, `audit/<split>.parquet`, `audit/<split>_rejected.txt` | evaluate | The ground-truth audit of `validation` and `test` |
| `scores/<accounts stem>_<date>.parquet`, `scores/<accounts stem>_<date>_rejected.txt` | score | The scores of the accounts of a file, and the ids TigerGraph rejected |
| `plots/<topic>_<figure>.png` | train, evaluate, report | The figures, PNG at 150 dpi |
| `report.md` | train, evaluate, report | The run's tables, with links to its figures |

Scores in every output are the sigmoid of the model's logit, computed in float64: in
float32 every logit above about 17 scored exactly 1, so the highest-scored accounts tied.
They rank accounts; under the balanced nnPU weight they are not calibrated probabilities.

### history.csv

The loss, objective and timing are means over the interval's steps; the counters from
`database_calls` on are totals of the run so far, over every segment of a resumed run. A
resumed run drops the rows logged after its resume position before it logs them again.

| Column | Meaning |
|---|---|
| `epoch`, `step` | The epoch (from 1) and the steps of it done when the interval ended |
| `date` | The train cutoff of the interval's last step |
| `loss` | The clamped nnPU loss that training minimised |
| `objective` | The unclamped nnPU risk |
| `corrected_steps` | Steps whose non-negative correction fired |
| `steps` | The steps of the epoch's schedule |
| `seconds_per_step`, `batch_wait_seconds` | Timing |
| `database_calls` | Successful REST calls |
| `contexts_requested`, `contexts_distinct` | Contexts the batches asked for, and the distinct ones |
| `memory_hits`, `disk_hits` | Contexts served from memory, and read from the context cache instead of requested |
| `rejected_roots`, `stub_children` | Training roots TigerGraph rejected, and hub children built as stubs |

### epochs.csv

`epoch`, `loss`, `steps`, `validation_ap` and `validation_roc_auc` (proxy metrics on
observed labels, of the weights `weights` names: `averaged` or `raw`), `selected` (the
epoch whose weights `model.pt` holds) and `stopped` (the epoch after which early stopping
ended the run).

### metrics.json

`status`, `dataset_id`, `seed`, `known_mules` (observed positives per split), `device`,
the loss (`loss`, `class_prior`, `positive_weight`, `objective`), `input_fingerprint`,
`parameter_count`, `revealed_training_accounts`, `best_epoch`, `observed_label_proxy`
(the proxy metrics of validation and test at the selected threshold),
`validation_proxy`, `evaluation_protocol` (`strict_inductive`), `performance_claim`,
`database_calls_during_training`, `elapsed_seconds` (wall-clock over every segment),
`contexts` (requested, distinct, `memory_hits`, `disk_hits` and `disk_hit_rate`: the share
of the contexts memory did not serve that the cache served, null when memory served them
all), `rejections`, `sampler_backend`, `sampler_totals`, `rejected_evaluation_rows`,
`rejected_roots` per split, `max_rejected_root_fraction` and `proxy_unlabeled_limit`.

The proxy metrics score each split's revealed positives against up to 2,000 unlabelled
accounts, which count as negatives: average precision, ROC AUC, precision, recall and F1
at the threshold, and precision and recall at the review budgets with unit weights. They
describe the observed labels, not the ground truth.

### The audit files

`audit/<split>.parquet` holds the scored sample: `account_id`, `is_mule`,
`inclusion_probability` (the chance the sample includes the account), `score`,
`revealed` (the graph revealed the account's label before the split's cutoff), `ring_id`
and `label_source`. `audit/<split>_rejected.txt` lists the sampled accounts TigerGraph
rejected, one per line.

`audit/<split>.json` holds `split`, `purpose` (`decisions` for validation, `reporting` for
test), `date`, `selection`, `population_accounts`, `metrics`, `intervals`, `constants`,
`revealed_positives`, `hidden_positives`, the rejection counts, `scope` and
`model_changed`.

- **`metrics`** estimate the split's whole population, each sampled account standing for
  1 / `inclusion_probability` accounts: `estimated_population`, `weighted_prevalence`,
  `average_precision`, `roc_auc`, and `precision`, `recall` and `f1` at the frozen
  threshold, plus `precision_at_1pct` and `recall_at_1pct` and the same at `5pct` and
  `10pct`: reviewing the highest-scored 1, 5 or 10% of the estimated population. A budget
  that ends inside a block of tied scores takes the same share of each account in it, so
  neither account ids nor row order matter. `evaluation_sample` names the sample, and
  ends in `_minus_rejected_negatives` when rejected non-mules were left out.
- **`intervals`** give each ranking metric its 90% bootstrap interval: 1,000 replicates
  drawn with seed 0, mules resampled by ring (a mule without a ring alone) and non-mules
  within their class, each keeping its inclusion weight.
- **`constants`** record the sample's size and seed, the review budgets and the bootstrap
  settings.

### The run's figures

| File | What it shows |
|---|---|
| `training_objective.png` | Loss and unclamped nnPU objective per interval, rolling mean, epoch boundaries |
| `training_corrections.png` | Share of steps whose non-negative correction fired |
| `validation_ranking.png` | Proxy AP and ROC AUC per epoch, the selected epoch, the prevalence line, which weights were validated |
| `training_throughput.png` | Seconds per step and batch wait; below, contexts requested, distinct and served from memory and from the cache |
| `proxy_precision_recall.png` | Validation and test precision and recall on observed labels, titled as a proxy |
| `run_health.png` | Rejections by split and status, stub children, sampler totals, database calls |
| `audit_precision_recall.png` | Weighted precision and recall per audited split, the chance line, AP with its interval |
| `audit_roc.png` | Weighted ROC per split with its AUC |
| `audit_capture.png` | Share of mules found against the top share of accounts reviewed (log axis), random and perfect lines, the review budgets labelled with recall and precision |
| `audit_threshold.png` | Weighted precision, recall and F1 of the test audit against the threshold (log10 odds), the selected threshold |
| `audit_score_distribution.png` | Weighted densities of the test audit's log10 odds, mules against non-mules, the threshold |
| `audit_revealed_hidden.png` | Where the test audit ranks revealed and hidden mules: the share of accounts ranked above each, on a log axis, with medians |

The first six are drawn once training has saved every other file, the audit figures
after the audits; `mule report` redraws them all.

## A suite: `results/experiments/<suite>/`

The runs of a suite are the run directories of the same `results/`, which `mule train`
and other suites share. A suite named by its variants is named by those names joined
with hyphens.

| File | Content |
|---|---|
| `summary.csv` | One row per run, split and metric: `variant`, `seed`, `split`, `metric`, `value`, `status` (`complete`, `failed` or `stopped`) and `commit` |
| `comparison.csv` | One row per variant, compared with the baseline |
| `events.jsonl` | The suite's own events: the plan (`suite`), each run's step that finished (`run_finished`) or failed (`run_failed`), the runs moved aside (`run_archived`), an outage (`suite_stopped`), and what came before the dataset's preparation recorded its own (the install, connecting) or a dataset found ready (`dataset`); preparation's go to the dataset's `events.jsonl`, and each run's to the run's |
| `plots/comparison_*.png` | Six figures |
| `report.md` | The variants ranked by the validation audit, with the tables and figures |

The metrics of `summary.csv` are the audit reports' ranking metrics per split, the run's
own `best_epoch`, `parameter_count` and `training_hours` (no split), the validation
`proxy_average_precision`, the validation `paired_average_precision` (on the accounts
every audit of the suite scored) and, but for the baseline, `average_precision_delta`
against the baseline's run of the same seed. A run that left no numbers keeps one row
without a metric.

`comparison.csv` has `variant`, `question`, `changes` and `seeds`; for each split the
seed-mean AP, its spread over seeds and the 90% interval of the seed mean
(`validation_ap`, `validation_ap_spread`, `validation_ap_low`, `validation_ap_high`, and
the same for `test_ap`); the paired validation delta against the baseline with its
interval (`validation_ap_delta`, `validation_ap_delta_low`, `validation_ap_delta_high`)
and `consistent` (every seed's delta has one sign and the interval excludes zero); the
seed means of `validation_roc_auc`, `test_roc_auc` and recall and precision at each
budget (`validation_recall_at_1pct`, `validation_precision_at_1pct` and so on for `5pct`,
`10pct` and the test split); `best_epoch`, `parameter_count` and `training_hours`;
`unpaired_accounts` (validation accounts left out of the pairing because some audit
rejected them); and `differs` (the runs whose commit, dirty state, device or sampler
backend differ from the suite's usual value). The seed-mean AP is the mean of each run's
own audit, while its interval and the delta come from the accounts every audit scored, so
when `unpaired_accounts` is above 0 the mean can fall outside its interval. The built-in
rejection limit of 0 fails any audit that rejects an account, so it is 0 for the built-in
variants.

| Figure | What it shows |
|---|---|
| `comparison_ap.png` | Validation and test audit AP per variant: a dot per seed, the seed mean, its interval, the baseline line |
| `comparison_delta.png` | Validation audit AP minus the baseline's: the paired interval, per-seed deltas, zero line; filled when consistent |
| `comparison_budget.png` | Validation audit recall at 1%, 5% and 10% per variant |
| `comparison_capture.png` | Seed-mean validation capture curves, one panel per variant with the baseline in each |
| `comparison_validation.png` | Seed-mean proxy AP per epoch, up to the last epoch every seed trained, one panel per variant with the baseline in each |
| `comparison_proxy_vs_audit.png` | Selected proxy AP against validation audit AP per run, with Spearman's rank correlation |

## A diagnostic study: `results/diagnostics/<dataset id>/`

| File | Content |
|---|---|
| `features.parquet` | The feature table: one row per sampled account of each split, with `account_id`, `split`, `date`, `is_mule`, `revealed`, `ring_id`, `label_source`, `inclusion_probability`, `weight` (1 / `inclusion_probability`), `rejected`, `context_contract` and `analytics_contract`, then one column per feature named `<family>__<name>` |
| `<analysis>.csv` | One long table per analysis (the analysis named with underscores, as in `learning_curve.csv`) |
| `study.json` | The dataset, the run compared, the reveal's salt and budget, and each analysis' last outcome (`written`, `kept` or `skipped` with its reason) |
| `events.jsonl` | The full record of every event of `mule diagnose`: each analysis' outcome (`diagnose`), the feature table's splits (`feature_table`), retries and warnings |
| `plots/<figure>.png` | The study's figures |
| `report.md` | The study's tables, with links to its figures |

The feature families are `model` (the root's model inputs), `messages` (summaries of the
hop-1 pool), `account` (the analytics query's account history) and `message_context`.
Every table is in long format, its key columns then `metric` and `value`:

| Table | Columns | Figures |
|---|---|---|
| `univariate.csv` | `feature`, `family`, `split`, `metric`, `value` | `univariate_auc.png` |
| `drift.csv` | `feature`, `family`, `model`, `setup`, `split`, `metric`, `value` | `drift.png` |
| `baselines.csv` | `baseline`, `features`, `model`, `split`, `metric`, `value`, `low`, `high` | `baselines.png` |
| `learning_curve.csv` | `model`, `labels`, `mules`, `repeat`, `split`, `metric`, `value` | `learning_curve.png` |
| `subgroups.csv` | `split`, `subset`, `rank`, `metric`, `value` | `ap_concentration.png`, `ring_coverage.png` |
| `proxy_validity.csv` | `split`, `subset`, `metric`, `value` | `proxy_validity.png` |
| `reveal_spread.csv` | `salt`, `split`, `metric`, `value` | `reveal_spread.png` |
| `nnpu_simulation.csv` | `positive_weight`, `seed`, `metric`, `value` | `nnpu_simulation.png` |

A key that does not apply to a row is empty; the baselines give each metric's bootstrap
interval in `low` and `high`.

## The archive: `results/archive/`

The experiments script moves a run whose settings differ from its variant's, whose
`config.json` cannot be read, or which was trained on another dataset, whole to
`results/archive/<variant>/seed-<n>/<UTC time>/` before training it again, with a
`run_archived` event naming what differed. Nothing there is deleted.

## The report of `mule check`: `results/check.json`

Each `mule check` replaces it with the report its checklist summarises: `graph`,
`scope_schema` (`present` or `missing`), `queries` (`up_to_date`, `stale` with each query's
issues, and `retired`), `cugraph` (the probe's `status`, device and reason), `dataset`,
`problems`, `status` (`ready` or `not_ready`), `peak_process_rss_bytes` and
`graph_writes` (always 0). Once the graph is ready it adds `source_open_seconds` and
`first_step`: the step's device, determinism and seed, the roots and accepted roots, the
batch's statistics, the rejections, context requests, REST calls and retries, the
seconds, the input and sampler fingerprints, the `tensor_bytes`, `tensor_digests` (each
tensor's dtype, shape and sha256, and summaries of the floating ones), their one
`batch_digest`, the `loss`, `objective` and `train_step_seconds`, the parameter count and
the accelerator's peak memory. Nothing else reads it.

## A command's own records: `results/events.jsonl`

Every `mule` command appends to it the records of the events it emits while no run,
dataset or study is recording its own, one JSON object each, which names the command
(`command`, such as `"train"`) after the time (`time`): the queries an install found
stale and up to date (`install`), the output TigerGraph gave each GSQL write (`gsql`: the
scope schema change, each `CREATE` and `DROP`), an install request left unanswered with
the error that ended it (`install_unanswered`), the wait for compilation and its end
(`install_wait`, `installed`), the retries of connecting with their whole error
(`retry`), a dataset found ready (`dataset`), a complete run reported
(`already_complete`), everything `mule install` and `mule check` emit, and the error that
stopped a command when no run, dataset or study was recording
([The error that stops a command](#the-error-that-stops-a-command)). So whatever the
console shows in a few words, or not at all, before preparation records in the dataset's
`events.jsonl` and training in the run's, is kept here. The time and the command tell one
command's records from another's: the install of `mule install` from the one `mule
train` began with, or the retries of connecting in one session from those of the next.
The file only grows; nothing reads it. The experiments script records the same events
in its suite's `events.jsonl`, without the command.

## The error that stops a command

A command that stops on an error records it before it exits, in the `events.jsonl` that
was recording when the error was raised: the run's, the dataset's or the study's, or
`results/events.jsonl` when none of them was, as for a single-attempt write such as the
scope schema change, a `CREATE QUERY` or a `DROP`. So the cause is in a file beside the
records that led to it, and not only on stderr. Its record names the command (`command`)
and gives the error's type and message on one line (`error`):

- **`command_stopped`**: TigerGraph's failures outlasted the retries, and stderr has the
  one line that says so; `error` keeps the whole of that line's error.
- **`command_failed`**: any other error, a bug, which Python shows with its traceback;
  `error` is cut to about 200 characters, and the record adds the error's `type`, named
  as the traceback names it, and its whole `message`.

The experiments script records its own the same way, as `run_experiments.py`, in the
dataset's or the suite's `events.jsonl`. An interruption (Ctrl-C) records nothing.

## events.jsonl and the console

Every event is one JSON object with an `event` name, led in the file by the UTC time it
was recorded (`time`, ISO 8601 to the second, such as `"2026-10-03T13:16:17+00:00"`):
preparation's stages, installs, retries (with a short `reason` beside the error), the
start or resume of a run (with its device, threads, determinism and plan), training
intervals, the scoring of each chunk of validation and test, epochs, completion, the
audits, scoring, the analyses, a suite's plan and runs, the sampler backend and warnings
(such as `cugraph_probe`, `context_cache_refused` and `host_settings`). The whole record
goes to the `events.jsonl` of what the command is working on: the run's for training,
the audits and scoring, the dataset's for its preparation, the suite's for the
experiments script's own events, the study's for `mule diagnose`. An event a command
emits outside them, such as an install or a retry before preparation, the `dataset`
event of a dataset found ready, or anything `mule install` and `mule check` emit, goes to
`results/events.jsonl`
([A command's own records](#a-commands-own-records-resultseventsjsonl)).

The console shows a short line for the events a person follows and nothing for the
others (`runtime.console.LINES` decides, by event name): the running totals, batch counts
and the scoring of each chunk stay in the file. The training steps, the wait for an
install to compile and how far scoring has come (validation and test in training, an
audit's sample, `mule score`) are rewritten in place on a terminal and not shown when
stdout is a file or a pipe; `history.csv` has every training interval either way.
