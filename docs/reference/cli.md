# Command line

One console script, `mule`, with seven commands; `python -m mule_pattern_learner` runs the
same where another tool also installs a `mule`. No command takes an option besides
`--help`: every setting is built in (`config.DEFAULT_CONFIG`,
[Configuration](configuration.md)) and `.env` holds only the TigerGraph connection. Before
any CUDA work every command sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (cuBLAS's
deterministic workspace) unless already set. `RUN` defaults to the built-in run,
`results/baseline/seed-42`. [Outputs](outputs.md) lists every file written.

| Command | What it does | TigerGraph work | Writes to TigerGraph |
|---|---|---|---|
| [`mule train`](#mule-train) | Trains the built-in run, resumes it, or reports it if complete | [Preparation](#preparation), then two rounds of context requests per step, less what the context cache holds | Preparation's |
| [`mule evaluate [RUN]`](#mule-evaluate-run) | Ground-truth audits of a run's model on validation and test | Reads the ground truth, pages the validation and test populations, requests the audit samples' contexts | No |
| [`mule report [RUN]`](#mule-report-run) | Redraws the figures and `report.md` of a run, suite or diagnostic study | None | No |
| [`mule score ACCOUNTS [DATE]`](#mule-score-accounts-date) | Scores the accounts in a file with the built-in run's model | Resolves the cutoff, builds an unscoped hub registry, requests the accounts' contexts | No |
| [`mule check`](#mule-check) | Read-only readiness, then one training batch's digests and first loss | Reads the schema and query catalog, then one batch's contexts | No |
| [`mule diagnose [ANALYSIS]`](#mule-diagnose-analysis) | The diagnostic study of the built-in run's dataset | Preparation, installing the analytics queries where their text differs, then the truth, the populations, both context queries for every split's audit sample, the reveal's inputs | Preparation's; the analytics queries |
| [`mule install`](#mule-install) | Installs the queries whose text differs, drops the retired ones | Creates and compiles the stale training queries and their callers, replaces an outdated scope vertex type, drops the retired names | The query catalog; the scope vertex type if missing or outdated |

## The console

- **Progress**: one short line per event worth reading (`runtime.console.LINES` decides by
  event name): TigerGraph not answering yet ("TigerGraph is not answering yet (starting
  workspace, HTTP 502): attempt 2, retrying in 6 s"), an install starting and ending,
  what preparation found ("Dataset 1a2b3c4d5e6f: 20 / 11 / 20 known mules in train /
  validation / test"), training's start, a line per epoch, each audit and analysis,
  warnings.
- **In place**: on a terminal, training steps ("epoch 3  step 60/100  loss 0.136  1.9
  s/step"), the wait for an install to compile, and scoring of validation and test, an
  audit's sample and `mule score` ("scoring validation 640/2,011") rewrite one line until
  the next line replaces it. When stdout is a file or pipe they are left out, so a
  `nohup mule train` log has no step lines; `history.csv` has every interval either way.
- **A summary** when done, given per command below. No command prints JSON.
- **The full records** stay in files, including the running totals, batch counts and
  chunk scoring the console omits
  ([events.jsonl and the console](outputs.md#eventsjsonl-and-the-console)).

**Exit status 1** when the graph is not ready (`mule check`), the study is incomplete
(`mule diagnose`), or TigerGraph's failures outlast the retries. That failure is one
stderr line after the retries' lines: the operation, the attempts and why they ended, then
the ending error in about 200 characters of TigerGraph's own words (an HTML page by its
title), so a cause such as "out of memory" is not cut. Any other error raises with its
traceback. Either way the error is recorded with the command in the `events.jsonl` then
recording, so a tmux or `nohup` session keeps its cause
([The error that stops a command](outputs.md#the-error-that-stops-a-command)).

## Preparation

`mule train`, `mule diagnose` and the experiments script prepare the built-in settings'
dataset in `data/<dataset id>/` unless a ready one exists (then nothing connects, but its
recorded query hashes must match the repository's). A stopped preparation resumes:

1. Install the stale training queries and their callers
   ([Installation](queries.md#installation)); retired names stay for `mule install`.
   Writes the query catalog, and the scope vertex type if missing.
2. If `scope.id` does not exist, create the scope: every Account and Party partitioned
   into train, validation and test. Writes one scope vertex and one membership edge per
   Account and Party.
3. On a graph without known labels, or with another reveal's, reveal every mule a bank
   would have discovered by each split's cutoff, by simulating each mule's discovery.
   Writes every internal Account's label fields.
4. Page the population, resolve the cutoffs, build the hub registry (no writes).

## mule train

Trains the built-in run into `results/baseline/seed-42/`. Run it in `tmux` or with
`nohup`.

1. Before connecting it checks the run directory. A complete run (`metrics.json` exists)
   of the same settings is reported from `metrics.json`, untouched. A run of other
   settings, complete or interrupted, is an error naming the differing settings; move it
   aside to train.
2. [Preparation](#preparation).
3. It trains, or continues from `resume.pt`, after checking the graph against the
   dataset's frozen source.
4. It draws the training figures and `report.md` once every run file is saved.

It shows the dataset, the plan ("Training on cuda (cuGraph sampler) into
results/baseline/seed-42: 100 steps per epoch, at most 30 epochs, early stop after 6
epochs without gain"), each epoch's loss, validation proxy AP and ROC AUC, time and
whether it is the best so far, and the early stop. The summary (for a complete run, from
its `metrics.json`): time taken, best epoch, validation and test proxy AP, ROC AUC and
recall at the top 1% with their known mules, the run directory. Proxy numbers count
unlabelled accounts as negatives; `mule evaluate` gives the ground-truth audit.

## mule evaluate [RUN]

Audits the run's frozen model against the ground truth on validation (for decisions) and
test (for reporting) into its `audit/`, then redraws the audit figures and `report.md`.
Each audit scores every mule of its split and 2,000 uniform non-mules at the split's
cutoff, weighted to the whole split
([Training](../explanation/training.md#the-ground-truth-audit)).

- An audited split is reported from its `audit/<split>.json`, never rewritten, and the
  command says so ("results/baseline/seed-42 is already audited on validation and test;
  its audit/validation.json and audit/test.json are summarised below"); with both
  recorded nothing connects.
- Otherwise it first checks the model, its dataset (the one `model.pt` names) and the hub
  registry; then one connection, with the model's retry budgets, checks the frozen
  source, reads the truth once for both splits and audits the missing ones.
- An audit fails before writing when a mule is rejected, or the rejected share exceeds
  the model's `runtime.max_rejected_root_fraction`.

The summary is a table, a column per split, marking the one for decisions: hidden and
revealed mules in the sample, then AP, ROC AUC, and recall and precision at 1%, 5% and
10% of the hidden mules (revealed mules removed from the ranking), then the same of every
mule, each with its 90% interval.

## mule report [RUN]

Redraws offline, from the saved files, `plots/` and `report.md` of:

- a run: a complete run's training figures and its audited splits' figures (the test
  split's need its audit);
- a suite, `results/experiments/<suite>/` (holds `summary.csv`): the seven comparison
  figures and the report;
- a study, `results/diagnostics/<dataset id>/` (holds `study.json`): the figures of the
  tables it has, and the report.

A figure that fails loses its older PNG; the rest and `report.md` are still written, then
the command fails naming every failed figure. Otherwise it says how many figures it drew,
and where.

## mule score ACCOUNTS [DATE]

Scores the accounts in file `ACCOUNTS` (one id per line) with the built-in run's
`model.pt` at `DATE` (ISO date; default the model's test cutoff), into the run's
`scores/<file stem>_<date>.parquet`, with ids TigerGraph rejected (missing, not yet
visible, over capacity) in `scores/<file stem>_<date>_rejected.txt`. Existing outputs are
refused before connecting.

It reads as an operational scorer would: history visible before the date, no experiment
scope, a hub registry for that cutoff (a date before the graph's first visible event is
refused). It needs no training dataset or label, but the installed queries must be the
repository's. The graph need not be the frozen source, so there is no context cache. The
summary gives the accounts scored and where, rejections by status, and child contexts
left out; the run's `events.jsonl` keeps the whole result, root and child rejections
apart. [Score new accounts](../how-to/score-new-accounts.md) has an example.

## mule check

Read-only readiness of the graph and the built-in run, as a checklist (`[x]` ready, `[ ]`
not ready, `[-]` information): whether the scope vertex type matches
`gsql/schema/scope_vertex.gsql` (if outdated, how it differs and that `mule install`
replaces it while the graph holds no scope vertex); whether the training queries have the
repository's text (naming stale ones); which retired queries are still installed; on a
CUDA host, the cuGraph probe's result; whether the built-in run's dataset is in `data/`.

When all is ready it builds the first training batch as training does and runs one
optimizer step on the configured device, requesting the contexts from TigerGraph (not the
cache), so the installed context query and its first Fourier spot check run. The last
item gives the batch's roots, context requests and seconds, the step's `loss` and
`objective` to six decimals, and one digest of the batch's tensors: two code versions
showing the same digest and loss on one machine and device built the same batch and step.

It ends "Ready to train.", or "Not ready" with what to do (`mule install`, then
`mule train`, after clearing and reloading the graph's data when scopes use an outdated
scope vertex type) and exits 1. It never writes to the graph or prepares a dataset. The
full report replaces `results/check.json`
([Outputs](outputs.md#the-report-of-mule-check-resultscheckjson)).

## mule diagnose [ANALYSIS]

Runs the diagnostic study of the built-in run's dataset into
`results/diagnostics/<dataset id>/`: every analysis in this order, or the one named:

`features`, `univariate`, `drift`, `baselines`, `learning-curve`, `subgroups`,
`activity-timing`, `proxy-validity`, `reveal-spread`, `nnpu-simulation`

After [preparation](#preparation) it writes the feature table, a long table per analysis,
`study.json`, `events.jsonl`, the figures and `report.md`. Only this command installs the
analytics queries (where their text differs). It shows a line per finished analysis
(written, kept or skipped, with rows and seconds). An analysis missing its inputs (no
built-in run on this dataset, or no audit) is skipped with its reason; the summary names
it, and the incomplete study exits 1.
[Run the diagnostics](../how-to/run-diagnostics.md) describes each analysis.

## mule install

Adds a missing `Temporal_Training_Scope` vertex type or replaces an outdated one
([The scope types](queries.md#the-scope-types)), installs the training queries
(`gsql/queries/` and `gsql/evaluation/`) whose text differs
([Installation](queries.md#installation)), drops the
[retired names](queries.md#the-retired-names), and lists, untouched, the installed
queries no repository file defines. Preparation installs the same way but drops nothing
and refuses an outdated scope type. It says what it installs, replaces and drops as it
goes; the summary gives how many training queries have the repository's text, whether it
replaced the scope vertex type, and which installed queries no repository file defines.

## The scripts

| Script | What it does |
|---|---|
| `python scripts/run_experiments.py [SUITE or VARIANT ...]` | Trains, audits and compares the control experiments: the `controls` suite by default, or the suites and variants named; `--help` lists them with their questions and changes, without connecting. Shows the run matrix, a line per finished run and the top of the comparison; exits 1 unless every run is trained and audited ([Run the control experiments](../how-to/run-control-experiments.md)) |
| `python scripts/render_queries.py` | Regenerates `gsql/queries/training_context.gsql` and `gsql/analytics/analytics_context.gsql` from `tigergraph.render`, naming the contract to set when a text changed; `--check` only compares, exiting 1 when one differs |

## Tests that need the graph or a GPU

Deselected by default; the suite never connects unless a marker selects these.

| Command | What it checks |
|---|---|
| `python -m pytest -m graph` | Read-only, against the graph in `.env`: the installed context query (text, time encodings, cutoff boundary), both context queries' features against the Python reference (the analytics one once installed), a dry run of the label reveal against its Python mirror |
| `python -m pytest -m cuda` | The cuGraph sampler on a CUDA host: probe, exact counts, strict cutoffs, uniform inclusion, determinism, then one real batch per backend from the prepared dataset and a deterministic step run twice |
| `python -m pytest -m graph_write --allow-graph-writes` | Scope isolation: writes fixture vertices and edges with a random prefix, checks that held-out and future data change no training input, deletes exactly those. Vertex counts change meanwhile, so never run it during training |
