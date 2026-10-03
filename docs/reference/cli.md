# Command line

One console script, `mule`, with seven commands; `python -m mule_pattern_learner` runs the
same ones, for a machine where another tool also installs a `mule`. The commands take no
option besides `--help`: every setting is built in (`config.DEFAULT_CONFIG`, see
[Configuration](configuration.md)) and `.env` holds only the TigerGraph connection.
Before any CUDA work every command reserves cuBLAS's deterministic workspace
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`, unless it is already set).

The console shows what a person follows, and the files hold the rest:

- **Progress**, one short line per event worth reading: TigerGraph not answering yet
  ("TigerGraph is not answering yet (starting workspace, HTTP 502): attempt 2, retrying in
  6 s"), an install starting and ending, what preparation found ("Dataset 1a2b3c4d5e6f: 20
  / 11 / 20 known mules in train / validation / test"), the start of training, one line
  per epoch, each audit and analysis, and warnings. On a terminal the training steps are
  rewritten in place on one line ("epoch 3  step 60/100  loss 0.136  1.9 s/step"); when
  stdout is a file or a pipe they are left out, so the log of `nohup mule train` has no
  step lines.
- **A summary** when the command is done: a few lines of its result, described with each
  command below. No command prints JSON.
- **The full records** stay in the files: every event's record, whole, in the
  `events.jsonl` of the run, dataset, suite or study it belongs to, with the running
  totals, batch counts and scoring progress the console leaves out (an install, or a
  retry, before any of them is known has its line only); `history.csv`, `epochs.csv` and
  `metrics.json` for training; `audit/<split>.json` for the audits; `results/check.json`
  for `mule check`; the suite's and the study's directories ([Outputs](outputs.md)).

A command exits 1 when the graph is not ready (`mule check`), the study is incomplete
(`mule diagnose`), or TigerGraph's failures outlast the retries: that failure is one line
on stderr, after the retries' lines, naming the operation, the attempts and why they
ended, then the error that ended them in about 200 characters of TigerGraph's own words
(an HTML page by its title), so a cause such as "out of memory" is not cut. Any other
error is raised with its traceback.

`RUN` defaults to the built-in run's directory, `results/baseline/seed-42`.
[Outputs](outputs.md) lists every file the commands write.

## What runs where

| Stage | Run by | TigerGraph work | Writes to TigerGraph |
|---|---|---|---|
| Install the queries | `mule install`, and every preparation that connects | Creates and compiles the stale training queries and their callers; `mule install` alone then drops the retired names still installed | The query catalog, and the scope vertex type if it is missing |
| Create the scope | the first preparation, when `scope.id` does not exist | Partitions every Account and Party into train, validation and test | One scope vertex and one membership edge per Account and Party |
| Reveal the known mules | the first preparation, on a graph without known labels | Simulates each mule's discovery and reveals up to 20 per split | The label fields of every internal Account |
| Prepare the dataset | `mule train`, `mule diagnose` and the experiments script when no ready dataset exists | Installs the stale queries (above), then pages the population, resolves the cutoffs, builds the hub registry | Only the install |
| Train | `mule train` | Two rounds of context requests per step, less what the context cache holds | No |
| Check readiness | `mule check` | Reads the schema and the query catalog, then the contexts of one batch | No |
| Audit | `mule evaluate` | Reads the ground truth, pages the validation and test populations, and requests the audit samples' contexts | No |
| Score | `mule score` | Resolves the cutoff, builds an unscoped hub registry and requests the accounts' contexts | No |
| Diagnose | `mule diagnose` | Installs the analytics queries where their text differs, then reads the truth, the populations, both context queries for every split's audit sample, and the reveal's inputs | The analytics queries in the catalog |

| Command | What it does | Writes to TigerGraph |
|---|---|---|
| [`mule train`](#mule-train) | Prepares the dataset as needed, then trains the built-in run, resumes it, or reports it when it is complete | on a fresh graph: the queries, the scope and the label reveal |
| [`mule evaluate [RUN]`](#mule-evaluate-run) | Ground-truth audits of a run's model on validation and test | no |
| [`mule report [RUN]`](#mule-report-run) | Redraws the figures and `report.md` of a run, a suite or a diagnostic study, offline | no |
| [`mule score ACCOUNTS [DATE]`](#mule-score-accounts-date) | Scores the accounts listed in a file with the built-in run's model | no |
| [`mule check`](#mule-check) | Read-only readiness, then one training batch's digests and first loss | no |
| [`mule diagnose [ANALYSIS]`](#mule-diagnose-analysis) | The diagnostic study of the built-in run's dataset | the analytics queries where their text differs |
| [`mule install`](#mule-install) | Installs the queries whose text differs and drops the retired ones | the query catalog, and the scope vertex type if missing |

## mule train

Trains the built-in run into `results/baseline/seed-42/`.

1. **Before anything connects**, it looks at the run directory. A complete run
   (`metrics.json` exists) of the same settings is reported from its `metrics.json` and
   left untouched. A run of other settings, complete or interrupted, is an error that
   names the settings that differ; move it aside to train these.
2. **It prepares the dataset** of the built-in settings in `data/<dataset id>/`, unless a
   ready one exists: then nothing connects, but the dataset's recorded query hashes must
   match the repository's. Otherwise it installs the stale queries (the retired ones stay
   for `mule install` to drop), creates the scope if it is missing, reveals the known
   mules if the graph has none, then pages the population, resolves the cutoffs and
   builds the hub registry. A preparation that stopped is resumed.
3. **It trains**, or continues an interrupted run from its `resume.pt`. The run checks the
   graph against the dataset's frozen source first.
4. **It draws the training figures** and `report.md` once every run file is saved.

Run it in `tmux` or with `nohup`. It shows the dataset, the plan ("Training on cuda
(cuGraph sampler) into results/baseline/seed-42: 100 steps per epoch, at most 30 epochs,
early stop after 6 without gain"), each epoch's loss, validation proxy AP and ROC AUC,
time and whether it is the best so far, and the early stop. Its summary gives the time
taken, the best epoch, the validation and test proxy AP, ROC AUC and recall at the top 1%
with their known mules, and the run directory; a complete run says so and gives the same
summary from its `metrics.json`. The proxy numbers count unlabelled accounts as
negatives: `mule evaluate` gives the ground-truth audit.

## mule evaluate [RUN]

Audits the run's frozen model against the ground truth on validation (for decisions) and
test (for reporting), into the run's `audit/`, then redraws the audit figures and
`report.md`. Each audit scores every mule of its split and 2,000 uniform non-mules, at
the split's cutoff, and weights them to the whole split ([Training](../explanation/training.md#the-ground-truth-audit)).

A split the run already has an audit for is reported from its `audit/<split>.json` and
never rewritten; when both are recorded nothing connects. Otherwise the model, its
dataset (the one `model.pt` names, in `data/`) and the hub registry are checked first,
then one connection, with the model's retry budgets, checks the frozen source, reads the
truth once for both splits and audits the missing ones. The summary is a table with a
column per split, saying which is for decisions: the mules in the sample, then AP, ROC
AUC and recall and precision at 1%, 5% and 10%, each with its 90% interval. An audit
fails before writing anything when a mule is rejected, or when the rejected share exceeds
the model's `runtime.max_rejected_root_fraction`.

## mule report [RUN]

Redraws, from the saved files and without connecting, the figures in `plots/` and
`report.md` of:

- a run directory: the training figures of a complete run and the audit figures of its
  audited splits (the test split's figures need its audit);
- a suite directory, `results/experiments/<suite>/` (it holds `summary.csv`): the six
  comparison figures and the suite's report;
- a diagnostic study, `results/diagnostics/<dataset id>/` (it holds `study.json`): the
  figures of the tables it has, and the study's report.

A figure that fails to draw loses its older PNG, the other figures and `report.md` are
still written, and the command then fails naming every failed figure. Otherwise it says
how many figures it drew, and where.

## mule score ACCOUNTS [DATE]

Scores the accounts listed in the file `ACCOUNTS`, one id per line, with the built-in
run's `model.pt` at `DATE` (an ISO date; the model's test cutoff by default). It writes
`scores/<file stem>_<date>.parquet` in the run, and the ids TigerGraph rejected (missing,
not yet visible, over capacity) to `scores/<file stem>_<date>_rejected.txt`. Existing
outputs are refused before connecting.

Scoring reads the graph as an operational scorer would: the history visible before the
date, without the experiment scope, and a hub registry computed for that cutoff (a date
before the graph's first visible event is refused). It needs neither the training dataset
nor any label, and the installed queries must be the repository's. The graph need not be
the dataset's frozen source, so scoring has no context cache. Its summary says how many
accounts it scored and where, how many TigerGraph rejected by status, and how many child
contexts it left out; the run's `events.jsonl` keeps the whole result, root and child
rejections apart. [Score new accounts](../how-to/score-new-accounts.md) has an example.

## mule check

Read-only readiness of the graph and the built-in run, shown as a checklist (`[x]` ready,
`[ ]` not ready, `[-]` for information): whether the scope vertex type exists, whether the
training queries are installed with the repository's text (naming the stale ones), which
retired queries are still installed, on a CUDA host what the cuGraph probe found, and
whether the built-in run's dataset is prepared in `data/`. When everything is ready it
builds the first training batch as training builds it and runs one optimizer step on the
configured device. The batch's contexts are requested from TigerGraph, not read from the
context cache, so the installed context query and its first Fourier spot check run.

The checklist's last item gives the batch's roots, context requests and seconds, the
step's `loss` and `objective` to six decimals, and one digest of the batch's tensors. Two
code versions that show the same digest and loss on one machine and device built the
same first batch and step. It ends "Ready to train.", or "Not ready" with the commands to
run (`mule install`, then `mule train`), and the command then exits 1. The full report
goes to `results/check.json`, replaced each time: `queries` (`up_to_date`, `stale` and
`retired`), `cugraph`, `dataset`, `first_step` (the REST calls, retries and seconds, the
stub and rejected counts, the sampler backend, the digest of every batch tensor in
`tensor_digests` and their one `batch_digest`, `loss` and `objective`), `problems` and
`status` (`ready` or `not_ready`). It never writes to the graph and never prepares a
dataset.

## mule diagnose [ANALYSIS]

Runs the diagnostic study of the built-in run's dataset into
`results/diagnostics/<dataset id>/`: every analysis by default, in this order, or the one
named:

`features`, `univariate`, `drift`, `baselines`, `learning-curve`, `subgroups`,
`proxy-validity`, `reveal-spread`, `nnpu-simulation`

It prepares the dataset as `mule train` does, then writes the feature table, one long
table per analysis, `study.json`, `events.jsonl`, the figures and `report.md`. It is the
only command that installs the analytics queries, where their text differs. It shows a
line per analysis as it ends (written, kept or skipped, with its rows and seconds). An
analysis whose inputs are missing (no built-in run on this dataset, or no audit) is
skipped with its reason; the study is then incomplete, the summary names what was
skipped and the command exits 1. [Run the diagnostics](../how-to/run-diagnostics.md)
describes each analysis.

## mule install

Adds the `Temporal_Training_Scope` vertex type if it is missing, installs the training
queries (`gsql/queries/` and `gsql/evaluation/`) whose text differs, then drops the
installed queries named on `contract.server.RETIRED_QUERIES`, callers first, and lists the
installed queries that no repository file defines without touching them. Every
preparation that connects installs the same way but drops nothing: code from before the
rename calls the retired names, so run `mule install` once no job of that code runs
anywhere. [Queries](queries.md#installation) describes staleness, the 90-minute wait and
the retired names. It says what it installs and drops as it goes, and its summary how many
training queries are installed with the repository's text and which installed queries no
repository file defines.

## The scripts

| Script | What it does |
|---|---|
| `python scripts/run_experiments.py [SUITE or VARIANT ...]` | Trains, audits and compares the control experiments: the `controls` suite by default, or the suites and variants named. `--help` lists them with their questions and changes, without connecting. It shows the run matrix, a line for each run as it finishes and the top of the comparison, and exits 1 unless every run is trained and audited ([Run the control experiments](../how-to/run-control-experiments.md)) |
| `python scripts/render_queries.py` | Regenerates `gsql/queries/training_context.gsql` and `gsql/analytics/analytics_context.gsql` from `tigergraph.render`, and names the contract to set when a text changed; `--check` only compares the rendered texts with the files and exits 1 when one differs |

## Tests that need the graph or a GPU

The test suite never connects unless a marker selects the tests that do; they are
deselected by default.

| Command | What it checks |
|---|---|
| `python -m pytest -m graph` | Read-only checks against the graph in `.env`: the installed context query (its text, time encodings and cutoff boundary), both context queries' features against the Python reference (the analytics one once it is installed), and a dry run of the label reveal against its Python mirror |
| `python -m pytest -m cuda` | The cuGraph sampler on a CUDA host: its probe, exact counts, strict cutoffs, uniform inclusion and determinism, then one real batch per backend from the prepared dataset and a deterministic step run twice |
| `python -m pytest -m graph_write --allow-graph-writes` | The scope isolation test: it writes fixture vertices and edges with a random prefix, checks that held-out and future data change no training input, and deletes exactly those. Vertex counts change while it runs, so never run it during a training run |
