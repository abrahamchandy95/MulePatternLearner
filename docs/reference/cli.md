# Command line

One console script, `mule`, with seven commands; `python -m mule_pattern_learner` runs the
same ones, for a machine where another tool also installs a `mule`. The commands take no
option besides `--help`: every setting is built in (`config.DEFAULT_CONFIG`, see
[Configuration](configuration.md)) and `.env` holds only the TigerGraph connection. Each
command prints one JSON result on its last line, after the structured event lines it
printed while it ran, and the commands that work on a run append those event lines to the
run's `events.jsonl`. Before any CUDA work every command reserves cuBLAS's deterministic
workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, unless it is already set).

`RUN` defaults to the built-in run's directory, `results/baseline/seed-42`.
[Outputs](outputs.md) lists every file the commands write.

## What runs where

| Stage | Run by | TigerGraph work | Writes to TigerGraph |
|---|---|---|---|
| Install the queries | `mule install`, and `mule train` when a text differs | Creates and compiles the stale training queries and their callers, then drops the retired names still installed | The query catalog, and the scope vertex type if it is missing |
| Create the scope | the first preparation, when `scope.id` does not exist | Partitions every Account and Party into train, validation and test | One scope vertex and one membership edge per Account and Party |
| Reveal the known mules | the first preparation, on a graph without known labels | Simulates each mule's discovery and reveals up to 20 per split | The label fields of every internal Account |
| Prepare the dataset | `mule train` (and `mule diagnose`) when no ready dataset exists | Pages the population, resolves the cutoffs, builds the hub registry | No |
| Train | `mule train` | Two rounds of context requests per step, less what the context cache holds | No |
| Check readiness | `mule check` | Reads the schema and the query catalog, then the contexts of one batch | No |
| Audit | `mule evaluate` | Reads the ground truth, pages the validation and test populations, and requests the audit samples' contexts | No |
| Score | `mule score` | Resolves the cutoff, builds an unscoped hub registry and requests the accounts' contexts | No |
| Diagnose | `mule diagnose` | Installs the analytics queries where their text differs, then reads the truth, the populations, both context queries for every split's audit sample, and the reveal's inputs | The analytics queries in the catalog (and the install's drop of the retired names) |

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
   match the repository's. Otherwise it installs the stale queries (and drops the retired
   ones, as `mule install` does), creates the scope if it is missing, reveals the known
   mules if the graph has none, then pages the population, resolves the cutoffs and
   builds the hub registry. A preparation that stopped is resumed.
3. **It trains**, or continues an interrupted run from its `resume.pt`. The run checks the
   graph against the dataset's frozen source first.
4. **It draws the training figures** and `report.md` once every run file is saved.

The result is the run's `metrics.json` record. Run it in `tmux` or with `nohup`;
progress goes to stdout and `events.jsonl`.

## mule evaluate [RUN]

Audits the run's frozen model against the ground truth on validation (for decisions) and
test (for reporting), into the run's `audit/`, then redraws the audit figures and
`report.md`. Each audit scores every mule of its split and 2,000 uniform non-mules, at
the split's cutoff, and weights them to the whole split ([Training](../explanation/training.md#the-ground-truth-audit)).

A split the run already has an audit for is reported from its `audit/<split>.json` and
never rewritten; when both are recorded nothing connects. Otherwise the model, its
dataset (the one `model.pt` names, in `data/`) and the hub registry are checked first,
then one connection, with the model's retry budgets, checks the frozen source, reads the
truth once for both splits and audits the missing ones. The result holds the reports by
split. An audit fails before writing anything when a mule is rejected, or when the
rejected share exceeds the model's `runtime.max_rejected_root_fraction`.

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
still written, and the command then fails naming every failed figure.

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
the dataset's frozen source, so scoring has no context cache. The result reports root
and child rejections apart. [Score new accounts](../how-to/score-new-accounts.md) has an
example.

## mule check

Read-only readiness of the graph and the built-in run. It reports the graph name, whether
the scope vertex type exists, which training queries are installed with the repository's
text (`queries.up_to_date` and `queries.stale`), which retired queries are still installed
(`queries.retired`), and on a CUDA host what the cuGraph probe found. When everything is
ready and the built-in run's dataset is prepared in `data/`, it builds the first training
batch as training builds it and runs one optimizer step on the configured device. The
batch's contexts are requested from TigerGraph, not read from the context cache, so the
installed context query and its first Fourier spot check run.

`first_step` holds the REST calls, retries and seconds, the stub and rejected counts, the
sampler backend, a digest of every batch tensor (`tensor_digests`), and the step's `loss`
and `objective`. Two code versions that print the same digests and loss on one machine and
device built the same first batch and step. The status is `ready`, or `not_ready` with
the `problems` found, and the command then exits 1. It never writes to the graph and
never prepares a dataset.

## mule diagnose [ANALYSIS]

Runs the diagnostic study of the built-in run's dataset into
`results/diagnostics/<dataset id>/`: every analysis by default, in this order, or the one
named:

`features`, `univariate`, `drift`, `baselines`, `learning-curve`, `subgroups`,
`proxy-validity`, `reveal-spread`, `nnpu-simulation`

It prepares the dataset as `mule train` does, then writes the feature table, one long
table per analysis, `study.json`, `events.jsonl`, the figures and `report.md`. It is the
only command that installs the analytics queries, where their text differs (that install
also drops the retired queries still installed). An analysis whose inputs are missing (no
built-in run on this dataset, or no audit) is skipped with its reason; the result's
status is then `incomplete` and the command exits 1. [Run the diagnostics](../how-to/run-diagnostics.md)
describes each analysis.

## mule install

Adds the `Temporal_Training_Scope` vertex type if it is missing, installs the training
queries (`gsql/queries/` and `gsql/evaluation/`) whose text differs, then drops the
installed queries named on `contract.server.RETIRED_QUERIES`, callers first, and lists the
installed queries that no repository file defines without touching them. `mule train`
does the same before it prepares a dataset. [Queries](queries.md#installation) describes
staleness, the 45-minute wait and the retired names. The result lists the queries
`installed`, `up_to_date`, `dropped` and `not_defined`.

## The scripts

| Script | What it does |
|---|---|
| `python scripts/run_experiments.py [SUITE or VARIANT ...]` | Trains, audits and compares the control experiments: the `controls` suite by default, or the suites and variants named. `--help` lists them with their questions and changes, without connecting. It exits 1 unless every run is trained and audited ([Run the control experiments](../how-to/run-control-experiments.md)) |
| `python scripts/render_queries.py` | Regenerates `gsql/queries/training_context.gsql` and `gsql/analytics/analytics_context.gsql` from `tigergraph.render`, and names the contract to set when a text changed; `--check` only compares the rendered texts with the files and exits 1 when one differs |

## Tests that need the graph or a GPU

The test suite never connects unless a marker selects the tests that do; they are
deselected by default.

| Command | What it checks |
|---|---|
| `python -m pytest -m graph` | Read-only checks against the graph in `.env`: the installed context query (its text, time encodings and cutoff boundary), both context queries' features against the Python reference (the analytics one once it is installed), and a dry run of the label reveal against its Python mirror |
| `python -m pytest -m cuda` | The cuGraph sampler on a CUDA host: its probe, exact counts, strict cutoffs, uniform inclusion and determinism, then one real batch per backend from the prepared dataset and a deterministic step run twice |
| `python -m pytest -m graph_write --allow-graph-writes` | The scope isolation test: it writes fixture vertices and edges with a random prefix, checks that held-out and future data change no training input, and deletes exactly those. Vertex counts change while it runs, so never run it during a training run |
