# Train and evaluate the built-in run

From a machine with the code to an audited model. You need a TigerGraph graph that holds
the loaded data ([Set up a graph](set-up-a-graph.md)) and its connection details. Every
setting is built in ([Configuration](../reference/configuration.md)); the commands take no
options.

## Install

Python 3.12 or newer, in an editable install, since the commands read `gsql/` from the
repository:

```bash
pip install -e ".[dev]"
```

torch and matplotlib are core dependencies. On a CUDA host, install the CUDA torch wheel
first and then the cuGraph extra that matches its CUDA major version
([On the CUDA host](#on-the-cuda-host)).

Copy `.env.example` to `.env` and fill in the connection; environment variables override
it. `GRAPHNAME` must be `Mule_Pattern_Learner`, and the connection refuses any other
graph.

```
HOST=https://your-tg-host
GRAPHNAME=Mule_Pattern_Learner
SECRET=your_restpp_secret
```

## Check the graph

```bash
mule check
```

It reports the graph, the scope vertex type, the installed queries and, on a CUDA host,
the cuGraph probe. Until the built-in run's dataset is prepared, it ends `not_ready` and
says so; that is expected on a new machine. Once a dataset exists it also builds the
first training batch and runs one optimizer step, which shows that the graph, the queries
and the device work before an hour of training.

To prepare the dataset without training (it installs, creates the scope and reveals on a
fresh graph, as `mule train` would, and only reads on a graph where they are in place;
about 6 minutes on the reference graph):

```bash
python -c "from mule_pattern_learner.config import DEFAULT_CONFIG; from mule_pattern_learner.pipeline.prepare import prepare_dataset; print(prepare_dataset(DEFAULT_CONFIG).root)"
```

## Train

```bash
mule train
```

Run it in `tmux` or with `nohup`; progress goes to stdout and to
`results/baseline/seed-42/events.jsonl`. It uses CUDA when available, then Apple MPS,
then the CPU. On the CUDA host a step takes about 3 seconds and a run about an hour with
early stopping.

- **Interrupted?** Run the same command again: it continues from `resume.pt` and
  reproduces the uninterrupted run exactly on the same device.
- **Complete?** The command prints the run's `metrics.json` and changes nothing.
- **Settings changed?** If the code's built-in settings now differ from the run's in a
  setting that changes results, the command fails and names the settings. Move
  `results/baseline/seed-42/` aside to train the new settings.

While it trains, watch `history.csv`: `objective` is the unclamped nnPU risk and
`corrected_steps` counts the steps whose non-negative correction fired. A loss far below
the first epoch's, rising corrections and a validation AP that peaks early point to
memorised positives ([Training](../explanation/training.md#the-loss)).

When it finishes, the run directory holds the model, its proxy predictions and metrics,
the training figures in `plots/` and `report.md` ([Outputs](../reference/outputs.md#a-run-resultsvariantseed-n)).
The proxy metrics count unlabelled accounts as negatives; judge the model by its audit.

## Audit

```bash
mule evaluate
```

It audits the model against the ground truth on validation and test, into `audit/`, and
redraws the audit figures and `report.md`. `mule evaluate results/<variant>/seed-<n>`
audits another run. Each audit scores every mule of its split and 2,000 uniform
non-mules, weighted to the whole split, with 90% intervals.

Read `report.md`, or `audit/validation.json` and `audit/test.json`:

- **Decide on validation.** `purpose` says it: `decisions` for validation, `reporting` for
  test. Comparing settings or picking a threshold on the test audit makes the test
  number optimistic.
- **Read the ranking.** Average precision against `weighted_prevalence`, ROC AUC, and
  recall and precision at reviewing the top 1, 5 and 10% of accounts, each with its
  interval. The F1 threshold was chosen on about a dozen validation mules and means
  little.
- **Compare revealed and hidden mules.** `revealed_positives` and `hidden_positives`, and
  `audit_revealed_hidden.png`: a model that finds only mules like the ones it was shown is
  not yet a detector.

A split already audited is never rewritten. To audit again, move its `audit/<split>.*`
files aside.

## Redraw the figures

```bash
mule report
```

It redraws the figures and `report.md` from the saved files, offline;
`mule report results/<variant>/seed-<n>` redraws another run. A figure that fails is
named, and the others are still drawn.

## On the CUDA host

A host that trained with earlier code starts from scratch: this code reads none of its
datasets or models ([Datasets and models of earlier code](#datasets-and-models-of-earlier-code)),
so leave `data/` and `results/` empty or move them aside.

1. **Environment.** Linux x86_64 with an NVIDIA driver for CUDA 12 (525.60 or newer) or
   CUDA 13 (580.65 or newer), and Python 3.12 to 3.14.
2. **The code.** Pull `main` (`git switch main && git pull`), or clone the repository.
3. **torch and cuGraph.** Install them again after the pull, so the `mule` command and
   the extras match the code. For CUDA 12, the cu129 torch wheel, then the `cuda12`
   extra, whose cuGraph wheels are on pypi.nvidia.com:

   ```bash
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
   pip install -e '.[dev,cuda12]' --extra-index-url=https://pypi.nvidia.com
   ```

   For CUDA 13, the cu130 wheel and the `cuda13` extra, which is on pypi.org:

   ```bash
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
   pip install -e '.[dev,cuda13]'
   ```

4. **`.env`.** Copy it; nothing else is copied. The settings are built in, the known mules
   are in TigerGraph, and the run prepares its own dataset.
5. **Check cuGraph** on the GPU. The tests skip when cuGraph cannot run on the host; the
   last one also needs the prepared dataset:

   ```bash
   python -m pytest -m cuda tests/integration/test_cugraph_sampler.py
   ```

6. **`mule check`.** Until the first `mule train` on a graph whose queries carry their
   old names, it ends `not_ready`: the renamed queries are stale, the old names are listed
   under `queries.retired`, and there is no dataset.
7. **`mule train`.** Its first run installs the renamed queries beside the old names
   (about 50 minutes, within a 90-minute wait), prepares the dataset (about 6 minutes)
   and trains the built-in run (about an hour). If the wait runs out, wait until `mule
   check` no longer lists the queries under `queries.stale`, then run `mule train` again. `mule check` now ends
   `ready`, still listing the old names under `queries.retired`.
8. **Drop the old names.** Stop every job of the earlier code, on every machine: it calls
   the old names. Then run `mule install`, which finds the renamed queries in place and
   drops the old names, callers first ([Queries](../reference/queries.md#the-retired-names)).
9. **`mule evaluate`, then `mule report`.**
10. **The control experiments:** `python scripts/run_experiments.py`, then
    `python scripts/run_experiments.py feature_drops`
    ([Run the control experiments](run-control-experiments.md)).
11. **`mule diagnose`.** Its first run installs the analytics queries, then
    `python -m pytest -m graph` checks them ([Run the diagnostics](run-diagnostics.md)).
12. **The cache cap.** Set `contract.bounds.CONTEXT_CACHE_ENTRIES` to about five times
    the baseline run's `contexts.distinct` (in its `metrics.json`). The runs of a dataset
    share entries when they request the same server-side groups, so the cache holds the
    baseline's contexts once, once more for each variant that drops a server-side group
    (`drop_entity_meta`, `drop_time_encoding`, `drop_pair_history`, `drop_flow_timing`),
    and the audits' samples beside them. At roughly 7 kB an entry, check that the disk
    has room, and watch the first run that fills the cache: each eviction scans the
    directory while the requests wait ([Outputs](../reference/outputs.md#a-prepared-dataset-datadataset-id)).

If cuGraph fails its probe, training warns (`cugraph_probe`) and samples with the torch
sampler, which draws from the same distribution.

## When TigerGraph rejects roots

The built-in run allows no rejected root (`runtime.max_rejected_root_fraction = 0.0`), so a
root TigerGraph rejects (for example `history_capacity_exceeded`) stops the run with the
statuses that caused it. The limit decides only whether a run may go on, never its
numbers, so the interrupted run may resume with a higher one, from Python, since the
commands take no options:

```python
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.pipeline.train import train_run

config = DEFAULT_CONFIG.with_changes({"runtime": {"max_rejected_root_fraction": 0.01}})
train_run(config=config, resume=True)
```

The test split is scored after `model.pt` is saved, so a test-split failure leaves
`model.pt` without `metrics.json`, and the same resume finishes it. An observed positive
that is rejected always fails. `mule evaluate` uses the model's own limit.

## Train another configuration

`mule train` trains only the built-in run. Another configuration trains from Python into a
directory of its own, on the dataset of its dataset settings (shared with the built-in
run when they are the same):

```python
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline.evaluate import evaluate_run
from mule_pattern_learner.pipeline.train import train_run

run = RunPaths.of("short_run", 42)
train_run(run, config=DEFAULT_CONFIG.with_changes({"training": {"epochs": 3}}), resume=True)
evaluate_run(run)
```

For comparisons over seeds with paired intervals, declare a variant and use the
experiments script instead ([Run the control experiments](run-control-experiments.md)).

## Datasets and models of earlier code

This code reads only the datasets and models it writes:

- **A dataset** is found by the dataset id of its settings in `data/<dataset id>/`, so a
  dataset kept anywhere else, or one that records no dataset settings, is never read, and
  `mule train` prepares a new one (about 6 minutes). A dataset prepared from other query
  texts is refused ("was built from different GSQL sources"): run `mule install`, move the
  directory aside (its `contexts/` goes with it; nothing deletes it), and run `mule train`
  to prepare it again.
- **A model** must record `SavedModel.FORMAT` and this code's contract fingerprint, so a
  `model.pt` of earlier code is refused. Train it again with this code.

## When something is refused

| Message | Meaning and fix |
|---|---|
| `Installed query differs from repository source or is not installed` | Run `mule install`; it recompiles only the stale queries |
| `Queries [...] are still not installed after ...s` | The 90-minute wait for compilation ran out, and the server may still be compiling. Once it has finished (`mule check` no longer lists the training queries under `queries.stale`; the GSQL shell's `ls` shows the analytics queries), run the same command again (`mule install`, `mule train`, `mule diagnose` or the experiments script): it installs only what is still stale |
| `Prepared dataset ... was built from different GSQL sources` | The query files changed after preparation; install them, then move the dataset aside so it is prepared again |
| `Graph counts changed; freeze the source and prepare a new dataset` | The graph was modified after preparation; freeze it and prepare a new dataset |
| `Scope ... was created with scope.unowned = ...` | The scope was created with another rule; use the stored rule or a new `scope.id` |
| `Account label contract violated after the reveal` | The label attributes are inconsistent; run `validate_label_contract` ([Labels](../reference/labels.md)) |
| `TigerGraph rejected ... training roots so far` or `validation: TigerGraph rejected ... roots` | Roots failed a per-request check beyond the rejection limit, or an observed positive was rejected; the statuses say why ([When TigerGraph rejects roots](#when-tigergraph-rejects-roots)) |
| `The run in ... is complete with other settings` | A finished run of other settings is in the directory; move it aside to train these |
| `Resumed configuration differs from the run` | An interrupted run of other settings is in the directory; move it aside, or resume it with its own settings |
| `... records no format` or `... is a model of format ...` | The model was saved by other code; train it again with this code |
| `The model's input groups or pool definitions differ from its configuration` | The model read a pool group whose definition has changed since; score with a model trained under the current one |
| `Training queries require Mule_Pattern_Learner` | `GRAPHNAME` in `.env` names another graph |
| A `cugraph_probe` warning | pylibcugraph or the GPU failed the probe; training continues with the torch sampler; run `mule check` and the `cuda` tests |
| Retry events in the log | TigerGraph was briefly unavailable or resuming; each operation waits up to `transport.max_outage_s` |
