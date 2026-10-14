# Train and evaluate the built-in run

From a machine with the code to an audited model, on a graph with the data loaded ([Set up
a graph](set-up-a-graph.md)). Every setting is built in
([Configuration](../reference/configuration.md)); the commands take no options and are
described in [Command line](../reference/cli.md).

## Install

Install as [Setup](../../README.md#setup) says (torch and matplotlib are core
dependencies; for a CUDA host see [On the CUDA host](#on-the-cuda-host)). In `.env`,
`SECRET` is the REST++ secret and `GRAPHNAME` must be `Mule_Pattern_Learner`: the
connection refuses any other graph.

## Check the graph

```bash
mule check
```

On a new machine it ends "Not ready" and says to run `mule train`, which is expected. Once
a dataset exists it also runs one training step, proving the graph, queries and device
before an hour of training ([mule check](../reference/cli.md#mule-check)). To prepare the
dataset without training (it installs, creates the split and reveals on a fresh graph, and
only reads where they are in place; about 6 minutes on the reference graph):

```bash
python -c "from mule_pattern_learner.config import DEFAULT_CONFIG; from mule_pattern_learner.pipeline.prepare import prepare_dataset; print(prepare_dataset(DEFAULT_CONFIG).root)"
```

## Train

```bash
mule train
```

Run it in `tmux` or with `nohup`. It uses CUDA, then Apple MPS, then the CPU; on the CUDA
host a step takes about 3 seconds and a run about an hour with early stopping. The console
([mule train](../reference/cli.md#mule-train)) shows the dataset, the plan and an epoch
per line:

```
epoch  1  loss 0.490  validation AP 0.452  ROC AUC 0.955  4.6 min  best so far
epoch  2  loss 0.212  validation AP 0.431  ROC AUC 0.951  3.1 min
```

"best so far" marks the epoch kept so far; other `training.selection` rules name their
criterion ("best ROC AUC so far", or "lowest risk so far" with the nnPU risk on the line),
and `"none"` (keep the last epoch) marks none.

- **Interrupted:** run it again; it resumes from `resume.pt` and reproduces the
  uninterrupted run exactly on the same device.
- **Complete:** it summarises `metrics.json` and changes nothing.
- **Settings changed:** if a built-in setting that changes results differs from the run's,
  it fails naming them; move `results/baseline/seed-42/` aside to train anew.
- **Memorised positives:** watch `history.csv`
  ([Training](../explanation/training.md#the-loss)).

The run's files are in [Outputs](../reference/outputs.md#a-run-resultsvariantseed-n).
Proxy metrics count unlabelled accounts as negatives: judge the model by its audit.

## Audit

```bash
mule evaluate
```

It audits validation and test against the ground truth into `audit/` and redraws the audit
figures and `report.md` ([mule evaluate](../reference/cli.md#mule-evaluate-run), [the
ground-truth audit](../explanation/training.md#the-ground-truth-audit));
`mule evaluate results/<variant>/seed-<n>` audits another run. To read it (`report.md`,
`audit/validation.json`, `audit/test.json`):

- **Judge by the hidden mules** (`hidden_metrics`). `metrics` (every mule) follows;
  training saw mules like the revealed ones, so it mixes finding new mules with ranking
  known ones again.
- **Decide on validation** (`purpose` `decisions`); test is for `reporting`. Comparing
  settings or picking a threshold on test makes it optimistic.
- **Read the ranking:** AP against `weighted_prevalence`, ROC AUC, recall and precision at
  the top 1, 5 and 10%. The F1 threshold was chosen on about a dozen validation mules and
  means little.
- **Compare revealed and hidden mules** (`revealed_positives`, `hidden_positives`,
  `audit_revealed_hidden.png`): a model that finds only mules like those it was shown is
  not yet a detector.

An audited split is never rewritten; to audit it again, move its `audit/<split>.*` aside.

## Redraw the figures

`mule report` redraws the figures and `report.md` offline
(`mule report results/<variant>/seed-<n>` for another run); a failing figure is named and
the rest are drawn ([mule report](../reference/cli.md#mule-report-run)).

## On the CUDA host

A host that trained with earlier code, on a graph that code built, upgrades in this order.
This code reads no dataset or model earlier code wrote ([Datasets and models of earlier
code](#datasets-and-models-of-earlier-code)), and the graph keeps its data: never run
`gsql/schema/schema.gsql` or `gsql/schema/loading.gsql` on it, since it was built with the
earlier schema, whose attributes they do not match.

1. **Stop** every job of the earlier code on every machine: it calls the query names
   `mule install` drops.
2. **Code:** `git switch main && git pull`, or clone. On Linux x86_64 with Python 3.12 to
   3.14 and an NVIDIA driver for CUDA 12 (525.60 or newer) or CUDA 13 (580.65 or newer),
   reinstall torch and the extras as [Setup](../../README.md#setup) says: the cu129 wheel
   and `cuda12` (from pypi.nvidia.com), or the cu130 wheel and `cuda13` (from pypi.org).
   Copy `.env`, nothing else: settings are built in and known mules are in TigerGraph.
3. **Outputs:** move `results/` and `data/` aside, keeping `results/archive/` (the
   archived diagnostic study, never read).
4. **`mule check`** reports the outdated scope types, the stale and retired queries and no
   dataset, and ends "Not ready".
5. **Delete the split** by hand ([After the data
   changes](set-up-a-graph.md#after-the-data-changes) has the request); its membership
   edges go with it.
6. **`mule install`** replaces the scope types, drops the retired queries
   ([Queries](../reference/queries.md#the-retired-names)) and installs every training
   query: about 50 minutes, within a 90-minute wait. If the wait runs out, rerun it once
   `mule check` lists no stale training query.
7. **`mule check`** lists no stale or retired query and ends "Not ready" only for the
   missing dataset.
8. **`mule train`** creates the split again, identical on the same loaded graph (same
   seed, shares and rule, and TigerGraph's internal ids), reveals the labels again for it
   ("again for the new split"), prepares the dataset (about 6 minutes) and trains (about an
   hour). The labels are unchanged when the `reveal` event of
   `data/<dataset id>/events.jsonl` gives the mules revealed per split of the earlier run's
   (`{"1": 39, "2": 21, "3": 38}` on the 300,000-person graph):
   `grep '"event": "reveal"' data/*/events.jsonl`.
9. **cuGraph on the GPU:**
   `python -m pytest -m cuda tests/integration/test_cugraph_sampler.py` (skips where
   cuGraph cannot run; the last test needs the prepared dataset).
10. **`mule evaluate`, then `mule report`.**
11. **Control experiments:** `python scripts/run_experiments.py`, then
    `python scripts/run_experiments.py methods` ([Run the control
    experiments](run-control-experiments.md)).
12. **`mule diagnose`:** the diagnostic study of the built-in run's dataset ([Run the
    diagnostics](run-diagnostics.md)).
13. **Cache cap:** set `contract.bounds.CONTEXT_CACHE_ENTRIES` to about five times the
    baseline's `contexts.distinct` (in its `metrics.json`). The cache holds the baseline's
    contexts once, once more per variant dropping a server-side group (`drop_entity_meta`,
    `drop_time_encoding`, `drop_pair_history`, `drop_flow_timing`), since runs share
    entries only for the same server-side groups, plus the audit samples. At about 7 kB an
    entry, check the disk, and watch the first run that fills it: each eviction scans the
    directory while requests wait
    ([Outputs](../reference/outputs.md#a-prepared-dataset-datadataset-id)).

## When TigerGraph rejects roots

The built-in run allows none (`runtime.max_rejected_root_fraction = 0.0`), so a rejected
root (such as `history_capacity_exceeded`) stops the run with its statuses ([Rejected
roots](../explanation/training.md#rejected-roots)). The limit never changes the numbers,
so resume with a higher one from Python:

```python
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.pipeline.train import train_run

config = DEFAULT_CONFIG.with_changes({"runtime": {"max_rejected_root_fraction": 0.01}})
train_run(config=config, resume=True)
```

A test-split failure leaves `model.pt` without `metrics.json` (test is scored after the
save); the same resume finishes it.

## Train another configuration

`mule train` trains only the built-in run. Another configuration trains from Python into
its own directory, on the dataset of its dataset settings (shared when they match the
built-in run's). For comparisons over seeds with paired intervals, declare a variant
instead ([Run the control experiments](run-control-experiments.md)).

```python
from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import RunPaths
from mule_pattern_learner.pipeline.evaluate import evaluate_run
from mule_pattern_learner.pipeline.train import train_run

run = RunPaths.of("short_run", 42)
train_run(run, config=DEFAULT_CONFIG.with_changes({"training": {"epochs": 3}}), resume=True)
evaluate_run(run)
```

## Datasets and models of earlier code

This code reads only what it writes:

- **A dataset** is found by its settings' id in `data/<dataset id>/`. One kept elsewhere,
  recording no dataset settings, or of code that named a scope id, is ignored, and
  `mule train` prepares a new one (about 6 minutes). One from other query texts ("was
  built from different GSQL sources"), or from earlier code's 70, 15 and 15% split
  ("records no scope shares"), is refused: run `mule install` (for the first), move the
  directory aside (with its `contexts/`; nothing deletes it) and run `mule train`.
- **A model** must record `SavedModel.FORMAT` and this code's contract fingerprint; an
  earlier `model.pt` is refused, so train again.

## When something is refused

| Message | Fix |
|---|---|
| `Installed query differs from repository source or is not installed` | `mule install`; it recompiles only the stale queries |
| `Queries [...] are still not installed after ...s` | The 90-minute wait ran out; the server may still be compiling. Once it has finished (`mule check` lists no stale training queries), rerun the command (`mule install`, `mule train`, `mule diagnose` or the experiments script); it installs only what is stale |
| `The graph's scope types differ from gsql/schema/scope_vertex.gsql: ...` | The scope types predate the file (for example a `Temporal_Training_Scope` keyed by a scope id, or a membership edge with `group_id`). `mule install` replaces them if the graph holds no scope vertex; otherwise delete those by hand first, as the message says ([After the data changes](set-up-a-graph.md#after-the-data-changes)); the next `mule train` creates the split again, with the same partitions on the same loaded graph |
| `Prepared dataset ... was built from different GSQL sources` | Query files changed after preparation: install, then move the dataset aside |
| `Graph counts changed; freeze the source and prepare a new dataset` | The graph changed after preparation: freeze it, prepare a new dataset |
| `The graph's split has the split seed ...` | The split has other settings: configure the split's, or delete the split by hand (and move aside the dataset the message names, if it names one) and let `mule train` create the configured one ([After the data changes](set-up-a-graph.md#after-the-data-changes)) |
| `TigerGraph rejected ... training roots so far` or `validation: TigerGraph rejected ... roots` | Rejected roots beyond the limit, or a rejected observed positive; see the statuses ([When TigerGraph rejects roots](#when-tigergraph-rejects-roots)) |
| `The run in ... is complete with other settings` | Move the finished run aside |
| `Resumed configuration differs from the run` | Move the interrupted run aside, or resume it with its own settings |
| `... records no format` or `... is a model of format ...` | Saved by other code: train again |
| `The model's input groups or pool definitions differ from its configuration` | A pool group's definition changed: score with a model trained under the current one |
| `Training queries require Mule_Pattern_Learner` | `GRAPHNAME` in `.env` names another graph |
| A `cugraph_probe` warning | pylibcugraph or the GPU failed the probe; training uses the torch sampler, which draws from the same distribution. Run `mule check` and the `cuda` tests |
| `TigerGraph is not answering yet (...): attempt 2, retrying in 6 s` | Briefly unavailable or resuming (TigerGraph Cloud: `starting workspace`); each operation waits up to `transport.max_outage_s` |
| `mule train stopped: TigerGraph stayed unavailable.` on stderr | Retries ran out; the line names the operation, the attempts and why they ended, and TigerGraph's error, kept as `command_stopped` in the run's or dataset's `events.jsonl` (or `results/events.jsonl`). Rerun once TigerGraph answers: the run resumes |
