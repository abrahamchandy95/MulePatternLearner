# Run the diagnostics

Study the built-in run's dataset against the ground truth, for analysis only: nothing the
diagnostics compute feeds a model. `mule diagnose` runs the analyses that should run again
whenever the dataset, the features or the model change; one-off findings are recorded in
[the diagnostic study](../research/diagnostic-study.md).

## Run them

Train and audit the built-in run first, since several analyses compare it:

```bash
mule train
mule evaluate
mule diagnose                   # every analysis, in order
mule diagnose baselines         # one analysis
```

The study goes to `results/diagnostics/<dataset id>/`, beside the run. `mule diagnose`
prepares the dataset as `mule train` does (a ready one needs no connection), then reads
the graph on one connection, after checking that it is still the dataset's frozen source.
It is the only command that installs the analytics queries (`gsql/analytics/`), where
their text differs.

**The first run** installs the three analytics queries. `fetch_analytics_context` is the
old all-groups context query under a new name, so expect a long install, of the order of
the 50 minutes a full install takes; the command waits up to 90 minutes for it. If that
wait runs out, the server may still be compiling: once the GSQL shell's `ls` shows the
analytics queries installed, run `mule diagnose` again, and it installs only what is still
stale. It then reads the feature table, about 6,200
accounts through both context queries. The training query's rows go through the
dataset's context cache; the analytics query's rows are not cached, so a rebuilt table
requests them all again. After that first run, `python -m pytest -m graph` also checks
the analytics query against its Python mirror (`tests/integration/test_feature_parity.py`).
Run it before reading the `account` family's results: offline, the fake graph serves the
mirror itself, so only this check compares the mirror with the GSQL.

## The analyses

| Analysis | What it asks | Needs |
|---|---|---|
| `features` | The feature table: each split's audit sample at its cutoff, with the training query's model inputs and candidate pool and the analytics query's account history | the graph |
| `univariate` | How well each feature alone ranks the mules of each split (weighted ROC AUC) | the feature table |
| `drift` | How each feature of the non-mules shifts between the splits' cutoffs, and what that costs a learner | the feature table |
| `baselines` | How well a table of the account's own activity ranks mules, with no neighbour, association or pool input, beside the run's audits | the feature table; the run's audits when it has them |
| `learning-curve` | How ranking quality grows with the number of labelled training mules | the feature table; the run's audits when it has them |
| `subgroups` | Which mules the run's audits find: revealed or hidden, how few make its AP, which rings | the run's audits |
| `proxy-validity` | How well the run's ranking, whose epoch the proxy chose, finds the ground truth: the hidden and revealed mules from its audit samples, which hold every hidden mule, and all of its proxy predictions | the run's audits and the graph's truth |
| `reveal-spread` | How the one-time label reveal's outcome varies with its salt, replayed offline over salts 0 to 999 | the reveal's inputs, read once |
| `nnpu-simulation` | Whether the nnPU positive weight alone explains the collapse of textbook nnPU, on a synthetic problem | nothing (offline, about 15 seconds) |

The analyses of the feature table build it first when it is missing or stale. "The run" is
the built-in run, and only if its `config.json` names this dataset. An analysis whose
inputs are missing (no such run, another dataset, no audit, no `metrics.json`) is skipped
with its reason, the study's status is `incomplete`, and the command exits 1; the
baselines and the curve then go without the run's rows. The console shows a line per
analysis as it ends (written, kept or skipped, its rows and seconds, or the reason it was
skipped) and then whether the study is complete and where it is; the analyses' records
are in the study's `events.jsonl`.

What some of them mean:

- **The feature table** samples each split as its audit does (every mule and 2,000
  uniform non-mules, with the split seed), so the baselines and the run's audits rank the
  same accounts. Each account is read at its split's cutoff and visibility. An account a
  query rejects keeps its row, marked `rejected`, without features. A table read with this
  code's two contracts and columns is kept, since the frozen graph would give the same one
  again; delete `features.parquet` to read it anew.
- **The baselines** answer the question of the retired `no_graph` control: PU logistic
  regression and gradient boosting, fitted at the train cutoff on the revealed train mules
  against the population-weighted rest, on the feature families `account` (the analytics
  features, which no model reads), `model`, `messages` and all of them; beside them the
  attribute-only floor, the five single features farthest from 0.5 on train, `chance` (a
  random ranking's expectation) and the run's audits, all with ring-clustered intervals on
  the validation and test audit samples.
- **The learning curve** fits on oracle-labelled train mules, hidden ones included, to see
  whether more labels would help: a question about the data, never a way to train.

## Read the study

`report.md` holds each analysis' outcome, the head of its table and its figures;
`study.json` records the dataset, the run compared, the reveal's salt and budget, and each
analysis' last outcome, kept for the analyses a later call does not run. [Outputs](../reference/outputs.md#a-diagnostic-study-resultsdiagnosticsdataset-id)
lists every table and figure, and `mule report results/diagnostics/<dataset id>` redraws
them offline.

Like the audits, every ranking of mules is measured on the hidden mules first, the
split's revealed mules removed from the ranking (metrics named `hidden_` and the metric),
then on every mule: the tables give both, the hidden mules' first, and the figures draw
the hidden mules', but for the ring coverage. Keep the study's warnings in mind: the
pool groups were designed after reading test-split mules, so their test numbers are
optimistic, and decisions use validation's hidden mules. A
figure whose table is gone keeps its old PNG in `plots/`, though `report.md` no longer
links it.
