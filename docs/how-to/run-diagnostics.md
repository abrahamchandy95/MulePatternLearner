# Run the diagnostics

Study the built-in run's dataset against the ground truth, for analysis only: nothing here
feeds a model. `mule diagnose` runs the analyses worth repeating whenever the dataset,
features or model change; one-off findings are in [the diagnostic
study](../research/diagnostic-study.md).

## Run them

Train and audit the built-in run first, since several analyses compare it:

```bash
mule train
mule evaluate
mule diagnose                   # every analysis, in order
mule diagnose baselines         # one analysis
```

The study goes to `results/diagnostics/<dataset id>/`, beside the run ([mule
diagnose](../reference/cli.md#mule-diagnose-analysis)). It prepares the dataset as
`mule train` does (a ready one needs no connection), then reads the graph on one
connection after checking it is still the dataset's frozen source.

**The first run** installs the three analytics queries (`gsql/analytics/`), which only
this command installs. `fetch_analytics_context` is the old all-groups context query
renamed, so the install is of the order of the 50-minute full install, within a 90-minute
wait; if that runs out, rerun once the GSQL shell's `ls` shows them installed, and it
installs only what is still stale. It then reads the feature table, about 6,200 accounts
through both context queries; the analytics query's rows skip the context cache, so a
rebuilt table requests them all again. Then run `python -m pytest -m graph` before reading
the `account` family's results: it checks the analytics query against its Python mirror
(`tests/integration/test_feature_parity.py`), which offline tests cannot, since there the
fake graph serves the mirror itself.

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

- **The feature table** is built first when missing or stale. It samples each split as its
  audit does (every mule and 2,000 uniform non-mules, with the split seed), so the
  baselines and audits rank the same accounts, each read at its split's cutoff and
  visibility. A rejected account keeps its row, marked `rejected`, without features. A
  table read with this code's two contracts and columns is kept, as the frozen graph would
  give the same; delete `features.parquet` to read it anew.
- **"The run"** is the built-in run, if its `config.json` names this dataset. An analysis
  missing inputs (no such run, another dataset, no audit, no `metrics.json`) is skipped
  with its reason, the study is `incomplete` and the command exits 1; the baselines and
  the curve then go without the run's rows.
- **The baselines** answer the retired `no_graph` control: PU logistic regression and
  gradient boosting, fitted at the train cutoff on the revealed train mules against the
  population-weighted rest, on the families `account` (the analytics features, which no
  model reads), `model`, `messages` and all of them. Beside them: the attribute-only
  floor, the five single features farthest from 0.5 on train, `chance` (a random ranking's
  expectation) and the run's audits, all with ring-clustered intervals on the validation
  and test audit samples.
- **The learning curve** fits on oracle-labelled train mules, hidden ones included, to see
  whether more labels would help: a question about the data, never a way to train.

## Read the study

`report.md` holds each analysis' outcome, the head of its table and its figures;
`study.json` keeps each analysis' last outcome, even for analyses a later call does not
run. [Outputs](../reference/outputs.md#a-diagnostic-study-resultsdiagnosticsdataset-id)
lists every file, table and figure; `mule report results/diagnostics/<dataset id>` redraws
them offline. A figure whose table is gone keeps its old PNG in `plots/`, unlinked from
`report.md`.

As in the audits, rankings measure the hidden mules first, then every mule. The pool
groups were designed after reading test-split mules, so their test numbers are optimistic;
decisions use validation's hidden mules.
