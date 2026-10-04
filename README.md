# Mule Pattern Learner

Ranks money-mule accounts by training a graph neural network on time-stamped payments,
directly from TigerGraph. Each account is scored at a calendar cutoff from its own and its
counterparties' payment history, exactly as it was before that cutoff. TigerGraph filters
events by time and experiment partition, computes the features and returns over REST a
bounded pool of candidate neighbours per account. The client resamples a fixed fan-out
(with cuGraph on a CUDA GPU) and trains a TGAT-style attention model over time-stamped
messages with a non-negative positive-unlabelled (nnPU) loss on the few mules the graph
reveals. No account id is a model parameter, so the same weights score accounts never seen
in training.

The repository holds the GSQL (`gsql/`: schema, training, evaluation and analytics
queries) and the Python package `mule_pattern_learner` with its command `mule`, which
installs the queries, prepares a bounded dataset, trains, audits against the ground truth,
scores and reports. It needs a TigerGraph graph `Mule_Pattern_Learner` with the data
loaded ([Set up a graph](docs/how-to/set-up-a-graph.md)).

## Setup

Python 3.12 or newer in a `.venv` (the development commands use it), installed editable,
since the commands read `gsql/` from the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On a CUDA host, install the CUDA torch wheel first, then the cuGraph extra of its CUDA
major version:

```bash
# CUDA 12
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev,cuda12]" --extra-index-url=https://pypi.nvidia.com
# CUDA 13
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,cuda13]"
```

Copy `.env.example` to `.env` and set `HOST`, `GRAPHNAME` and `SECRET`; environment
variables override it. Nothing else is configured: settings are built in (`DEFAULT_CONFIG`
in `src/mule_pattern_learner/config.py`), and no command takes an option but `--help`.

## Train, audit and report

```bash
mule check                  # read-only readiness: the graph, its queries, cuGraph, one batch
mule train                  # prepare as needed, then train the built-in run
mule evaluate               # ground-truth audits of validation and test
mule report                 # redraw the figures and report.md, offline
mule score ACCOUNTS [DATE]  # score the accounts listed in a file
mule install                # install the queries ahead of time
```

`mule train` writes `results/baseline/seed-42/` (model, proxy predictions and metrics,
history, `plots/`, `report.md`) and resumes when run again; a complete run is only
summarised. On a fresh graph its first run installs the queries, creates the frozen
experiment scope and reveals the mules a bank would have discovered. Each run prepares its
dataset in `data/<dataset id>/`, whose context cache spares later runs and audits the same
requests.

`mule evaluate` audits validation (for decisions) and test (for reporting) on every mule
and 2,000 sampled non-mules, weighted to the whole split, with 90% intervals. It leads with
the hidden mules (unknown at the cutoff), which the model exists to find: the revealed
mules leave the ranking, as an investigator removes known cases, and decisions use the
hidden mules' validation AP.

`python -m mule_pattern_learner` runs the same commands. The console shows progress and a
summary; full records go to `events.jsonl`, `history.csv`, `epochs.csv`, `metrics.json` and
`audit/`. See [Train and evaluate](docs/how-to/train-and-evaluate.md) and
[Command line](docs/reference/cli.md).

## Control experiments

```bash
python scripts/run_experiments.py                            # the controls suite
python scripts/run_experiments.py methods                    # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                     # suites, variants, questions
```

It trains variants of the built-in run (`src/mule_pattern_learner/experiments/variants.py`)
over seeds 42 to 51, audits them and compares each with the baseline on the same accounts,
with paired intervals, in `results/experiments/<suite>/`. Complete runs are kept, so fewer
seeds are only topped up; runs whose settings differ move to `results/archive/`, never
deleted ([Run the control experiments](docs/how-to/run-control-experiments.md)).

## Diagnostics

```bash
mule diagnose                # every analysis
mule diagnose baselines      # one of them
```

It studies the built-in run's dataset against the ground truth, for analysis only (single
features, drift between cutoffs, own-activity baselines, a learning curve, which mules the
run finds, proxy metric validity, the label reveal over salts, the nnPU positive weight on
a synthetic problem), into `results/diagnostics/<dataset id>/`. It is the only command that
installs the analytics queries ([Run the diagnostics](docs/how-to/run-diagnostics.md)).

## Starting again on the CUDA host

This code reads no dataset or model earlier code wrote: start from scratch with `data/`
and `results/` empty or moved aside, except `results/archive/` (the archived diagnostic
study, never read). Then follow [On the CUDA host](docs/how-to/train-and-evaluate.md#on-the-cuda-host):
reinstall with the CUDA extra ([Setup](#setup)), `mule train` (its first run installs the
renamed queries beside the old names, about 50 minutes within a 90-minute wait), stop
every job of the earlier code on every machine, `mule install` (it drops the old names),
then `mule evaluate`, `mule report`, the control experiments and `mule diagnose`.

The earlier code is commit 08b487e, kept in `main`'s history. It calls the old query
names, so it runs only until `mule install` drops them; the owner retrains from scratch,
so nothing is compared with it, and a comparison on the graph would have to run before
that drop. Its outputs lie in folders this code never writes (`models/`, `artifacts/`,
`outputs/`, `runs/`, `logs/`, `docs/experiments/`, `*.sqlite` files), which `.gitignore`
still hides so none is committed: move what you keep out of the repository or under
`results/archive/`, then drop their lines from `.gitignore`.

## Documentation

- [Architecture](docs/architecture.md): layers, ports, import contracts, the data flow
  from TigerGraph to `results/`, and the decisions behind them.
- How-to: [Set up a graph](docs/how-to/set-up-a-graph.md),
  [Train and evaluate](docs/how-to/train-and-evaluate.md),
  [Run the control experiments](docs/how-to/run-control-experiments.md),
  [Score new accounts](docs/how-to/score-new-accounts.md),
  [Run the diagnostics](docs/how-to/run-diagnostics.md).
- Reference: [Command line](docs/reference/cli.md),
  [Configuration](docs/reference/configuration.md), [Outputs](docs/reference/outputs.md),
  [Features](docs/reference/features.md), [Schema](docs/reference/schema.md),
  [Labels](docs/reference/labels.md), [Queries](docs/reference/queries.md).
- Explanation: [Training](docs/explanation/training.md),
  [Sampling](docs/explanation/sampling.md),
  [Feature design](docs/explanation/feature-design.md),
  [Time encoding](docs/explanation/time-encoding.md),
  [Leakage and scaling](docs/explanation/leakage-and-scaling.md),
  [Label reveal](docs/explanation/label-reveal.md).
- Research: [the reference runs](docs/research/reference-run.md),
  [the diagnostic study](docs/research/diagnostic-study.md),
  [the mule profile](docs/research/mule-profile.md),
  [the nnPU positive weight](docs/research/nnpu-positive-weight.md),
  [the control experiments' first three seeds](docs/research/control-experiments.md).
- [The GSQL folder](gsql/README.md): which query lives where, and who installs it.

## Development

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/basedpyright src
.venv/bin/basedpyright tests scripts
.venv/bin/python -m pytest -q
.venv/bin/python scripts/render_queries.py --check
.venv/bin/lint-imports
```

The tests never connect to TigerGraph unless a marker selects them: `-m graph` for the
read-only checks against the graph in `.env`, `-m cuda` for the cuGraph checks on a CUDA
host, `-m graph_write --allow-graph-writes` for the scope isolation test, which writes
fixture vertices and removes them
([Command line](docs/reference/cli.md#tests-that-need-the-graph-or-a-gpu)).
`scripts/render_queries.py` regenerates the two context queries after their Python
contracts change.

## License

MIT (author: Abraham Chandy).
