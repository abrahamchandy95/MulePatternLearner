# Mule Pattern Learner

Learns to rank money-mule accounts from payment data by training a graph neural network
over time-stamped payments, directly from TigerGraph.

Every account is scored at a calendar cutoff from its own payment history and the history
of its counterparties, exactly as they looked before that cutoff. TigerGraph does the
heavy work: it filters events by time and by experiment partition, computes the features
and returns a bounded pool of candidate neighbours for each account over REST. The client
resamples a fixed fan-out from those pools (with cuGraph on a CUDA GPU) and trains a
TGAT-style attention model over time-stamped messages with a non-negative
positive-unlabelled (nnPU) loss on the few mules the graph reveals. No account id is a
model parameter, so the same weights score accounts that never appeared in training.

The repository holds the GSQL (`gsql/`: the schema, the training and evaluation queries,
and the analytics queries) and the Python package `mule_pattern_learner` with its command
`mule`, which installs the queries, prepares a bounded dataset, trains, audits against the
ground truth, scores and reports. You need a TigerGraph graph `Mule_Pattern_Learner` with
the data loaded ([Set up a graph](docs/how-to/set-up-a-graph.md)).

## Setup

Python 3.12 or newer, in a virtual environment `.venv` (the development commands below
use it), installed in editable mode, since the commands read `gsql/` from the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On a CUDA host, install the CUDA torch wheel first, then the cuGraph extra that matches
its CUDA major version. For CUDA 12:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev,cuda12]" --extra-index-url=https://pypi.nvidia.com
```

For CUDA 13:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,cuda13]"
```

Copy `.env.example` to `.env` and fill in the connection (`HOST`, `GRAPHNAME`, `SECRET`);
environment variables override it. Nothing else is configured: the settings are built in
(`DEFAULT_CONFIG` in `src/mule_pattern_learner/config.py`), and no command takes an option
besides `--help`.

## Train, audit and report

```bash
mule check       # read-only readiness: the graph, its queries, cuGraph, one batch
mule train       # prepare as needed, then train the built-in run
mule evaluate    # ground-truth audits of validation and test
mule report      # redraw the figures and report.md, offline
```

`mule train` writes the run to `results/baseline/seed-42/`: the model, its proxy
predictions and metrics, its history, the training figures in `plots/` and `report.md`.
On a fresh graph its first run installs the queries, creates the frozen experiment scope
and reveals the mules a bank would have discovered; every run then prepares its dataset
in `data/<dataset id>/`, whose context cache spares later runs and audits the same
requests. Run it again to resume an interrupted run; on a complete run it prints the
run's `metrics.json` and changes nothing.

`mule evaluate` audits the model against the ground truth on validation (for decisions)
and test (for reporting), weighting a sample of every mule and 2,000 non-mules to the
whole split, with 90% intervals. `mule score ACCOUNTS [DATE]` scores the accounts listed
in a file, and `mule install` installs the queries ahead of time. `python -m
mule_pattern_learner` runs the same commands; each prints one JSON result.
[Train and evaluate](docs/how-to/train-and-evaluate.md) walks through it, the CUDA host
included, and [Command line](docs/reference/cli.md) lists every command.

## Control experiments

```bash
python scripts/run_experiments.py                            # the controls suite
python scripts/run_experiments.py feature_drops              # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                     # suites, variants, questions
```

The script trains variants of the built-in run (declared in
`src/mule_pattern_learner/experiments/variants.py`) over the seeds 42, 43 and 44, audits
them, and compares each with the baseline on the same accounts, with paired intervals, in
`results/experiments/<suite>/`. Complete runs are kept, and runs whose settings differ are
moved to `results/archive/`, never deleted. [Run the control
experiments](docs/how-to/run-control-experiments.md) has the details.

## Diagnostics

```bash
mule diagnose                # every analysis
mule diagnose baselines      # one of them
```

`mule diagnose` studies the built-in run's dataset against the ground truth, for analysis
only: the features one at a time, their drift between cutoffs, baselines on the account's
own activity, a learning curve, which mules the run finds, how valid its proxy metrics
are, the label reveal over salts and the nnPU positive weight on a synthetic problem. It
writes to `results/diagnostics/<dataset id>/` and is the only command that installs the
analytics queries. [Run the diagnostics](docs/how-to/run-diagnostics.md) describes each
analysis.

## Starting again on the CUDA host

This code reads no dataset or model that earlier code wrote, so a host that trained with
it starts from scratch, with its `data/` and `results/` empty or moved aside:

1. Pull `main` (`git switch main && git pull`) and reinstall with the CUDA extra
   ([Setup](#setup)).
2. `mule check`. Until the first `mule train` it ends `not_ready`: the renamed queries are
   not installed yet, the old names are listed under `queries.retired`, and there is no
   dataset.
3. `mule train`. Its first run installs the renamed queries beside the old names (about
   50 minutes, within a 90-minute wait), prepares the dataset (about 6 minutes) and
   trains the built-in run (about an hour). If the wait runs out, wait until `mule check`
   no longer lists the queries under `queries.stale`, then run `mule train` again.
4. Stop every job of the earlier code, on every machine, since it calls the old names.
   Then `mule install`: it finds the renamed queries in place and drops the old names,
   and `mule check` lists no `queries.retired` from then on.
5. `mule evaluate`, then `mule report`.
6. `python scripts/run_experiments.py`, then `python scripts/run_experiments.py
   feature_drops`.
7. `mule diagnose`. Its first run installs the analytics queries, an install of the
   order of 50 minutes within the same 90-minute wait; if that runs out, run `mule
   diagnose` again once the compilation has finished.

[Train and evaluate](docs/how-to/train-and-evaluate.md#on-the-cuda-host) walks through
each step.

## Documentation

- [Architecture](docs/architecture.md): the layers, ports and import contracts, the data
  flow from TigerGraph to `results/`, and the decisions behind them.
- How-to guides: [Set up a graph](docs/how-to/set-up-a-graph.md),
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
  [the mule profile](docs/research/mule-profile.md) and
  [the nnPU positive weight](docs/research/nnpu-positive-weight.md).
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

The tests never connect to TigerGraph unless a marker selects them: `-m graph` runs the
read-only checks against the graph in `.env`, `-m cuda` the cuGraph checks on a CUDA host,
and `-m graph_write --allow-graph-writes` the scope isolation test, which writes fixture
vertices and removes them again ([Command line](docs/reference/cli.md#tests-that-need-the-graph-or-a-gpu)).
`scripts/render_queries.py` regenerates the two context queries after their Python
contracts change.

## License

MIT (author: Abraham Chandy).
