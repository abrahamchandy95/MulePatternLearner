# Mule Pattern Learner

Learns mule and money-laundering patterns from payment data by training a temporal graph
neural network directly from TigerGraph.

Every account is scored at a calendar cutoff from its own payment history and the history
of its counterparties, exactly as it looked before that cutoff. TigerGraph does the heavy
work: it filters events by time and by experiment partition, computes the features and
returns a bounded pool of candidate neighbours for each account over REST. The client
resamples a fixed fanout from those pools (with cuGraph on the GPU when CUDA is available)
and trains a TGAT-style temporal attention model with a non-negative positive-unlabeled
(nnPU) loss. No account ID is a model parameter, so the same weights score accounts that
never appeared in training.

The project contains:

1. **A GSQL layer** (`gsql/`): the temporal payment schema and Account label contract,
   the cutoff-aware training queries, the experiment scope, the hub registry, the Fourier
   time encoding and the one-time label reveal.
2. **A Python layer** (`src/mule_pattern_learner/`, command `mule`)
   that installs those queries, prepares a bounded dataset, streams batches from
   TigerGraph, trains, scores and evaluates.

You need a working TigerGraph instance whose graph `Mule_Pattern_Learner` follows the
[temporal payment schema](docs/temporal_schema.md) and holds the loaded data.

## Setup

Python 3.12 or newer.

```bash
pip install -e .                 # training
pip install -e ".[dev]"          # training and dev tools
pip install -e ".[dev,cuda12]" --extra-index-url=https://pypi.nvidia.com  # CUDA 12 hosts
pip install -e ".[dev,cuda13]"   # CUDA 13 hosts (install the matching CUDA torch wheel first)
```

Install in editable mode: the commands read `gsql/` from the repository.

Copy `.env.example` to `.env` and fill in the TigerGraph connection (read by
`Settings`; environment variables override it):

```
HOST=https://your-tg-host
GRAPHNAME=Mule_Pattern_Learner
SECRET=your_restpp_secret
```

The graph is created from `gsql/schema/schema.gsql`. This repository ships
only the Account loading contract (`gsql/schema/account_loading.gsql`); the
payment events, associations, tokens and other entities are loaded by the external
data producer (for the reference snapshot, an export of the PhantomLedger
simulator).

## Commands

With the data loaded in TigerGraph, one command trains the model. It needs nothing but
`.env`: no configuration file, no dataset identifier, no label file and no prepared
artifacts to copy.

```bash
mule train
```

It uses CUDA when available (then Apple MPS, then CPU) and writes the run to
`results/baseline/seed-42/`: `model.pt`, `metrics.json`, `history.csv`, `epochs.csv`,
`events.jsonl` and the other files of the run directory, then the training figures in
`plots/` and `report.md`, the run's tables with links to its figures. On a fresh graph the first run
installs the training queries, creates the frozen scope and reveals the known mules in
the graph ([label reveal](docs/label_reveal.md)); every run then prepares its dataset in
`data/<dataset id>/`, where `contexts/` keeps the contexts runs of the dataset read from
the graph, so later runs and audits request them no more. Run the same command again to resume an interrupted run; on a
complete run it prints the run's `metrics.json` and changes nothing. The
settings are built in:
`DEFAULT_CONFIG` in `src/mule_pattern_learner/config.py`, frozen dataclasses with one
section per concern. No command reads a configuration file, and no command takes an
option besides `--help`.

| Command | What it does |
|---|---|
| `mule train` | Prepares as needed (install, scope, reveal, dataset), then trains the built-in run into `results/baseline/seed-42/` with its training figures in `plots/` and `report.md`, resumes it, or reports it when it is complete |
| `mule evaluate [RUN]` | Ground-truth audits of the run's model on the frozen validation and test partitions, written to the run's `audit/` with the audit figures in `plots/`: decisions use the validation audit, and the test audit is for reporting; `RUN` defaults to `results/baseline/seed-42` |
| `mule score ACCOUNTS [DATE]` | Scores the accounts listed in a file (one id per line) with the built-in run's model; `DATE` defaults to the test cutoff. Writes `scores/<file stem>_<date>.parquet` in the run, and the ids TigerGraph rejects to `scores/<file stem>_<date>_rejected.txt` |
| `mule report [RUN]` | Redraws the run's figures (`plots/<topic>_<figure>.png`) and `report.md` from the files it saved, offline: the training figures of a complete run and the audit figures of its audits. Given a suite's directory (`results/experiments/<suite>`), it redraws the suite's comparison figures and `report.md` instead |
| `mule check` | Read-only readiness: the graph, its installed queries and the cuGraph probe, then one training batch with its tensor digests and the first loss. The batch needs the built-in run's prepared dataset in `data/` (see below) |
| `mule install` | Adds the scope vertex type if it is missing, installs the queries whose text differs, drops the queries retired by the rename that are still installed (`train` does this too) and lists installed queries that no file defines |

`mule check` reads its batch from the built-in run's prepared dataset, which `mule train`
prepares before it trains. To prepare the dataset without training, for example to check
a graph before a long training run, call the same step from Python:

```bash
python -c "from mule_pattern_learner.config import DEFAULT_CONFIG; from mule_pattern_learner.pipeline.prepare import prepare_dataset; print(prepare_dataset(DEFAULT_CONFIG).root)"
```

Like `mule train`, it installs the queries, creates the scope and reveals the known mules
on a fresh graph; on a graph where they are in place it only reads.

`python -m mule_pattern_learner` runs the same commands. Each prints one JSON result.
The end-to-end guide describes
[what runs where](docs/temporal_training_end_to_end.md#what-runs-where).

## Control experiments

The control experiments train variants of the built-in run over the seeds 42, 43 and 44,
audit them and compare them with the baseline:

```bash
python scripts/run_experiments.py                           # the controls suite
python scripts/run_experiments.py feature_drops             # a suite by name
python scripts/run_experiments.py no_attention prior_weight  # chosen variants
python scripts/run_experiments.py --help                    # suites, variants, questions, changes
```

The variants are declared in `src/mule_pattern_learner/experiments/variants.py`: the
controls (`no_attention`, `no_slot_sum`, `no_pool_counts`, `prior_weight`,
`no_weight_average` and `drop_time_encoding`) and one `drop_<group>` variant per feature
group of the built-in run. Every run of a suite trains on the built-in run's dataset, the
baseline's seed 42 is the run `mule train` makes, and each run goes to
`results/<variant>/seed-<n>/` with its own figures and audits. The script validates every
variant offline and prints the run matrix with a time bound before it trains; it keeps
complete runs, moves a run whose settings differ to `results/archive/` (nothing is deleted)
and trains it again, and stops on a TigerGraph outage while carrying on past a run's own
error. It always rewrites `summary.csv`, `comparison.csv`, the comparison figures and
`report.md` under `results/experiments/<suite>/`. The report ranks the variants by the
validation audit with paired intervals against the baseline; the test audit is for
reporting, not selection.

## Documentation

- [End-to-end guide](docs/temporal_training_end_to_end.md): the data in TigerGraph, every
  query and what it pulls, per-batch sampling with cuGraph, the training loop and the
  CUDA runbook. Start here.
- [Live temporal training](docs/live_temporal_training.md): behaviour details and
  configuration semantics.
- [GSQL feature catalog](docs/gsql_feature_catalog.md): the exact model inputs.
- [Feature redesign](docs/feature_redesign.md): the window-free feature groups and the
  sampler options.
- [Leakage and scaling](docs/leakage_and_scaling.md): read before choosing an
  evaluation protocol.
- [Label reveal](docs/label_reveal.md) and [Account mule labels](docs/account_mule_labels.md):
  how known mules become observed positives, and the Account label contract.
- [Temporal payment schema](docs/temporal_schema.md) and
  [GSQL temporal encoding](docs/temporal_encoding.md): the graph and the 64-dimensional
  payment-gap and cutoff-age encodings.
- [GSQL guide](gsql/README.md): which query files exist and how they are installed.

## Development

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/basedpyright src
.venv/bin/lint-imports
.venv/bin/python scripts/render_queries.py --check
```

The tests never connect to TigerGraph unless a marker selects them. The integration
tests in `tests/integration/` are deselected by default: `-m graph` runs the read-only
checks against the graph in `.env`, `-m cuda` the cuGraph checks on a CUDA host, and
`-m graph_write --allow-graph-writes` the scope isolation test, which writes temporary
fixture vertices and removes them again:

```bash
.venv/bin/python -m pytest -m graph
.venv/bin/python -m pytest -m cuda
.venv/bin/python -m pytest -m graph_write --allow-graph-writes
```

`scripts/render_queries.py` regenerates the context query,
`scripts/run_experiments.py` runs the control experiments, and
`scripts/simulate_label_reveal.py` shows how the label reveal varies with its salt; each
prints its purpose with `--help` without connecting. The one-time schema installer
scripts, already run against the live graph, are kept in git history.

## License

MIT (author: Abraham Chandy).
