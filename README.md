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
2. **A Python layer** (`src/mule_pattern_learner/temporal/live/`, command `mule-temporal`)
   that installs those queries, prepares a bounded cohort, streams batches from
   TigerGraph, trains, scores and evaluates.

You need a working TigerGraph instance whose graph `Mule_Pattern_Learner` follows the
[temporal payment schema](docs/temporal_schema.md) and holds the loaded data.

## Setup

Python 3.12 or newer.

```bash
pip install -e ".[model]"        # training
pip install -e ".[all]"          # training and dev tools
pip install -e ".[all,cuda12]" --extra-index-url=https://pypi.nvidia.com  # CUDA 12 hosts
pip install -e ".[all,cuda13]"   # CUDA 13 hosts (install the matching CUDA torch wheel first)
```

Install in editable mode: the commands read `gsql/` from the repository.

Copy `.env.example` to `.env` and fill in the TigerGraph connection (read by
`Settings`; environment variables override it):

```
HOST=https://your-tg-host
GRAPHNAME=Mule_Pattern_Learner
SECRET=your_restpp_secret
```

The graph is created from `gsql/schema/temporal_schema.gsql`. This repository ships
only the Account loading contract (`gsql/schema/temporal_account_loading.gsql`); the
payment events, associations, tokens and other entities are loaded by the external
data producer (for the reference snapshot, an export of the PhantomLedger
simulator).

## Train

With the data loaded in TigerGraph, one command trains the live temporal model. It
needs nothing but `.env`: no configuration file, no dataset identifier, no label file
and no prepared artifacts to copy.

```bash
mule-temporal train
```

It uses CUDA when available (then Apple MPS, then CPU) and writes `models/temporal/model.pt`.
On a fresh graph the first run installs the training queries, creates the frozen
scope and reveals the known mules in the graph ([label reveal](docs/label_reveal.md));
every run then prepares its cohort inside `models/temporal/model_run/`. Run the same
command again to resume an interrupted run. The settings are built in
(`DEFAULT_RUN` in `src/mule_pattern_learner/temporal/live/config_schema.py`); an
optional `--config overrides.toml` changes only the keys it sets (tables such as
`[sampler]` merge key by key).

The other commands (`install`, `prepare`, `score`, `score-new`, `evaluate` and
`evaluate-final`) are described in the end-to-end guide under
[what runs where](docs/temporal_training_end_to_end.md#what-runs-where);
`mule-temporal --help` lists their options.

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
.venv/bin/python scripts/temporal/render_training_queries.py --check
```

The tests never connect to TigerGraph. The scripts in `scripts/temporal/` install
schema additions and run live checks against the graph; each prints its purpose with
`--help` without connecting. The verification scripts that record a report write it
under `artifacts/temporal/` (git-ignored); `--output` chooses another path. The one-time
schema installers keep their preflight backup next to their documentation in `docs/`.

## License

MIT (author: Abraham Chandy).
