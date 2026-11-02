# Mule Pattern Learner

Finds money-mule accounts that a bank has not found yet with a temporal heterogeneous graph
neural network on TigerGraph, trained and scored on data simulated by PhantomLedger.

What it does and every decision behind it is in [the design](docs/design.md), and every number
that reaches the model is in [the features](docs/features.md). The queries are GSQL under
`gsql/` and are the main deliverable; the Python package under `src/mule_pattern_learner/` fetches
what they print, builds batches, trains, compares and evaluates.

## Setup

Python 3.12 or newer in a `.venv`, installed editable:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run `pip install -e ".[dev]"` again after every pull: the `mule` command, the import contracts
and the dependencies change with the code.

Copy `.env.example` to `.env` and set:

| Setting | What it is |
|---|---|
| `HOST`, `GRAPHNAME`, `SECRET` | the TigerGraph Cloud workspace of this project and its graph, `Mule_Pattern_Learner` |
| `MULE_EXPORT_DIR` | the parent of the loader's export folders (`tf_gnn_loader_v2`), which `mule parity` reads |
| `MULE_PL_RECORDS_DIR` | the folder the loader writes each world's identity file and gate record into, from PhantomLedger's records |

A relative folder is taken from the repository root, and an environment variable overrides
`.env`. Everything else is a setting in [`config.py`](src/mule_pattern_learner/config.py): the
worlds with their seeds and sizes, the rotations of their roles, the calendar, the caps and
thresholds, the model and the training. Commands take no options.

## The commands

| Command | What it does |
|---|---|
| `mule install` | installs the queries of `gsql/queries/` and `gsql/evaluation/` that are not current |
| `mule parity` | runs every installed query on the loaded world and on the reference graph of the loader's export of it, and writes `gsql/parity.toml` |
| `mule prepare` | matches the loaded world to a configured one, checks its calendar and labels, writes the training examples and weights, and registers the world |
| `mule fetch` | fills the cache for every example the world's roles need |
| `mule audit` | reads the answer key into the evaluation store, runs the label checks, the clock gain test (it fails below `audit.clock_min_mule_clusters`, 30, mule clusters in the audit sample) and, on development worlds, the feature audit, on the test world stores the audit sample with the start of each row's week and fetches its rows there for the clock invariance test, and writes pass records |
| `mule train` | trains the GNN over the prior grid and five seeds, its four ablations and the two XGBoost comparisons, from the cache |
| `mule evaluate` | scores the test world, runs the clock invariance test of the selected GNN (the same account-weeks at the random scoring times and at the week starts, failing below the same minimum of mule clusters), and writes the metrics, groups, uncertainty, missed-mule report and identification bias |
| `mule score` | scores the accounts and dates set in `config.py` (`score.accounts`, `score.dates`, any date, each scored at 00:00 UTC) with the full model of the best-selection seed of the run `score.run` names |
| `mule bench` | measures what the loaded world costs and projects it to other sizes |

`mule install`, `mule parity`, `mule prepare`, `mule fetch`, `mule audit` and `mule bench` need
the world in TigerGraph, and `mule score` any graph loaded with the repository's queries
installed. `mule train` and `mule evaluate` never query it.

Every long command first runs a read-only readiness check, and stops at the first check that
fails, naming it:

- the commands that read the loaded world check that the installed queries are the repository's,
  that the loaded world is a configured one with the calendar's 455 exported days, that one
  account can be fetched (into a scratch folder, never the cache) and that one training step
  runs;
- `mule score` checks only that the installed queries are the repository's: it matches no world,
  so it scores any loaded graph, a bank's included, and fetches into a scratch folder that the
  run alone reads;
- `mule train` and `mule evaluate` check, for every rotation before the first one starts, that
  every world of it was prepared, that one example can be read from the cache of each world the
  command reads (`mule train` the training world and the selection world, `mule evaluate` the test
  world) and that one training step runs.

`mule train` also reads the pass or fail of the audit records of every rotation's worlds before it
trains anything, and stops, naming the checks, on any failed check of a reported world and on a
failed label check or feature audit of a development world. Run `mule audit` on each world first.

### Rotations and results

Three worlds make a rotation: one trains, one selects and one tests. `config.worlds.rotations`
holds them, and `mule train` and `mule evaluate` run every one in order, so set it to the
rotations whose worlds you have prepared. The results of a rotation are in
`results/<train>-<select>-<test>/`: the model files, the tables, the manifest and `events.jsonl`.
`mule evaluate` also writes the spread over the rotations of each kind of world with two or more
rotations, to `results/rotations/rotation_spread_<kind>.parquet`: the three reported rotations make
one, and the development rotation is never pooled with them. `results/bench/` holds the
benchmark's tables and `results/commands/` the events of the other commands. Nothing under
`data/` or `results/` is committed.

## One world at a time

Only one world is loaded in TigerGraph at a time, and each is read out before the next is loaded.
For each world of a rotation, in turn:

1. PhantomLedger generates it, with its fixed seed and size from `config.py`.
2. `mule-load push` (in `tf_gnn_loader_v2`) creates the schema on the empty graph, loads the world,
   verifies it, and writes its identity file and gate record into `MULE_PL_RECORDS_DIR`.
3. `mule prepare` matches the loaded world to its configured name by the identity file, checks the
   calendar and the labels, and writes the examples of the world's roles.
4. `mule fetch` fills the cache for those examples. It stops at a query version with no passing
   parity record.
5. `mule audit` reads the answer key and runs the checks.
6. `mule-load clear` empties the graph. The schema and the queries stay.

Then, once every world of a rotation is read out, `mule train` and `mule evaluate`. The first
time on a workspace, `mule install` installs the queries, and `mule parity` is run once on the
5,000-person test world.

## The 5,000-person test, from this laptop

World T (seed 3001, 5,000 people) is the test of the whole chain on TigerGraph Cloud before any
300,000-person world is generated. PhantomLedger, `tf_gnn_loader_v2` and this repository all run
from the laptop, against the workspace `.env` names.

1. `mule-load wipe` empties the workspace, queries and schema included.
2. Generate world T with PhantomLedger, then `mule-load push`, which creates the schema and
   verifies the load, the 455 exported days and every attribute filled from the export.
3. `mule install`, run every query, `mule parity`, and the two demo queries of `gsql/demo/`.
4. `mule prepare`, `mule fetch` and `mule audit` as a readiness run. World T plays no role in a
   rotation, so `mule fetch` says it has nothing to fetch; `mule prepare` and `mule audit` run in
   full.
5. Commit `gsql/parity.toml`, the record of the query versions that matched the reference graph.
6. `mule-load clear`.

## The benchmark and the recommended training architecture

`mule bench` measures a sample of the loaded world and writes `results/bench/<world>.parquet`, a
table with a row per number (`section`, `name`, `variant`, `value`, `unit`, `note`). It decides
three things before any 300,000-person world is generated, each projected at the sizes of
`config.bench.projected_people`:

- whether whole-world selection is affordable: selection builds every selection example anew on
  every epoch of every prior, seed and ablation, so a rotation makes `passes_per_rotation` passes
  (1,050 with the prior grid, 750 with the prior fixed) over `examples_per_pass` examples, and the
  `selection` section prices them from the measured seconds to build one example;
- what `mule audit` costs: it fetches every active true-mule account-week and 200,000 non-mule
  account-weeks on every world, priced in the `audit` section from the measured seconds to
  fetch a sparse account-week (the mule term counts only the identified mules, since the bench
  never reads the truth, so it is a lower bound);
- what `mule evaluate`'s paired bootstrap costs: 1,000 replicates over every run (27: five seeds
  of the full model and of each ablation, and each XGBoost comparison once) over every ranked
  account-week, priced in the `bootstrap` section.

Beside them it holds the fetch time and bytes per query at 1, 2 and 4 concurrent requests, the
subgraph sizes, the cache bytes per account, the seconds to build a batch and take a training
step, the XGBoost file hand-off, `profile_hubs`' histograms and the whole-world scoring run
against its fallback. TigerGraph's own memory in use is not in the table: the metrics endpoint
answers 403 for this workspace's secret, so read it in the Cloud console while the bench runs.

**The recommended training architecture is not written yet.** It follows from these tables, which
need a loaded world, and none has been benchmarked: whether built batches stay on the GPU or in
pinned host memory, the accounts per request and the parallel requests, whether selection scores
the whole selection world or its fallback, and the number of passes. When the bench has run on a
30,000-person world, this section is replaced with what the tables show.

## Development

```bash
.venv/bin/ruff check
.venv/bin/ruff format --check
.venv/bin/basedpyright
.venv/bin/lint-imports
.venv/bin/pytest -q -p no:warnings
```

The tests never connect to TigerGraph: tests that need it are marked `graph` and run only with
`pytest -m graph`. `tests/gsql/test_live_compile.py` compiles every written query on the live
workspace, and `tests/testing/test_parity.py` runs `mule parity` on a loaded world once one is
there. The import contracts in `pyproject.toml` run as one of the tests, and every `.gsql` file
passes `gsql-lsp check` against `gsql/schema/`.

## License

MIT (author: Abraham Chandy).
