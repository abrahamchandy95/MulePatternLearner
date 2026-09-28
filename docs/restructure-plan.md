# Restructuring plan (approved 2026-09-27)

This is the design record for moving the repository to its layered layout. It is
temporary: the migration's docs step turns it into `docs/architecture.md` and deletes it.
Where the owner decisions below differ from the proposal that follows, the decisions win.

## Owner decisions

1. **Layout approved** as proposed below.
2. **Commands take no flags.** Inputs come from built-in settings and the latest run:

   | Command | What it does |
   |---|---|
   | `mule train` | Prepares as needed (install, scope, reveal, dataset), trains the one model into `results/baseline/seed-42/`, resumes an interrupted run, writes the training plots |
   | `mule evaluate [RUN]` | Ground-truth audit of validation and test with intervals and plots |
   | `mule report [RUN]` | Redraws every figure and `report.md` from saved files, offline |
   | `mule score ACCOUNTS [DATE]` | Scores the accounts listed in a file; the date defaults to the test cutoff |
   | `mule check` | Read-only readiness: connection, installed queries, cuGraph probe, one batch with tensor digests and the first loss |
   | `mule diagnose [ANALYSIS]` | Re-runs the diagnostic study (all analyses by default) |
   | `mule install` | Installs queries whose text differs (train already does this), then drops the installed queries that no file defines |

   `RUN` defaults to the baseline run. There is no `--config`, `--split`, `--truth`, `--force`,
   `--drop`, `--batch` or `prepare` command. Retiring installed queries is not a flag either:
   `mule install` drops the ones no file defines (decided later, see "`mule install` drops the
   legacy queries" below).
3. **Control experiments** run as `python scripts/run_experiments.py [SUITE or VARIANT ...]`
   with no flags: suite `controls` by default, seeds fixed in code (42, 43, 44), suites and
   variants listed under `--help`, completed runs skipped, mismatched runs moved to
   `results/archive/` automatically (never deleted), comparison tables and plots always written.
   Variants are declared in `experiments/variants.py`. Nothing is read from or written to `/tmp`.
4. **TigerGraph queries are renamed after their responsibility** (verb first, no prefix), and
   only queries the project uses remain. Pipeline queries live in `gsql/queries/`, the
   ground-truth query in `gsql/evaluation/`, and queries used only for analysis in
   `gsql/analytics/`:

   | Current | New | Folder |
   |---|---|---|
   | `temporal_training_context` | `fetch_training_context` | `queries/` (generated) |
   | `temporal_fourier64_values` | `encode_fourier64` | `queries/` (subquery of the context query) |
   | `temporal_create_training_scope` | `create_training_scope` | `queries/` |
   | `temporal_finalize_training_scope` | `finalize_training_scope` | `queries/` |
   | `temporal_scope_population` | `list_scope_accounts` | `queries/` |
   | `temporal_scope_policy` | `summarize_scope_policy` | `queries/` |
   | `temporal_training_cutoffs` | `resolve_split_cutoffs` | `queries/` |
   | `temporal_hub_registry` | `list_hub_accounts` | `queries/` |
   | `temporal_reveal_mule_labels` | `reveal_mule_labels` | `queries/` |
   | `temporal_reveal_uniforms` | `draw_reveal_uniforms` | `queries/` (subquery of the reveal) |
   | `temporal_validate_account_supervision` | `validate_label_contract` | `queries/` |
   | `temporal_get_account_supervision` | `read_ground_truth` | `evaluation/` |
   | `zelle_pair_time64` | `encode_zelle_pair_gaps` | `analytics/` |
   | `payment_pair_time64` | `encode_payment_pair_gaps` | `analytics/` |
   | `temporal_training_population` | retired (unused once `shared_history` is removed) | |
   | `temporal_fourier64` (public wrapper) | retired (only a deleted verification script called it) | |

   The rename happens once, in the server step, after live parity with the unchanged queries.
   Files are named after the responsibility their queries share.
5. **Window and account-aggregate computations the model does not use belong to analytics.**
   In the server step, feature groups that neither the built-in run nor a declared training
   variant uses move out of the training context query into a query under `gsql/analytics/`,
   used by `mule diagnose`. The exact split is decided (and confirmed with the owner) in that step;
   until then all groups stay in the context query unchanged. Narrowed later (see "Training keeps
   only the feature groups it trains on" below): only the built-in run's groups stay.
6. **Decisions use the validation ground-truth audit.** The test audit is for reporting only.
7. **Replacing main** is a fast-forward; pushing to `origin` and `learner` needs the owner's
   confirmation at that time.
8. **Names for the data and the outputs.**
   - **"Dataset" replaces "cohort"** in code, configuration, docs and paths. A dataset is what
     preparation stages for training: the accounts of each split, their revealed labels, the
     cutoffs and the hub list. Prepared datasets live in `data/<dataset id>/`. Examples:
     `prepare_dataset()`, `DatasetConfig`, and `data/accounts.py` for the seed reservoirs and the
     positive pool.
   - **"Source id"** names the identity of the data loaded into the graph (today's `dataset_id`,
     stored as the scope's `source_id`). The dataset id is the fingerprint of the source id,
     `scope`, `dataset` and the sampler's pool parameters.
   - **`results/` replaces `runs/`** for everything the commands write:
     `results/baseline/seed-42/` (what `mule train` writes), `results/<variant>/seed-<n>/`,
     `results/experiments/<suite>/`, `results/diagnostics/<dataset id>/` and `results/archive/`.
     A run is still one training run, so `RunPaths` keeps its name. `DATA_DIR` and
     `RESULTS_DIR` replace `RUNS_DIR`. Both folders are gitignored.
9. **One branch at the end: `main`.** After the fast-forward and the confirmed pushes, and after
   the owner confirms, every other branch is deleted: local `restructure`, `temporal`,
   `archive/diagnostic-study`, `claude/great-mirzakhani-802edd` with its worktree, and
   `origin/temporal`. The diagnostic study reaches `main` through the diagnostics step (modules,
   and notes with figures under `docs/research/`) before its archive branch goes. The
   `pre-restructure` tag is deleted after the live parity and server steps unless the owner keeps
   it.

Decided on 2026-09-28, on questions the mid-migration review raised:

10. **Every run reads the graph's labels.** Every run reads the labels revealed in the graph
    (`pu_label`), and the model trains only on them, so there is no table label reader.
    `data.ports.ObservedLabelReader` has no `from_graph` flag and no `positive_ids()`, and
    `data.accounts.select_accounts` always reads the scope population with its observed labels.
    Tests serve their observed positives through the fake graph's population, so
    `testing.builders.FrameObservedLabels` is deleted. `metrics.json` drops its constant
    `label_policy`, and `gsql/README.md` its external label providers. The golden literals and
    the saved-model scores stay identical.
11. **The old saved-settings conversion stays until the new baseline run has a ground-truth
    audit**, so the owner's current best model can still be compared, and the replace-main step
    deletes it. Until then it lives in a module of its own, `inference/saved_settings.py`
    (`SAVED_SETTINGS`, the `variant`, `tabular` and `cohort_seed` values), so `SavedModel`
    stays small.
12. **The dataset id's scope settings are `scope.id` and `scope.unowned` only**, as implemented
    (see Configuration).

Decided on 2026-09-28, after the audit and plots steps:

13. **Training keeps only the feature groups it trains on.** From the server step on, the training
    context query and `contract.feature_groups.FEATURE_GROUPS` hold the groups of the built-in run
    (`BUILT_IN_GROUPS`) and nothing else. Every other group is analytics: its computation moves
    out of the context query into a query under `gsql/analytics/`, which `mule diagnose` uses.
    This replaces the proposal's "all 20 feature groups stay" and the rule on window and
    account-aggregate computations above ("neither the built-in run nor a declared training
    variant"). Until the server step the query text stays byte-identical, so all groups stay in
    it. The variants that read other groups cannot train from the context query any more: the
    `no_graph` control (its account aggregates are mostly outside the built-in run) and the
    `feature_adds` suite (`OPTIONAL_GROUPS`). The experiments step says what becomes of them,
    for example an analysis of `mule diagnose` over the analytics query; drops of built-in groups
    stay variants.
14. **`mule install` drops the legacy queries.** After installing the queries whose text differs,
    it drops the installed queries that no repository file defines (the old names, once the server
    step has installed the new ones), with no flag. This replaces the server step's one-off call.
    Like the rename it waits until no old-code run is active anywhere, since the old code calls the
    old names.
15. **`learner` and `origin` end with the same `main`.** The replace-main step pushes the one
    fast-forwarded `main` to both remotes, so `learner/main` and `origin/main` name the same
    commit. The pushes still wait for the owner's confirmation at that time.

## Cost correction

Measured on the CUDA host, a graph run takes about 3 s per step and about 1 hour with early
stopping (the reference run stopped at epoch 11), not the 11 s per step and 12.5 hours an earlier draft assumed.

## Part 2: Final proposal

### Decisions at a glance

**Names.**
- The distribution stays `mule-pattern-learner` and the package stays `mule_pattern_learner`.
- There is one console script, `mule`, and `python -m mule_pattern_learner` also works.
- No path, identifier or command contains `temporal`, `live`, `v5` or `legacy`. The exceptions are the allow-listed server values (see Naming).

**One model, `TGAT`.**
- `SummaryMLP` exists only for the controls.
- These are deleted: the `variant` axis, the `single` architecture, the `LEGACY_GROUPS` constant, the `recent`/`stratified` samplers, SQLite storage, `shared_history`, the parquet label policy and the table label reader.
- Until the server step all 20 feature groups stay in the registry and in the query; from then on only the built-in run's groups do, and the others move to `gsql/analytics/` (see Owner decisions).

**Two roots, both gitignored.** Prepared datasets go under `data/<dataset id>/`, and everything the commands write goes under `results/`. `mule train` writes `results/baseline/seed-42/`.

**Configuration is Python.**
- Frozen dataclasses in `config.py`; their defaults are the built-in run.
- `.env` holds only the connection.
- There is no `--config`, and no tracked TOML, YAML or JSON.

**Control experiments.**
- Variants are declared in `experiments/variants.py`.
- `scripts/run_experiments.py` trains the chosen variants and seeds on one dataset, audits validation and test, and writes the tables and figures to `results/experiments/<suite>/`.
- Decisions use the validation audit. The test audit is for reporting.

**Plots.** Only `reporting/` imports matplotlib, and it reads saved files only. `mule report DIR` rebuilds any figure offline.

**Layers** are enforced by import-linter in the test gate, with an AST test as the fallback.

**Server step.** Query names are renamed once, as in the owner decisions.

**Replacing main** is a fast-forward. No push happens without the owner's confirmation.

### Directory tree

```
MulePatternLearner/
├── pyproject.toml          hatchling; `mule` script; torch and matplotlib as core deps; extras cuda12, cuda13, dev
│                           (pytest, ruff, basedpyright, import-linter); pytest importlib mode and markers
├── README.md               the model, setup, `mule train`, `mule evaluate`, the experiments script, doc index
├── LICENSE
├── .env.example            HOST, GRAPHNAME, SECRET (GRAPHNAME must equal contract.server.GRAPH_NAME; checked at connect)
├── .gitignore              unchanged
├── src/mule_pattern_learner/
│   ├── __init__.py         package version
│   ├── __main__.py         `python -m mule_pattern_learner` runs cli.main
│   ├── py.typed
│   ├── cli.py              argparse; calls pipeline, reporting and diagnostics; prints one JSON result
│   ├── config.py           RunConfig and its sections; DEFAULT_CONFIG (the built-in run); fingerprint(); to_dict/from_dict
│   ├── paths.py            REPOSITORY_ROOT, GSQL_DIR, DATA_DIR, RESULTS_DIR; DatasetPaths, RunPaths (every file name); suite, diagnostics dirs
│   ├── artifacts.py        column schemas and read/write of every run file; atomic_write and file_digest (the only copies)
│   ├── metrics.py          pure numpy/sklearn: threshold, weighted AP/ROC AUC, PR/ROC/capture curves, tie-aware recall and
│   │                       precision at review budgets, stratified, ring-clustered and paired bootstrap, weighted quantiles
│   ├── contract/           definitions shared with GSQL; no I/O, no torch
│   │   ├── server.py           GRAPH_NAME, SCOPE_VERTEX, query names and parameter sets, CONTEXT_CONTRACT
│   │   ├── graph_schema.py     node types, relations, associations, rails, channels, strata, splits and phases, ContextKey,
│   │   │                       context_scope, row columns (hub, oracle, account load, score)
│   │   ├── feature_groups.py   FeatureGroup, FEATURE_GROUPS (the built-in run's groups), BUILT_IN_GROUPS, pool-count names,
│   │   │                       bands and pass-through constants, FeaturePlan
│   │   ├── sampler_plan.py     PoolPlan, SamplerPlan (also the config section type), selection-key version
│   │   ├── bounds.py           every numeric bound once: request cap 64, fan-out 1 to 64, batch up to 128, seed limits,
│   │   │                       pool ranges
│   │   ├── time_basis.py       Fourier64 in numpy, BASIS_ID
│   │   ├── fingerprints.py     fingerprint (sha256 of JSON), stable_hash (blake2b, sampler keys), hash64; the docstring says
│   │   │                       why there are three (persisted draws)
│   │   ├── clock.py            timestamp parsing, cutoff_ms
│   │   └── salts.py            frozen RNG salt values
│   ├── runtime/
│   │   ├── device.py           CUDA, then MPS, then CPU; determinism; reserve_deterministic_cublas() (called first by every
│   │   │                       entry point); threads
│   │   ├── workers.py          DaemonPool: the one bounded pool (context requests and batch prefetch)
│   │   └── progress.py         emit(): one structured stdout line, also appended to the current run's events.jsonl
│   ├── tigergraph/         the only code that speaks REST or GSQL
│   │   ├── connection.py       ConnectionSettings from .env, loaded when called
│   │   ├── client.py           pyTigerGraph connection, finite timeouts, HTTP error raising
│   │   ├── executor.py         QueryExecutor protocol; TigerGraphExecutor: failure classes, retry and outage budgets, paging
│   │   ├── gsql_text.py        read, strip comments from and normalise GSQL; query signatures
│   │   ├── installer.py        install queries whose text differs; the one scope schema change when its vertex type is
│   │   │                       missing; drop the installed queries no file defines
│   │   ├── render.py           build-time generator of gsql/queries/training_context.gsql (the runtime never imports it)
│   │   ├── context_query.py    TigerGraphContextFetcher: validation, bisection on timeout, per-row contract and encoding check
│   │   ├── scope.py            TigerGraphScope: header, create, finalize, population pages, policy
│   │   ├── cutoffs.py          TigerGraphCutoffs
│   │   ├── hubs.py             TigerGraphHubs: hub query, rows parsed into data.hub_registry.HubRegistry
│   │   ├── labels.py           TigerGraphObservedLabels; label-contract audit
│   │   ├── reveal.py           the one-time reveal (writes; runs only when the graph has no revealed mules)
│   │   ├── oracle.py           TigerGraphTruth: is_mule, ring id, label source (evaluation and diagnostics only)
│   │   └── provenance.py       vertex counts, derived source id, frozen-source check
│   ├── data/
│   │   ├── ports.py            ContextFetcher, ScopeReader, CutoffReader, HubReader, ObservedLabelReader
│   │   ├── accounts.py         seed reservoirs and positive pool (paging through ScopeReader, the one pager)
│   │   ├── splits.py           split dates, cutoff clocks, context keys per split
│   │   ├── observed_labels.py  observed-label frame, alignment, summaries
│   │   ├── hub_registry.py     HubRegistry, parquet save/load, stub warning
│   │   ├── manifest.py         dataset manifest, dataset id, integrity gate
│   │   ├── preparation.py      prepare_dataset(): reservoirs, labels, hubs, manifest (resumable; read ports only)
│   │   ├── contexts.py         ContextSource: request windows over DaemonPool, LRU, coverage check, rejection counts
│   │   └── context_cache.py    ContextSource's cache tiers: the LRU (MemoryTier) and the disk tier (DiskTier) of a
│   │                           dataset's ContextCache under data/<dataset id>/contexts/
│   ├── sampling/           candidates.py, torch_sampler.py, cugraph_sampler.py (pylibcugraph on first use), backend.py
│   ├── model/              torch modules; imports contract and config only
│   │   ├── inputs.py           ModelInputs
│   │   ├── tgat.py             TGAT-style attention over hop-1 and hop-2 slots, summary branch, slot sum
│   │   ├── summary_mlp.py      SummaryMLP (the no_attention and no_graph controls); not summary.py, which training/ has
│   │   ├── loss.py             NonNegativePULoss
│   │   └── build.py            build_model(), probabilities_from_logits()
│   ├── batching/
│   │   ├── features.py         vectorised node, base and edge matrices
│   │   ├── time_encoding.py    Fourier64 in torch for client-computed edge ages (tested against contract.time_basis)
│   │   ├── pool_counts.py      pool counts over the root's candidate pool
│   │   ├── limits.py           BatchLimits, BatchIndex (bounds from contract.bounds)
│   │   └── assemble.py         RootBatch, build_batch(), to_device() (the one device rule)
│   ├── inference/          saved_model.py, predictor.py (the one scoring loop), rejections.py, score_accounts.py,
│   │                       saved_settings.py (configurations saved before RunConfig; until main is replaced)
│   ├── training/           trainer.py, objective.py, averaging.py, schedule.py, checkpoint.py, history.py, summary.py
│   ├── evaluation/
│   │   ├── truth.py            TruthReader protocol, ParquetTruth
│   │   ├── sample.py           audit_sample(split): every positive plus uniform negatives with inclusion probabilities;
│   │   │                       AUDIT_NEGATIVES = 2000
│   │   └── audit.py            audit(run, split): score, rejection limit, weighted metrics with ring-clustered intervals,
│   │                           revealed flags
│   ├── pipeline/           composition root: the only place adapters are built
│   │   ├── connect.py          settings, executor with config.transport, adapters, graph-name check
│   │   ├── prepare.py          install, scope, reveal, prepare_dataset, frozen-source checks
│   │   ├── train.py            train_run(): trainer, then run report
│   │   ├── evaluate.py         evaluate_run(): audits, then audit report
│   │   ├── score.py            score_accounts use case
│   │   └── check.py            read-only readiness; one batch's digests and first loss
│   ├── reporting/          style.py, training.py, ranking.py, scores.py, comparison.py, diagnostics.py, report.py
│   ├── experiments/        variants.py, runner.py, comparison.py
│   ├── diagnostics/        feature_table.py, baselines.py, learning_curve.py, univariate.py, drift.py, subgroups.py,
│   │                       proxy_validity.py, reveal_spread.py, nnpu_simulation.py
│   ├── reference/          CPU mirrors, never imported by runtime layers except diagnostics
│   │   ├── gsql_features.py    mirror of the context query's features
│   │   ├── label_reveal.py     mirror of the reveal job
│   │   └── batch_features.py   scalar node, base and edge features (oracle for batching.features)
│   └── testing/            fakes and builders (the PyG testing/ pattern)
│       ├── fake_graph.py       FakeTigerGraph: a ConnectionExecutor answering the repository's queries, asserting GSQL
│       │                       signatures, with SHOW QUERY, endpoints, counts and scope headers on its connection
│       ├── fake_connection.py  fake pyTigerGraph connections for client and executor tests
│       └── builders.py         accounts with their revealed labels, messages, payments, associations, contexts, test RunConfig
├── gsql/
│   ├── README.md
│   ├── queries/            the training pipeline's queries; reinstalled when their text changes
│   │   ├── training_context.gsql   fetch_training_context; generated by scripts/render_queries.py, do not edit
│   │   ├── fourier64.gsql          encode_fourier64, the subquery of fetch_training_context
│   │   ├── training_scope.gsql     create_training_scope, finalize_training_scope, list_scope_accounts,
│   │   │                           summarize_scope_policy
│   │   ├── split_cutoffs.gsql      resolve_split_cutoffs
│   │   ├── hub_accounts.gsql       list_hub_accounts
│   │   ├── label_reveal.gsql       draw_reveal_uniforms, reveal_mule_labels
│   │   └── label_contract.gsql     validate_label_contract
│   ├── evaluation/
│   │   └── ground_truth.gsql       read_ground_truth (the oracle; evaluation and diagnostics only)
│   ├── analytics/          queries used only for analysis (`mule diagnose`); training never calls them
│   │   ├── zelle_pair_gaps.gsql    encode_zelle_pair_gaps
│   │   └── payment_pair_gaps.gsql  encode_payment_pair_gaps
│   └── schema/
│       ├── schema.gsql             fresh graph DDL (run by a person)
│       ├── account_loading.gsql
│       └── scope_vertex.gsql       applied by `mule install` when the scope vertex type is missing
├── scripts/
│   ├── run_experiments.py  control-experiment entry point (thin wrapper over experiments.runner)
│   └── render_queries.py   regenerate, or `--check`, gsql/queries/training_context.gsql
├── tests/
│   ├── conftest.py         fixtures: test RunConfig, FakeTigerGraph, temporary data and results dirs
│   ├── contract/ tigergraph/ data/ sampling/ model/ batching/ inference/ training/ evaluation/ pipeline/
│   │   reporting/ experiments/ diagnostics/ reference/ runtime/      test_<module>.py, mirroring src
│   ├── test_config.py, test_metrics.py, test_artifacts.py, test_cli.py, test_scripts.py, test_naming.py,
│   │   test_layers.py (AST fallback of the contracts), test_doc_links.py
│   └── integration/        markers graph, graph_write (needs --allow-graph-writes), cuda; excluded by default
├── docs/
│   ├── architecture.md
│   ├── how-to/             set-up-a-graph.md, train-and-evaluate.md, run-control-experiments.md, score-new-accounts.md,
│   │                       run-diagnostics.md
│   ├── reference/          cli.md, configuration.md, outputs.md, features.md, schema.md, labels.md, queries.md
│   ├── explanation/        training.md, sampling.md, feature-design.md, time-encoding.md, leakage-and-scaling.md,
│   │                       label-reveal.md
│   └── research/           diagnostic-study.md, mule-profile.md, nnpu-positive-weight.md, reference-run.md,
│                           figures/ (committed PNGs of recorded findings)
├── data/                   gitignored: prepared datasets, the inputs to training
│   └── <dataset id>/                 manifest.json, accounts.parquet, observed_labels.parquet, hubs.parquet, contexts/
└── results/                gitignored: everything the commands write
    ├── baseline/seed-42/             what `mule train` writes
    ├── <variant>/seed-<n>/           one training run of a control experiment
    ├── experiments/<suite>/          summary.csv, comparison.csv, report.md, plots/
    ├── diagnostics/<dataset id>/     features.parquet, <analysis>.csv, report.md, plots/
    └── archive/                      results moved aside because their settings changed (never deleted)
```

### Run directory (`results/<variant>/seed-<n>/`, names fixed in `paths.RunPaths`)

| File | Written by | Content |
|---|---|---|
| `config.json` | train | `{"config": …, "fingerprint": …, "provenance": {git commit, dirty flag, versions of the package, torch, numpy, scikit-learn and pyTigerGraph, device, threads, determinism, sampler backend, dataset id, started}}` |
| `model.pt` | train | selected weights, RunConfig, feature plan, threshold, dataset id, `SavedModel.FORMAT` |
| `resume.pt` | train | optimizer, weight average, RNG, epoch, step, selection state, dataset id and manifest sha256 |
| `history.csv` | train | `epoch, step, date, loss, objective, corrected_steps, steps, seconds_per_step, batch_wait_seconds, database_calls, contexts_requested, contexts_distinct, memory_hits, disk_hits, rejected_roots, stub_children` |
| `epochs.csv` | train | `epoch, loss, steps, validation_ap, validation_roc_auc, weights, selected, stopped` |
| `events.jsonl` | every command | resume (with the segment's device, threads and determinism, and any change of them), backend choice, warnings, rejections (the lines `emit()` prints) |
| `predictions/validation.parquet`, `predictions/test.parquet` | train | proxy scores on observed labels |
| `metrics.json` | train | proxy metrics, totals, context counts with the disk cache's hit rate, rejections, sampler totals, wall-clock seconds over every segment |
| `audit/<split>.json`, `audit/<split>.parquet`, `audit/<split>_rejected.txt` | evaluate | ground-truth audit of `validation` and `test`. Parquet columns: `account_id, is_mule, inclusion_probability, score, revealed, ring_id, label_source`. The JSON records the audit constants |
| `scores/<accounts stem>_<date>.parquet`, `scores/<accounts stem>_<date>_rejected.txt` | score | scores of arbitrary accounts, and the accounts TigerGraph rejected |
| `plots/*.png`, `report.md` | train, evaluate, report | figures and tables |

### CLI (`mule`)

| Command | What it does | Graph writes |
|---|---|---|
| `mule install` | Adds the scope vertex type if missing, installs pipeline queries whose text differs, and drops installed queries that no file defines (from the server step on; until then it lists them). | yes |
| `mule train` | Prepares the dataset in `data/<dataset id>/` as needed, then the built-in run into `results/baseline/seed-42/`; resumes an interrupted run; a complete matching run is reported and left untouched; a mismatching one is an error that names the differing keys. Writes the training plots. | first run only: install, scope and reveal |
| `mule evaluate [RUN]` | Ground-truth audits of validation and test, plus audit plots. | no |
| `mule score ACCOUNTS [DATE]` | Scores the accounts listed in a file, one id per line; DATE defaults to the test cutoff. Replaces `score` and `score-new`. | no |
| `mule report [RUN]` | Redraws figures and `report.md` from the files already in a run, suite or diagnostics directory. Offline. | no |
| `mule diagnose [ANALYSIS]` | `features`, `baselines`, `learning-curve`, `univariate`, `drift`, `subgroups`, `proxy-validity`, `reveal-spread` or `nnpu-simulation`; all of them by default. | no |
| `mule check` | Connection, graph-name match, scope schema, installed query text, sampler probe, then one batch and one training step with tensor digests and the first loss. | no |

`RUN` defaults to `results/baseline/seed-42`.

**Why these names.**
- **`mule`, not `mule-pattern-learner`.** The PyPA guide pairs a console script with `__main__.py` (https://packaging.python.org/en/latest/guides/creating-command-line-tools/), and a short command matches Ludwig's CLI (`ludwig train`, https://github.com/ludwig-ai/ludwig/tree/main/ludwig). `python -m mule_pattern_learner` covers a machine where another tool (for example a MuleSoft runtime) also installs `mule`.
- **`score`, not `predict`.** The output is a risk score.
- **Experiments are a script, not a subcommand.** The owner asked for a script.

### Layering rules

```
cli                                              entry point
experiments | diagnostics                        research use cases
pipeline                                         composition root: the only place adapters are built
training | evaluation | reporting | tigergraph   use cases, figures, the TigerGraph adapter
inference                                        saved model, the one scoring loop
batching                                         contexts to ModelInputs
data | sampling | model                          dataset, ports and contexts; neighbour sampling; nn modules
runtime                                          device, worker pool, progress
artifacts | metrics                              file schemas; pure metrics
config | paths                                   built-in run; filesystem layout
contract                                         definitions shared with GSQL
(reference, testing, __main__: outside the stack)
```

**The rules.**
- **One way down.** A module imports only from layers below it. Modules separated by `|` are independent of each other.
- **Ports belong to their consumers.**
  - `data/ports.py` holds the read ports that preparation, training, inference and evaluation use.
  - `evaluation/truth.py` holds `TruthReader`, so ground truth is not even on training's import surface.
  - `tigergraph` satisfies them structurally.
  - Only `pipeline` builds adapters, with the `config.transport` retry budgets. That removes the hidden `TigerGraphExecutor()` calls (`L/evaluation.py:50`, `L/cli.py:126`) and the `live_executor` calls inside `L/source.py:583`, the predictor and `prepare_live`.
  - Sources: Cosmic Python's composition root (https://www.cosmicpython.com/book/chapter_13_dependency_injection.html). PyG shapes its remote-backend port around reads (`FeatureStore`/`GraphStore`, https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/feature_store.py), and its docs name TigerGraph as a GraphStore (https://pytorch-geometric.readthedocs.io/en/latest/advanced/remote.html).
- **Graph writes** (install, scope creation, reveal) happen only in `pipeline/prepare.py` and `tigergraph`. `data.preparation` gets read ports only.
- **`model` imports only `contract` and `config`.** `batching` adapts data to `ModelInputs`. Precedents: PyG's `nn/models/tgn.py` imports only `nn.inits` and `utils` (https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric), and in Transformers configuration never imports modeling (https://github.com/huggingface/transformers/tree/main/src/transformers).
- **Metrics are pure and sit low.** This replaces the three or four copies the inventory found. Models: sklearn's split between `_ranking.py` and `_plot/` (https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/metrics), and GraphStorm's `eval_func.py` (https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/eval/eval_func.py).
- **Reporting reads saved files only.** Sources: CCDS (https://cookiecutter-data-science.drivendata.org/opinions/) and PyKEEN's `plot_utils.py` (https://github.com/pykeen/pykeen/blob/master/src/pykeen/pipeline/plot_utils.py).
- **`pipeline` is the one wiring point**, as in GraphStorm's `gsgnn_np` (https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/run/gsgnn_np/gsgnn_np.py) and PyKEEN's `pipeline()` (https://github.com/pykeen/pykeen/blob/master/src/pykeen/pipeline/api.py). The CLI and the runner both call `pipeline.train.train_run` and `pipeline.evaluate.evaluate_run`.

**Enforcement.** Layer contracts: https://github.com/seddonym/import-linter/blob/main/docs/contract_types/layers.md. Forbidden contracts check indirect imports unless told otherwise: https://github.com/seddonym/import-linter/blob/main/docs/contract_types/forbidden.md

```toml
[tool.importlinter]
root_package = "mule_pattern_learner"
include_external_packages = true

[[tool.importlinter.contracts]]
name = "Layers run one way"
type = "layers"
containers = ["mule_pattern_learner"]
layers = ["cli", "experiments | diagnostics", "pipeline", "training | evaluation | reporting | tigergraph",
          "inference", "batching", "data | sampling | model", "runtime", "artifacts | metrics",
          "config | paths", "contract"]

[[tool.importlinter.contracts]]
name = "Use cases reach TigerGraph only through ports"
type = "forbidden"
source_modules = ["mule_pattern_learner.training", "mule_pattern_learner.evaluation", "mule_pattern_learner.data",
                  "mule_pattern_learner.batching", "mule_pattern_learner.inference", "mule_pattern_learner.sampling",
                  "mule_pattern_learner.model", "mule_pattern_learner.reporting", "mule_pattern_learner.diagnostics"]
forbidden_modules = ["mule_pattern_learner.tigergraph", "pyTigerGraph"]

[[tool.importlinter.contracts]]
name = "Training never reads ground truth"
type = "forbidden"
source_modules = ["mule_pattern_learner.training", "mule_pattern_learner.data", "mule_pattern_learner.batching",
                  "mule_pattern_learner.inference", "mule_pattern_learner.sampling", "mule_pattern_learner.model",
                  # the pipeline's preparing, training, checking and scoring use cases
                  "mule_pattern_learner.pipeline.connect", "mule_pattern_learner.pipeline.prepare",
                  "mule_pattern_learner.pipeline.train", "mule_pattern_learner.pipeline.check",
                  "mule_pattern_learner.pipeline.score"]
forbidden_modules = ["mule_pattern_learner.evaluation", "mule_pattern_learner.diagnostics",
                     "mule_pattern_learner.tigergraph.oracle"]

[[tool.importlinter.contracts]]
name = "Only reporting draws"
type = "forbidden"
source_modules = ["<every package except reporting, listed>"]
forbidden_modules = ["matplotlib"]
allow_indirect_imports = true          # pipeline, cli, experiments and diagnostics call reporting

[[tool.importlinter.contracts]]
name = "Reporting reads files, not models or the graph"
type = "forbidden"
source_modules = ["mule_pattern_learner.reporting"]
forbidden_modules = ["torch", "mule_pattern_learner.model", "mule_pattern_learner.inference",
                     "mule_pattern_learner.training"]

[[tool.importlinter.contracts]]
name = "Fakes stay out of the package"
type = "forbidden"
source_modules = ["<every package except testing, listed>"]
forbidden_modules = ["mule_pattern_learner.testing"]

[[tool.importlinter.contracts]]
name = "Only diagnostics uses the verification mirrors"
type = "forbidden"
source_modules = ["<every package except diagnostics and reference, listed>"]
forbidden_modules = ["mule_pattern_learner.reference"]
allow_indirect_imports = true          # cli reaches reference only through diagnostics

[[tool.importlinter.contracts]]
name = "The model knows nothing about storage"
type = "forbidden"
source_modules = ["mule_pattern_learner.model"]
forbidden_modules = ["pandas", "pyarrow", "requests", "mule_pattern_learner.data"]
```

If import-linter does not install on Python 3.14, `tests/test_layers.py` enforces the same contracts from an AST import graph.

### Configuration

- **`config.py` holds the built-in run** as frozen dataclasses. Each component receives only its own section.
  - Validation happens in `__post_init__`, and bounds come from `contract.bounds`.
  - The sampler section *is* `contract.SamplerPlan`.
  - This follows Twelve-Factor (https://12factor.net/config) and Transformers' dataclass defaults.
- **`.env` holds the connection only.** It is read when `pipeline.connect` runs, never at import.
- **Deleted:** `FALLBACKS`, `OPERATIONAL_DEFAULTS`, the pydantic `LiveConfig`, `run_config`, `setting()`, and the TOML/JSON loaders. nanoGPT's own docstring calls exec'd override files "Probably a terrible idea" (https://github.com/karpathy/nanoGPT/blob/master/configurator.py).

```python
@dataclass(frozen=True)
class RunConfig:
    scope: ScopeConfig = ScopeConfig()          # id="strict_mule_v2", create=True, unowned="linked",
                                                # reveal_per_split=20, reveal_salt=42
    dataset: DatasetConfig = DatasetConfig()    # dates, seed_limits, seed=42, split_seed=42
    sampler: SamplerPlan = BUILT_IN_SAMPLER     # fanouts (16, 4), roots and children pools, relation_fanouts (8, 4),
                                                # association_fanout 1, association_slots 2, backend "auto", evaluation_seed 0
    features: tuple[str, ...] = BUILT_IN_GROUPS
    model: ModelConfig = ModelConfig()          # architecture "tgat" | "summary", hidden 64, heads 4, dropout 0.15, slot_sum True
    loss: LossConfig = LossConfig()             # class_prior 0.001, positive_weight "balanced" | "prior" | float
    training: TrainingConfig = TrainingConfig() # seed 42, epochs 30, steps_per_epoch 100, batch_size 64, patience 6,
                                                # learning_rate 1e-3, weight_decay 1e-4, weight_average_decay 0.99,
                                                # proxy_unlabeled_limit 2000
    transport: TransportConfig = TransportConfig()  # request_batch_size 8, query_concurrency 16 (keep the measurement
                                                # comment), context_lru_capacity 256, encoding_check_every 64,
                                                # max_query_attempts 6, max_outage_s 900, prepare_batch_size 16
    runtime: RuntimeConfig = RuntimeConfig()    # device "auto", threads 4, deterministic True, prefetch_batches 2,
                                                # checkpoint_every_steps 0, log_every_steps 10, max_rejected_root_fraction 0.0

    def fingerprint(self) -> str: ...           # every section except transport and runtime (today's RUNTIME_KEYS plus
                                                # device, threads, deterministic, which provenance records)
DEFAULT_CONFIG = RunConfig()
```

- **The dataset id** is the fingerprint of the source id, `scope`, `dataset` and the sampler's pool parameters. It is exactly today's `PREPARATION_KEYS` minus the deleted ones.
  - Decided by the owner (see Owner decisions): of `scope`, that means only `scope.id` and `scope.unowned`, as in `PREPARATION_KEYS` (`scope_id`, `scope_unowned`). `scope.create`, `scope.reveal_per_split` and `scope.reveal_salt` act once on the graph (whether a missing scope is created, and the one-time reveal), so changing them later names no other dataset.
- **Audit constants are not run configuration.** `AUDIT_NEGATIVES = 2000` (`evaluation/sample.py`), `REVIEW_BUDGETS = (0.01, 0.05, 0.10)`, `BOOTSTRAP_REPLICATES = 1000` and `INTERVAL = 0.90` (`metrics.py`) are recorded in each audit JSON.
- **Component selection.** One `match` per real choice: `model.build.build_model` (tgat or summary), `sampling.backend.choose_sampler` (auto, torch or cugraph) and `training.objective` (positive weight).
  - The only registries are two plain tables: feature groups and variants.
  - There are no class resolvers. PyKEEN's `class_resolver`, GraphGym's `register_*` and GraphStorm's `BUILTIN_*` solve a problem that a single model does not have.
- **GSQL stays at the repository root** and is found through `paths.GSQL_DIR`.
  - The repository is used as an editable install, `data/` and `results/` already tie runtime to the root, reviewers read GSQL as files, and a byte-identity test guards the generated query.
  - CCDS uses the same single root constant (https://github.com/drivendataorg/cookiecutter-data-science/blob/master/%7B%7B%20cookiecutter.repo_name%20%7D%7D/%7B%7B%20cookiecutter.module_name%20%7D%7D/config.py).
  - TigerGraph warns that schema changes invalidate queries (https://www.tigergraph.com/docs/gsql-ref/4.2/ddl-and-loading/modifying-a-graph-schema), so `mule install` applies the scope schema change before installing any query.
  - A numbered migrations folder, in the style of Flyway's versioned migrations (https://documentation.red-gate.com/fd/versioned-migrations-273973333.html), comes back only if a second schema change appears.
- **Packaging.**
  - Keep the src layout (https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) and pip.
  - Make torch and matplotlib core dependencies; remove the `model` and `all` extras; keep `dev`, `cuda12` and `cuda13`.
  - Configure pytest with `--import-mode=importlib`, and drop `pythonpath` and basedpyright's `extraPaths` (https://docs.pytest.org/en/stable/explanation/goodpractices.html, https://docs.pytest.org/en/stable/explanation/pythonpath.html).
  - Do not commit a lock file: CUDA torch wheels come from per-CUDA indexes. Provenance records the package versions instead.

### Naming conventions

| Thing | Rule | Examples |
|---|---|---|
| Packages, modules | Lowercase role nouns. Never `utils`, `common`, `helpers`, `live`, `temporal`, `v5` or `legacy`. No two modules with the same name (https://github.com/wemake-services/wemake-python-styleguide/blob/master/wemake_python_styleguide/constants.py; https://peps.python.org/pep-0008/) | `training/trainer.py`, `data/contexts.py` |
| Classes | CapWords, role suffix, no project prefix | `Trainer`, `Predictor`, `SavedModel`, `ResumeState`, `ContextSource`, `TorchNeighborSampler` |
| Ports and adapters | Port = role noun + `Reader`/`Fetcher`/`Executor`; adapter = `<Technology><Port>` (Cosmic Python's `SqlAlchemyRepository`); fake = `Fake<Technology>` | `ContextFetcher` / `TigerGraphContextFetcher` / `FakeTigerGraph` |
| Functions | Verbs for use cases and factories; `*_curve` returns arrays, `bootstrap_*` intervals, `plot_*` draws | `prepare_dataset`, `train_run`, `audit`, `capture_curve`, `plot_capture` |
| Constants | UPPER_CASE, defined once | `BUILT_IN_GROUPS`, `GRAPH_NAME`, `REQUEST_CAP` |
| Config keys | Section-qualified snake_case, no repeated section name, units as suffixes | `scope.id`, `loss.positive_weight`, `transport.max_outage_s` |
| Variants | "Variant" everywhere (never "arm"). `baseline`, `no_<mechanism>`, `drop_<group>`, `add_<group>`, or a control's own name | `no_attention`, `no_graph`, `drop_pair_history`, `add_rolling_windows`, `prior_weight` |
| Concepts | One name each: "dataset" (not cohort or preparation); "source id" is the identity of the data loaded into the graph; "audit" is the ground-truth report and `evaluate` the command that writes it; the context source parameter is always `contexts` | |
| Runs and figures | `results/<variant>/seed-<n>/`; `plots/<topic>_<figure>.png` | `audit_capture.png` |
| GSQL | A file is named after the responsibility its queries share; query names are verb-first snake_case with no prefix (the graph is dedicated) | `hub_accounts.gsql` defines `list_hub_accounts` |
| Tests | `tests/<package>/test_<module>.py`; markers `graph`, `graph_write`, `cuda` | `tests/sampling/test_cugraph_sampler.py` |
| Docs | kebab-case in the Diataxis folders (https://diataxis.fr/) | `docs/how-to/run-control-experiments.md` |

`tests/test_naming.py` checks file and folder names, Python identifiers (read from the AST), CLI commands, run paths and GSQL query names. It does not check prose.

**Renames**

| Current | New |
|---|---|
| `mule-temporal`, `mule_pattern_learner.temporal.live.*` | `mule`, the layered packages |
| `LiveTGAT` with `split`, `summary`, `single` | `TGAT` (`tgat`), `SummaryMLP` (`summary`); `single` deleted |
| `variant` (`temporal`, `no_fourier`, `tabular`) | deleted; variants `drop_time_encoding` and `no_attention` |
| `DEFAULT_RUN`, `FALLBACKS`, `OPERATIONAL_DEFAULTS`, `TRANSPORT_DEFAULTS`, `LiveConfig` | `RunConfig`, `DEFAULT_CONFIG` |
| `make_live_batch`, `live_executor`, `prepare_live`, `pipeline.run` | `build_batch`, `pipeline.connect`, `prepare_dataset`, `train_run` |
| `cohort` (the proposal's word), `L/cohort.py` | dataset, `data/accounts.py` |
| `dataset_id` (the identity of the data loaded into the graph) | `source_id` |
| `TemporalPredictor`, `_TrainingRun.score` | `Predictor` |
| `ModelCheckpoint`, `checkpoint_last.pt` | `SavedModel` (`model.pt`), `ResumeState` (`resume.pt`) |
| `StreamingContextSource`; `store`, `source`, `contexts` | `ContextSource`; `contexts` |
| `GraphEvaluationTruth`, `ParquetEvaluationTruth`, `evaluate_final_population`, `evaluate-final` | `TigerGraphTruth`, `ParquetTruth`, `audit`, `mule evaluate` |
| `grouped_ap_interval`, `weighted_top_fractions`, `evaluate_weighted` | `bootstrap_intervals` (stratified or ring-clustered), `capture_at_budgets`, `audit_metrics` |
| `sampler.py`/`sampling.py`, `memory.py`, `policy.py`, `contract.py`, `config_schema.py`, `common.py` | `sampling/`, `training/schedule.py`, `batching/limits.py`, split up, `contract/`, `config.py`, split up |
| `CONTRACT_VERSION` | `CONTEXT_CONTRACT` in `contract/server.py` (value unchanged until the server step) |
| `TRAINING_PROTOCOL`, `CHECKPOINT_FORMAT` | `SavedModel.FORMAT = 1`, `ResumeState.FORMAT = 1` |
| `progress.jsonl`, `metrics.json["history"]`, `<split>_predictions.parquet`, `final_eval.*` | `history.csv` plus `events.jsonl`, `epochs.csv`, `predictions/<split>.parquet`, `audit/<split>.*` |
| `models/temporal/model.pt`, `<run>/prepared`, `artifacts/temporal/<id>` | `results/baseline/seed-42/model.pt`, `data/<dataset id>/` |
| Queries `temporal_*`, `*_pair_time64` (server step) | the verb-first names in the owner decisions' query table; the `temporal_fourier64` wrapper and `temporal_training_population` are retired |

**Kept on purpose** (allow-listed with reasons in `tests/test_naming.py`):
- the `Temporal_Training_Scope` vertex type and its edges;
- the scope id `strict_mule_v2`;
- the salt values in `contract/salts.py` (`"temporal_live_step"`, the reservoir and split salts);
- until the server step: the query names and the `CONTEXT_CONTRACT` value.

The constants that hold these values get new names; only the persisted values stay.

`CONTEXT_CONTRACT` stays a literal in `contract/server.py`. From the server step on, a render test asserts that it equals `"context_" + sha256(normalised rendered query without the literal)[:12]`, so a query change cannot ship without a new value.

### Plots

**Principles.**
- Every figure function has the shape `plot_<thing>(ax: Axes, data) -> Axes`. It takes computed inputs, never reads files and never saves. This is matplotlib's object-oriented style (https://matplotlib.org/stable/users/explain/quick_start.html) and PyKEEN's shape.
- `reporting/report.py` alone reads files and saves figures.
- Figures are built with `matplotlib.figure.Figure()` and saved through the Agg canvas, so pyplot is never imported (https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html, https://matplotlib.org/stable/users/explain/figure/backends.html).
- Curves come from `metrics.py` (sklearn curves with `sample_weight`). sklearn Display classes are not used, because they import pyplot.
- Output is PNG at 150 dpi with fixed colours; the baseline, mule and non-mule colours never change.
- **When figures are written:**
  - after `model.pt` and every run file are saved (so a plotting error cannot lose a model, and it makes the command exit non-zero only after everything else is written);
  - after audits;
  - by the runner;
  - by `diagnose`.
- `report.md` holds the tables with relative image links.
- Findings recorded in `docs/research/` embed PNGs copied into `docs/research/figures/`, which are tracked.

| File | Function | Source | What it shows |
|---|---|---|---|
| `training_objective.png` | `plot_objective` | history.csv | Loss and unclamped nnPU objective per interval, rolling mean, epoch boundaries |
| `training_corrections.png` | `plot_corrections` | history.csv | Share of steps whose non-negative correction fired |
| `validation_ranking.png` | `plot_validation_ranking` | epochs.csv | Proxy AP and ROC AUC per epoch, selected epoch, prevalence line, which weights were validated |
| `training_throughput.png` | `plot_throughput`, `plot_context_counts` | history.csv | Seconds per step and batch wait; below, contexts requested, distinct and cached |
| `proxy_precision_recall.png` | `plot_precision_recall` | predictions/*.parquet | Validation and test PR on observed labels, titled as a proxy |
| `run_health.png` | `plot_run_health` | metrics.json, history.csv | Rejections by split and status, stub children, sampler backend totals, database calls |
| `audit_precision_recall.png` | `plot_precision_recall` | audit/*.parquet, *.json | Weighted PR per split, chance line, AP with its ring-clustered 90% interval |
| `audit_roc.png` | `plot_roc` | audit/*.parquet | Weighted ROC per split with AUC |
| `audit_capture.png` | `plot_capture` | audit/*.parquet | Cumulative gains per split, random and perfect lines, markers at 1%, 5% and 10% labelled with recall and precision |
| `audit_threshold.png` | `plot_threshold_metrics` | audit/test.parquet | Weighted precision, recall and F1 against threshold, the selected threshold |
| `audit_score_distribution.png` | `plot_score_distribution` | audit/test.parquet | Weighted densities of log10 odds, mules against non-mules, threshold line |
| `audit_revealed_hidden.png` | `plot_revealed_vs_hidden` | audit/test.parquet | ECDF of the rank from the top (one minus the percentile rank, on a log axis), revealed against hidden mules, points and medians |
| `comparison_ap.png` | `plot_comparison` | summary.csv, comparison.csv | Validation-audit and test-audit AP per variant (two panels): one dot per seed, seed mean, interval, baseline line |
| `comparison_delta.png` | `plot_paired_delta` | comparison.csv | Validation-audit AP minus baseline: paired interval, per-seed deltas, zero line, sorted |
| `comparison_budget.png` | `plot_budget_recall` | summary.csv | Recall at 1%, 5% and 10% per variant (validation audit) |
| `comparison_capture.png` | `plot_capture_overlay` | audit parquets | Seed-mean capture curves, baseline emphasised |
| `comparison_validation.png` | `plot_validation_overlay` | epochs.csv files | Seed-mean proxy AP per epoch per variant |
| `comparison_proxy_vs_audit.png` | `plot_proxy_vs_audit` | epochs.csv, audit/validation.json | Selected proxy AP against validation-audit AP per run: is the proxy informative? |
| `label_curve.png` | `plot_label_curve` | learning_curve.csv | Test audit AP against oracle-labelled training mules (log x), LR and HGB bands, model line, revealed-count marker |
| `univariate_auc.png` | `plot_univariate` | univariate.csv | Weighted ROC AUC per feature and split, top 30 |
| `drift.png` | `plot_drift` | drift.csv | Standardised mean difference of non-mule features at validation and test cutoffs against train |
| `baselines.png` | `plot_baselines` | baselines.csv | AP with intervals per baseline family, single-feature rankings, the model |
| `ap_concentration.png` | `plot_ap_concentration` | subgroups.csv | Cumulative AP against number of top-ranked mules |
| `ring_coverage.png` | `plot_ring_coverage` | subgroups.csv | Share of test rings with a member in the top 1%, 5% and 10% |
| `proxy_validity.png` | `plot_proxy_validity` | proxy_validity.csv | Oracle metrics of the proxy predictions: all, hidden only, revealed only |
| `reveal_spread.png` | `plot_reveal_spread` | reveal_spread.csv | Per-split reveal outcomes over salts |
| `nnpu_simulation.png` | `plot_nnpu_simulation` | nnpu_simulation.csv | Simulated ranking quality and collapse rate, prior against balanced positive weight |

### Experiments

**Variants are frozen dataclasses of changes against the built-in run.** Seeds are a separate axis. A variant never sets a seed, and `dataset.seed`, `dataset.split_seed` and `scope.reveal_salt` are explicit fields that no variant touches.

```python
@dataclass(frozen=True)
class Variant:
    name: str
    question: str
    change: Callable[[RunConfig], RunConfig]          # built with dataclasses.replace

    def config(self, base: RunConfig, seed: int) -> RunConfig:
        return with_seed(self.change(base), seed)     # sets training.seed only

ACCOUNT_AGGREGATES = ("entity_meta", "entity_age", "rolling_windows", "amount_ratios", "recency",
                      "decayed_activity", "history_support")      # server-computed from the account's own events
BASELINE = Variant("baseline", "The built-in run", lambda c: c)
CONTROLS = (
    Variant("no_attention", "Does attention over sampled neighbours add anything beyond the root's own inputs, "
            "pool counts included?", lambda c: with_model(c, architecture="summary", slot_sum=False)),
    Variant("no_graph", "How well does a table of the account's own activity aggregates rank mules, with no "
            "neighbour, association or pool input?",
            lambda c: with_model(with_groups(c, ACCOUNT_AGGREGATES), architecture="summary", slot_sum=False)),
    Variant("no_slot_sum", "Does the per-slot MLP sum help beyond attention?", lambda c: with_model(c, slot_sum=False)),
    Variant("no_pool_counts", "How much of the ranking comes from the candidate-pool counts?",
            lambda c: without_groups(c, POOL_GROUPS)),
    Variant("prior_weight", "Does the balanced positive weight beat textbook nnPU across seeds?",
            lambda c: with_loss(c, positive_weight="prior")),
    Variant("no_weight_average", "Does selecting on the moving average of the weights help?",
            lambda c: with_training(c, weight_average_decay=0.0)),
    drop_group("time_encoding"),
)
FEATURE_DROPS = tuple(drop_group(g) for g in BUILT_IN_GROUPS if g != "message_core")   # drops dependents too
FEATURE_ADDS = tuple(add_group(g) for g in OPTIONAL_GROUPS)                           # adds requirements too
SUITES = {"controls": (BASELINE, *CONTROLS),
          "feature_drops": (BASELINE, *FEATURE_DROPS),
          "feature_adds": (BASELINE, *FEATURE_ADDS)}
SUITES["all"] = unique_by_name(*SUITES.values())
SEEDS = (42, 43, 44)      # 42 is the built-in seed: `mule train` is baseline seed 42 and is reused
```

**What the generated variants are.**
- **Drops:** `drop_entity_meta`, `drop_hub_indicator`, `drop_time_encoding`, `drop_pair_history` (also removes both pool groups; its question says so), `drop_flow_timing` (also removes `pool_activity`), `drop_pool_activity` and `drop_pool_internal_inflows` (replaces `/tmp` `no_internal.toml`).
- **Additions:** `add_entity_age`, `add_rolling_windows`, `add_amount_ratios` (adds `rolling_windows`), `add_recency`, `add_association_counts`, `add_pair_window_counts`, `add_decayed_activity`, `add_history_support`, `add_identity_order`, `add_device_ip_context`, `add_event_channel` and `add_sampler_meta`.
- **Replaced `/tmp` files:** `tabular.toml` becomes `no_attention`, and `seed7.toml` is covered by the fixed seeds.
- **Cost:** `no_attention` and `no_graph` fetch no children, so they cost about a fifteenth per batch.

**The script** (`scripts/run_experiments.py`, about 30 lines, no flags besides `--help`):
```
python scripts/run_experiments.py                          # suite "controls", seeds 42 43 44
python scripts/run_experiments.py feature_drops            # a suite by name
python scripts/run_experiments.py no_attention no_graph    # chosen variants (baseline always included)
python scripts/run_experiments.py --help                   # suites, variants, their questions and config changes
```
Before training it validates every variant offline and prints the run matrix with a time bound
from the last run's `history.csv`. Completed runs are skipped, runs whose settings differ are
moved to `results/archive/` (never deleted), and the comparison tables and plots are always
rewritten.

**What `run_suite` does, in order.**
1. **Resolve the variants.** Take the named suites and variants (suite `controls` when none are named), and always include `baseline`.
2. **Validate offline.** Build every variant's `RunConfig`, `FeaturePlan` and model on CPU. Every variant must share one dataset id and have a distinct fingerprint. Failures name the variant.
3. **Connect once**, run `pipeline.prepare` once (install, scope, reveal and dataset, as `mule train` does), and read the oracle truth once.
4. **Train, seeds outer and variants inner.** Each run goes to `results/<variant>/seed-<n>/`.
   - A complete run with an equal `fingerprint()` is skipped.
   - A differing one is moved to `results/archive/`, with the differing keys printed, and trained again.
   - An error specific to one variant is recorded as `failed` and the suite continues.
   - A TigerGraph outage (the executor's availability budget exhausted) stops the suite.
   - Exit status is 1 if anything failed.
5. **Audit** validation and test for every complete run that lacks them. The audit sample depends only on scope, truth and `split_seed`, so every variant is scored on the same accounts.
6. **Compare.** Write `summary.csv` (long format: variant, seed, split, metric, value, status, commit) and `comparison.csv`.
   - `comparison.csv` holds, per variant: the question, seeds, validation and test AP means, spread over seeds, intervals, paired validation delta, ROC AUC, recall and precision at the three budgets, mean best epoch, parameter count and training hours.
   - It flags runs that differ in git commit, dirty state, device or sampler backend.
   - Layout follows Ludwig's `MetricDiff` (https://github.com/ludwig-ai/ludwig/tree/main/ludwig) and GADBench's mean and spread over fixed seeds (https://github.com/squareRoot3/GADBench/tree/master).
7. **Report** with `reporting.write_suite_report`. `report.md` ranks variants by the validation audit. It marks the test audit "for reporting, not selection", and warns that the pool groups were designed after reading test-split mules.

**Statistics.**
- **Per run, per split:** weighted AP, ROC AUC, and recall and precision at the budgets. Each has a 90% bootstrap interval with 1,000 replicates and seed 0.
  - Positives are resampled by ring (`ring_id`), and negatives within class. Inclusion weights are kept.
  - The level and replicate count come from `/tmp/mpl_arms/audit_summary.py`.
- **Paired delta:** every replicate draws one resample of the shared accounts and rings and applies it to every run (`metrics.paired_replicates`). The statistic is seed-mean AP of the variant minus seed-mean AP of the baseline.
  - The interval covers audit-sample uncertainty for these seeds, not seed-to-seed variation. The per-seed deltas are plotted beside it.
  - A variant is marked "consistent" only when every seed's delta has the same sign and the interval excludes zero.
  - With about 18 variants at 90%, about two will exclude zero by chance, so results are exploratory until repeated with more seeds.
- **Rejections:** accounts rejected in any run are left out of the pairing, and their count is reported.
- **Sample size:** the diagnostic sample held 233 mules across the three splits (`mule_profile.md`), so expect wide intervals.

**Cost.**
- Measured on the CUDA host: about 3 s per step, and about 1 hour per graph run with early stopping.
- The `controls` suite over three seeds is 24 runs: 18 graph runs and 6 summary runs. That is roughly a day back to back without the cache.
- The script prints this bound from the last run's `history.csv` before it starts.
- The context cache lands before the first suite (see the migration plan).

**Tests.**
- `tests/experiments/test_variants.py` is parametrised over every variant. Each must build a valid config, plan and model offline, share the baseline's dataset id, and have a distinct fingerprint. This follows PyKEEN's `test_experiment_integrity.py` (https://github.com/pykeen/pykeen/blob/master/tests/test_experiment_integrity.py).
- `tests/experiments/test_runner.py` runs 2 variants × 2 seeds × 1 epoch on `FakeTigerGraph` and checks:
  - the run files, both CSVs and the plots;
  - skip-if-done;
  - archiving of a run whose settings differ;
  - stop on outage.

No experiment writes to `/tmp`.

### Diagnostics

**The rule.**
- An analysis becomes a module and a `mule diagnose` command when it should rerun whenever the dataset, features or model change.
- It becomes a research note when it answered a one-off question, or when the path it tested is gone or has become a variant.

**How the modules behave.**
- They read the graph only through ports and use truth only through `TruthReader`.
- They write to `results/diagnostics/<dataset id>/`, with CSVs in the long format of `summary.csv`.
- `reveal_spread` may import `reference`.

| Source | Becomes | Notes |
|---|---|---|
| `mpl_diag/stage_*.py`, `common.py`, `probe.py` | `diagnostics/feature_table.py`, `mule diagnose features` | Public functions of `evaluation.sample`, `ContextSource` and `batching.features` instead of `source._canonical`; train, validation and test samples |
| `mpl_diag/bl_lib.py` | `metrics.py`; `split_rank_transform` to `diagnostics/drift.py` | removes the duplicate metrics |
| `bl_models.py` A | `diagnostics/baselines.py` | adds single-feature rankings and the attribute-only floor |
| `bl_models.py` B | `diagnostics/learning_curve.py` | |
| `bl_models.py` D, `bl_shift.py` | `diagnostics/drift.py` | |
| `bl_models.py` A3, C, D1 | `docs/research/diagnostic-study.md` | one-off answers recorded |
| `bl_univariate.py` | `diagnostics/univariate.py` | |
| `bl_subgroup.py`, `mpl_arms/audit_summary.py` (revealed/hidden part) | `diagnostics/subgroups.py` | adds ring coverage |
| `mpl_arms/audit_summary.py` (intervals) | `evaluation/audit.py` | non-tie-aware `top()` replaced by `metrics.capture_at_budgets`; ring-clustered |
| dropped `evaluate` command | `diagnostics/proxy_validity.py` | oracle metrics of the proxy predictions, hidden and revealed |
| `bl_report.py` | `reporting/report.py` (`write_diagnostics_report`) | |
| `pool_activity_check*.py`, `pool_activity_offline.py`, `pool_activity_passthrough.py`, `pool_activity_check.md` | `docs/research/diagnostic-study.md` | superseded by `univariate`, `baselines` and the `drop_pool_*` variants |
| `binormal_ap.py` | `docs/research/diagnostic-study.md` | expected-AP reasoning recorded |
| `mpl_diag/profile/p1` to `p11`, `load_messages.py`, `mule_profile.md` | `docs/research/mule-profile.md` | records the typology findings that shaped the pool groups, and the resulting test-audit optimism |
| `extract_notes.md`, `shift_review.md`, `baselines.md`, `bl_tables.md`, `bl_template.md` | the two research notes | numbers copied in |
| `flags_check.py`, `head_src/`, `mpl_arms/fake/` | nothing | a HEAD-versus-tree flag check superseded by the variant tests; a source copy; a fixture |
| `mpl_arms/*.toml` | variants `no_attention`, `drop_pool_internal_inflows`; the fixed seeds | files not moved |
| `nnpu_sim/sim.py`, `grid.py`, `traj.py` | `diagnostics/nnpu_simulation.py` (offline) and `docs/research/nnpu-positive-weight.md` | the live test is `prior_weight` |
| `scripts/temporal/simulate_label_reveal.py` | `diagnostics/reveal_spread.py` | uses `reference/label_reveal.py` |

### Mapping from current files to new homes

**Python modules** (`L/` = `src/mule_pattern_learner/temporal/live/`)

| Current | New home |
|---|---|
| `__init__.py`, `py.typed` | same place |
| `configuration.py` | `paths.py`; TOML/JSON loaders deleted |
| `device.py` | `runtime/device.py`; `cli.py:14` deleted (entry points call `reserve_deterministic_cublas()` first) |
| `tigergraph/__init__.py`, `settings.py`, `client.py` | `tigergraph/__init__.py`, `connection.py`, `client.py` |
| `temporal/__init__.py`, `L/__init__.py` | deleted |
| `temporal/common.py` | `contract/clock.py`, `contract/fingerprints.py`, `artifacts.file_digest` |
| `temporal/encoding.py` | `contract/time_basis.py` (numpy, `BASIS_ID`), `batching/time_encoding.py` (torch) |
| `temporal/loss.py` | `model/loss.py` |
| `temporal/metrics.py` | `metrics.py`, merged with the weighted metrics of `L/evaluation.py`; `grouped_ap_interval` becomes the clustered bootstrap |
| `L/contract.py` | `contract/server.py`, `graph_schema.py`, `feature_groups.py` (all groups; pool constants), `sampler_plan.py`, `bounds.py`, `fingerprints.py`; `from_config` deleted; `extraction_plan` kept in `feature_groups.py` (the groups the context source requests: the model's without the client-computed ones); `LEGACY_GROUPS`, `FEATURE_NAMES` ordering, `HopBound`, positional pool arguments and `per_relation` deleted |
| `L/config_schema.py` | `config.py`; `FALLBACKS` and `OPERATIONAL_DEFAULTS` deleted |
| `L/executor.py` | `tigergraph/executor.py`; `transport_settings`/`live_executor` to `pipeline/connect.py` |
| `L/installation.py` | `tigergraph/gsql_text.py`, `installer.py`, `provenance.py`; installs through the executor, not `executor.client.conn` |
| `L/queries.py` | `tigergraph/render.py` |
| `L/context_query.py` | `tigergraph/context_query.py`; `query_context_batch` to `testing/builders.py` |
| `L/source.py` | `data/contexts.py`; `_DaemonPool` to `runtime/workers.py`; `ContextStore` deleted; `open_context_source` to `pipeline/connect.py`; `rejection_summary` to `inference/rejections.py` |
| `L/scope.py` | `tigergraph/scope.py`; `strict_mule_v1`-era inference deleted |
| `L/cohort.py` | `data/accounts.py`; stale-query guards deleted |
| `L/policy.py` | `context_scope` to `contract/graph_schema.py`; rejection limit to `inference/rejections.py`; protocol validation deleted |
| `L/labels.py` | `tigergraph/reveal.py`, `tigergraph/labels.py`; `ACCOUNT_LOAD_COLUMNS` to `contract/graph_schema.py` |
| `L/reveal_model.py` | `reference/label_reveal.py` |
| `L/supervision.py` | `data/observed_labels.py`; graph source to `tigergraph/labels.py`; `FrameObservedLabels` and the parquet policy deleted (every run reads the graph's labels) |
| `L/hubs.py` | `data/hub_registry.py`, `tigergraph/hubs.py`; `HUB_COLUMNS` once in contract; `scan_cap` guard deleted |
| `L/dataset.py` | `data/manifest.py`, `preparation.py`, `splits.py`; legacy export, union-find split, SQLite fill and `preparation_fingerprint` deleted |
| `L/batching.py` | `batching/features.py`, `pool_counts.py`, `assemble.py`; backend policy to `sampling/backend.py`; scalar features to `reference/batch_features.py`; legacy selection deleted; `PinnedRoots.__getattr__` replaced by fields |
| `L/memory.py` | `batching/limits.py` |
| `L/sampler.py` | `sampling/candidates.py`, `torch_sampler.py`, `cugraph_sampler.py`, `backend.py` |
| `L/sampling.py` | `training/schedule.py`; `BatchPrefetcher` merged into `runtime/workers.py` |
| `L/model.py` | `model/inputs.py`, `tgat.py`, `summary_mlp.py` (module names stay unique beside `training/summary.py`), `build.py` (submodule creation order preserved) |
| `L/training.py` | `training/trainer.py`, `objective.py`, `averaging.py`, `history.py`, `summary.py`; scoring to `inference/predictor.py` |
| `L/checkpoint.py` | `inference/saved_model.py`, `training/checkpoint.py`; `_result_view` deleted |
| `L/predictor.py` | `inference/predictor.py`, `inference/score_accounts.py` |
| `L/inference.py` | deleted (training writes split predictions; `mule score` covers accounts) |
| `L/evaluation.py` | `evaluation/sample.py` (any split), `audit.py`, `truth.py`; graph truth to `tigergraph/oracle.py`; `evaluate_predictions` to `diagnostics/proxy_validity.py` |
| `L/pipeline.py` | `pipeline/` package; path policy to `paths.py` |
| `L/experiments.py` | `experiments/variants.py` |
| `L/history_reference.py` | `reference/gsql_features.py` |
| `L/cli.py` | `cli.py` |
| new | `__main__.py`, `data/ports.py`, `data/context_cache.py`, `artifacts.py`, `runtime/progress.py`, `reporting/`, `experiments/runner.py`, `comparison.py`, `diagnostics/`, `testing/` |

**Scripts** (`scripts/temporal/`)

| Current | New home |
|---|---|
| `run_live_experiments.py`, `feature_experiments.py` | `scripts/run_experiments.py` |
| `render_training_queries.py` | `scripts/render_queries.py` |
| `benchmark_live_batch.py` | `mule check` (it gained digest output in the safety-net step) |
| `verify_live_training.py` | `mule check`; `tests/integration/test_context_query.py` |
| `verify_cugraph_sampler.py` | probe in `mule check`; `tests/integration/test_cugraph_sampler.py` (`cuda`) |
| `verify_strict_isolation.py` | `tests/integration/test_scope_isolation.py` (`graph_write`, `--allow-graph-writes`) |
| `verify_feature_redesign.py` | `tests/integration/test_feature_parity.py` (`graph`) |
| `verify_label_reveal.py` | `tests/integration/test_label_reveal.py` (`graph`, `apply=FALSE`) |
| `simulate_label_reveal.py` | `diagnostics/reveal_spread.py` |
| `convert_mule_label_to_integer.py`, `install_account_supervision.py`, `install_time_encoding.py`, `verify_account_supervision.py`, `verify_time_encoding.py` | deleted (applied migrations; in git history) |

**GSQL**

| Current | New home |
|---|---|
| `gsql/README.md` | rewritten; fresh-graph procedure to `docs/how-to/set-up-a-graph.md` |
| `features/temporal_fourier64.gsql` | `queries/fourier64.gsql` (text unchanged until the server step, which drops the wrapper) |
| `features/zelle_pair_time64.gsql`, `payment_pair_time64.gsql` | `analytics/zelle_pair_gaps.gsql`, `analytics/payment_pair_gaps.gsql`; installed only by the command that uses them, never by `mule install` or `mule train` |
| `schema/temporal_schema.gsql`, `temporal_account_loading.gsql` | `schema/schema.gsql`, `schema/account_loading.gsql` (the unused `pair_*` attributes stay, documented as unused) |
| `schema/migrations/temporal_training_scope.gsql` | `schema/scope_vertex.gsql` |
| `schema/migrations/temporal_valid_time.gsql`, `temporal_encoding_attributes.gsql`, `account_mule_supervision.gsql` | deleted (applied; in the fresh DDL) |
| `temporal/training_context.gsql` | `queries/training_context.gsql` (byte-identical until the server step) |
| `temporal/training_scope.gsql`, `label_reveal.gsql` | `queries/`, same file names |
| `temporal/training_cutoffs.gsql`, `hub_registry.gsql` | `queries/split_cutoffs.gsql`, `queries/hub_accounts.gsql` |
| `temporal/account_supervision.gsql` | split: `evaluation/ground_truth.gsql` (the oracle read) and `queries/label_contract.gsql` (the label-contract validation) |
| `temporal/training_population.gsql` | deleted from the tree with `shared_history`; the installed query is retired in the server step |

GSQL files move, split and take these names in the move step with every query's text unchanged, so
nothing is reinstalled. The server step renames the queries inside them.

**Tests** (`tests/temporal/`)

| Current | New home |
|---|---|
| `conftest.py` | `tests/conftest.py` with fixtures; `sys.path` insert deleted |
| `temporal_fakes.py` | `testing/fake_graph.py` (`FakeExecutor` becomes `FakeTigerGraph`, absorbing the inline executors), `testing/builders.py`; `LEGACY_PROFILE` and `profile_config` deleted; test config defaults to `DEFAULT_CONFIG` |
| `test_transport.py` | `tests/tigergraph/test_client.py`, `test_executor.py`, `test_context_query.py`, `test_installer.py`, `test_scope.py`; `tests/data/test_contexts.py`, `test_accounts.py`, `test_observed_labels.py`; `tests/pipeline/test_connect.py`; server fakes to `testing/fake_connection.py` |
| `test_sampler.py` | `tests/sampling/*`, `tests/batching/test_assemble.py`; `recent`/`stratified` tests deleted |
| `test_training_runtime.py` | `tests/training/*`, `tests/runtime/*`, `tests/evaluation/test_audit.py`, `tests/inference/test_score_accounts.py`, `tests/test_cli.py` |
| `test_gsql_v5.py` | `tests/tigergraph/test_render.py`, `test_gsql_text.py`, `test_query_files.py` |
| `test_live_training.py` | `tests/pipeline/test_train.py`, `tests/training/test_golden_run.py` |
| `test_live_pipeline.py` | `tests/contract/test_feature_groups.py`, `tests/data/test_manifest.py`, `tests/model/test_build.py`, `tests/experiments/test_variants.py` |
| `test_pool_activity.py` | `tests/batching/test_pool_counts.py` |
| `test_strict_backend.py` | `tests/data/test_scope_isolation.py` |
| `test_slot_sum.py` | `tests/model/test_tgat.py`, `tests/experiments/test_variants.py` |
| `test_scripts.py` | `tests/test_scripts.py` |
| `test_feature_redesign.py` | `tests/reference/test_gsql_features.py`, `tests/test_metrics.py` |
| `test_label_reveal.py` | `tests/tigergraph/test_reveal.py`, `tests/reference/test_label_reveal.py` |
| `test_loss.py`, `test_score_precision.py` | `tests/model/test_loss.py`, `tests/model/test_build.py`, `tests/test_metrics.py` |
| new | `tests/reference/test_batch_features.py` (vectorised against scalar), `test_naming.py`, `test_layers.py`, `test_config.py`, `test_doc_links.py`, `tests/reporting/`, `tests/experiments/test_runner.py`, `tests/diagnostics/`, `tests/integration/` |

The five message and payment builders, two `Hubs` fakes, two `hub_registry` helpers and two `live_config` helpers merge into `testing/builders.py`.

**Docs**

| Current | New home |
|---|---|
| `temporal_training_end_to_end.md`, `live_temporal_training.md` | README quickstart, `how-to/train-and-evaluate.md`, `explanation/training.md`, `explanation/sampling.md`, `reference/configuration.md`; "Upgrading earlier preparations" deleted |
| `gsql_feature_catalog.md` | `reference/features.md` (the built-in run's groups; the others with the analytics query), `reference/queries.md` |
| `feature_redesign.md` | `explanation/feature-design.md`; commands replaced by `how-to/run-control-experiments.md`; "Validation status" deleted |
| `leakage_and_scaling.md` | `explanation/leakage-and-scaling.md` (adds the context cache) |
| `temporal_schema.md` | `reference/schema.md` |
| `temporal_encoding.md` | `explanation/time-encoding.md` |
| `account_mule_labels.md` | `reference/labels.md` |
| `label_reveal.md` | `explanation/label-reveal.md` |
| `feature_plan_v4.md`, `temporal_gsql_review.md` | deleted |
| new | `architecture.md`, `how-to/set-up-a-graph.md`, `score-new-accounts.md`, `run-diagnostics.md`, `reference/cli.md`, `reference/outputs.md`, `research/*.md` and `figures/` |

**Top level and local files**

| Current | New |
|---|---|
| `pyproject.toml` | as under Configuration (no lock file) |
| `README.md` | rewritten |
| `.gitignore`, `LICENSE`, `.env.example` | unchanged |
| `models/temporal/` (local) | keep until the new baseline has an audit, then delete by hand |
| `models/*.pt` (main era), `artifacts/`, `walkthrough.ipynb`, `docs/*.json` (local) | delete by hand when convenient |

### Migration plan

**Working rules.**
- Branch: `git switch -c restructure temporal`.
- Gate for every commit: `.venv/bin/python -m ruff check`, `ruff format --check`, `basedpyright` and `pytest`. From the layered-tree step on, add `lint-imports` (or `tests/test_layers.py`).
- Commit messages carry no attribution trailers.
- A commit that deliberately changes numbers says so.
- Keep running old-code jobs (for example on the CUDA machine) on `temporal`. They are unaffected until the server step.

The steps, in order:

0. **Preserve the out-of-repo work now.**
   - Commit the `.py` and `.md` files from `/private/tmp/mpl_diag` (including `profile/`), `mpl_arms` and `nnpu_sim` to a side branch `archive/diagnostic-study` that is never merged. The data files are ignored and stay local.
   - Draft `docs/research/reference-run.md` from `models/temporal/metrics.json` and `final_eval.*`.
1. **Safety net.**
   - Add `test_golden_run.py`: the built-in profile on the fakes, CPU, deterministic, 2 epochs of 4 steps. It asserts exact sha256 digests of the first batch's tensors, loss and objective per interval (relative tolerance 1e-5), the selected epoch, the validation AP and the rendered query's sha256. The values are Python literals.
   - Switch `temporal_fakes.live_config` to the built-in profile. Mark the tests that still need a legacy path `@pytest.mark.legacy`.
   - Make `benchmark_live_batch.py` print the tensor digests and the first-step loss.
   - Tag this commit `pre-restructure`.
   - Gate: suite green, golden test green twice.
2. **Packaging.**
   - torch and matplotlib become core dependencies; import-linter goes into `dev`; remove `model` and `all`.
   - Add `mule` beside `mule-temporal` and add `__main__.py`.
   - pytest uses importlib mode and excludes the markers by default.
   - Gate: `pip install -e '.[dev]'`, suite, `python -m mule_pattern_learner --help`, and a check that import-linter runs on 3.14.
3. **Remove one-off material.** Delete the five migration scripts, the three applied migrations, the two obsolete docs, and their `test_scripts` entries. The pair_time64 queries stay, for `gsql/analytics/`. Gate: `git grep` finds no references.
4. **Remove legacy runtime paths**, one commit each, with the golden test identical after each:
   - SQLite storage;
   - `shared_history` and the population query file (it leaves `QUERY_FILES`, so live datasets need re-preparation but nothing is reinstalled);
   - the parquet label policy;
   - the `recent`/`stratified` samplers and their compatibility arguments;
   - the `single` architecture, the `variant` axis, `legacy_no_fourier`, `LEGACY_GROUPS` and the `FEATURE_NAMES` ordering (all groups stay);
   - `FALLBACKS`, the re-applied defaults, checkpoint back-compat and the upgrade guards;
   - scalar features moved to `reference/`;
   - `extraction_groups` out of the preparation key and the config.

   Gate for each: `git grep` for the removed names is empty.
5. **Move into the layered tree.**
   - Pure `git mv`, bottom-up. GSQL files move to the folders and names of the GSQL mapping with every query's text unchanged, and query names stay in `contract/server.py`.
   - Tests move with their modules; `temporal_fakes.py` becomes `testing/`.
   - Add the import contracts with an `ignore_imports` list of today's violations, each with its reason.
   - Gate: golden identical.
6. **Split mixed modules and remove duplication.** One commit per item, each deleting one ignore line or one duplicate:
   - the training god class;
   - one scoring loop;
   - one worker pool;
   - one metrics module;
   - one `atomic_write`;
   - one population pager;
   - hub columns once;
   - `contract/bounds.py`;
   - consumer-owned ports instead of `getattr`/`inspect` probes;
   - adapters built only in `pipeline`;
   - no environment mutation at import.

   Gate: golden identical; the ignore list ends empty.
7. **Typed configuration.** Add the `config.py` dataclasses with explicit `dataset.seed` and `scope.reveal_salt`, `fingerprint()`, and the sampler section as `SamplerPlan`. Delete pydantic `LiveConfig`, `run_config`, `setting()`, the loaders and `--config`. Gate: a test maps `DEFAULT_CONFIG` field by field to the old `DEFAULT_RUN` through an explicit table; golden identical.
8. **Run layout, renames and the `mule` CLI.**
   - `RunPaths`, `DatasetPaths`, `results/`, `data/<dataset id>/`, `history.csv` with the context counters, `epochs.csv`, `events.jsonl`.
   - Renamed identifiers (including `split` to `tgat`) and the new subcommands.
   - Remove `mule-temporal`; add `test_naming.py` with the server names allow-listed.
   - Update the README's commands in the same commit.
   - Gate: an offline end-to-end test asserts the run directory's file set; golden identical.
   - Names kept on purpose: `data.preparation.prepare` stays `prepare`, because `pipeline.prepare.prepare_dataset` is the use case callers run and two functions of one name would blur the layers. `_TrainingRun.score` keeps its name: it already runs the one scoring loop (`inference.predictor.score_batches`), which is what the renames table asks of it.
   - Left for a later code step: `build_batch` and `build_root_batch` still take `fanouts` beside the `sampler` section that holds them (both are required now, with the feature plan). Drop the parameter and read `sampler.fanouts` before the experiments step; about 50 test calls pass it.
   - Left for the server step: the stale-query guards the removed `legacy` marker tracked, `_check_label_fields` in `data/accounts.py` (its masked-label check) and `check_graph_label_rows` in `tigergraph/labels.py`. Delete them once that step has installed every query under its new name and re-prepared the dataset: from then on no installed query and no prepared dataset can predate the masked-label predicate.
   - Kept until the owner decides, in the server step: the scope-rule inference the marker also tracked (`inferred_scope_policy` in `tigergraph/scope.py`, from the `strict_mule_v1` era). The scope vertex stores no `scope.unowned` rule, so inferring it from the membership is the only check that an existing scope was created with the configured rule. Deleting it means storing the rule on the vertex (a schema change) or giving up that check.
   - Done in the audit step (see Audit additions): the fake graph absorbed the inline executors, and "final audit" became the ground-truth audit. The stub in `tests/pipeline/test_connect.py` stays: it replaces the `TigerGraphExecutor` class to record its constructor arguments and runs no query.
   - Left for the docs step: "arm" in `docs/feature_redesign.md` and `docs/temporal_training_end_to_end.md` (the owner's word is "variant"), and `<output>.rejected.txt` in `docs/live_temporal_training.md` (now `scores/<stem>_<date>_rejected.txt`). That guide also says to resume a run that stopped on rejected roots with a higher limit. The commands take no options, so only a Python caller can, and the guide must say how: `train_run(config=DEFAULT_CONFIG.with_changes({"runtime": {"max_rejected_root_fraction": 0.01}}), resume=True)` from `pipeline.train`.
   - Commits that fail the per-commit gate (the history is not rewritten, so bisect should skip them): `b0bbbaf`, `eeaf5d5` and `30d1461` of the deduplication step fail `ruff check --select I`, which `68e8297` fixes; `55bb769` and `90c0ed4` fail it too, which `6ad436f` fixes; `519d897` and `455654e` fail `tests/test_naming.py`, which `17cbaf3` fixes. Ruff's configuration now selects the import rules, so the plain `ruff check` catches this.
   - No experiment runner exists on `restructure` until the experiments step writes `experiments/variants.py`, `runner.py` and `scripts/run_experiments.py` (the old matrix `feature_experiments` is deleted, not ported). Until then control experiments run from `temporal`.
   - The mid-migration review (of the move, deduplication, typed-configuration and run-layout steps) settled, before later steps build on them: resume refuses a dataset other than the run's (`resume.pt` records its id and manifest sha256); config.json and the start and resume events record each segment's device, threads and determinism, and a changed one is a `host_settings` event; old saved configurations without fan-outs or sampler are refused; the graph and query names live in `contract/server.py`; every context source is built by `pipeline.connect.context_source`; every output line is an `emit` record with an `event` name (warnings and retries included) and the CLI prints its result on one line; audits and scoring run inside the saved runtime settings; `history.csv` counts `memory_hits`; `metrics.json` records `elapsed_seconds`.
9. **Live parity with unchanged queries (owner-run, read-only).**
   - (a) `mule check` reports every query up to date. The text did not change, so nothing is installed.
   - Then prepare the new dataset, which (b) compares and (c) needs: `mule check` refuses the batch without it, and `mule train` would go on to the hour-long baseline run, which waits for the server step. With the queries, the scope and the reveal already in place, preparing only reads the graph (about 6 minutes):
     `python -c "from mule_pattern_learner.config import DEFAULT_CONFIG; from mule_pattern_learner.pipeline.prepare import prepare_dataset; print(prepare_dataset(DEFAULT_CONFIG).root)"`
   - (b) The new dataset's accounts and observed labels have the same rows as the old preparation's.
   - (c) `mule check` prints the same digests and first loss as `benchmark_live_batch.py --train-step` at `pre-restructure`, on the same machine and device.
   - (d) Optional: the first three log intervals of `mule train` equal those of `mule-temporal train` at the tag. Stop both after 30 steps, and write the old run's output under `results/parity/`.
   - Models trained before the restructure (such as the current best model on the CUDA host) are audited from `temporal`: their datasets record no dataset settings, so this code refuses to audit them. The saved-model test shows that such models still load and give their recorded scores offline.
10. **Server rename** (owner decision, owner-run). Precondition: no old-code run is active anywhere.
    - Render with the new names, the derived `CONTEXT_CONTRACT`, and without the `fourier64` wrapper.
    - In the same render, fix the generated header of `gsql/queries/training_context.gsql` (written by `tigergraph/render.py`): it still names `scripts/temporal/render_training_queries.py`, which is now `scripts/render_queries.py`. Byte identity keeps it until then.
    - Run `mule install` (about 50 minutes; rerun if the 45-minute wait expires). The new dataset is re-prepared (about 6 minutes).
    - Repeat check (c): the digests must be unchanged.
    - Then drop the retired names, callers first: `mule install` drops every installed query that no file defines (owner decision), so this step gives it that behaviour and runs it.
    - Move the groups outside `BUILT_IN_GROUPS` out of the context query and `FEATURE_GROUPS` into a query under `gsql/analytics/` (owner decision), and confirm the split with the owner before installing.
    - The renames are the constants of `contract/server.py` (and the allow-list of `tests/test_naming.py`, which a test keeps equal to them); some test texts of GSQL still spell the old names and fail until updated.
    - From the mid-migration review: the installer still writes (the scope schema change, CREATE, `installQueries`) through `executor.client.conn` rather than the executor, because the executor's retry errors would hide the install timeout the installer polls on. Route the writes through the executor with one attempt when this step runs `mule install` live. After the dataset is re-prepared, consider comparing the manifest's query hashes by file again (`data.manifest.changed_query_files` matches on content alone so that files moved in the layered restructure stay valid).
    - Gate offline: render check; golden identical except the query hash literal; the allow-list shrinks to the vertex, scope id and salts.
    - If declined, skip this step.

    The baseline run (`mule train`, about an hour) can start once this step is done.
11. **Audit additions.**
    - Audits of any split; `revealed`, `ring_id` and `label_source` columns.
    - Ring-clustered intervals, curve arrays, tie-aware budgets everywhere, `diagnostics/proxy_validity.py`.
    - Gate: hand-computed small cases; audits on `FakeTigerGraph`.
    - From the mid-migration review:
      - `audit()` still audits the test split only and does seven jobs: give it the signature `audit(run, split, *, truth, scope, contexts)`, split it into `audit_population`, `score_sample` and `write_audit`, pass the loaded inputs in instead of loading them again (`pipeline.evaluate` and `audit` both call `audit_inputs`), define `AUDIT_NEGATIVES` once in `evaluation/sample.py`, reuse `data.splits.eligible_mask` and import at module level.
      - The proxy metrics' top 1 and 5% counts (`metrics.proxy_metrics`) still ignore ties; make them tie-aware with the audit's budgets.
      - `evaluate_predictions` is deleted; write `diagnostics/proxy_validity.py` anew.
      - `FakeTigerGraph` answers only five read queries and has no `call` or `gsql`: make it a `ConnectionExecutor` (SHOW QUERY, the endpoint listing, vertex counts, the scope header), and give the executor protocol a `graph_name` so `pipeline/check.py` stops reading `executor.client.graphname`.
      - Move the `capture_at_budgets` tests out of `tests/evaluation/test_audit.py` when the audit tests are rewritten, and merge the test builders (`testing/builders.py` has three configuration builders with three source ids, about a dozen message and context builders and several `SamplerPlan`s; `testing/sampler_checks.py` defines `message` again).
    - Settled in this step, for the steps after it:
      - `evaluation.audit.audit_inputs(run)` loads and checks the model, its dataset, the manifest and the hub registry once into an `AuditedRun`; `audit(run, split, *, truth, scope, contexts)` takes it, the truth table and the caller's context source, which the caller closes. `pipeline.evaluate.evaluate_run` returns the reports by split, audits only the splits whose `audit/<split>.json` is missing (the report is written last) and connects only if one is.
      - The truth table has `contract.graph_schema.TRUTH_COLUMNS` (`account_id, is_mule, ring_id, label_source`), read by `TigerGraphTruth` and checked by `evaluation.truth.checked_truth`. The report holds `split`, `purpose` (`decisions` or `reporting`), `date`, `population_accounts`, `metrics`, `intervals` (ring-clustered), `constants`, the revealed and hidden positives and the rejection counts. The rejection counts are the audit's own: `evaluation.audit.score_sample` reports what the shared context source served while it scored (`inference.rejections.SourceRejections`), so a split's report does not depend on whether another split was audited first.
      - The proxy metrics in `metrics.json` report the audit's review budgets with unit weights: 1, 5 and 10%, where a budget that ends inside an account or a block of tied scores takes a share of it. The commit that changed them lists the golden run's old and new values.
      - `FakeTigerGraph` is the one fake executor: `before` and `answers` hooks replace scripted executors, `delay` and `encodings` the context faults. `testing.builders.unit_config` is the one configuration builder (`unit_config(RUNTIME_CHANGES)` for the runtime tests), `UNIT_SOURCE` the one source id, and `ground_truth_rows` the truth of a scope population.
    - Left for later steps:
      - The plots step draws the audit figures from `metrics.precision_recall_curve`, `roc_curve` and `capture_curve`, and the report's intervals.
      - The experiments step computes paired deltas with `metrics.paired_replicates` and still has to give `evaluate_run` a shared connection and truth for a suite.
      - The diagnostics step adds `mule diagnose proxy-validity` over `diagnostics.proxy_validity.proxy_validity(run, truth)`, which returns the long table and writes nothing.
      - The fake graph does not answer the reveal or the label-contract query, so the pipeline's end-to-end test still replaces the reveal; `tests/tigergraph/test_reveal.py`, `test_labels.py` and `test_installer.py` and `tests/test_scripts.py` keep their own executors or connections for the queries and writes they script.
12. **Plots and reports.** `reporting/`, `mule report`, automatic plots. Gate: every figure smoke-renders from synthetic files to a non-empty PNG under its fixed name; the matplotlib and torch contracts pass.
    - Settled in this step, for the steps after it:
      - `reporting/style.py` holds the fixed colours (mule, non-mule, baseline, each audited split; the measures a figure compares take `MEASURES` in order, each with its own line style), the sizes, `DPI = 150` and the matplotlib settings, applied with `matplotlib.rc_context`. `training.py`, `ranking.py` and `scores.py` hold the plot functions; `ranking.SplitScores` is a split's scored accounts (unit weights for the proxy, inverse inclusion probabilities for an audit) with the metrics and intervals its file records, which the labels print, so a figure shows the recorded numbers.
      - `reporting/report.py` alone reads files and saves figures (`paths.RunPaths.figure(name)`). `pipeline.train.train_run` calls `write_training_report` after the trainer returns, and `pipeline.evaluate.evaluate_run` calls `write_audit_report` after the audits it ran; a complete run reported, or a run whose audits are all recorded, is left as it was. `mule report [RUN]` is `report_run`: the training figures of a complete run, the audit figures of the audited splits (those of the test split alone need its audit), then `report.md`. A figure that fails is recorded and loses any older PNG of it, the others and `report.md` are still written, and the call then raises naming every failed figure.
      - Scores sit near 0 and 1, so the threshold and density figures use log10 odds (within plus or minus 16, since a float64 score rounds to 1 above a logit of about 37), and the capture and revealed-and-hidden figures a log axis of the top share of accounts, labelled at the review budgets. The precision-recall steps keep only the blocks that add recall, so the area under them is the recorded AP.
      - `metrics.budget_name` names a review budget's metrics; `testing.builders.write_run_files` writes a complete, audited run shaped like the reference runs (47,749 test accounts, 40 mules, scores near 0 and 1) for the reporting tests.
    - Left for later steps: `mule report` reports a run directory; the experiments step adds `reporting/comparison.py` and suite directories (`write_suite_report`), and the diagnostics step `reporting/diagnostics.py` and diagnostics directories, each with its own figures from the Plots table.
13. **Context cache**, before any suite.
    - A disk tier inside `ContextSource` under `data/<dataset id>/contexts/`, with a size cap.
    - Keyed by hop, `ContextKey`, requested group flags, pool fingerprint, `CONTEXT_CONTRACT` and dataset id. It stores raw TigerGraph rows compressed, and the frozen-source check invalidates it.
    - The baseline run's `contexts_distinct` sets the cap.
    - Gate: golden identical with the cache on and off; hit rate in `metrics.json`.
    - From the mid-migration review: `ContextSource.fetch` writes its LRU inline and `_canonical` drops the encodings before caching, while this step stores raw rows; extract a cache-tier interface first. Add the disk tier's hits to `history.csv` beside `memory_hits`.
    - Settled in this step, for the steps after it:
      - `data/context_cache.py` holds the cache-tier interface `ContextTier` (`get` the rows a tier holds of some keys, `put` the rows one fetch obtained, `close`), `MemoryTier` (the LRU, moved out of `ContextSource.fetch` with its rules unchanged) and `DiskTier`. `ContextCache(directory, dataset_id, source, capacity)` names a dataset's cache, and `ContextCache.of(dataset, manifest)` puts it in `DatasetPaths.contexts`. `ContextSource(..., cache=...)` (and `build_context_source`, `pipeline.connect.context_source`) reads memory, then disk outside its lock, then TigerGraph; disk rows are served through `_canonical` like requested ones and both go into the LRU.
      - An entry is one gzip-compressed JSON file per context, `<first two hex digits>/<name>.json.gz`, holding its name and the fetcher's raw row (request position and spot-check vectors included; rejected contexts too). The name is the fingerprint of the hop, the `ContextKey`, `query_flags(hop)`, the fingerprint of `query_params(hop)`, `CONTEXT_CONTRACT`, the dataset id (`data.manifest.recorded_dataset_id`) and the frozen source (`data.manifest.source_fingerprint`: the manifest's counts, scope, source id and split seed). An entry that cannot be read, records another name, or holds no row with a status or an ok row of another key or contract is refused with a `context_cache_refused` warning and requested again; its entry is then replaced. A directory that cannot be written warns once (`context_cache_unwritable`) and is then only read.
      - The frozen-source check invalidates the cache in two ways: the source fingerprint names every entry, and only a connection `verify_frozen_source` has passed opens the cache (`open_context_source` for training and `mule check`, `evaluate_run` for the audits). `mule score` has no cache.
      - The cap is `contract.bounds.CONTEXT_CACHE_ENTRIES = 1_500_000`, estimated from the reference run (at most 1,088 contexts per step over 11 epochs of 100 steps, plus about 70,000 each for the proxy evaluations and the audits; roughly 10 GB). Beyond it the least recently used entries go (a hit refreshes an entry's time) until 90% remain.
      - `history.csv` has `disk_hits` after `memory_hits`, `metrics.json`'s `contexts` hold `disk_hits` and `disk_hit_rate` (the share of the contexts memory did not serve that the disk served; null when memory served all), `report.md` prints both, and `training_throughput.png` draws the disk hits as a fourth line in the palette's magenta (`MEASURES[3]`, dash-dot).
      - The golden run's literals hold with the cache off (the existing test), cold and warm (`test_the_golden_run_is_the_same_with_the_context_cache_cold_and_warm`); the warm run requests no context from the fake graph.
    - Left for later steps:
      - Set `CONTEXT_CACHE_ENTRIES` from the baseline run's `contexts.distinct` in `metrics.json`, the audits' contexts and the bytes per entry its `contexts/` directory shows.
      - The experiments step: seeds and the variants that request the baseline's groups share entries, training and audits alike. That includes the controls that change only the loss, the weight average or the slot sum, and the drops of client-computed groups (the hub indicator and both pool groups), which are never requested. A variant that requests other groups at a hop has entries of its own at that hop; TGAT children skip summary groups, so a change of summary groups still shares the children. Suites should still pass one connection and truth, as the audit step left.
      - The server step: the new `CONTEXT_CONTRACT` and the narrower flags of the owner decision on feature groups name new entries, so every context is requested once more; the old entries are never read again and age out under the cap. Removing `data/<dataset id>/contexts/` by hand reclaims the disk at once (the owner's call; nothing deletes it).
      - A run started before this step has a `history.csv` without `disk_hits`, and resuming it would append rows of another width; no such run exists outside tests, so there is no conversion.
      - The docs step: `docs/leakage_and_scaling.md` and the limits table of `docs/live_temporal_training.md` still give old request sizes and concurrency (16 contexts per request and a concurrency of 2 to 4, or 8), while the defaults are 8 and 16.
14. **Experiments.** `variants.py`, `runner.py`, `comparison.py`, `scripts/run_experiments.py` and the tests described under Experiments.
    - From the mid-migration review: the pipeline functions connect on every call (`connect()` in each use case, and `evaluate_run` reads the whole truth each time); give them an optional connection or session so a suite connects once, prepares once and reads truth once. The executor raises `TransientQueryError` both when the outage budget runs out and when a suspected-deterministic failure repeats; add a subclass for the outage alone (for example `TigerGraphUnavailableError`) so the suite stops on an outage but carries on past one variant's own error. The feature additions and `OPTIONAL_GROUPS` are gone (owner decision: only the built-in run's groups stay in training); say what becomes of `no_graph` and `feature_adds`.
15. **Diagnostics.** The modules, `mule diagnose`, and the research notes (with figures) filled from `archive/diagnostic-study`, after which `main` holds everything worth keeping from that branch. Gate: each analysis runs on a synthetic features table, and `feature_table` on `FakeTigerGraph` matches the batching features of the same keys.
    - From the mid-migration review: `scripts/simulate_label_reveal.py` runs its interpreted query through `client.conn`; `diagnostics/reveal_spread.py` replaces it and reads through the executor. The import contracts already name `diagnostics` (the ports, fakes and matplotlib contracts), so its modules are checked from their first commit.
16. **Docs.** The Diataxis tree and `architecture.md`. Gate: `test_doc_links.py` and the naming test.
    - From the mid-migration review: tell users that a dataset prepared before the restructure records no dataset settings, so training refuses it and it is prepared again; and choose one adapter naming rule in `architecture.md` (the naming table asks for `<Technology><Port>`, for example `TigerGraphScopeReader`, while the tree and the code use `TigerGraphScope` and `ParquetTruth`).
17. **Replace main.** Gate: the full offline gate, `pytest -m cuda` on the CUDA host, and read-only `pytest -m graph`. No push until the owner confirms.
    - First delete the old saved-settings conversion (owner decision), once the new baseline run has a ground-truth audit: `inference/saved_settings.py` and its test file, and the conversion in `SavedModel.config`, which then reads `RunConfig.from_dict` alone. The models saved before the restructure no longer load, so their `.pt` fixtures go too, with the saved-model test's cases that load them (the scores of the three models, the old dataset's scores, and the format and directory check); the case that prepares the old dataset's accounts and labels needs no model and can stay. Drop the module from `architecture.md` in the same commit.

```
git fetch origin && git fetch learner
git merge-base --is-ancestor origin/main restructure && echo origin-ff
git merge-base --is-ancestor learner/main restructure && echo learner-ff
git switch main
git merge --ff-only restructure
# after confirmation, and after checking branch protection and the default branch on both remotes:
git push origin main
git push learner main            # local main tracks learner/main; name each remote explicitly
git push origin pre-restructure  # optional: keeps the parity reference reachable
```

**If a remote main has moved.**
- Read `git log restructure..<remote>/main` first.
- Only if nothing there should be kept, run `git merge -s ours <remote>/main -m "Replace the old main with the restructured model"` on `restructure`, rerun the gate, and fast-forward.

Both remotes end with the same `main` (owner decision): check afterwards that `git rev-parse origin/main learner/main main` prints one commit three times.

**Afterwards.** With the owner's confirmation, delete every branch other than `main`, as in the owner decisions, so `main` is the only branch locally and on both remotes.

### Decisions for the owner to approve

All five were decided on 2026-09-27, and the three the mid-migration review raised later (the
table label reader, the old saved-settings conversion and the dataset id's scope settings) on
2026-09-28; see Owner decisions. On 2026-09-28 the owner also narrowed the one to keep all 20
feature groups: only the built-in run's groups stay in training, and the others move to analytics.


1. **The server rename** (the server-rename step: about an hour of installing and re-preparing, done once), or keep the query names as allow-listed data.
2. **Keep all 20 feature groups** and add the window-based `no_graph` control, instead of deleting the "legacy" groups.
3. **Validation audits for decisions**, with the test audit for reporting only.
4. **Commit the `/tmp` scripts and notes to `archive/diagnostic-study` now.**
5. **Push the fast-forwarded `main`** to `origin` and `learner`.
