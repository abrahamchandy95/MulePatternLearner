# Temporal account learning

**For the current large live graph, use [the live temporal training pipeline](live_temporal_training.md).**
It fetches cutoff-aware GSQL features, uses event-time recursive attention, and
streams bounded batches. The loader-export snapshot workflow below is legacy
experimental code, not the production observed-label path. Its masking remains
coupled to experiment setup; use the live path for the decoupled interfaces.

The `temporal` branch adds a separate training pipeline under
`src/mule_pattern_learner/temporal/`. The original static training modules are
unchanged. The new pipeline uses the live graph's verified loader export and the
existing GSQL time basis. It stages observations once, trains on bounded
neighborhoods, and writes checkpoints and learned embeddings locally.

## What the model sees

A training example is an internal deposit account at an exclusive UTC timestamp
cutoff. Payment events remain individual vertices in
TigerGraph. The sampler projects their observed roles into directed messages;
this is a modeling reduction, not a schema migration. It retains repeated
payments and uses observed account recipients, falling back to tokens when the
recipient account cannot be resolved. It never resolves a historical token
through today's registration.

Each entity has 108 features: type and subtype; first-observation age; 1-, 7-,
30- and 90-day counts, amounts, distinct counterparties and rail activity;
last-payment recency; and active/ended association counts. These are recomputed
at each cutoff. No whole-history centrality, fraud flags, ring IDs, labels,
identifier embeddings or database train/test flags enter model inputs.
First-observation age is not a claim about the account's actual opening date.
Amounts currently require one source currency; mixed-currency input fails until
cutoff-available currency conversion is supplied.

Messages carry relation type, rail, observed amount and presence indicators.
The temporal variant adds two separate 64-coordinate vectors:

- **Age:** scoring cutoff minus payment time.
- **Directed pair gap:** payment time minus the preceding payment time for the
  same sender, observed recipient and rail.

Integer millisecond subtraction happens before floating-point conversion.
Missing preceding payments produce a missingness flag and a zero vector; an
observed zero gap produces the full sine/cosine vector. This distinction agrees
with the GSQL queries. The exact fixed basis is
`log1p_s_400d_32x_sincos_v1`, documented in
[temporal_encoding.md](temporal_encoding.md). The network learns how to combine
these inputs; the GSQL Fourier vectors are not trained account embeddings.

Associations use `valid_from_seq <= seed_seq` and
`valid_to_seq == 0 OR seed_seq < valid_to_seq`. Re-enrollments remain distinct
tenures. At a timestamp cutoff, `seed_seq` is the last sequence witnessed strictly
before that timestamp in payments or entity first-seen records. This avoids
exposing an association that begins at the next, excluded event. Association
start-time encodings are present only when the exact sequence has a timestamp
witness. Sequence differences are never treated as elapsed time.

The two message-passing layers compute entity state at the same scoring cutoff.
They do not claim to reconstruct a neighbor's embedding at each older event's
time. This is a bounded temporal heterogeneous attention model, not an exact
TGAT implementation or a recurrent TGN. Default learned entity embeddings have
32 dimensions; the time encodings have 64 dimensions each. `hidden` controls the
learned embedding size independently.

## Sampling and memory

Each cutoff keeps at most four recent messages per relation, interleaves
relations, and stores at most 32 messages per entity. Training uses fanouts
`[8, 4]`, deduplicating intermediate nodes within a batch. Only the selected
features and messages are transferred to the model device. The entire graph is
never put into GPU memory. Snapshots are read through memory-mapped arrays.
The implemented deterministic sampler emphasizes recent history, so it can miss
older rare relationships; it does not claim complete neighborhood coverage.

The current preprocessing implementation reads the staged event table into CPU
memory and builds each cutoff offline. This is adequate for this corpus, not a
claim of billion-event preprocessing scalability. A larger implementation should
partition event storage by time and source, maintain temporal adjacency indexes,
and fetch bounded batches through a sampler service or GPU-adjacent cache.
TigerGraph can serve cutoff-bounded subgraphs or incremental staged exports;
reading the same full neighborhoods from the database every epoch adds network
latency and load. The existing GSQL pair queries verify exact pair histories but
are not a production-scale batched training sampler.

Cache keys must include dataset/export identity, cutoff, sampler policy and time
basis. Late-arriving/backdated data invalidates affected snapshots and pair gaps.
Known-time attributes are absent, so the current backtest assumes event facts and
registrations were available when recorded as valid. It cannot establish
historically available information under delayed master-data corrections.

## Running the pipeline

From the repository root, with the existing virtual environment:

```bash
.venv/bin/python -m mule_pattern_learner.temporal.cli stage \
  --manifest ../tf_gnn_loader_v2/artifacts/mule_temporal/export_manifest.json

.venv/bin/python scripts/temporal/verify_training_source.py

.venv/bin/python -m mule_pattern_learner.temporal.cli snapshots \
  --config configs/temporal/zelle_forecast.toml

.venv/bin/python scripts/temporal/run_experiments.py \
  --config configs/temporal/zelle_forecast.toml \
  --output artifacts/temporal/runs/zelle_forecast_new

OMP_NUM_THREADS=4 .venv/bin/python scripts/temporal/forecast_tree_baseline.py \
  --output artifacts/temporal/runs/zelle_forecast_new/rolling_trees

MPLCONFIGDIR=/tmp/temporal_matplotlib .venv/bin/python scripts/temporal/summarize_experiments.py \
  --runs artifacts/temporal/runs/zelle_forecast_new \
  --output artifacts/temporal/reports/zelle_forecast_new
```

Staging verifies every source shard checksum and requires a matching successful
loader verification. The live check uses `.env`, reads vertex counts and sampled
pair histories, and compares GSQL/Python encodings without writes. It does not
perform a full live attribute checksum. Training requires no database connection.
Choose a new output directory for a new experiment; existing results are not
silently overwritten. The matrix runner resumes completed matching runs.

```bash
.venv/bin/python -m mule_pattern_learner.temporal.inference \
  --checkpoint artifacts/temporal/runs/zelle_forecast_example/temporal_seed42_reveal1/model.pt \
  --snapshot artifacts/temporal/snapshots_v2/2025-01-01 \
  --output artifacts/temporal/predictions/zelle_2025-01-01.parquet
```

The score's meaning comes from its training task. This checkpoint forecasts
outgoing Zelle activity; it does not score mule risk. Scoring exports learned
embeddings, scores, cutoff and validation-threshold flags without modifying the
live graph. A future snapshot may be scored but requires future observations
before its forecast can be evaluated.

## Mule labels and masking

Use `configs/temporal/mule_graph_mask.toml` to honor the stored Account mask, or
`mule_pu.toml` to resample masks for label-budget experiments. Both read the
separate label table staged from Account. See [the mask contract](account_mule_labels.md).

The `mule_pu.toml` configuration and nnPU training path are implemented. Labels
are stored on Account and staged separately from features. The fifteen-column
Account export automatically produces the label table below. Alternatively supply a
Parquet table with one row per account:

| Field | Meaning |
|---|---|
| `account_id` | Exact opaque Account ID in the staged graph |
| `target` | Explicit evaluation ground truth, 0 or 1 |
| `effective_ts_ms` | First time the positive account target applies |
| `available_ts_ms` | When its adjudication became available for training |
| `ring_id` | Group for ring holdouts; `0` is valid, `-1` means none |

An adjacent `.json` file must specify `target_definition` as `confirmed_mule` or
`synthetic_laundering_intermediary` or `account_mule`, and `complete_negative_ground_truth: true`.
This assertion must reflect the source. Unreviewed real accounts are not confirmed
negatives; an evaluation needs an adjudicated subset. Accounts omitted from the
label table can still be unlabeled training examples but are excluded from
validation/test metrics. The target is cumulative participation by the cutoff;
a future-positive account is not positive before its effective time. The supplied
importer supports one ring membership per account and fails on overlapping rings.

The optional historical `scripts/temporal/import_laundering_proxy.py` is unnecessary
when regenerated data supplies direct Account mule truth. It uses a fixed read-only
source query and requires the explicit `--allow-laundering-proxy` flag. It includes
only enumerated layering/scatter-gather intermediary roles and cycle participants.
It excludes arbitrary fraud incidence, invoice recipients, source victims and
terminal cash-out destinations unless they independently fill an included role.
It checks every source event ID, timestamp and participant against the staged
graph. Its default seven-day adjudication delay is simulated, not investigator
history. It has not been used to manufacture mule results in the current report.

```bash
# After choosing the target and preparing its external labels:
.venv/bin/python -m mule_pattern_learner.temporal.cli snapshots \
  --config configs/temporal/mule_pu.toml

.venv/bin/python scripts/temporal/run_experiments.py \
  --config configs/temporal/mule_pu.toml \
  --fractions 0.1 0.25 0.5 1.0 --seeds 42 43 44 \
  --output artifacts/temporal/runs/mule_label_budget
```

In `mask_source="resample"` mode, a stable hash selects accounts for revelation.
The 10% set is contained in the
25%, 50% and 100% sets for the same seed. Hidden accounts remain hidden across all
cutoffs; no hidden account labels or label-derived features are supplied to the
network. Revealed positives must also be available by their training cutoff.
The loss draws positives separately and draws an independent marginal sample of
all training accounts, including both revealed and hidden positives. This follows
the nnPU estimator; hidden positives are never relabeled as negatives.

Co-owned accounts are grouped before splitting. All ownership tenures are used
only to conservatively prevent owner overlap across splits, including future
co-ownership; features still observe their own cutoff. The split is fixed across
masking seeds and budgets. `dark_ring_ids` forces entire co-owner groups into test;
run those as separate configurations. Verify enough independent rings exist before claiming a three-way ring
generalization study. Reports distinguish hidden
training-account recovery from held-out test performance and include per-ring
recall. A zero-positive mask or validation set without both classes skips the run
without changing its mask or split.

`class_prior` is an explicit assumed target prevalence, never an estimate from
hidden test truth. Run sensitivity checks, for example 0.005, 0.01 and 0.02, before
interpreting results. The masking fraction changes training positive supervision;
validation still uses its fixed adjudicated ground truth for checkpoint and
threshold selection. Therefore the experiment is not a claim that the same
fraction covers the total labeling cost of the system. Scarce validation labels
need a separately budgeted study.

An optional `pretrained_checkpoint` initializes the encoder while resetting its
prediction head. Its normalization is restored with it. The trainer rejects a
pretraining checkpoint whose validation/forecast horizon reaches beyond the
fine-tuning validation boundary. The supplied full-year pretraining run therefore
must not be used for the earlier October mule validation cutoff; train an earlier
pretraining schedule before that comparison.

## Evaluation and validation

The configurations separate both accounts/owner groups and time. Forecast
training cutoffs are April through September, validation is October, and test is
November/December. A 30-day target horizon may not cross the next split boundary.
The source must extend through the entire target horizon. Mule cutoffs and label
availability are handled separately. Held-out nodes can appear in past graph
neighborhoods: this is transductive topology with held-out labels, not strict
unseen-node inductive evaluation.

Checkpoint selection uses validation average precision. Threshold selection uses
validation F1. Test is first scored after both choices. Reports include average
precision, AUROC, precision/recall at 1% and 5% review budgets, threshold metrics,
per-cutoff metrics, and 200-draw owner-group bootstrap intervals. Seed standard
deviations measure initialization/training variation, not population uncertainty.
Random-ranking average precision equals target prevalence.

Tests cover future-payment and future-end-time invariance, exclusive tied-time
cutoffs, repeated tenures, unresolved tokens, repeated pair events, nested masks,
label availability, bounded sampling, missing-neighbor behavior, nnPU gradients,
checkpoint round trips and zero-label skips:

```bash
.venv/bin/pytest -q tests/temporal
.venv/bin/ruff check src/mule_pattern_learner/temporal tests/temporal scripts/temporal
.venv/bin/basedpyright src/mule_pattern_learner/temporal tests/temporal scripts/temporal
```

## Research basis

[TGAT](https://arxiv.org/abs/2002.07962) motivates functional time representations
and temporal neighborhood attention. This implementation shares a fixed basis
with GSQL rather than learning the frequencies.
[TGN](https://arxiv.org/abs/2006.10637) introduces event-stream memory; that would
require explicit state updates, replay/reset rules and state management during
training and inference, which this initial model avoids.
[nnPU](https://arxiv.org/abs/1703.00593) supplies the nonnegative positive-unlabeled
risk estimator. The [Temporal Graph Benchmark](https://arxiv.org/abs/2307.01026)
supports the need for temporal evaluation and strong baselines; complexity alone
is not evidence of improved detection.
