# Configuration

Every setting of a run, with its built-in value. The settings are Python:
`config.RunConfig` is a frozen dataclass with one section per concern, and its defaults
are the built-in run, `config.DEFAULT_CONFIG`. No command reads a configuration file or
takes an option, so `mule train` always trains these values. `.env` holds only the
TigerGraph connection (`HOST`, `GRAPHNAME` and `SECRET`; environment variables override
it), and it is read when a command connects, never at import.

A test (`tests/test_config.py`) checks this page against `config.py`: every setting is
listed once, with the value `config.json` records for it.

## Changing a setting

Another run is built in Python. `RunConfig.with_changes` takes a table of the shape
`config.json` records and changes only the settings it names; `dataclasses.replace`
changes a section. Each section checks its values when it is built, with the ranges of
`contract.bounds`, so a bad value fails before any database work, and an unknown key is
refused by its dotted name.

```python
from mule_pattern_learner.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.with_changes({"training": {"epochs": 3}, "model": {"slot_sum": False}})
```

A control experiment declares its change in `experiments/variants.py` instead (see
[Run the control experiments](../how-to/run-control-experiments.md)).

## Which settings matter where

- **The fingerprint** (`RunConfig.fingerprint`) covers every setting that can change a
  run's results: every section except `transport` and `runtime`, and the sampler without
  `sampler.backend`. `config.json` records it, a resumed run must match it, and a complete
  run of another fingerprint is never overwritten.
- **The dataset settings** name the prepared dataset: the source id (read from the
  graph, never configured), `scope.id`, `scope.unowned`, the whole `dataset` section and
  the sampler's candidate pools (`sampler.roots` and `sampler.children`). Their
  fingerprint is the dataset id, the name of `data/<dataset id>/`. Runs that share them
  share one dataset; any other setting, the feature groups and the training seed
  included, may differ between runs of one dataset.
- **`scope.create`, `scope.reveal_per_split` and `scope.reveal_salt`** act once on the
  graph, when a missing scope is created and in the one-time label reveal, so changing
  them later names no other dataset.
- **`transport` and `runtime`** never change a run's numbers, with one exception:
  `runtime.device`, `runtime.threads` and `runtime.deterministic` can change floating-point
  results. `config.json` records the values a run started with, and a resumed segment
  that changes them says so in `events.jsonl` (a `host_settings` event).

## scope

The frozen experiment scope the splits come from, and the first run's label reveal.

| Setting | Default | Meaning |
|---|---|---|
| `scope.id` | `"strict_mule_v2"` | The `Temporal_Training_Scope` vertex the run samples in (at most 256 bytes) |
| `scope.create` | `true` | The first run creates a missing scope; `false` forbids that write |
| `scope.unowned` | `"linked"` | Where accounts no party owns go: `"independent"`, `"shared"` or `"linked"` (see [the scope queries](queries.md#create_training_scope)) |
| `scope.reveal_per_split` | `20` | Known mules the first run reveals per split, among those a bank would have discovered before the split's cutoff (0 to 1,000) |
| `scope.reveal_salt` | `42` | Seed of the reveal's deterministic draws |

## dataset

What preparation stages inside the scope: the split cutoffs and the seed reservoirs.

| Setting | Default | Meaning |
|---|---|---|
| `dataset.dates.train` | `["2024-07-01"]` | The train cutoffs, as ISO dates; history is what happened before midnight UTC |
| `dataset.dates.validation` | `["2024-10-01"]` | The validation cutoffs, all after every train cutoff |
| `dataset.dates.test` | `["2025-01-01"]` | The test cutoffs, all after every validation cutoff |
| `dataset.seed_limits.train` | `20000` | Accounts in the train split's label-blind seed reservoir (1 to 20,000) |
| `dataset.seed_limits.validation` | `2000` | The same for validation |
| `dataset.seed_limits.test` | `2000` | The same for test |
| `dataset.seed` | `42` | Seed of the reservoirs' hash ranks |
| `dataset.split_seed` | `42` | Seed of the scope's partition into splits, and of the audit and proxy samples |

## sampler

The candidate pools TigerGraph returns per context and hop, and the client's
resampling of them into fan-out slots ([Sampling](../explanation/sampling.md) explains
them). The section is `contract.sampler_plan.SamplerPlan` itself.

| Setting | Default | Meaning |
|---|---|---|
| `sampler.fanouts` | `[16, 4]` | Slots per context at hop 1 and hop 2 (1 to 64 each) |
| `sampler.roots.recent` | `8` | A root's most recent visible events per payment relation (1 to 32) |
| `sampler.roots.older` | `4` | Older events per payment relation, at evenly spaced recency ranks (0 to 16) |
| `sampler.roots.distinct` | `4` | Recent events with counterparties new to the pool, per payment relation (0 to 16) |
| `sampler.roots.associations` | `2` | Most recent active tenures per association relation (0 to 8) |
| `sampler.roots.max_history` | `2048` | Visible events per payment relation above which a context is rejected (32 to 4,096) |
| `sampler.children.recent` | `4` | The same pool for a hop-1 child |
| `sampler.children.older` | `2` | |
| `sampler.children.distinct` | `2` | |
| `sampler.children.associations` | `0` | Children have no association candidates: hop 2 samples payments only |
| `sampler.children.max_history` | `2048` | |
| `sampler.relation_fanouts` | `[8, 4]` | Payment candidates kept per context and relation at hop 1 and hop 2 |
| `sampler.association_fanout` | `1` | Candidates kept per association relation at hop 1 (0 to 8) |
| `sampler.association_slots` | `2` | Hop-1 slots that associations may fill, at most a quarter of them (0 to 16) |
| `sampler.backend` | `"auto"` | `"auto"` (cuGraph on CUDA when its probe passes, else torch), `"cugraph"` or `"torch"` |
| `sampler.evaluation_seed` | `0` | Seed of the hash keys that evaluation and scoring draw with |

The smaller of the two `max_history` values is also the hub threshold: an account with
more visible events than that in one payment relation is a hub, and a child hub becomes
a stub instead of a request.

## features

| Setting | Default | Meaning |
|---|---|---|
| `features` | `["entity_meta", "hub_indicator", "message_core", "time_encoding", "pair_history", "flow_timing", "pool_activity", "pool_internal_inflows"]` | The feature groups the model reads, every group of `contract.feature_groups.FEATURE_GROUPS` ([Features](features.md)) |

A group's dependencies must come with it (`pool_activity` reads `pair_history` and
`flow_timing`), and the graph model needs `message_core`.

## model

| Setting | Default | Meaning |
|---|---|---|
| `model.architecture` | `"tgat"` | `"tgat"` attends over sampled neighbours; `"summary"` reads only the root's own inputs (the `no_attention` control) |
| `model.hidden` | `64` | Hidden width (8 to 512, divisible by the heads) |
| `model.heads` | `4` | Attention heads (1 to 16) |
| `model.dropout` | `0.15` | Dropout rate, from 0 up to but excluding 1 |
| `model.slot_sum` | `true` | Feed the head a sum of a small MLP of each hop-1 slot beside attention; the summary architecture ignores it |

## loss

| Setting | Default | Meaning |
|---|---|---|
| `loss.class_prior` | `0.001` | The assumed share of mules among all accounts, strictly between 0 and 1 |
| `loss.positive_weight` | `"balanced"` | The weight of the revealed positives' risk: `"balanced"` (imbalanced nnPU, 1 minus the prior), `"prior"` (textbook nnPU) or a number strictly between 0 and 1 |

## training

| Setting | Default | Meaning |
|---|---|---|
| `training.seed` | `42` | Seed of the epoch schedules, the model's initial weights and every step's draws |
| `training.epochs` | `30` | Most epochs a run trains |
| `training.steps_per_epoch` | `100` | Steps per epoch; `null` trains on every marginal account of an epoch |
| `training.batch_size` | `64` | Roots per step (1 to 128): a quarter observed positives, the rest from the marginal |
| `training.patience` | `6` | Epochs without a better validation AP before training stops; 0 never stops early |
| `training.learning_rate` | `0.001` | AdamW learning rate |
| `training.weight_decay` | `0.0001` | AdamW weight decay |
| `training.weight_average_decay` | `0.99` | Decay per step of the moving average of the weights that validation scores and `model.pt` keeps; 0 validates the raw weights |
| `training.proxy_unlabeled_limit` | `2000` | Unlabeled accounts per validation and test cutoff that the proxy evaluation scores beside every observed positive; `null` scores them all |

## transport

How contexts are requested, and the retry budgets. Changing them never changes a run's
results, so a resumed run may change them.

| Setting | Default | Meaning |
|---|---|---|
| `transport.request_batch_size` | `8` | Contexts per REST request (1 to 64) |
| `transport.query_concurrency` | `16` | Requests one context source keeps in flight (1 to 16) |
| `transport.context_lru_capacity` | `256` | Contexts the source keeps in memory (0 to 4,096) |
| `transport.encoding_check_every` | `64` | Every n-th context request also asks for the Fourier vectors, which are checked against the client's |
| `transport.max_query_attempts` | `6` | Attempts that count per query (1 to 20) |
| `transport.max_outage_s` | `900` | Seconds one operation keeps retrying while TigerGraph is unavailable (0 to 86,400) |

The measured choice: 8 contexts per request and 16 in parallel built a 64-root batch in
about 11 seconds on the reference graph, against about 20 seconds for 16 per request and
8 in parallel, and about 22 seconds for 4 per request and 16 in parallel.

## runtime

The host. Only the first three can change floating-point results.

| Setting | Default | Meaning |
|---|---|---|
| `runtime.device` | `"auto"` | CUDA when available, then Apple MPS, then CPU; or `"cpu"`, `"mps"` or `"cuda"` |
| `runtime.threads` | `4` | CPU threads torch uses |
| `runtime.deterministic` | `true` | Deterministic algorithms, warning on CUDA-only gaps; `"strict"` fails on them; `false` turns them off |
| `runtime.prefetch_batches` | `2` | Batches built ahead of the one in use (0 to 8) |
| `runtime.checkpoint_every_steps` | `0` | Also save `resume.pt` every n steps; 0 saves it once per epoch |
| `runtime.log_every_steps` | `10` | Steps per row of `history.csv` |
| `runtime.max_rejected_root_fraction` | `0.0` | Largest share of an epoch's or a split's roots TigerGraph may reject; it decides only whether a run may go on, never its numbers |

## Constants that are not settings

The audit's constants are not run configuration: every audit of every run uses them, and
each audit report records them under `constants`.

| Constant | Value | Where |
|---|---|---|
| `evaluation.sample.AUDIT_NEGATIVES` | 2,000 uniform non-mules per audited split | the audit sample |
| `metrics.REVIEW_BUDGETS` | 1%, 5% and 10% of the estimated population | precision and recall at a review budget |
| `metrics.BOOTSTRAP_REPLICATES` | 1,000 | the bootstrap intervals |
| `metrics.INTERVAL` | 0.90 | the bootstrap intervals |
| `metrics.BOOTSTRAP_SEED` | 0 | the bootstrap intervals |

Every numeric bound of the pipeline (request size, fan-outs, batch size, seed limits, the
context cache's 1,500,000 entries) is defined once in `contract.bounds`.
