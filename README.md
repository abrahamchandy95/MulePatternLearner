# Mule Pattern Learner

For the populated temporal graph, start with the
[live TGAT-style training guide](docs/live_temporal_training.md): cutoff-aware
GSQL features, bounded streaming, observed-label supervision and embedding export.
Read the [leakage and scaling assessment](docs/leakage_and_scaling.md) before
choosing an evaluation protocol, and the [GSQL feature catalog](docs/gsql_feature_catalog.md)
for the exact model inputs. The default strict experiment excludes held-out
Account/Party contributions before GSQL aggregation and sampling. The backend
uses bounded seed cohorts and batch-local temporal IDs, and supports new-account
prediction without a training-ID lookup table.

For the new temporal payment and Zelle schema, see
[Temporal payment schema](docs/temporal_schema.md) and
[`temporal_schema.gsql`](gsql/schema/temporal_schema.gsql). The
[GSQL temporal encoding guide](docs/temporal_encoding.md) covers the installed
64-dimensional payment-gap and cutoff-age queries. The separate
[temporal training pipeline](docs/temporal_training.md) includes model training,
legacy experiment masking and inference. Dataset-specific reports are local artifacts.
The snapshot pipeline described below is retained as a legacy implementation.

Learns mule and money-laundering patterns from transactional data by training a graph neural network (GNN) directly from TigerGraph.

The project contains:

1. **A GSQL layer** on TigerGraph that resolves entities, detects WCC communities, and derives node/edge features on the graph.
2. **A Python (PyTorch Geometric) layer** that uses TigerGraph as a remote backend, samples neighbourhoods, and trains a node classifier that ranks accounts by how mule-like they are.

For this project, you would need to be connected to a working instance of
TigerGraph. The target graph name used in this project is `Mule_Pattern_Learner`.


## Setup

```bash
pip install -e ".[model]"        # add ,baseline,dev as needed, or use [all]
```

Create a `.env` for the TigerGraph connection (read by `Settings`):

```
HOST=https://your-tg-host
GRAPHNAME=Mule_Pattern_Learner
SECRET=your_restpp_secret
```

## How the queries are organised

The legacy snapshot queries live under `gsql/` and are registered in `src/mule_pattern_learner/tigergraph/gsql_paths.py`, which maps a short **registry name** to its `.gsql` file. The temporal encoding queries use the separate installer linked above. A registry name is not always the installed query name; the ones that matter most differ:

| Registry name | Installed query | Group |
| --- | --- | --- |
| `match_parties` | `match_parties` | entity resolution |
| `unify_parties` | `unify_parties` | entity resolution |
| `weight_account_edges` | `account_account_with_weights` | community detection |
| `cluster_with_wcc` | `tg_wcc_account_with_weights` | community detection |
| `pagerank` | `tg_pagerank_wt_account` | features |
| `fastrp` | `tg_fastRP` | features |
| `money_flow` | `account_money_flow_features` | features |
| `temporal_features` | `account_temporal_features` | features |
| `triangle_clustering` | `account_triangle_clustering` | features |
| `identity_sharing` | `account_identity_sharing_features` | features |
| `account_account_degree` | `account_aa_degree_feature` | features |
| `derive_reference_epoch` | `derive_reference_epoch` | features (derivation) |
| `derive_max_bins` | `derive_max_bins` | features (derivation) |
| `get_split_accounts` | `get_split_accounts` | masking |
| `sample_khop_neighborhood` | `sample_khop_neighborhood` | sampling (runtime) |
| `fetch_account_features` | `fetch_account_features` | sampling (runtime) |
| `fetch_has_paid_features` | `fetch_has_paid_features` | sampling (runtime) |
| `export_account_features` / `export_edges_by_type` / `export_has_paid_edges` | same | export |

### Installing queries

`src/mule_pattern_learner/tigergraph/gsql_install.py` installs a query by reading its file in this repository, 
extracting the `CREATE [OR REPLACE] [DISTRIBUTED] QUERY <name>` name, and installing it onto the graph:

```python
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_install import install_query

client = Client(Settings())
install_query(client, "match_parties", drop_first=True)
# ...install the rest before running them.
```

Install a query before you run it.

## Query execution order

The graph must be prepared in stages, because later queries read structure that earlier queries write. Run them in this order.

### 0. Schema and load

- `gsql/schema/schema.gsql` defines the vertices and edges (accounts, parties, PII values, transactions, and the derived types `Resolved_Entity`, `Connected_Component`, and so on).
- `gsql/schema/loading_job.gsql` (registry `loading_job`) loads the transaction/account/PII data into that schema. Files were loaded into the GUI in GraphStudio, adjust accordingly.

`scripts/pipeline/pre_graph/temporal_flow_aggs.py` precomputes temporal flow aggregates before/with loading.

### 1. Entity resolution (order is mandatory)

Two queries, run strictly in this order:

1. **`match_parties`** scans every pair of `Party` vertices that share a PII value (email, phone, birthdate, address parts, or name/address/street MinHash buckets), each reachable through the `Has_*` edges. It accumulates a weighted match score per shared attribute, skips PII values whose degree is implausibly high (the `pii_*_connections_limit` parameters, so that a shared city does not link thousands of people), and writes a **`Same_As`** edge between any two parties whose combined score clears `threshold`.
2. **`unify_parties`** runs a weakly-connected-components pass over those `Same_As` edges (iterative minimum-component-id propagation), then materialises one **`Resolved_Entity`** vertex per component and a **`Party_In_Entity`** edge from each party to its entity.

`unify_parties` reads the `Same_As` edges that `match_parties` produces, so running it first, or without `match_parties`, yields no resolved entities.

### 2. Community detection / WCC (order is mandatory)

Two queries, run strictly in this order:

1. **`account_account_with_weights`** (file `weight_account_edges.gsql`) deletes any existing `Account_Account` edges, then walks the `HAS_PAID` transaction edges and re-inserts a symmetric `Account_Account` edge between each pair of accounts, weighted by the total number of transactions between them (`min_edge_weight` drops trivial links).
2. **`tg_wcc_account_with_weights`** (file `cluster_with_wcc.gsql`) runs weakly-connected components over those `Account_Account` edges, keeping only edges with `weight >= min_link_weight`. It writes `com_id` / `com_size` back onto each `Account` and materialises `Connected_Component` vertices with `Account_In_Ring` edges.

The WCC query consumes the weighted `Account_Account` edges the first query builds, so the weighting step has to run first.

### 3. Feature derivation

With entities resolved and communities detected, derive the node and edge features. These are largely independent of one another and can run in any order, with a few dependencies:

- `account_temporal_features` (registry `temporal_features`) builds per-account temporal transaction features (binned amounts/counts). Run it **before** `derive_max_bins`, which reads the resulting `num_bins`.
- `account_identity_sharing_features` (registry `identity_sharing`) derives features from shared identity, so it expects entity resolution (stage 1) to have run.
- `account_aa_degree_feature` (registry `account_account_degree`) reads the weighted `Account_Account` graph from stage 2.
- `tg_pagerank_wt_account` (weighted PageRank), `tg_fastRP` (FastRP structural embeddings), `account_money_flow_features` (in/out money-flow aggregates), and `account_triangle_clustering` (triangle/clustering features) can run any time after stages 1 and 2.
- `derive_reference_epoch` returns the latest transaction time in the graph, the snapshot epoch that recency features are measured against.

`derive_max_bins` and `derive_reference_epoch` are also invoked at training time by `src/mule_pattern_learner/tigergraph/derivation.py`, which folds them into a single `GraphTemporalSpec` so the loader's edge-feature padding and the model's `edge_dim` cannot disagree.

### 4. Masking and splits

`get_split_accounts` reads existing split and observed-PU flags; it does not
assign them. Legacy synthetic masking utilities remain local and Git-ignored.
Supply observed labels and split metadata from the data producer. The live
temporal path uses an injected observed-label provider and a separate oracle
evaluator, described in the guide above.

### 5. Runtime queries (used during training, not a one-off step)

Installed once and called repeatedly by the PyG remote backend while training:

- `sample_khop_neighborhood` samples a k-hop neighbourhood around a batch of seed accounts.
- `fetch_account_features` and `fetch_has_paid_features` fetch node and edge features for the sampled subgraph.

The `export_*` queries are an alternative to runtime sampling: they bulk-export account features and edges for fully offline training.

## Scripts

- **Install queries:** `tigergraph/gsql_install.py` (`install_query`), with the file registry in `gsql_paths.py`.
- **Pre-graph:** `scripts/pipeline/pre_graph/temporal_flow_aggs.py` (temporal flow aggregates before load).
- **After load:** supply observed labels and split metadata from your source. Synthetic masking is a separate local experiment.
- **Train:** `mule-train` (entry point `mule_pattern_learner.training.train:main`). The training loop (`training/loop.py`) drives a PyG model (`pyg/model.py`) over a TigerGraph-backed remote dataset (`pyg/backend.py`, `graph_store.py`, `feature_store.py`, `sampler.py`), using `derive_temporal_spec` for edge widths.
- **Evaluate:** `scripts/experiments/evaluate_hidden.py` (recall on hidden, never-revealed positives, the generalization headline) and `scripts/experiments/check_val_mules.py`.
- **Diagnostics and demos:** `scripts/experiments/diagnose*.py`, `sample_probe.py`, `khop_probe.py`, and runnable walk-throughs under `scripts/demos/` (backend, sampler, feature store, loader, model, and so on). Not required to run, but helps visualize data. 

End-to-end flow:

```
install queries
  -> load data (+ pre_graph aggregates)
  -> match_parties -> unify_parties                               # entity resolution
  -> account_account_with_weights -> tg_wcc_account_with_weights  # WCC
  -> feature queries (pagerank, fastrp, money_flow, temporal, triangle, identity, aa-degree)
  -> external split/observed-label setup -> get_split_accounts   # metadata
  -> mule-train                                                   # GNN training (runtime sampling)
  -> evaluate_hidden                                              # generalization check
```
## Notes

- Connection settings come from `.env` via `Settings` (`host`, `graphname`, `secret`); all three are required.
- Runs are seeded for reproducibility (`training/seeds.py`).
- Several queries carry a raised `query_timeout`, since entity resolution and WCC scan the full graph.

## Known limitations and planned work

The items below concern the **legacy snapshot pipeline** and remain
**TO BE IMPLEMENTED** there. The separate live temporal path has its own
limitations documented above. Each item names the file that would
change. They are grouped by the concern they address, and ordered within each
group by value.

### Single-ended transactions and counterparty sinks

Cash withdrawals, card purchases and other outflows to an unidentifiable party
are captured, but as edges to a small number of shared placeholder accounts. Shared placeholders can create hubs that distort structural features; the
legacy representation also discards the channel label.

- [ ] **Set `is_external` on the placeholder accounts.** `gsql/schema/loading_job.gsql`
  skips that column for both the external-account and merchant loads (VALUES
  position 5 is `_`), so the placeholders carry the schema default of 0 and are
  indistinguishable from customers. This is the only feature in the node vector
  that could mark them.
- [ ] **Exclude placeholders from `Account_Account` and WCC.** A `WHERE`
  clause in `gsql/community_detection/weight_account_edges.gsql`. Without it,
  every account that used a shared sink twice is merged into one connected
  component, which makes the `com_size` node feature near-constant and turns
  `Account_In_Ring` into a single giant ring. This is the highest-value item in
  this group.
- [ ] **Exclude placeholders from splits, seeds and the class prior.**
  `gsql/masking/get_masking_inputs.gsql` selects every Account with no filter,
  so sinks become nnPU training seeds, and `_resolve_prior` in
  `training/train.py` counts them in the prior denominator. They are not mule
  candidates.
- [ ] **Guard against a sink being assigned to val or test.** A placeholder has
  no owning party, so `data/splitting.py` treats it as a freely assignable solo
  group. If it lands in a held-out split, the sampler's `allow_val` /
  `allow_test` gate removes all of its edges from training neighbourhoods while
  keeping them in validation, silently changing the graph topology between the
  two regimes. Excluding sinks from split assignment fixes this.
- [ ] **Carry the transaction channel onto the edge.** The raw ledger has a
  `channel` column (`atm_withdrawal`, `card_purchase`, `p2p`, and so on) that
  the pre-graph aggregation never reads, so the model cannot distinguish a cash
  withdrawal from a transfer to another customer. For mule detection that is
  arguably the most important distinction available.
- [ ] **Promote `external_type` and `category` to real features.** They are
  loaded and then parked in `DEFERRED_CATEGORICAL_ATTRS` (`schema/specs.py`),
  and no query or model code reads them.

### Temporal representation and time-inductiveness

The model is inductive over unseen **accounts** within a snapshot. It is not
inductive over **time**. The items below are what would extend it.

- [ ] **Anchor the bins to the scoring moment rather than the dataset start.**
  Bins currently tile forward from a global earliest timestamp
  (`scripts/pipeline/pre_graph/temporal_flow_aggs.py`), so bin 0 is a fixed
  calendar fortnight forever and an account created after the training window
  falls off the end of the grid. A rolling fixed-K window measured backwards
  from the as-of date fixes three problems at once: accounts from different
  eras become directly comparable, the representation becomes shift-invariant
  so a dormant-burst-abandoned pattern is the same shape whenever it occurred,
  and bin overflow becomes structurally impossible. **Highest-value change in
  this document.**
- [ ] **Make `reference_epoch_s` an as-of parameter instead of a derived
  constant.** It is currently read from the graph at job start and frozen into
  the checkpoint. The plumbing already exists, since
  `pyg/transform.py` accepts it as a constructor argument. At training it
  should be the snapshot date; at scoring, the scoring date.
- [ ] **Reject transactions newer than the reference epoch.**
  `features/temporal.py` clamps `recency_days` at zero, so any transaction
  after a frozen reference epoch reads as maximally recent rather than raising.
  This fails silently.
- [ ] **Compress and cap the time scalars.** `recency_days`, `duration_days`,
  `span_days` and `active_bin_count` enter the model raw and unbounded, while
  amounts and counts are already `log1p` compressed (`features/temporal.py`).
  Edge features are also not standardized at all, unlike node features.
- [ ] **Consume the bins as an ordered sequence.** `flat_edge_features`
  flattens the `[E, bins, channels]` tensor into a flat vector for
  `GATv2Conv`'s `edge_dim`, so the ordering carries no weight. A 1D
  convolution or small GRU over `EdgeFeatures.bin_seq`, which is already the
  right shape, would let the model learn temporal shape rather than
  per-position magnitude.
- [ ] **Vary the as-of date across training examples.** Every example currently
  shares one reference epoch, so the model has never seen that value change.
  Sampling `(account, as_of_date)` pairs across several historical cut points
  is the standard construction for a model meant to run forward in time.
- [ ] **Add a temporal holdout split.** `data/splitting.py` splits by party
  group with no time dimension, so forward generalization is currently
  unmeasured. Train as-of T, evaluate as-of T plus delta.

### Bin overflow and data integrity

- [ ] **Keep the newest bins on truncation, not the oldest.** `_pad_bins` in
  `features/temporal.py` copies the first `max_bins` entries, and bin 0 is the
  oldest, so overflow discards the most recent activity. Zero-padding a short
  array has the mirror problem: it fills the recent end, making an older edge
  look recently dormant. Largely moot once the rolling window above lands.
- [ ] **Make truncation fatal.** It currently raises a `warnings.warn` that
  nothing escalates, so real temporal data is dropped with only a message on
  stderr.
- [ ] **Replace the three-maxima consistency check with a per-edge one.**
  `derive_max_bins.gsql` compares three global maxima, which agree even when
  individual edges disagree. Counting edges where `num_bins` differs from
  `amount_bins.size()` costs the same traversal and actually catches the
  mixed-vintage case that produces overflow.

### Training throughput

Each batch makes three sequential TigerGraph calls before the GPU runs, with no
prefetching.

- [ ] **Fix the retry replay.** `_resilient_batches` in `training/loop.py`
  recovers from a transient error by skipping already-delivered batches, but it
  skips the *results* rather than the *requests*: reaching batch N means
  batches 0 through N-1 were already pulled from the loader, and each pull
  issues all three queries. One network blip re-queries the epoch so far. The
  fix is to pass a skip count into the loader factory so it advances past those
  seed batches without building a loader or issuing a query. **This is a bug,
  not a tuning item.**
- [ ] **Collapse the three per-batch queries into one.** The sampling query
  already holds the neighbourhood's account ids, so returning node and edge
  attributes in the same response removes two round-trips and deletes the
  edge re-fetch and re-alignment in `pyg/transform.py`.
- [ ] **Cache node features for the run.** They do not change during training,
  and `training/train.py` already fetches every train account once to fit the
  normalizer.
- [ ] **Instrument the database versus GPU split.** Only total wall-clock per
  batch is measured today, via the tqdm rate in `training/loop.py`. A timer
  around each query call, accumulated per epoch, would give the real ratio.
- [ ] **Prefetch one batch ahead.** Requires making the `NodeIDMapper`
  per-batch first: it is currently a single shared object reset at the top of
  every `sample_from_nodes`, and `pyg/backend.py` documents that the design
  depends on the loader being synchronous.

### Job-start scans

`derive_max_bins` and `derive_reference_epoch` each traverse every `HAS_PAID`
edge once per run. Both values are properties of the dataset rather than of the
graph, and are known by whatever job builds the binned edges.

- [ ] **Read both from a small config file, with the scan as a fallback.**
  The file is two integers and is a constant size regardless of graph size,
  while the scans it replaces grow with the edge count. Note that this
  introduces a new way to trigger the bin-overflow failure described above: the
  file has to be rewritten on every load, including incremental ones, or
  training runs against a stale window and silently drops the newest bins.
  Validating the file against the graph, or stamping it with the load it
  describes, should land with it rather than after it.
- [ ] **Merge the two queries into a single traversal** if the values must stay
  graph-derived. They currently scan the same edges twice.
- [ ] **Move the bin-length integrity check into a periodic validation job**
  rather than running it at every training start.

### Smaller items

- [ ] **Derive the account feature count instead of hardcoding it.**
  `_ACCOUNT_FEATURES = 31` appears in `training/train.py` and
  `scripts/demos/smoke.py` while `NUM_ACCOUNT_FEATURES` is derived from the
  schema spec. Changing the feature list silently desynchronizes them.
- [ ] **Reduce the export payload.** `export_account_features.gsql` returns
  whole vertices including the FastRP embedding, whether or not the caller
  needs it.

## License

MIT (author: Abraham Chandy).
