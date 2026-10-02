# GSQL for the temporal graph

The live graph uses `schema/schema.gsql`. Every file in this directory
belongs to it. The schema, loading and encoding files are:

| File | Purpose |
| --- | --- |
| `schema/schema.gsql` | Fresh graph DDL, including integer Account mule truth and boolean masking |
| `schema/account_loading.gsql` | Fifteen-column Account CSV contract |
| `queries/fourier64.gsql` | Shared 64-dimensional Fourier calculator (a subquery) |
| `analytics/analytics_context.gsql` | Every feature group of a context, for analysis only (generated) |
| `analytics/zelle_pair_gaps.gsql` | Zelle pair gaps, cutoff ages, and 1h/24h/7d counts |
| `analytics/payment_pair_gaps.gsql` | Equivalent calculations for other payment rails |
| `evaluation/ground_truth.gsql` | Paginated truth/mask export, for evaluation only |
| `queries/label_contract.gsql` | Label-contract validation |

The populated instance reached this schema through one-off migrations that were
applied once and are kept in git history; only the training scope schema change,
`schema/scope_vertex.gsql`, remains, because the installer still applies it. Do not
run fresh graph DDL on the populated instance. Schema changes may invalidate
compiled queries and positional loading jobs; verify and restore both afterwards.

The analytics queries are installed only by `mule diagnose`, where their text differs
(`install(executor, analytics=True)`); training never calls them. `analytics/analytics_context.gsql` is generated beside the
training context query: it computes the training groups as that query does, and the
groups training does not read (windows, recency, amount ratios, association counts,
decayed sums, identity order, pair window counts and device and IP ages). The pair
queries default to `persist=false`. Their `max_events` limit bounds pair results and
sorting, but still requires traversing the sender's candidate history. They are POC
extraction queries for analysis, not a batched temporal training sampler. The live
trainer uses the separate context sampler described below.

See [encoding semantics](../docs/temporal_encoding.md) and
[account labels](../docs/account_mule_labels.md).

## Live temporal training queries

The training path installs `queries/fourier64.gsql`,
`queries/training_context.gsql`, `queries/training_scope.gsql`,
`queries/split_cutoffs.gsql`, `queries/hub_accounts.gsql`,
`evaluation/ground_truth.gsql`, `queries/label_contract.gsql` and
`queries/label_reveal.gsql` (`TRAINING_QUERY_FILES` in
`src/mule_pattern_learner/contract/server.py`). `mule train`
installs whatever is stale; to install ahead of time:

```bash
python -m mule_pattern_learner install
```

The installer applies `schema/scope_vertex.gsql` when
needed. The first run writes to the graph in two steps: scope creation and
finalization write experiment metadata only, and the one-time label reveal
(`queries/label_reveal.gsql`) writes the Account label contract of every internal
account on a graph without known labels. The population, context, cutoff and hub
queries are read-only. Relationships are never changed.

The context query is generated from the shared Python relation/window contract and
computes only the feature groups of the built-in run.
Strict mode removes excluded Account/Party contributions before aggregation,
sampling and predecessor searches. The paged scope population exports observed
supervision only. Only the label reveal, the `label_contract.gsql` audit and the
`ground_truth.gsql` evaluation query read complete truth; no feature or preparation
query calls them.

Existing pair queries provide independent timing checks. See the
[live training guide](../docs/live_temporal_training.md) and
[GSQL feature catalog](../docs/gsql_feature_catalog.md) for dimensions,
all-payment time-encoding support and remaining server scan limits.
