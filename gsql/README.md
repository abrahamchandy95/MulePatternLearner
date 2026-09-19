# GSQL for the temporal graph

The live graph uses `schema/temporal_schema.gsql`. Use the following files on the
`temporal` branch:

| File | Purpose |
| --- | --- |
| `schema/temporal_schema.gsql` | Fresh graph DDL, including integer Account mule truth and boolean masking |
| `schema/temporal_account_loading.gsql` | Fifteen-column Account CSV contract |
| `features/temporal_fourier64.gsql` | Shared 64-dimensional Fourier calculator and public wrapper |
| `features/zelle_pair_time64.gsql` | Zelle pair gaps, cutoff ages, and 1h/24h/7d counts |
| `features/payment_pair_time64.gsql` | Equivalent calculations for other payment rails |
| `temporal/account_supervision.gsql` | Paginated truth/mask export and label-contract validation |

For existing graphs, use the scripts in `scripts/temporal/` and each migration's
stated prerequisites. Do not run fresh graph DDL or the original empty-graph
migration on the populated instance. Schema changes may invalidate compiled
queries and positional loading jobs; verify and restore both afterwards.

The other files remain from the static implementation. In particular,
`schema/schema.gsql`, `schema/loading_job.gsql`, `features/temporal_features.gsql`,
the static sampling/export queries, and the legacy Python GSQL registry target
the old schema. Do not bulk-install that registry on the temporal graph.

The pair queries default to `persist=false`. Their `max_events` limit bounds
pair results and sorting, but still requires traversing the sender's candidate
history. They are POC extraction queries, not a batched temporal training
sampler. The live trainer uses the separate context sampler described below.

See [encoding semantics](../docs/temporal_encoding.md),
[account labels](../docs/account_mule_labels.md), and the
[live MCP review](../docs/temporal_gsql_review.md).

## Live temporal training queries

The training path installs `temporal/training_context.gsql`,
`temporal/training_scope.gsql`, `temporal/training_population.gsql`,
`temporal/training_cutoffs.gsql` and `features/temporal_fourier64.gsql`:

```bash
python -m mule_pattern_learner.temporal.live.cli install
```

The installer applies `schema/migrations/temporal_training_scope.gsql` when
needed. Scope creation/finalization write experiment metadata only. The population,
context and cutoff queries are read-only. Business labels and relationships are
not changed by preparation.

The context query is generated from the shared Python relation/window contract.
Strict mode removes excluded Account/Party contributions before aggregation,
sampling and predecessor searches. The paged scope population exports observed
supervision only and skips label reads for external observed-label providers.
Complete synthetic truth/masking queries are excluded from the normal installer.

Existing pair queries provide independent timing checks; the legacy stored
whole-history feature queries are not used by this model. See the
[live training guide](../docs/live_temporal_training.md) and
[GSQL feature catalog](../docs/gsql_feature_catalog.md) for dimensions,
all-payment time-encoding support and remaining server scan limits.
