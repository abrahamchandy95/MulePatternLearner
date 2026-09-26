# Temporal feature redesign

This implements the first feature experiment stage of the
[feature plan v4 draft](feature_plan_v4.md). The built-in run (`DEFAULT_RUN` in
`config_schema.py`) keeps memoryless TGAT and removes every hard-window input.
Features are hypotheses; no mule-detection lift has been established. The legacy
feature profile is what the components choose when a raw configuration has no
`feature_groups`; `mule-temporal train` always starts from the built-in groups, so
select legacy groups explicitly in an overrides file to compare against them.

## Implemented inputs

| Group | Model input | Meaning and limits |
|---|---|---|
| `message_core` | Amount, presence, event indicator; relation and rail embeddings | Direction and payment rail remain on each message. This POC includes USD events only. |
| `time_encoding` | 64 payment-age and 64 directed-pair-gap coordinates, gap presence | Age is assessment cutoff minus event time. Pair gap is event time minus the preceding same-pair, same-rail event. Never compared with today's date. |
| `pair_history` | Prior count, first-prior age in seconds, presence | Strictly before the sampled event. Canonical Account recipient wins over Token; unresolved tokens remain distinct. |
| `flow_timing` | Forward/backward delay, observation duration, presence/censoring, amount ratio/presence, same rail | Incoming payment to next outgoing, or outgoing to previous incoming. Both clocks must qualify. This is timing coincidence, not tracing the same funds. Missing outgoing history is censored, not a negative finding. Only Account contexts have this meaning. |
| `event_channel` | Channel embedding | Includes P2P, ATM withdrawal and card purchase. Unrecognized source values use an explicit `other` category. Audit coverage before interpreting this feature. |
| `device_ip_context` | Device/IP age at payment time, presence | Optional. Uses event participation edges and entity first observation; chooses the youngest eligible observation if multiple endpoints exist. Does not claim association tenure age. |
| `entity_meta` | Entity type, external/deposit indicators | Can be removed entirely in the zero-node-input arm. |
| `entity_age` | Age at assessment | Optional legacy control, absent from the new default. No claim that the current cohort tests cold starts. |
| `history_support` | Visible payment participations, fewer-than-five flag | Optional; self-transfers participate in both directions. Counts only visible USD history. |
| `decayed_activity` | Incoming/outgoing counts and amounts at half-lives 1, 7, 30, 90 days | Optional smooth summaries: contribution is `value * 2**(-age/half_life)`. These half-lives still need sensitivity tests. |
| `identity_order` | Starts/ends of owner, token and device tenures since the tenth most recent payment | Ordinal, not hours or days. With fewer than ten payments, starts at the earliest available payment; with none, absent/zero. Never converts sequence differences to time. |
| `rolling_windows`, `amount_ratios`, `recency`, `association_counts`, `pair_window_counts` | Existing summary and pair inputs | Reproducible controls. `amount_ratios` requires `rolling_windows`. Out/in ratio is not pass-through speed. |
| `sampler_meta` | Sampling-stratum embedding | Optional. Describes selection, not behavior. |

The registry in `temporal/live/contract.py` defines groups, dimensions,
transforms, dependencies and query flags. Input dimensions are derived, including
memory admission estimates. Disabled groups have no model columns or projection
weights. The `split` architecture sends entity metadata through attention and
optional **root-only** summaries through a separate MLP. The `single` architecture
retains the old all-node-features-through-attention control. `summary` skips child
fetches and creates no attention parameters. A zero-node arm has no node/base
projection parameters. The old `no_fourier` variant remains compatibility-only;
it is not a no-window comparison.

## History and GSQL

The default sampler requests four recent, three older rank-stratum and two
additional distinct-peer events per payment relation, plus two active associations
per association relation. Duplicate events are suppressed. Python reserves payment
slots before associations and interleaves recent/older/distinct strata; second-hop
sampling uses payments only. Default fanouts are 16/4. Actual event coverage depends
on available relations and must be measured; no fixed claim of fourteen payments
per account is made.

GSQL fills bounded per-relation heaps from existing account participation scans.
Canonical peer joins support both unique-peer summaries and diversity selection.
For Accounts, an ascending traversal computes pair predecessor, count and first
observation without a sender-adjacency scan per sampled event. Token contexts and
the optional legacy pair-window baseline retain the sender scan. Flow timing
searches the bounded retained root history, at most selected-events × history
operations, with no extra payment-adjacency traversal. It is not yet the proposed
linear pending-receipt algorithm. Decayed sums accumulate during the event scans.
Optional device/IP lookups traverse only sampled payments.

**No silent history truncation:** more than `max_history` visible USD events in
one relation returns `history_capacity_exceeded`, which fails preparation/training
with a clear diagnostic. This avoids calling a truncated count “all-time”, losing
an older stratum to a large burst, or fabricating a missing predecessor. It is an
admission guard, not a solution for high-degree hubs. Adjacency candidate scans and
legacy Token pair scans can still be expensive. Temporal indexing, partitioned
history retrieval or an explicitly evaluated approximation are needed at larger
scale. The client cannot promise that TigerGraph or other host processes never
run out of memory.

All event features use `event_seq < cutoff_seq` AND `event_ts_ms <= cutoff_ms`.
Child contexts keep the connecting payment's cutoff. A peer first observed on
that exact payment can contribute its immutable identity metadata, with an empty
prior history; rejecting such a peer would break first-interaction inference.
Association visibility uses
`cutoff_seq - 1`. Strict endpoint membership is applied before counts or sampling.
Legacy pair-window boundaries now consistently use age **less than** the window.
Non-USD events are excluded individually and counted in diagnostics, never summed
with USD or allowed to abort otherwise valid USD contexts. These diagnostics are
not model inputs. There is no FX conversion or multi-currency modeling yet.

## Transport, reproducibility and evaluation

Streaming remains the default: bounded request size, concurrency and LRU, with
batch-local typed/time-qualified tensor IDs. No full-graph export or ID table was
added. SQLite remains an explicit option. Cache provenance now includes extraction
groups and sampler parameters, independent of the model seed. `extraction_groups`
can request a superset for feature-only ablations; the model groups must be a subset.
A different sampler or missing child context requires fresh preparation. This
change does **not** silently build the union of both samplers' neighborhoods or
stage the entire population. Prepared labels/manifests remain immutable.

Checkpoints store the selected input fingerprint and sampler. Old query responses
and old checkpoints fail the new contract check. New batches use current registry
widths for admission. Existing trained weights cannot be reused with changed inputs.

`positive_weight = "prior"` preserves textbook nnPU; `"balanced"` (the built-in
run) is imbalanced nnPU (Su, Chen and Xu, 2021); 0.1 and 0.5 are explicit
cost-sensitive alternatives. A higher weight is not assumed to fix rankings.
Compare them using the same revealed labels, dates, prior and observed validation
proxy, then freeze this setting before feature comparisons. Hidden truth is never
read by preparation, training, early stopping or threshold selection.

The separate `evaluate-final` command enumerates the whole frozen test partition,
includes every truth-positive test account and a uniform sample of truth-negative
accounts, then computes inverse-inclusion-probability-weighted metrics at the
frozen checkpoint threshold. This avoids depending on the few hidden positives
that happen to fall into the training preparation reservoir. It requires complete
binary truth for that test population and one test date. It is a bounded POC audit
(up to one million test metadata rows and 100,000 scored rows), not a production
truth service. Weighted metrics are sample estimates; uncertainty intervals and
ring-level bootstrap remain future evaluation work. Test reports cannot be used
for feature selection. Which mules are observed is decided once in the graph by the
[label reveal](label_reveal.md).

## Commands and experiment order

Use a fresh prepared-data directory for the new contract. Keep the immutable
source identity, scope and revealed labels; do not reveal labels again between
feature arms. The settings are the built-in run; an arm's `--config overrides.toml`
sets only the keys it changes (tables such as `[sampler]` merge key by key).
`--dataset <run>_run/prepared` trains another arm on an existing preparation.

```sh
# Review dimensions and parameter counts; does not train or open truth.
.venv/bin/python scripts/temporal/feature_experiments.py

# Install only after query parity/validation is satisfactory.
.venv/bin/python -m mule_pattern_learner.temporal.live.cli install

# A label-blind account audit; choose the account without consulting truth.
.venv/bin/python scripts/temporal/verify_feature_redesign.py --account ACCOUNT_ID --date 2025-01-01

# After scoped isolation and batch-cost qualification, train an arm.
.venv/bin/python -m mule_pattern_learner.temporal.live.cli train --config overrides.toml --output models/temporal/feature_v4.pt

# Final-only; do not run during feature selection. Truth comes from the graph.
.venv/bin/python -m mule_pattern_learner.temporal.live.cli evaluate-final --checkpoint models/temporal/feature_v4.pt --output artifacts/temporal/final_audit.json
```

Run parity and cost qualification first, then nnPU/noise-floor comparisons, then
summary-only, legacy single/split, event-core/zero-node, individual groups and
combined winners. `feature_experiments()` builds these configurations while holding
source/labels/clocks fixed. Repeat the event and hybrid comparisons under both
samplers, with separately prepared contexts where necessary. The utility does not
launch expensive runs automatically. No synthetic results are promoted as mule
results.

Deferred until the first wave is measured: full counterparty concentration and
reciprocity summaries, thresholded forwarding summaries, root-only causal walks,
neighbor-at-root-clock and external-peer-stub arms, association time brackets,
streamed final-truth joins and cache unions across samplers. Actual association
timestamps, mule onset/discovery clocks, ring IDs and post-cutoff account openings
still require source/schema/generator work; no timestamps were invented here.

GSQL accumulator behavior was checked against the [official reference](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/accumulators): numeric MapAccum values add, so pair counts increment by one and last-event timestamps use MaxAccum values.


## Validation status (20 September 2026)

The temporal and PU-loss suite passed 86 tests. Ruff passed on changed live
pipeline modules and scripts. The full repository suite stops during collection
in nine older tests that import absent legacy modules/functions; none of those
modules were removed by this change.

The core query passed live GSQL semantic checking under
`temporal_training_context_v4_check`. Its installation exceeded the MCP call's
300-second timeout. A later direct status check reported `installed=false`,
`installing=true`, `status=VALID`. Runtime parity and cost qualification are
therefore **not yet passed**. The interpreter fallback reports `Attribute not
exists` for both the modified query and the unmodified legacy query. The optional
device/IP extension is in the repository and its live validation is pending.
The installed production `temporal_training_context` has not been replaced by
this work. Do not start a training run until the exact repository source has
compiled and the parity/isolation/batch checks pass.

The live graph has populated payment-to-device edges, so that feature is a
reasonable optional experiment; presence coverage and numerical age parity
still need measurement. No oracle labels were opened in the live checks.

The waiting local installation retry was stopped; this does not cancel the original server-side compilation. Only the temporary validation query was submitted for installation.
