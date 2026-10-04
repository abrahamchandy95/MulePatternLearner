# Leakage and scaling

What keeps the future, held-out accounts and hidden labels out of training, and what
bounds the work on the client and on TigerGraph. Read it before choosing an evaluation
protocol.

## The experiment scope

`strict_inductive`, the only protocol, withholds whole ownership groups. The scope (a
`Temporal_Training_Scope` vertex, one `Entity_In_Training_Scope` edge per Account and
Party) keeps experiment membership apart from the business data. Accounts and Parties
joined by any ownership tenure form a component, which a seeded hash places in train,
validation or test by the split shares (half, a quarter and a quarter of the groups in the
built-in run). Ownership history serves grouping only, conservatively, never as a
feature, and mule truth is never read. Membership is verified before the scope becomes
ready.

`scope.unowned` places accounts no Party owns
([`create_training_scope`](../reference/queries.md#create_training_scope) has the rules).
The built-in `"linked"` shares unowned external and ledger accounts (partition 1, visible
in every phase) and gives an unowned internal account the partition and group of its only
owned internal deposit counterparty, if it has exactly one:

- Ledger accounts (`account_type = "gl"`, the bank's fee and interest books) belong to no
  customer; hashing them would hide every fee posting to a held-out one from training.
- Internal credit accounts, unowned here, almost always transact with exactly one deposit
  account, their holder: linking keeps a held-out holder's card activity out of training
  and a training root's own card payments in it.
- Links come after the components are final, so a component with a Party keeps its
  component, partition and group under every rule. The rule is checked on every use
  ([`summarize_scope_policy`](../reference/queries.md#summarize_scope_policy)).

Visibility is cumulative:

| Phase | A context may use | Optimizer updates? |
|---|---|---|
| Train (1) | Train members only | Yes |
| Validation (2) | Train and validation members | No |
| Test (3) | Every member of the frozen scope | No; the model and threshold are already fixed |

Shared tokens, devices, IPs and addresses are allowed, and accounts outside the frozen
scope cannot silently enter training. The benchmark is of unseen existing groups: held-out
accounts need not have been opened after training. Membership lives on the graph and a
request names only its scope and phase: a per-request exclusion list, or filtering only
the returned neighbours, would leave held-out accounts in the aggregates. Context keys
carry scope and phase, so no context crosses scopes and no feature computed on the full
graph is reused.

The isolation requirements:

- changing, adding or deleting held-out accounts leaves every training feature,
  neighbour and time vector unchanged;
- future transactions and future or late-arriving associations leave earlier inputs
  unchanged;
- changing hidden truth with the observed labels fixed leaves the training weights and
  selection unchanged;
- the splits keep their ownership groups apart, and available positives are counted
  without budgets changing on their own;
- scoped pair gaps, frequency counts and visibility match a small independent reference.

`tests/integration/test_scope_isolation.py` (marker `graph_write`) checks the first two on
the graph: it adds and changes held-out payments and shared-identifier associations,
compares complete training contexts, checks that future events stay out, scores an
account inserted after an accelerator update, and removes its fixtures. Unit tests check
scope propagation, label separation, bounded seed selection and batch-local ids. None of
it shows production-scale performance or rules out missing source availability data.

## Leakage is broader than labels

A feature can carry an excluded account's activity when its node is absent, so held-out
accounts must add nothing to neighbourhoods, payments, associations, counts, amounts,
distinct-counterparty counts, degrees, pair histories or association summaries. A
payment with an excluded endpoint account is removed, and the visibility predicate
applied, before every aggregation and sampling step. A shared device or token of the
training graph can be context for a new test account, but what that account added cannot
reach training features.

**Event time is not knowledge time.** A relationship effective in March but received in
May must not appear in a March backtest, and valid-time filters cannot enforce that: it
needs arrival or discovery timestamps, or immutable snapshots with an ingestion
watermark. No configuration flag makes unknown availability safe, and the graph has no
such clocks ([Schema](../reference/schema.md#clocks)).

Other channels are label availability and effective times, preprocessing, model
selection and repeated looks at hidden test scores. The pipeline splits forward in time,
fits nothing on held-out facts, and fixes the class prior, threshold and model before the
audit.

| Risk | Control |
|---|---|
| Future events | `event_seq < cutoff_seq AND event_ts_ms <= cutoff_ms` in every scan; associations at `cutoff_seq - 1`; cuGraph asserts strictly earlier edge times |
| A neighbour sees its own future | A payment's counterparty is read at that payment's cutoff |
| Held-out groups | Scope partitions; an event with a held-out From or To account is removed before features and sampling |
| A hub decision uses the future | Hub counts use only events before the root's cutoff and visible in its phase |
| Label leakage | No label attribute in any feature, population, cutoff or hub query; training reads only the revealed positives |
| Selection on test | The model and threshold come from validation; the test audit is for reporting |

## Three evaluation protocols

| Protocol | What training may see | The question it answers |
|---|---|---|
| New-account chronological holdout | Only accounts and events observable before training ends | Can the model score accounts first observed later? |
| Withheld existing ownership groups | An induced training graph with held-out groups and their contributions removed | Can the model generalise to unseen existing groups? |
| Shared-history chronological evaluation | Earlier unlabelled context from accounts later used as test seeds | Can the model forecast risk on the observable network? |

The first is the operational inductive test, the second a controlled stress test, and
neither is recovering masked labels of training accounts. The pipeline implements the
second (`strict_inductive`); the third was its `shared_history` protocol, now removed, and
a separate run of it must be reported under its own name. If a generator creates every
mule before the training cutoff there may be no positive cold-start sample: change the
simulation or observation period rather than moving old accounts into a supposedly
new-account sample. `mule score` scores accounts outside the scope from the history
before any date, so the architecture is inductive; quality on newly arriving accounts
needs its own chronological sample.

## Observed labels and truth are different interfaces

Training reads labels only through `data.ports.ObservedLabelReader` (`account_id`,
`known_positive`, `known_from_ms`), which refuses oracle columns and takes an unlisted or
zero account as unlabelled, not legitimate. The rows come from the graph's label contract
([Labels](../reference/labels.md)), so another label feed writes there and the model and
loss stay as they are. Ground truth is read only by the one-time [label
reveal](label-reveal.md) and, after selection, by the audits and diagnostics through
`evaluation.truth.TruthReader`, which import contracts keep off training's import surface
with the oracle adapter. The audit applies the already selected threshold and never takes
an unknown label as a negative: a bare 0/1 column cannot tell unknown from adjudicated
negative without a contract saying so.

## The frozen source

A dataset records the graph's vertex counts, its scope and the hashes of the query files
it was prepared with. Every run reading the graph for it (training, its resume, the
audits, the diagnostics) first checks the installed query texts, the vertex counts (scope
vertices aside, as experiment metadata) and the scope's header and rule, and refuses a
change. A same-count edit, or a change during a run, goes undetected: keep the graph
frozen for the experiment, and after reloading or materially changing it use a new scope
id and prepare a new dataset.

## Bounded transport

Contexts are streamed, never replicated. `data.contexts.ContextSource`, the one source of
contexts (a `ContextReader` to batching, training and scoring), requests each batch's
contexts through bounded installed-query REST calls and returns rows in key order, `None`
where TigerGraph rejected a request, counting rejections by status and hop.

| Bound | How |
|---|---|
| Requests | At most `transport.request_batch_size` contexts per call (8; at most 64) and `transport.query_concurrency` calls in flight (16; at most 16), in one bounded pool shared by every batch thread; a key another thread is reading or requesting is awaited, not requested again |
| Memory | An LRU of `transport.context_lru_capacity` contexts (256; at most 4,096), keyed by hop and context |
| Disk | The dataset's context cache (`data/<dataset id>/contexts/`), read before requesting and keeping every returned row as it came, so later runs of the dataset request nothing an earlier run fetched; never holds labels, and only connections that passed the frozen-source check open it, so `mule score` has none ([Outputs](../reference/outputs.md#a-prepared-dataset-datadataset-id)) |
| Seeds | Preparation pages the scope population 10,000 rows at a time, keeps label-blind hash reservoirs (20,000 train, 2,000 validation, 2,000 test) plus the observed positives, and drops each page; the wider graph is reached only through server-side membership |
| Batches | Admitted before any request or allocation ([Sampling](sampling.md#bounds)) |
| Shutdown | The pool's workers are daemon threads: after an error or Ctrl-C, training and scoring close the source without waiting, cancelling queued requests; those in flight finish or are dropped at exit |

### Failures and retries

`tigergraph.executor` classifies every failure before retrying, each class with its own
budget:

| Class | Examples | Budget |
|---|---|---|
| Availability | Connection errors; HTTP 408, 429 and every 5xx but a bare 500; HTML error pages (including TigerGraph Cloud's while a workspace starts); chunked-encoding errors; overload and not-ready messages | Jittered exponential backoff (4 s, capped at 60 s) until `transport.max_outage_s` (900 s) after the operation's first such failure; then `TigerGraphUnavailableError` |
| Server timeout | REST-3002 (also inside a JSON 5xx body), a timeout message, a client read timeout | One retry, then `ServerTimeoutError` |
| Suspected deterministic | A bare HTTP 500, a response neither JSON nor HTML, query out of memory | One retry |

Every other error (contract, validation, per-request statuses) is permanent and raises
at once; writes get one attempt. `transport.max_query_attempts` (6) caps the attempts that
count, all but availability failures within 30 seconds, so the wall clock bounds a
fast-failing outage and the cap a request that always hangs. Worker threads share one
"back off until" time: one finding TigerGraph unavailable pauses the others, and any
success ends the pause. The client raises the HTTP error of a 5xx, 408 or 429 before
pyTigerGraph reads the body (else a JSON-bodied 503 looks like a permanent query error),
and every HTTP session times out (30 s to connect, 600 s to read).

A timed-out block of keys is split in half at once and each half requested alone,
isolating a slow key in logarithmically many extra calls. A single key gets one retry,
then raises `ContextTimeoutError` with the key and hop. That is fatal on purpose: which
keys time out depends on server load, and dropping them would make the training data
depend on it. Retry when the server is less busy, or prepare again with a lower
`max_history`.

## What scale is shown, and what is not

Measured on 24 September 2026 against the installed queries (TigerGraph Cloud, from a
MacBook with an M2 Max), and later on the CUDA host:

| Measurement | Result |
|---|---|
| One root context | 0.25 s |
| 64 root contexts in one request | 3.5 s |
| One 64-root batch, 8 contexts per request, 16 in parallel, cold | 11 to 12 s (76 REST calls, about 960 contexts, 357 stubs) |
| The same batch, 16 contexts per request, 8 in parallel | 19.6 s |
| A warm step with prefetch | 4.9 s |
| Scope creation plus preparation | 6 min 19 s |
| Installing every training query | about 50 min, most of it the context query |
| A training step on the CUDA host | about 3 s; a run with early stopping takes about an hour (the reference run stopped at epoch 11) |

Training is bound by TigerGraph, not the GPU. Scans are not indexed by time, so server
work per context grows with its all-time adjacency in each payment relation; a heap
bounds the rows returned, not the history scanned, and hubs never reach the server as
children. Scope creation visits every ownership component, population pages can rescan
membership, and cutoff resolution scans every event clock. A bounded response bounds
neither server memory nor latency, and the client cannot promise that TigerGraph or other
processes never run out of memory.

Scale far beyond this graph is not shown. It needs indexed cutoff watermarks,
time-organised adjacency, scalable partition and seed selection, and maintained pair
predecessor state, handling late events and backfills explicitly. Levers before that: on
a larger instance raise the bound on `transport.query_concurrency`
(`contract.bounds.QUERY_CONCURRENCY`, which the default of 16 already reaches) and then
the setting; shrink the child pool; process a call's requests together; or materialise
the event-intrinsic pair features, which depend on neither scope nor cutoff.

## Transport alternatives

| Option | Advantages | Costs and open work |
|---|---|---|
| Bounded custom REST requests (this pipeline) | Simple, minimal client retention | Round trips, JSON overhead and repeated scans |
| TigerGraph GDS with Kafka-backed batches | Bounded delivery and prefetch, decoupled producer and consumer | Broker, security and Cloud configuration; custom time and visibility semantics still needed |
| Scoped sharded exports to storage near the GPUs | Repeatable multi-epoch training, distributed readers | Snapshot refresh and storage cost; export only the relevant partitions |
| A sampler service with a shared cache | Reuse across GPU workers and epochs | Operational complexity; keys must include snapshot, scope and cutoff |

Check the installed version's documentation before choosing GDS: `filter_by` selects
seeds, not every traversed neighbour, so a seed filter is no inductive boundary, and a
time filter after fetching cannot undo leaked server-side aggregates. Per the
[documented HTTP and Kafka difference](https://www.tigergraph.com/docs/pytigergraph/1.6/gds/dataloaders),
HTTP may collect batches before iteration while Kafka delivers them incrementally, so an
iterator API does not mean bounded streaming.

TigerGraph applies scope and time predicates, finds neighbours and predecessors,
aggregates and computes the requested Fourier vectors; forward passes, gradients and
optimisation stay in PyTorch. A GSQL calculation is not a learned embedding.

## References

- [TGAT (Xu et al., ICLR 2020)](https://arxiv.org/abs/2002.07962)
- [TGB, the benchmark (Huang et al., 2023)](https://arxiv.org/abs/2307.01026)
- [TGB evaluation rules](https://tgb-website.pages.dev/docs/leader_rules/)
- [TigerGraph data loaders](https://www.tigergraph.com/docs/pytigergraph/1.8/gds/dataloaders)
- [TigerGraph GDS factory functions](https://www.tigergraph.com/docs/pytigergraph/1.8/gds/factory-functions)
- [GSQL SELECT evaluation and sampling](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/select-statement/)
