# Leakage and scaling

What keeps the future, held-out accounts and hidden labels out of training, and what
bounds the work on the client and on TigerGraph. Read it before choosing an evaluation
protocol.

## The experiment scope

Every run samples inside a frozen experiment scope, `strict_inductive` being the only
protocol: entire ownership groups are withheld from training. The scope is a
`Temporal_Training_Scope` vertex with one `Entity_In_Training_Scope` edge per Account and
Party, holding experiment membership apart from the business data. Its creation groups
the Accounts and Parties connected by any ownership tenure into components and gives each
a deterministic train, validation or test partition (70, 15 and 15%) from a seeded hash.
It never reads mule truth; all ownership history is used, conservatively, for grouping,
never as a feature. Membership is verified before the scope becomes ready.

An Account without an ownership edge has no Party in its component. `scope.unowned`
decides where such accounts go when the scope is created:

| `scope.unowned` | Unowned external accounts | Unowned internal accounts |
|---|---|---|
| `"independent"` | Their own hashed partition | Their own hashed partition |
| `"shared"` | Partition 1, visible in every phase (group `shared:<component>`) | Their own hashed partition |
| `"linked"` (built in) | As `"shared"` | The partition and group of their only owned internal deposit counterparty, when there is exactly one; otherwise their own hashed partition |

Unowned bank ledger accounts (`account_type = "gl"`, the bank's income books for fees and
interest) are shared like external ones under `"shared"` and `"linked"`: they belong to no
customer, and a hashed partition would hide every fee posting to a held-out ledger
account from training. Internal credit accounts are not owned in this data but almost
always transact with exactly one deposit account, their holder; linking them keeps a
held-out holder's card activity out of training and a training root's own card payments
in it. The link is assigned after the ownership components are final, so components with
a Party keep the same component, partition and group under every rule. The scope's rule
is checked on every use ([`summarize_scope_policy`](../reference/queries.md#summarize_scope_policy)).

Visibility is cumulative:

| Phase | A context may use | Optimizer updates? |
|---|---|---|
| Train (1) | Train members only | Yes |
| Validation (2) | Train and validation members | No |
| Test (3) | Every member of the frozen scope | No; the model and threshold are already fixed |

Shared tokens, devices, IPs and addresses are allowed. An excluded Account's or Party's
payments, associations, amounts and pair histories are removed from training contexts:
a payment with any excluded account endpoint is removed entirely, before any feature or
sampling. New accounts outside the frozen scope cannot silently enter training. This is
an unseen-existing-group benchmark: it does not imply the held-out accounts were opened
after training.

Membership lives on the graph, so a request names only its scope and phase: sending a
long exclusion list with every request, or filtering only the neighbours a query
returns, would not keep a held-out account out of the aggregates. Every context key
carries its scope and phase, so a context of one scope is never reused for another, and
no feature computed on the full graph is reused.

These requirements define the isolation; the scope isolation test checks the first two
on the graph:

- adding, deleting or changing held-out accounts leaves every training feature,
  neighbour and time vector unchanged;
- future transactions and future or late-arriving associations leave earlier inputs
  unchanged;
- changing hidden truth with the observed labels fixed leaves the training weights and
  the selection unchanged;
- train, validation and test keep their ownership groups apart, and the available
  positives are counted without budgets changing on their own;
- scoped pair gaps, frequency counts and visibility match a small independently
  computed reference.

That test, `tests/integration/test_scope_isolation.py` (marker `graph_write`), adds and
changes held-out payments and shared-identifier associations on the graph and compares
complete training contexts, checks that future events stay out, and scores an account
inserted after an accelerator update; its fixture vertices are removed afterwards. Unit
tests check scope propagation, label separation, bounded seed selection and batch-local
ids. None of it establishes production-scale performance or rules out missing source
availability data.

## Leakage is broader than labels

For a strict experiment, held-out accounts must contribute nothing to training
neighbourhoods, nor to counts, amounts, distinct-counterparty counts, degrees, pair
histories or association summaries: a model can exploit an excluded account's activity
through a feature even when the account's node is absent. So the visibility predicate
applies before every aggregation and every sampling step. A shared device or token that
existed in the training graph can be legitimate context for a new test account, but what
that test account added cannot appear in training features.

**Event time is not knowledge time.** A relationship effective in March but received in
May must not appear in a March backtest, and valid-time filters alone cannot enforce that.
It needs arrival or discovery timestamps, or immutable snapshots with an ingestion
watermark; no configuration flag makes unknown availability safe. The graph has no such
clocks ([Schema](../reference/schema.md#clocks)).

Label availability, label effective times, preprocessing, model selection and repeated
looks at hidden test scores are further channels. The pipeline uses forward time splits,
fits nothing on held-out facts, and fixes the class prior, the threshold and the model
before the audit.

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

The first is the operational inductive test and the second a controlled stress test;
both differ from recovering masked labels of training accounts. The pipeline implements
the second (`strict_inductive`). The third, its earlier `shared_history` protocol, was
removed; a separate run of it must be reported under its own name. A generator that
creates every mule before the training cutoff may give no positive cold-start sample;
change the simulation or the observation period rather than moving old accounts into a
supposedly new-account sample. `mule score` can score accounts outside the scope from the
history before any date, which shows the architecture is inductive; measuring quality on
newly arriving accounts needs its own chronological sample.

## Observed labels and truth are different interfaces

Training depends on `data.ports.ObservedLabelReader`, whose rows hold `account_id`,
`known_positive` and `known_from_ms`; an unlisted or zero account is unlabelled, not a
confirmed legitimate account, and oracle columns are refused. Every run reads the labels
revealed in the graph's label contract ([Labels](../reference/labels.md)), so a
production system with another label feed writes its known positives and discovery times
there, and the model and loss need no rewrite.

Ground truth is read only by the one-time [label reveal](label-reveal.md) before
training, and by the audits and diagnostics after selection, through
`evaluation.truth.TruthReader`. Import contracts keep training's modules from importing
the evaluation, the diagnostics and the oracle adapter, so the truth is not even on
training's import surface. The audit applies the model's already selected threshold;
an unknown label is never a negative, and a bare 0/1 column cannot tell unknown from
adjudicated negative without a contract that says so.

## The frozen source

A dataset records the graph's vertex counts, its scope and the hashes of the query files
it was prepared with. Every run that reads the graph for a dataset (training, its resume,
the audits, the diagnostics) first checks the installed query texts, the vertex counts
(scope vertices aside, since they are experiment metadata) and the scope's header and
rule, and refuses to run if any changed. Counts cannot detect a same-count edit, and a
change made while a run is in progress is not detected, so keep the graph frozen for the
experiment. After reloading or materially changing the graph, use a new scope id and
prepare a new dataset.

## Bounded transport

Contexts are streamed, never replicated. `data.contexts.ContextSource` is the one source
of contexts (batching, training and scoring see it as a `ContextReader`): it requests each
batch's contexts through bounded installed-query REST calls and returns rows in key order,
`None` where TigerGraph rejected a request, counting rejections by status and by hop.

- **Requests.** At most `transport.request_batch_size` contexts per call (8; at most 64)
  and `transport.query_concurrency` calls in flight (16; at most 16), shared by every
  batch-building thread through one bounded pool. A key another thread is already
  reading or requesting is awaited, not requested twice.
- **Memory.** An LRU of `transport.context_lru_capacity` contexts (256; at most 4,096),
  keyed by hop and context.
- **Disk.** The prepared dataset's context cache (`data/<dataset id>/contexts/`): a context
  the LRU lacks is read from the cache before it is requested, and every row TigerGraph
  returns is kept there as it came, so the next run of the dataset, another seed, a
  variant requesting the same groups, or the audits, request none of the contexts an
  earlier run fetched. An entry is named by the hop, the context key, the flags and pool
  requested at that hop, the context contract, the dataset id and the frozen source, so a
  variant of other groups has entries of its own and a changed graph fails the
  frozen-source check before any entry is read. It never holds labels. Only connections
  that passed that check open it; `mule score` reads whatever graph it connects to, which
  need not be any dataset's source, and has none
  ([Outputs](../reference/outputs.md#a-prepared-dataset-datadataset-id)).
- **Seeds.** Preparation pages the scope population 10,000 rows at a time and keeps
  label-blind hash reservoirs (20,000 train, 2,000 validation and 2,000 test accounts)
  plus the observed positives, discarding each page after selection. The wider graph is
  reached only through server-side membership.
- **Batches** are admitted before any request or allocation ([Sampling](sampling.md#bounds)).
- **Shutdown.** The pool's workers are daemon threads. After an error or Ctrl-C, training
  and scoring close the source without waiting: queued requests are cancelled, and
  requests in flight finish on their own or are dropped when the process exits.

### Failures and retries

Every failure is classified before it is retried (`tigergraph.executor`), and each class
has its own budget:

| Class | Examples | Budget |
|---|---|---|
| Availability | Connection errors, HTTP 408, 429 and every 5xx but a bare 500, HTML error pages (including the page TigerGraph Cloud serves while a workspace starts), chunked-encoding errors, overload and not-ready messages | Retried with jittered exponential backoff (from 4 s, capped at 60 s) until `transport.max_outage_s` (900 s) have passed since the operation's first such failure; then `TigerGraphUnavailableError` |
| Server timeout | Code REST-3002 (also inside a JSON 5xx body), a timeout message, a client read timeout | Retried once, then `ServerTimeoutError` |
| Suspected deterministic | A bare HTTP 500, a response that is neither JSON nor HTML, query out of memory | Retried once |

Contract and validation errors, per-request statuses and every other error are permanent
and raise at once, and writes get exactly one attempt. `transport.max_query_attempts` (6)
caps the attempts that count: every attempt but an availability failure that failed
within 30 seconds, so a fast-failing outage is bounded by the wall clock and a request
that hangs on every attempt by the attempt cap. Worker threads share one "back off until"
time, so when one finds TigerGraph unavailable the others pause too, and any success ends
the pause. The client raises the HTTP error of a 5xx, 408 or 429 response before
pyTigerGraph reads its body, which would otherwise make a JSON-bodied 503 look like a
permanent query error; every HTTP session has finite timeouts (30 s to connect, 600 s to
read).

A context request that times out is not repeated whole: a block of several keys is split
in half at once and each half requested on its own, which isolates a slow key in a
logarithmic number of extra calls. A single key gets one retry and then raises
`ContextTimeoutError` naming the key and hop. That is fatal on purpose: which keys time
out depends on server load, so dropping one would make the training data depend on it.
Retry when the server is less busy, or prepare again with a lower `max_history`.

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

Training is bound by TigerGraph, not the GPU. Server work per context grows with the
context's all-time adjacency in each payment relation, since scans are not indexed by
time; a heap bounds the rows returned, not the history scanned, and hubs never reach the
server as children. Scope creation visits every ownership component, population pages
can rescan membership, and cutoff resolution scans every event clock. A bounded response
is no bound on server memory or latency, and the client cannot promise that TigerGraph or
other processes never run out of memory. Scale far beyond this graph has not been
demonstrated: it needs indexed cutoff watermarks, time-organised adjacency, scalable
partition and seed selection, and maintained pair predecessor state, with late events and
backfills handled explicitly. Levers before that: on a larger instance, raise the bound
on `transport.query_concurrency` (`contract.bounds.QUERY_CONCURRENCY`, which the default
of 16 already reaches) and then the setting, shrink the child pool, process a call's requests together, or
materialise the event-intrinsic pair features, which do not depend on scope or cutoff.

## Transport alternatives

| Option | Advantages | Costs and open work |
|---|---|---|
| Bounded custom REST requests (this pipeline) | Simple, minimal client retention | Round trips, JSON overhead and repeated scans |
| TigerGraph GDS with Kafka-backed batches | Bounded delivery and prefetch, decoupled producer and consumer | Broker, security and Cloud configuration; custom time and visibility semantics still needed |
| Scoped sharded exports to storage near the GPUs | Repeatable multi-epoch training, distributed readers | Snapshot refresh and storage cost; export only the relevant partitions |
| A sampler service with a shared cache | Reuse across GPU workers and epochs | Operational complexity; keys must include snapshot, scope and cutoff |

Check the installed version's documentation before choosing GDS: `filter_by` selects
seeds, not every traversed neighbour, so a seed filter alone is no inductive boundary,
and a time filter after fetching cannot undo leaked server-side aggregates. The
[documented HTTP and Kafka difference](https://www.tigergraph.com/docs/pytigergraph/1.6/gds/dataloaders)
matters too: HTTP may collect batches before iteration, while Kafka delivers them
incrementally. Do not infer bounded streaming from an iterator API.

TigerGraph applies scope and time predicates, finds neighbours and predecessors,
aggregates and computes the requested Fourier vectors; the network's forward passes,
gradients and optimisation stay in PyTorch. A GSQL calculation is not a learned
embedding.

## References

- [TGAT (Xu et al., ICLR 2020)](https://arxiv.org/abs/2002.07962)
- [TGB, the benchmark (Huang et al., 2023)](https://arxiv.org/abs/2307.01026)
- [TGB evaluation rules](https://tgb-website.pages.dev/docs/leader_rules/)
- [TigerGraph data loaders](https://www.tigergraph.com/docs/pytigergraph/1.8/gds/dataloaders)
- [TigerGraph GDS factory functions](https://www.tigergraph.com/docs/pytigergraph/1.8/gds/factory-functions)
- [GSQL SELECT evaluation and sampling](https://www.tigergraph.com/docs/gsql-ref/4.2/querying/select-statement/)
