# Set up a graph

From an empty TigerGraph 4.2.5 instance to a graph `mule train` can train on. A person
creates the schema and loads the data; the first `mule train` (or `mule install`) does the
rest. [Schema](../reference/schema.md) describes the graph,
[Labels](../reference/labels.md) the account label contract.

Never run these steps on a populated graph: `schema.gsql` is fresh-graph DDL, and a schema
change can invalidate compiled queries and positional loading jobs. To load again,
recreate the graph and follow every step, or [reuse the graph](#reuse-a-graph).

## 1. Create the graph

```bash
gsql gsql/schema/schema.gsql
```

Run it on the empty instance, or paste it into TigerGraph Cloud's GSQL editor. It creates
the graph `Mule_Pattern_Learner` (`contract.server.GRAPH_NAME`, where every query is
installed) with eight vertex and nineteen edge types and their reverse edges, through a
schema-change job it then drops.

## 2. Load the data

Load the tables of PhantomLedger's mule-temporal export (payments, accounts, parties,
tokens, devices, IPs, addresses, participation edges, association tenures) through Kafka
or the loading UI of GraphStudio or TigerGraph Cloud. The data producer owns those jobs;
this repository defines only the Account loader, for a file:

```bash
gsql gsql/schema/account_loading.gsql
```

It creates `load_accounts`, which reads the export's Account table as it is: the fifteen
label-contract columns in order (`contract.graph_schema.ACCOUNT_LOAD_COLUMNS`). A Kafka
job or UI mapping maps each column to the same-named Account attribute ([Loading
accounts](../reference/labels.md#loading-accounts) gives the columns and the positional
order). Map every column, `mule_ring_id` included: unmapped, it stays -1 (no ring) and the
audits resample each mule alone instead of with its ring.

The rest must keep the schema's contract ([Associations and valid
time](../reference/schema.md#associations-and-valid-time)): one chronological sequence
domain for payments and association changes, positive sequences, edge clocks equal to
their event's, the canonical role counts per event, and valid-time tenures with explicit
start sequences. A Zelle payment is a `Zelle_Transfer` only; only USD amounts are used.

## 3. Connect

Copy `.env.example` to `.env` and fill in `HOST`, `GRAPHNAME` (`Mule_Pattern_Learner`) and
`SECRET` (the REST++ secret). Nothing else is configured.

## 4. Install the queries

```bash
mule install
```

It adds (or [replaces](#reuse-a-graph)) the `Temporal_Training_Scope` vertex type of
`gsql/schema/scope_vertex.gsql`, then installs the training queries of `gsql/queries/` and
`gsql/evaluation/`: about 50 minutes, mostly the context query, within a 90-minute wait.
If the wait runs out, rerun once `mule check` lists no stale training queries; it installs
only what is still stale ([Installation](../reference/queries.md#installation)).
`mule train` installs anything stale too, so this step only does it ahead of time. The
analytics queries of `gsql/analytics/` wait for `mule diagnose`.

## 5. Check the labels

Run `validate_label_contract` and read a page of `read_ground_truth`, as
[Loading accounts](../reference/labels.md#loading-accounts) shows. A PhantomLedger load
marks every label unknown and masked, so until the first `mule train` reveals the known
mules, `invalid_unknown` counts every mule and `revealed_positives` is 0. Every other
count must be zero; after the reveal, all of them are.

## 6. Prepare and train

```bash
mule check
mule train
```

`mule check` confirms the graph, the scope vertex type and the installed queries, and ends
"Not ready" until a dataset exists. The first `mule train` then:

1. creates the frozen scope `strict_mule_v3`, splitting every Account and Party by
   ownership group into train, validation and test (half, a quarter and a quarter of the
   groups: `scope.train_share`, `scope.validation_share`, `scope.test_share`), with the
   `linked` rule for accounts no party owns. It writes only one scope vertex, which
   records the shares, and one membership edge per Account and Party;
2. reveals the known mules once ([Label reveal](../explanation/label-reveal.md)), writing
   the label fields of every internal Account, and checks the label contract;
3. prepares `data/<dataset id>/`: seed reservoirs, observed labels, cutoff sequences and
   the hub registry (scope and preparation took 6 minutes 19 seconds on the reference
   graph);
4. trains ([Train and evaluate](train-and-evaluate.md)).

The scope and label fields are the only writes: relationships and business attributes
never change, and a graph that already has the scope and known labels is only read.

## Reuse a graph

To reload a graph that has the schema and queries without recreating it:

1. Clear its data in the GSQL shell with `CLEAR GRAPH STORE`. It deletes every vertex and
   edge (scopes and revealed labels included) of every graph on the instance, keeping the
   schema and installed queries: if another graph shares the instance, recreate the graph
   and follow every step instead.
2. Push the data again ([Load the data](#2-load-the-data)).
3. Run `mule install`. Beyond installing what is stale, it replaces a
   `Temporal_Training_Scope` type that differs from `gsql/schema/scope_vertex.gsql` (such
   as one from before the scope recorded its split shares, which `mule check` reports as
   outdated): it drops the repository queries using the scope types, drops and reapplies
   the types and installs every training query, about 50 minutes. It changes nothing while
   the graph holds a scope vertex, or while a query or edge type that no repository file
   defines uses the scope types ([The scope
   types](../reference/queries.md#the-scope-types)). `mule train` and `mule diagnose`
   refuse an outdated type and say to run `mule install`.
4. Run `mule train`, which creates the scope and reveals the known mules on the new data
   ([Prepare and train](#6-prepare-and-train)). A dataset prepared before the clear with
   the same `scope.id` describes a scope that is gone: give the run a new `scope.id` first
   ([After the data changes](#after-the-data-changes)).

## After the data changes

Keep the graph frozen while in use: every run checks the vertex counts, installed query
texts and scope against its dataset's record and refuses a changed graph ("Graph counts
changed"). After a reload or material change the old dataset and scope describe data that
is gone, so give the built-in run a new `scope.id` (the default of `config.ScopeConfig`):
the next `mule train` creates a scope and prepares a new dataset. The old scope vertex is
harmless experiment metadata. After a schema change on a populated graph, verify and
restore the compiled queries (`mule install`) and positional loading jobs.
