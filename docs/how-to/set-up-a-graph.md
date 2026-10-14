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

Export the 27 tables of PhantomLedger's mule-temporal export (payments, accounts, parties,
tokens, devices, IPs, addresses, participation edges, association tenures) as CSV files
with their header line, each named after its table, such as
`\copy mule_temporal."mt_Party" TO 'Party.csv' WITH (FORMAT csv, HEADER true)`. Put them
on the TigerGraph server, then create the loading job and run it with every file:

```bash
gsql gsql/schema/loading.gsql
gsql 'USE GRAPH Mule_Pattern_Learner RUN LOADING JOB load_graph USING party_csv="/data/Party.csv", account_csv="/data/Account.csv", ...'
```

`load_graph` has one file variable per table, its name in lower case and `_csv`
(`Party_Owns_Account.csv` is `party_owns_account_csv`), and loads each attribute from its
same-named column (an edge's endpoints from `from_id` and `to_id`), skipping the columns
the schema keeps no attribute for and the reverse edges, which `REVERSE_EDGE` creates.
TigerGraph reads a column name from a file's own header line only for a file the job
names when it is created, so the job declares each table's header in the export's column
order: a file of another order needs its header changed. A run that names some files
loads only those. A Kafka job or the loading UI of GraphStudio or TigerGraph Cloud maps
each column the same way ([Loading accounts](../reference/labels.md#loading-accounts)
gives the Account columns and their positions). Map `mule_ring_id`: unmapped, it stays -1
(no ring) and the audits resample each mule alone instead of with its ring.

Loading jobs written for the earlier schema, which kept attributes such as the payments'
`channel`, do not fit a fresh graph of this one: load it with `load_graph`. A graph built
with the earlier schema keeps its data; never run this loading file on it.

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
`mule train` installs anything stale too, so this step only does it ahead of time.

## 5. Check the labels

Read a page of the loaded truth in the GSQL shell, `RUN QUERY read_ground_truth("", 100)`
([Loading accounts](../reference/labels.md#loading-accounts)).

## 6. Prepare and train

```bash
mule check
mule train
```

`mule check` confirms the graph, the scope vertex type and the installed queries, and ends
"Not ready" until a dataset exists. The first `mule train` then:

1. creates the graph's one split, every Account and Party by ownership group into train,
   validation and test (half, a quarter and a quarter of the groups: `scope.train_share`,
   `scope.validation_share`, `scope.test_share`), placing accounts no party owns by one
   rule ([`create_training_scope`](../reference/queries.md#create_training_scope)). It
   writes only one scope vertex, which records the split seed and shares, and one
   membership edge per Account and Party;
2. reveals every mule a bank would have discovered by each split's cutoff ([Label
   reveal](../explanation/label-reveal.md)), writing the label fields of every internal
   Account;
3. prepares `data/<dataset id>/`: seed reservoirs, observed labels, cutoff sequences and
   the hub registry (split and preparation took 6 minutes 19 seconds on the reference
   graph);
4. trains ([Train and evaluate](train-and-evaluate.md)).

The split and label fields are the only writes: relationships and business attributes
never change, and a graph that already has the split and the configured reveal's labels
is only read. Labels another reveal wrote are revealed again when a dataset is prepared,
and so are the reveal's own when the run creates the split, whose partitions they follow.

## Reuse a graph

To reload a graph that has the schema and queries without recreating it:

1. Clear its data in the GSQL shell with `CLEAR GRAPH STORE`. It deletes every vertex and
   edge (the split and revealed labels included) of every graph on the instance, keeping
   the schema and installed queries: if another graph shares the instance, recreate the
   graph and follow every step instead.
2. Push the data again ([Load the data](#2-load-the-data)).
3. Run `mule install`. Beyond installing what is stale, it replaces scope types that
   differ from `gsql/schema/scope_vertex.gsql` (such as a `Temporal_Training_Scope` keyed
   by a scope id, from when a graph held several splits, or a membership edge with
   `group_id`, which `mule check` reports as outdated): it drops the repository and
   retired queries using the scope types, drops and reapplies the types and installs
   every training query, about 50 minutes. It changes nothing while the graph holds a
   scope vertex, or while a query neither the repository's nor retired, or an edge type
   the file does not declare, uses the scope types ([The scope
   types](../reference/queries.md#the-scope-types)). `mule train` and `mule diagnose`
   refuse outdated types and say to run `mule install`.
4. Move the datasets prepared before the clear out of `data/`
   ([After the data changes](#after-the-data-changes)), then run `mule train`, which
   creates the split and reveals the known mules on the new data
   ([Prepare and train](#6-prepare-and-train)).

A graph kept loaded whose scope types are outdated needs only its split deleted by hand
(the first step of [After the data changes](#after-the-data-changes)), then `mule install`
and `mule train`: on the same loaded graph the new split has the same partitions, with the
same seed, shares and rule, since its hash uses TigerGraph's internal vertex ids, and so
the same labels ([On the CUDA host](train-and-evaluate.md#on-the-cuda-host)).

## After the data changes

Keep the graph frozen while in use: every run checks the vertex counts, installed query
texts and split against its dataset's record and refuses a changed graph ("Graph counts
changed"). A graph holds one split, which nothing changes or deletes but its owner. After
a reload or material change the split and the datasets describe data that is gone:

1. Delete the graph's `Temporal_Training_Scope` vertices if they remain (their membership
   edges go with them), with pyTigerGraph's `delVertices("Temporal_Training_Scope")` or
   RESTPP `DELETE /graph/Mule_Pattern_Learner/vertices/Temporal_Training_Scope` (through
   port 14240 or TigerGraph Cloud, `/restpp/graph/...` with a token).
2. Move the old `data/<dataset id>/` aside. Its id comes from the vertex counts, so a
   reload of the same data keeps it, though the new split may differ: the split hashes
   TigerGraph's internal ids, which a load assigns.
3. Run `mule train`: it creates the split and prepares a new dataset.

To split the same data again (other shares or seed), do all three steps:
the old split's datasets no longer match the graph, and `mule train` reuses a ready
dataset without creating a split. After a schema change on a populated graph, verify and
restore the compiled queries (`mule install`) and positional loading jobs.
