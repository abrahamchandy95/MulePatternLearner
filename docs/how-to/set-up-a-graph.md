# Set up a graph

From an empty TigerGraph 4.2.5 instance to a graph `mule train` can train on. A person
creates the schema and loads the data; the first `mule train` (or `mule install`) does the
rest. [Schema](../reference/schema.md) describes the graph and
[Labels](../reference/labels.md) the account label contract.

Never run these steps on a populated graph: `schema.gsql` is fresh-graph DDL, and a
schema change can invalidate compiled queries and positional loading jobs. The populated
reference graph reached this schema through one-off migrations kept in git history.

## 1. Create the graph

Run the DDL with the GSQL client on the empty instance (or paste it into the GSQL editor
of TigerGraph Cloud):

```bash
gsql gsql/schema/schema.gsql
```

It creates the graph `Mule_Pattern_Learner` (the name every query is installed on,
`contract.server.GRAPH_NAME`) with its eight vertex types and nineteen edge types and
their reverse edges, through a schema-change job it drops again.

## 2. Load the data

Load the payments, accounts, parties, tokens, devices, IPs, addresses, participation
edges and association tenures. The data producer owns those loading jobs (for the
reference graph, an export of the PhantomLedger simulator); this repository defines only
the Account loader:

```bash
gsql gsql/schema/account_loading.gsql
```

It creates `load_accounts`, which reads an Account CSV with the fifteen columns
of the label contract in a fixed order ([Labels](../reference/labels.md#loading-accounts)).
Whatever loads the rest must keep the schema's contract: one chronological sequence
domain for payments and association changes, positive sequences, edge clocks equal to
their event's, the canonical role counts per event, and valid-time tenures with explicit
start sequences ([Schema](../reference/schema.md#associations-and-valid-time)). A Zelle
payment is a `Zelle_Transfer` only, and only USD amounts are used.

## 3. Connect

Copy `.env.example` to `.env` in the repository and fill in `HOST`, `GRAPHNAME`
(`Mule_Pattern_Learner`) and `SECRET`, the REST++ secret. Nothing else is configured.

## 4. Install the queries

```bash
mule install
```

It adds the `Temporal_Training_Scope` vertex type (`gsql/schema/scope_vertex.gsql`), then
creates and installs the training queries of `gsql/queries/` and `gsql/evaluation/`.
Compiling them takes about 50 minutes, most of it the context query. The command waits
45 minutes for the compilation; if that runs out, run it again later, and it installs only
what is still stale. `mule train` installs whatever is stale too, so this step only does
it ahead of time. The analytics queries of `gsql/analytics/` wait for `mule diagnose`.

## 5. Check the labels

With the queries installed, check the label contract the load wrote (every violation
count must be zero) and read a page of the oracle export:

```gsql
RUN QUERY validate_label_contract()
RUN QUERY read_ground_truth("", 100)
```

A PhantomLedger load masks every mule, so `revealed_positives` is 0 until the first
preparation reveals them.

## 6. Prepare and train

```bash
mule check
mule train
```

`mule check` confirms the graph, the scope vertex type and the installed queries, and
reports `not_ready` until a dataset exists. The first `mule train` then, in order:

1. creates the frozen scope `strict_mule_v2`, partitioning every Account and Party by
   ownership group into train, validation and test (the `linked` rule for accounts no
   party owns); this writes one scope vertex and one membership edge per Account and
   Party, and nothing else;
2. reveals the known mules once ([Label reveal](../explanation/label-reveal.md)), writing
   the label fields of every internal Account, and checks the label contract;
3. prepares the dataset in `data/<dataset id>/`: the seed reservoirs, the observed
   labels, the cutoff sequences and the hub registry (scope creation and preparation took
   6 minutes 19 seconds on the reference graph);
4. trains ([Train and evaluate](train-and-evaluate.md)).

Relationships and business attributes are never changed: the scope and the label
fields are the only writes, and a graph that already has the scope and known labels is
only read.

## After the data changes

Keep the graph frozen while it is used: every run checks the vertex counts, the installed
query texts and the scope against what its dataset recorded and refuses a changed graph
("Graph counts changed"). After a reload or a material change, the old dataset and
scope describe data that is gone. Give the built-in run a new `scope.id` (the default of
`config.ScopeConfig`), so the next `mule train` creates a scope on the new data and
prepares a new dataset; the old scope vertex is experiment metadata and does no harm.
A schema change on a populated graph may invalidate compiled queries and positional
loading jobs: verify and restore both afterwards (`mule install` reinstalls the queries).
