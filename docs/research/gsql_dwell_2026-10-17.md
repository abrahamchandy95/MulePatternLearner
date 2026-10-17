# GSQL dwell research (2026-10-17)

How B1's last-in, first-out dwell can be computed in GSQL on TigerGraph 4.2.5, researched on
2026-10-17 and checked the same day by an independent reviewer. This file holds the research
with every correction of the check applied, then the check itself.

- The research was written against the October 17 query layout, where the walk sat inside
  `fetch_subgraphs` (commit c6376eb). The design now puts it in a query of its own,
  `read_money_flow` (see the design's
  [B1 section](../design.md#read_money_flow-b1-velocity) and the B1 walk in
  [the features](../features.md#account-node-309-columns)), and has settled the decisions the
  research left to the owner (see [Decisions taken since](#decisions-taken-since)).
- Line numbers refer to the files as they were on 2026-10-17: `fetch_subgraphs.gsql` and
  `config.py` at c6376eb, and PhantomLedger's exporter and pass-through header at that day's
  commit.
- The scratch scripts named below (`/tmp/dwell_check/check.py`, `cap.py`, `nocut.py`, and the
  checker's `/tmp/dwell_review/scripts/port.py` and `cost.py`) were run outside every
  repository and are kept in none.

## Recommendation

Build the walk in plain GSQL, with no user-defined function and no subquery. The new steps use
local scalars, FOREACH RANGE and WHILE inside POST-ACCUM, reading ListAccums that an earlier
SELECT filled.

**Why this shape.** The documentation guarantees immediate updates for local variables. It
also shows an `=` assignment to the clause vertex's own accumulator being read later in the
same POST-ACCUM (PageRank, in Operators and Expressions). Whether a `+=` append, a mutator
(`update`, `remove`, `pop`) or a global accumulator written earlier in one POST-ACCUM is seen
later in it is not documented. So the sequential computation keeps its state in local
variables, or passes it between separate SELECTs, and relies on nothing undocumented.

**Last in, first out needs no stored stack.** Measure money in cents, as heights in a column.
Inflow k sits in the band of cents from L (the level just before it arrived) to L + c. It
still holds `max(0, min(the running minimum of the level since k, L + c) - L)` cents. Each
fall of that running minimum inside the band is a piece that left at that outflow's time.

The capped computation is exact if two rules hold:

- The dwell heaps order by `(ts DESC, id DESC)`, the exact reverse of event order.
- When outflows were capped (more sent in the 30 days than were kept), the kept inflows that
  come before the oldest kept outflow are dropped.

Both rules are checked in the findings below: no mismatch against an explicit stack, and none
against full-history last-in, first-out matching.

**The suffix-minimum stop is part of the design**, not a profiling option: without it a mule
that keeps a cut of every credit is the most expensive account (see
[Cost per account](#cost-per-account)).

**New to the query.** The sketch adds a HeapAccum whose capacity comes from a variable,
`MapAccum.containsKey` (documented, but not used in `fetch_subgraphs`) and `minute_ms`, which
the query never declared (it had only `hour_ms` and `day_ms`). These belong in the compile
probe. Before building, also try the first of the
[alternatives](#alternatives-if-the-above-fails-a-compile-or-profile), or an explicit stack in
vertex ListAccums inside one POST-ACCUM, as a compile probe: if mutator effects turn out to be
visible within the clause, it is O(n + m) and avoids the n x m scans.

## Findings

### Accumulator reads inside one ACCUM

- **Claim.** Inside one ACCUM clause, an accumulator update is not visible to other reads in
  that clause: every match reads the value from before the clause, and the documentation's fix
  is to move the dependent statement into POST-ACCUM. A global (non-accumulator) variable
  assigned in ACCUM takes effect only when the query completes; the footnote that says so
  covers ACCUM only and says nothing about POST-ACCUM.
- **Numbers.** The 4.2 SELECT page, in the note on its wrong-usage example: "The read value of
  an accumulator does not change until the ACCUM phase ends." The 3.10 page makes only an
  ordering point (the second statement may not run after the first), not this rule.
- **Source.** GSQL Language Reference, SELECT statement, the subsection on parallelism in the
  ACCUM clause (4.2 and 3.10):
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/select-statement/>,
  <https://www.tigergraph.com/docs/gsql-ref/3.10/querying/select-statement/>.
- **Confidence.** High. The design relies on neither the 3.10 wording nor the footnote.

### What a POST-ACCUM statement sees

- **Claim.** POST-ACCUM runs once per vertex and can read what ACCUM aggregated. Several
  POST-ACCUM clauses in one SELECT run in parallel in no set order; each clause uses one vertex
  alias, and no assignment uses two. Operators and Expressions, in its subsection on the
  previous value of an accumulator (3.10 and 4.2), shows PageRank's POST-ACCUM assigning
  `s.@score` with `=` and then comparing `s.@score` with `s.@score'` later in the same clause,
  calling the first the score after the update; TigerGraph's own `tg_pagerank.gsql` depends on
  it. So an `=` assignment to the clause vertex's own accumulator is shown to be visible later
  in the same POST-ACCUM. Still undocumented: `+=` appends, mutators and global accumulators.
- **Numbers.** Mutator functions such as `update`, `remove` and `pop` work on vertex-attached
  accumulators only in POST-ACCUM.
- **Source.** SELECT statement, POST-ACCUM and multiple POST-ACCUM subsections (4.2, 3.10);
  Operators and Expressions (3.10, 4.2); Accumulators (4.2); the accumulators tutorial (4.3):
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/select-statement/>,
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/operators-and-expressions>,
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/accumulators>,
  <https://www.tigergraph.com/docs/gsql-ref/4.3/tutorials/accumulators-tutorial>.
- **Confidence.** High on what is documented. The design does not depend on the undocumented
  cases.

### Local variables in ACCUM and POST-ACCUM

- **Claim.** ACCUM and POST-ACCUM can declare three kinds of local variable, and the
  documentation says updates to each take effect immediately. Local base variables are INT,
  UINT, FLOAT, DOUBLE, BOOL, STRING, DATETIME, VERTEX, EDGE and JSON; local containers are SET,
  BAG, LIST and MAP; local tuples are the third kind. A LIST can hold base types, VERTEX, EDGE,
  tuples, or ListAccum nested up to 3 levels; MAP values can be any accumulator except
  HeapAccum. In POST-ACCUM, a local can be used with only one alias. The pages show a container
  only being declared with an initializer, iterated with FOREACH and added into an accumulator
  with `+=`; they never document `get`, `update`, `remove` or `size` on a local LIST.
- **Numbers.** Documented in both 3.10 and 4.2. The grammar requires an initializer:
  `containerType<type> name = expr`. One rule on the SELECT page, under updating
  vertex-attached accumulators and worded for ACCUM blocks, says a local is declared once and
  not reassigned, and claims to apply to all local variables; the declaration page contradicts
  it. The repository is no evidence either way: no server has compiled the c6376eb queries.
- **Source.** Declaration and Assignment Statements, sections on local base type, local
  container and local tuple variables (4.2 and 3.10):
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/declaration-and-assignment-statements>,
  <https://www.tigergraph.com/docs/gsql-ref/3.10/querying/declaration-and-assignment-statements>.
- **Confidence.** Medium: local container methods need a compile probe.

### Loops inside a clause

- **Claim.** FOREACH and WHILE both work inside ACCUM and POST-ACCUM. FOREACH can loop over a
  RANGE or a container accumulator, and WHILE takes an optional LIMIT. BREAK and CONTINUE work
  inside either loop. Inside a clause a FOREACH may never change its loop variable, statements
  are separated by commas, and nesting appears in the examples.
- **Numbers.** The c6376eb `fetch_subgraphs` uses WHILE inside FOREACH RANGE inside
  POST-ACCUM on reassigned local scalars (AccountRows, lines 824 to 903, with the flatten
  SELECT at 815 to 819). That is not a working precedent: c6376eb was never compiled on a
  server (its `tests/tigergraph/test_gsql.py` says no server compiles the GSQL files, and
  `docs/temporal_schema_deployment.json` records that only the schema was installed), and the
  queries once installed on `main` never reassign a local inside a POST-ACCUM WHILE. The
  pattern is documented, but a compile on 4.2.5 is still needed.
- **Source.** Control Flow Statements (4.2), the WHILE and FOREACH statements, FOREACH's
  limitations, CONTINUE and BREAK:
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/control-flow-statements>.
- **Confidence.** High on the documentation; the compile is unproven.

### HeapAccum and ListAccum

- **Claim.** HeapAccum is declared with a capacity and then its sort fields, read left to
  right. The prose calls the capacity optional, but the accumulator grammar requires it, as the
  term `simpleSize`, which Control Flow Statements defines as an integer, a variable name or a
  parameter name. So a query parameter as capacity is allowed, and TigerGraph's own
  `tg_pagerank.gsql` declares `HeapAccum<Vertex_Score>(top_k, score DESC)` with `top_k` a query
  parameter. Its functions are `size`, `top`, `pop`, `resize` and `clear`. When it is full,
  `+=` drops the tuple that sorts last, and PRINT outputs it sorted. The order FOREACH walks a
  HeapAccum in is not documented: the page only says the output is sorted, and tg_pagerank's
  FOREACH over its heap only fills a set. ListAccum has `get` (which returns the default value
  for any out-of-range or negative index, so the sketch's out-of-range reads are safe), `size`,
  `contains`, `update(index, value)`, `remove(index)`, `removeOne`, `removeAll` and `clear`.
  Neither ListAccum nor ArrayAccum has a sort.
- **Numbers.** `recent_payments` defaults to 256 per direction (`config.py` line 128 at
  c6376eb; the design's caps table holds the same value).
- **Source.** Accumulators (4.2), the HeapAccum, ListAccum and ArrayAccum subsections:
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/accumulators>.
- **Confidence.** High on the functions and the parameter capacity; FOREACH order over a heap
  rests on practice, not documentation, so the sketch checks it at run time.

### Subqueries

- **Claim.** A query with a RETURNS clause is a subquery. Its parameters can be primitives,
  VERTEX, or a set or bag of either; it cannot take a ListAccum, so it would have to re-read
  the account's payments. It can return any base type or accumulator, including a ListAccum of
  an anonymous tuple. It can be called in ACCUM (for example `@@acc += sub(src)`), and a local
  container can hold what it returns. Subqueries do not run in interpreted mode and must be
  installed before or together with the query that calls them. A distributed query cannot call
  another distributed query, and a non-distributed query cannot call a distributed one in
  ACCUM or POST-ACCUM. So a per-vertex call can go only to a non-distributed subquery: one full
  query execution per account, with no documented overhead figure.
- **Numbers.** A per-account subquery means thousands of query executions per batch, each
  re-reading that account's payments.
- **Source.** Operators and Expressions, the Subqueries section and its limitations,
  parameter types and return types (4.2); Create and Install Queries (4.2); Query operations
  (3.10); Distributed Query Mode (4.2):
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/operators-and-expressions>,
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/create-and-install-queries>,
  <https://www.tigergraph.com/docs/gsql-ref/3.10/querying/query-operations>,
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/distributed-query-mode>.
- **Confidence.** High on the rules; the per-call cost is unknown.

### Distributed mode

- **Claim.** Distributed mode rules out `Evaluate()`, the vertex methods `edgeAttribute`,
  `neighbors`, `neighborAttribute` and `size`, methods called on an accumulator nested inside
  another, and the subquery calls above. The page lists no limit on local variables, FOREACH,
  WHILE or HeapAccum. The recommended design uses none of the excluded features. The
  repository installs every query non-distributed (`CREATE OR REPLACE QUERY`).
- **Numbers.** The sketch adds 2 new SELECTs, or 3 with the suffix-minimum pass, and changes 5
  existing ones: Outflows and Inflows (step 1), the flatten SELECT (step 2), AccountRows and
  PaymentRows (step 5).
- **Source.** Distributed Query Mode (4.2), its Unsupported Features section:
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/distributed-query-mode>.
- **Confidence.** Medium.

### C++ user-defined functions

- **Claim.** C++ UDFs are written in two files. `ExprFunctions.hpp` holds the functions a query
  can call; `ExprUtil.hpp` holds helpers and structs, which queries cannot call. Every function
  must be declared inline, and names may not start with `tg_` (from 3.9). Arguments and return
  values can be bool, int, uint, float, double, string or an accumulator; `std::string` and
  other C++ types are allowed only in the function body, and the body cannot change a for
  loop's variable. Headers are checked against an allowlist. TigerGraph's own graph data
  science library passes `ListAccum<int64_t>&` and `ListAccum<double>` arguments, returns
  `ListAccum<int64_t>`, and reaches tuples only through templates such as `MaxAccum<tup>`. So
  parallel `ListAccum<int64_t>` inputs and a packed `ListAccum<int64_t>` output are the safe
  shape for this feature.
- **Numbers.** A GSQL INT is `int64_t` and a UINT is `uint64_t`. That only one UDF file is
  active per instance is an inference from the fixed file names and from GitHub taking
  precedence over other sources, not a stated rule.
- **Source.** Query User-Defined Functions (4.2 and 3.10), its allowed data types;
  tigergraph/gsql-graph-algorithms `UDF/tg_ExprFunctions.hpp` and `tg_ExprUtil.hpp`:
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/func/query-user-defined-functions>,
  <https://www.tigergraph.com/docs/gsql-ref/3.10/querying/func/query-user-defined-functions>,
  <https://raw.githubusercontent.com/tigergraph/gsql-graph-algorithms/master/UDF/tg_ExprFunctions.hpp>.
- **Confidence.** High.

### Installing a UDF

- **Claim.** On premises a UDF install needs the TigerGraph Linux user and `gadmin`. One route
  is `PUT ExprFunctions` from a file, off by default and switched on with
  `GSQL.UDF.EnablePutExpr`; the other, recommended, is the `GSQL.Github*` keys, which take
  precedence over any other source. PUT copies the files to every node of a cluster. Any
  change to the UDFs disables every installed query that uses one until it is reinstalled. The
  GSQL reference says TigerGraph Cloud customers cannot install UDFs themselves and must ask
  support; the Savanna documentation instead describes a workspace setting (the Graph Admin
  tab's GSQL section takes a GitHub repository, branch, path and token, and saving it may
  restart services). This project runs TigerGraph 4.2.5 in a Cloud workspace.
- **Numbers.** `server_version` 4.2.5 in `docs/temporal_schema_deployment.json`; the
  executor handles the Cloud "Starting workspace" page.
- **Source.** Query User-Defined Functions (4.2, 3.10); Savanna, Configure UDF:
  <https://www.tigergraph.com/docs/savanna/main/graph-development/advanced-features/configure-udf>,
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/func/query-user-defined-functions>.
- **Confidence.** High.

### The formula without a stack

- **Claim.** Last in, first out can be computed without storing a stack. Inflow k occupies
  heights L to L + c, L being the level just before it arrived. A cent at height h is still
  held at a later event exactly when the level has not fallen to h or below since. So the
  cents inflow k still holds are `max(0, min(running minimum of the level since k, L + c) -
  L)`, and each time that running minimum falls inside k's band, the cents that leave do so at
  that outflow's time. Outflows that exceed the tracked money floor the level at 0. One pass
  computes the level after each outflow, which is all the formula needs.
- **Numbers.** 20,000 random event sequences of up to 40 events each, with same-time ties,
  zero-cent payments and outflows larger than the tracked money: the formula and an explicit
  stack agreed on every per-inflow half time and 24-hour cents, every window sum, held total
  and median, with 0 mismatches. Matched pieces never exceeded n + m. The checker reproduced
  this, and also ported the sketch line by line (the newest-first lists, the cap-drop WHILE,
  the merge with out-of-range `get` returning defaults, the n-1-k indexing after the drop):
  against full-history matching it had 0 wrong inflows in 20,000 worlds, with and without
  self-payments.
- **Source.** Scratch scripts `/tmp/dwell_check/check.py` and
  `/tmp/dwell_review/scripts/port.py`.
- **Confidence.** High.

### Exactness under the caps

- **Claim.** Last in, first out spends the newest money first, so money that arrived before
  the first event read sits at the bottom of the stack and matters only once everything read
  is spent. The capped computation is therefore exact for every inflow it reads, as long as
  the events read are every visible event after some point in event order. Two rules
  guarantee that: both heaps order by `(ts DESC, id DESC)`, the exact reverse of event order
  (the `id ASC` of c6376eb breaks it on a tie at the cap); and when the account sent more than
  `recent_payments` payments in the 30 days, the kept inflows that come before the oldest kept
  outflow are dropped.
- **Numbers.** With both rules: 20,000 random worlds, 90,583 kept inflows, 0 mismatches against
  full-history matching. Without the drop: 4,649 of 20,000 worlds had a wrong inflow. With
  `id ASC` heaps: 353 of 20,000. The checker reproduced all three.
- **Source.** Scratch scripts `/tmp/dwell_check/cap.py` and `/tmp/dwell_check/nocut.py`.
- **Confidence.** High.

### Ties within one second

- **Claim.** Ordering ties inside one second by payment id conflicts with the design's Time
  section, which says payments in the same second have no order. PhantomLedger's payment ids
  were then `T` followed by the 12-digit row number of a chronological stream, so ordering by
  id within one second imported the simulator's row order. The account-level outputs (window
  shares, held share, median) do not depend on the order among same-second inflows or among
  same-second outflows, only on whether an outflow is placed before or after an inflow of the
  same second: same-time inflows share one arrival time, and same-time outflows remove the
  same set of heights in either order. The per-inflow outputs also depend on the order among
  same-second inflows.
- **Numbers.** PhantomLedger's `eventId` was `std::format("T{:012}", ++rows)`
  (`src/exporter/mule_temporal/streaming.cpp`, line 589), one counter shared by Zelle and card
  or ACH rows, with gaps where rows are dropped.
- **Source.** PhantomLedger's exporter; the design's Time section.
- **Confidence.** High.

### Cents, ids and integer types

- **Claim.** Payment amounts are stored as DOUBLE dollars. The query turns them into integer
  cents with `floor(p.amount * 100 + 0.5)`, and Python's `math.floor(amount * 100 + 0.5)` gives
  the same IEEE result. The ids are zero-padded ASCII and GSQL compares strings
  lexicographically, so ordering by `(ts, id)` is identical in GSQL and Python. Mixing INT and
  UINT in GSQL turns the arithmetic into UINT, so levels and cents must stay INT.
- **Numbers.** `Payment_Transaction` and `Zelle_Transfer` have `amount DOUBLE` and
  `event_ts_ms UINT`.
- **Source.** The repository's schema and query; Operators and Expressions (4.2), comparison
  and type promotion:
  <https://www.tigergraph.com/docs/gsql-ref/4.2/querying/operators-and-expressions>.
- **Confidence.** High.

### Visibility

- **Claim.** The dwell reads the same Inflows and Outflows traversals as the rest of the
  subgraph, which filter `window_start <= event_ts_ms < as_of_ms`, and the 30-day test uses the
  same `since_30d` bound. So it sees exactly the payments the subgraph treats as visible, and
  nothing at or after t.
- **Numbers.** A self-payment's own outflow takes its cents at dwell 0, so it looks like an
  instant pass-through and inflates the 15-minute share. The same was true of `passed_on` and
  `next_out` at c6376eb.
- **Confidence.** High.

## The sketch

Additions to `fetch_subgraphs` as it was at c6376eb. INT for every level and cent.

```gsql
TYPEDEF TUPLE<UINT dwell_ms, INT cents> Chunk;
TYPEDEF TUPLE<STRING id, INT cents, BOOL half_present, UINT half_ms, INT left_24h> InflowDwell;
UINT minute_ms = 60000;
INT chunk_cap = 2 * recent_payments;   /* pieces never exceed n + m */
HeapAccum<Recent>(recent_payments, ts DESC, id DESC) @dwell_in, @dwell_out;
HeapAccum<Chunk>(chunk_cap, dwell_ms ASC) @chunks;
ListAccum<UINT> @din_ts, @dout_ts;  ListAccum<STRING> @din_id, @dout_id;
ListAccum<INT> @din_cents, @dout_cents, @din_base, @din_next, @dout_level, @dout_sufmin;
SumAccum<INT> @d_total, @d_15m, @d_1h, @d_6h, @d_1d, @d_3d, @d_7d, @d_held;
MapAccum<STRING, ListAccum<InflowDwell>> @@inflow_dwell;
OrAccum @@unsorted;
SetAccum<VERTEX<Account>> @@unsorted_accounts;
```

The capacity of `@chunks` is a variable because the grammar's `simpleSize` admits an integer,
a variable name or a parameter name, not an expression; that this form compiles belongs in
the compile probe.

**Step 1**, in the existing Inflows and Outflows ACCUM, inside the `since_30d` branch:

```gsql
a.@dwell_in += Recent(p.event_ts_ms, p.id, floor(p.amount * 100 + 0.5))
a.@dwell_out += Recent(p.event_ts_ms, p.id, floor(p.amount * 100 + 0.5))
```

**Step 2**, in the existing flatten POST-ACCUM: copy each heap into its lists, newest at index
0. While copying, compare each tuple with the previous one; if a heap does not iterate in sort
order, set `@@unsorted` and add the account to `@@unsorted_accounts`. A POST-ACCUM cannot PRINT
or RETURN (neither is a DML-sub statement in the SELECT grammar), so right after the SELECT:
if `@@unsorted`, PRINT the status `unsorted_heap` with the accounts, and RETURN. The same check
covers the FOREACH order of `@chunks`, which step 5's median walk depends on.

**Step 3**, a new SELECT that merges the two lists and records the levels:

```gsql
Full = SELECT a FROM Full:a POST-ACCUM
  INT n = a.@din_ts.size(), INT m = a.@dout_ts.size(), INT i = 0, INT j = 0,
  INT level = 0, BOOL out_first = FALSE,
  IF a.@out_30d > m AND m > 0 THEN
    WHILE i < n AND (a.@din_ts.get(n-1-i) < a.@dout_ts.get(m-1)
      OR (a.@din_ts.get(n-1-i) == a.@dout_ts.get(m-1)
          AND a.@din_id.get(n-1-i) < a.@dout_id.get(m-1))) DO i = i + 1 END
  END,
  WHILE i < n OR j < m DO
    out_first = j < m AND (i >= n OR a.@dout_ts.get(m-1-j) < a.@din_ts.get(n-1-i)
      OR (a.@dout_ts.get(m-1-j) == a.@din_ts.get(n-1-i)
          AND a.@dout_id.get(m-1-j) < a.@din_id.get(n-1-i))),
    IF out_first THEN
      level = CASE WHEN level > a.@dout_cents.get(m-1-j)
              THEN level - a.@dout_cents.get(m-1-j) ELSE 0 END,
      a.@dout_level += level, j = j + 1
    ELSE
      a.@din_base += level, a.@din_next += j,
      level = level + a.@din_cents.get(n-1-i), i = i + 1
    END
  END;
```

The drop keeps an inflow whose key equals the oldest kept outflow's (a self-payment's inflow,
which comes before its own outflow), and keeping it is exact.

**Step 3b, the suffix minima**, a new SELECT: walk `@dout_level` from its last outflow back to
its first, keeping the lowest level seen, and append it to `@dout_sufmin`, so that
`a.@dout_sufmin.get(m-1-j)` is the lowest level from outflow j to the last. No outflow from j
on can lower an inflow's running minimum once that value is at or above it.

**Step 4**, a new SELECT with one scan per kept inflow. The kept inflows are the newest n, so
the k-th oldest is at index n-1-k:

```gsql
Full = SELECT a FROM Full:a POST-ACCUM
  INT n = a.@din_base.size(), INT m = a.@dout_level.size(),
  INT j = 0, INT c = 0, INT base = 0, INT low = 0, INT left = 0, INT gone = 0,
  INT left_24h = 0, UINT x = 0, UINT d = 0, BOOL half = FALSE, UINT half_ms = 0,
  IF n > 0 THEN FOREACH k IN RANGE[0, n - 1] DO
    x = a.@din_ts.get(n-1-k), c = a.@din_cents.get(n-1-k), base = a.@din_base.get(k),
    low = base + c, gone = 0, left_24h = 0, half = FALSE, half_ms = 0,
    j = a.@din_next.get(k),
    WHILE j < m AND low > base AND a.@dout_sufmin.get(m-1-j) < low DO
      IF a.@dout_level.get(j) < low THEN
        low = a.@dout_level.get(j),
        left = c - (CASE WHEN low > base THEN low - base ELSE 0 END),
        d = a.@dout_ts.get(m-1-j) - x,
        a.@chunks += Chunk(d, left - gone),
        IF d < 15 * minute_ms THEN a.@d_15m += left - gone END,
        /* the same line for 1 hour, 6 hours, 1 day, 3 days and 7 days */
        IF d < day_ms THEN left_24h = left END,
        IF NOT half AND 2 * left >= c THEN half = TRUE, half_ms = d END,
        gone = left
      END,
      j = j + 1
    END,
    a.@d_total += c, a.@d_held += c - gone,
    IF @@transactions_in.containsKey(a.@din_id.get(n-1-k))
       OR @@transfers_in.containsKey(a.@din_id.get(n-1-k)) THEN
      @@inflow_dwell += (a.@din_id.get(n-1-k) ->
        InflowDwell(a.@din_id.get(n-1-k), c, half, half_ms, left_24h))
    END
  END END;
```

The third condition of the WHILE is the suffix-minimum stop of step 3b.

**Step 5**, in the existing AccountRows POST-ACCUM: walk `a.@chunks` (dwell ascending) adding
up cents. The median is the dwell of the first piece where `2 x cumulative >= a.@d_total`; if no
piece gets there (more than half the cents still held), the median is absent. Then add
`d_total`, `d_15m` to `d_7d`, `d_held`, `median_present` and `median_ms` to the row. Payment
rows read `@@inflow_dwell` for `half_present`, `half_ms` and `left_24h`; every incoming payment
not tracked gets `present = false`.

Integer cents leave GSQL, and the shares are divided in one place in Python. Window bounds use
`dwell < W`, which matches the design's "within w of x" rule (x inclusive, x + w exclusive).

## Cost per account

With n inflows and m outflows in the 30 days, each capped at 256:

- No extra edge reads.
- n + m heap inserts, each O(log 256), and an n + m step merge.
- The scan, without the suffix-minimum stop, is at most n x m steps: an inflow never emptied
  scans every later outflow, and when every inflow comes before the outflows that is m each,
  65,536 steps at 256 and 256.
- The mule is the expensive case, not the cheap one. PhantomLedger's pass-through mules keep
  5% to 10% of every credit (`kRetainMin` 0.05 and `kRetainMax` 0.10 in
  `include/phantomledger/transfers/fraud/pass_through.hpp`), and layering and cycle hops take
  haircuts, so every mule inflow keeps a residue, is never emptied, and its scan runs to the
  last outflow. Simulated with 256 interleaved credits and 90% to 95% forwards: 32,896 steps.
  Inflows first with tiny outflows: 65,536. Forwarding 100%: 256.
- The suffix-minimum stop brings the retaining mule to about 256 steps, but not an account
  later emptied by a sweep (still about 32,896).
- At most n + m piece inserts, and about 13 lists of at most 256 values each.

All accounts of a batch share t and run in the same POST-ACCUMs in parallel. At an estimated
tens of nanoseconds a step, the worst account takes a few milliseconds; the step time is
unmeasured. Source: scratch script `/tmp/dwell_review/scripts/cost.py`.

## The Python mirror

The reference graph uses a plain stack written from the specification, an independent
implementation, so the parity check tests the GSQL's algebra as well as its plumbing. Integer
cents and the same order keys:

- `key = (ts, id)`; `cents = math.floor(amount * 100 + 0.5)`; `since = t - 30 days`.
- `ins` are the visible payments received with `ts >= since`, sorted by key, the last cap of
  them; `outs_all` the same for payments sent; `outs` the last cap of `outs_all`.
- If `len(outs_all) > cap`, keep only the `ins` whose key is at or after `key(outs[0])`, as
  the GSQL does. Keeping only those strictly after drops a self-payment's inflow: with
  self-payments in the generator, 2,297 of 20,000 worlds disagreed with the sketch that way,
  and 0 with "at or after". PhantomLedger's `railFor` checks `tx.source == tx.target`, so
  self-payments could reach the export.
- Events sorted by `(key, 0 for an inflow, 1 for an outflow)`, so a self-payment's inflow comes
  first, as the GSQL's ELSE branch does.
- A stack of `[payment, cents left]`. On an outflow, take `min(need, top)` from the top, record
  `(dwell, take)` only when `take > 0`, and pop at zero.

Then everything comes from the pieces by the same rules: the per-inflow half time is the first
piece where `2 x cumulative >= c`; `left_24h` the pieces with dwell under a day; window sums
the pieces with dwell under W; held the total minus all pieces; the median the pieces sorted by
dwell, stopping at `2 x cumulative >= total`.

**Tests.** Hand cases (a pass-through within minutes, a split outflow, an outflow larger than
the tracked money, same-second ties, zero cents, the cap cut, a self-payment); a property test
of the reference against a port of the GSQL formula; the cap exactness property; the live
parity check against TigerGraph.

## Alternatives, if the above fails a compile or profile

1. An explicit stack in one POST-ACCUM, in a local LIST. Shorter and O(n + m), but the
   documentation never says local LISTs support `get`, `update` or `remove`; try it with a
   ten-line probe query first. An explicit stack in vertex ListAccums inside one POST-ACCUM is
   the same probe for mutators.
2. A query-level WHILE that runs one SELECT per event. The stack lives in vertex ListAccums
   changed with `update` and `remove`, whose changes become visible at each clause boundary,
   all documented. O(n + m) work, but up to 512 SELECT rounds per request.
3. A non-distributed subquery called once per account. Allowed, but thousands of query
   executions per batch, and a second query to version and parity-check.
4. A C++ UDF, for example `inline ListAccum<int64_t> mpl_lifo_dwell(const ListAccum<int64_t>&
   ts, const ListAccum<int64_t>& cents, int64_t t)` returning a packed list. O(n + m), and it
   can sort without relying on heap order. But on this Cloud workspace it means the Graph Admin
   GitHub setting (whose save may restart services) or a support ticket; it holds the
   instance's UDF file; every UDF change disables the queries that use it; and it adds C++ to
   a deliverable meant to be GSQL.

## Gaps left by the research

- Whether a `+=` append, a mutator or a global accumulator written earlier in one POST-ACCUM is
  seen later in it. The design never relies on it.
- The order FOREACH walks a HeapAccum in. The walk relies on newest-first order; the sketch
  checks it at run time and stops with `unsorted_heap`, but the order is unconfirmed on 4.2.5.
- Which functions local LIST variables support (`get`, `update`, `remove`, `size`); the first
  alternative needs a compile probe.
- Step costs are counted, not measured. No GSQL was compiled or run against TigerGraph 4.2.5,
  and per-call subquery overhead is unknown.
- Whether the GSQL reference (UDFs on Cloud need a support request) or the Savanna
  documentation (a self-service GitHub setting) applies to this workspace's plan.

The research also named the parameter capacity of a HeapAccum as a gap; the check closed it
(see [HeapAccum and ListAccum](#heapaccum-and-listaccum)).

## Decisions the research left to the owner

As the research listed them, with the self-payment question the check added:

1. Adopt the cap rule and the `id DESC` order for these heaps.
2. The tie rule: order same-second payments by payment id, or follow the Time section's rule
   that same-second payments have no order. An id-free version: within one second, all
   inflows go before all outflows, and same-second inflows are one lot sharing its half time
   and 24-hour share.
3. Censoring: an inflow younger than W at t counts as not left within W, or each window's
   share uses only cents at least W old.
4. The median: over all tracked cents, absent when more than half are still held.
5. Zero-cent inflows get no half time.
6. Whether payments with sender equal to receiver are excluded, since a self-payment registers
   as dwell 0.

Also proposed and not decided then: where the per-inflow outputs go in the response.

## Decisions taken since

The design settles each of them:

- **The walk in GSQL.** The owner chose to write it in TigerGraph, in plain GSQL with no
  user-defined function and no subquery, in its own query `read_money_flow`; the Python
  reference is an independent explicit stack of lots written from the specification.
- **Caps and order.** Heaps ordered `(ts DESC, id DESC)` with explicit capacities. The design's
  whole-second span rule generalises the drop: when either list is cut, every payment of either
  list at or before the second of the newest payment left out is dropped, so both lists are
  complete inside the span.
- **Ties.** Inflows of one second form one lot, which comes before every outflow of that
  second, so same-second order never matters (the Order-free ties guard). PhantomLedger now
  exports payment ids as BLAKE2b pseudonyms, so ids carry no row order.
- **Censoring.** Only outflows before t count: an inflow received less than w before t shows
  only what left by t, and the payment's age on its edge tells the model how much of its curve
  it could see.
- **The median.** Over all tracked cents, absent when less than half left before t.
- **Zero cents.** Only inflows with positive cents are tracked.
- **Self-payments.** Read by no query.
- **The suffix-minimum stop.** Standard; `mule bench` measures the worst account.
- **Per-inflow outputs.** On each incoming payment's B4 edge, as `dwell_tracked`, `left_24h`
  and `dwell`; `read_money_flow` prints one row per tracked lot.

## The check

An independent reviewer checked every claim against the cited pages and reran the scratch
scripts. Verdict: the recommendation is supported, and the formula and the sketch are exact.
Its corrections, all applied above:

1. **POST-ACCUM visibility is partly documented.** The PageRank example shows an `=` assignment
   to the vertex's own accumulator read later in the same POST-ACCUM, so the claims that only
   local variables update immediately, and that a separate SELECT is the only documented
   route, were wrong. `+=`, mutators and global accumulators stay undocumented. The "since 3.5"
   for several POST-ACCUM clauses is on no cited page, and the global variable footnote covers
   ACCUM only, with the change taking effect when the query completes.
2. **The repository is no precedent.** The c6376eb queries were never compiled on a server, so
   the walk "uses" the pattern rather than "already runs" it, and needs the same compile probe.
3. **The HeapAccum parameter capacity is closed.** The grammar's `simpleSize` admits a
   parameter name, and tg_pagerank declares `HeapAccum(top_k, ...)`. Give `@chunks` an
   explicit capacity of twice `recent_payments`, since the grammar does not show the capacity
   as optional, and check the FOREACH order of `@chunks` too.
4. **Cost.** The worst case is n x m (65,536 steps), not half of it, and the retaining mule is
   the expensive case (about 32,896 steps). The suffix-minimum stop becomes standard. The count
   is 2 new SELECTs (3 with suffix minima) and 5 changed ones.
5. **A POST-ACCUM cannot RETURN or PRINT.** Flag an unsorted heap in a global OrAccum and stop
   after the SELECT. Declare `minute_ms`. `containsKey` and a heap capacity from a variable are
   new to the query.
6. **The Python mirror** keeps inflows whose key is at or after the oldest kept outflow's, as
   the GSQL does, with a self-payment hand case; and whether self-payments are excluded goes to
   the owner.
7. **Small points.** The 3.10 SELECT page makes an ordering point, not the rule about values
   from before the clause. Calling the capacity optional in the prose is contradicted by the
   grammar. "Only one UDF file per instance" is an inference, not a stated rule.

Claims the check confirmed as written: the ACCUM read rule on the 4.2 page; local variables and
their documented gaps; the documented loop rules; the HeapAccum and ListAccum functions; the
subquery and distributed-mode rules; the UDF rules and install routes; the formula without a
stack (0 mismatches, reproduced, and the line-by-line port); the cap exactness results
(90,583 inflows, 4,649 and 353, reproduced exactly); the tie analysis; cents, ids and types;
and the visibility of the walk.
