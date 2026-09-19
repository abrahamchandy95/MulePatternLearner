# Temporal GSQL review through TigerGraph MCP

The official `tigergraph-mcp` package, version 1.0.3, is installed in an isolated
Python environment under `~/.local/share/tigergraph-mcp/`. It is registered as
`tigergraph` in the desktop app's MCP configuration. The launcher reads this
project's existing `.env` and maps `HOST`, `GRAPHNAME`, and `SECRET` to the
server's `TG_*` settings at startup. Credentials are not copied into the app
configuration or repository. The project training environment is unchanged.

A real stdio MCP session initialized, discovered 69 tools, fetched the live
schema and all seven installed query definitions, and ran read-only query
checks. The six repository-managed query definitions matched their live
definitions after removing comments and whitespace. The additional
`mt_validate_graph` query belongs to the data-loading workflow; its installed
definition was also reviewed.

## Results

- `Account.is_mule` is `INT`; `is_mule_masked` is `BOOL`. Label-contract
  validation returned zero violations. Supervision pagination and rejection of
  invalid page sizes worked.
- All seven association types use `valid_from_seq` as a discriminator and
  retain `valid_to_seq` for closed tenures.
- The 64-dimensional GSQL Fourier output matched a Python mathematical
  reference across zero, millisecond, daily, and very large gaps. The maximum
  absolute coordinate error was below `4e-7`, within the `1e-5` tolerance.
- Pair queries enforce both sequence and timestamp cutoffs. Observed recipient
  Accounts take precedence over routing Tokens, preventing double counting.
  Unresolved recipient Tokens remain supported without present-day registration
  lookup.
- Label attributes are not read by the Fourier or pair-feature queries. The
  supervision export remains separate from model features.

The review found a cached-count issue in the live encoding verifier. Its before
and after counts now use `realtime=True`. The verifier also checks exact
inclusive 1h/24h/7d frequency boundaries for both Zelle and non-Zelle payments.

## Scope and remaining limits

The pair queries calculate fixed time features, not learned account embeddings.
They do not traverse association tenures or implement a production batched
sampler. The staged training pipeline applies association visibility; a future
GSQL sampler must apply the same predicate in every traversal direction.

The history cap does not bound sender adjacency traversal. Large-scale training
still needs staged/cached temporal indexes and bounded sampling. Valid-time
data also cannot reconstruct when a backdated correction became known; frozen
extracts or source-supported known-time history are needed for that case.

No GSQL algorithm or live schema change was needed for this review. The older
static GSQL files are identified separately in [the GSQL guide](../gsql/README.md).

The underlying MCP and integration reports remain local, Git-ignored artifacts.
This is a historical review; use the [current query catalog](gsql_feature_catalog.md)
and [leakage assessment](leakage_and_scaling.md) for the live training path.

Installation references:
[official TigerGraph MCP server](https://github.com/tigergraph/tigergraph-mcp)
and [package distribution](https://pypi.org/project/tigergraph-mcp/).
