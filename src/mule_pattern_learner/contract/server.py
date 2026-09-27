"""Values the installed TigerGraph queries share with this package.

CONTRACT_VERSION is printed by the context query and recorded by saved models, so it
keeps its value until the query names change on the server. QUERY_FILES are the
files of the queries preparation runs. GSQL files are named by their path relative to
the repository's gsql folder (paths.GSQL_DIR).
"""

from __future__ import annotations

CONTRACT_VERSION = "temporal_live_v5_candidate_pools"
# The generated context query (tigergraph.render, scripts/render_queries.py).
CONTEXT_QUERY_FILE = "queries/training_context.gsql"
# The queries preparation runs; a prepared dataset records their source hashes.
QUERY_FILES = (
    "queries/fourier64.gsql",
    CONTEXT_QUERY_FILE,
    "queries/training_scope.gsql",
    "queries/split_cutoffs.gsql",
    "queries/hub_accounts.gsql",
)
