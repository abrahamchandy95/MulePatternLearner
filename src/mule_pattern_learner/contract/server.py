"""Values the installed TigerGraph queries share with this package.

CONTRACT_VERSION is printed by the context query and recorded by saved models, so it
keeps its value until the query names change on the server. QUERY_FILES are the
repository files of the queries preparation runs.
"""

from __future__ import annotations

CONTRACT_VERSION = "temporal_live_v5_candidate_pools"
# The queries preparation runs; a prepared dataset records their source hashes.
QUERY_FILES = (
    "gsql/queries/fourier64.gsql",
    "gsql/queries/training_context.gsql",
    "gsql/queries/training_scope.gsql",
    "gsql/queries/split_cutoffs.gsql",
    "gsql/queries/hub_accounts.gsql",
)
