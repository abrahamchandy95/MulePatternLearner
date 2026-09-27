"""Values the installed TigerGraph queries share with this package.

GRAPH_NAME is the graph every query is installed on; the connection refuses another
(tigergraph.executor). The query names are the names the repository's GSQL files
install, and every adapter, fake and installer takes them from here, so the server
step renames each in one place. CONTEXT_CONTRACT is printed by the context query and
recorded by saved models, so it keeps its value until the query names change on the
server. GSQL files are named by their path relative to the repository's gsql folder
(paths.GSQL_DIR).
"""

from __future__ import annotations

GRAPH_NAME = "Mule_Pattern_Learner"
CONTEXT_CONTRACT = "temporal_live_v5_candidate_pools"
# The vertex type of the experiment scopes (gsql/schema/scope_vertex.gsql). It is part of
# the graph's schema, so it keeps the name it was created with.
SCOPE_VERTEX = "Temporal_Training_Scope"

# The installed queries, by what they do. They keep the names they were installed with
# until the server step renames them (the owner decision on query names).
CONTEXT_QUERY = "temporal_training_context"
# The Fourier encoding the context query calls for every message age and gap.
FOURIER_QUERY = "temporal_fourier64_values"
CREATE_SCOPE_QUERY = "temporal_create_training_scope"
FINALIZE_SCOPE_QUERY = "temporal_finalize_training_scope"
POPULATION_QUERY = "temporal_scope_population"
SCOPE_POLICY_QUERY = "temporal_scope_policy"
CUTOFF_QUERY = "temporal_training_cutoffs"
HUB_QUERY = "temporal_hub_registry"
REVEAL_QUERY = "temporal_reveal_mule_labels"
# The per-mule draws the reveal calls.
REVEAL_UNIFORMS_QUERY = "temporal_reveal_uniforms"
LABEL_CONTRACT_QUERY = "temporal_validate_account_supervision"
# The oracle: evaluation and diagnostics only.
TRUTH_QUERY = "temporal_get_account_supervision"
# Analytics: the persisted pair encodings, checked against the context query.
ZELLE_PAIR_QUERY = "zelle_pair_time64"
PAYMENT_PAIR_QUERY = "payment_pair_time64"

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
# What `mule install` installs: the preparation queries plus the oracle export for
# audits, the label-contract validation and the one-time reveal job (the first run
# reveals known mules; see tigergraph.reveal).
TRAINING_QUERY_FILES = (
    *QUERY_FILES,
    "evaluation/ground_truth.gsql",
    "queries/label_contract.gsql",
    "queries/label_reveal.gsql",
)
# Analytics queries: parity tools for the persisted pair encodings. Training never calls
# them, so only the code that uses them installs them (installer.install with
# analytics=True).
ANALYTICS_QUERY_FILES = (
    "analytics/zelle_pair_gaps.gsql",
    "analytics/payment_pair_gaps.gsql",
)
