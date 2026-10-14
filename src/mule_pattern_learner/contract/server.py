"""Values the installed TigerGraph queries share with this package.

GRAPH_NAME is the graph every query is installed on; the connection refuses another
(tigergraph.executor). The query names are the names the repository's GSQL files
install, verb first and without a prefix since the graph is dedicated, and every
adapter, fake and installer takes them from here (the Fourier subquery's, which only
tests name, is testing.gsql_blocks.FOURIER_QUERY). CONTEXT_CONTRACT names the text of
the context query: every row the query returns prints it, the client refuses a row
without it, saved models record it through their contract fingerprint and the context
cache names its entries with it. It is "context_" and the first 12 hex digits of the
sha256 of the query file's text, normalised, without it (tigergraph.gsql_text.query_contract),
and a test keeps the two equal. RETIRED_QUERIES are the names the queries were
installed under before, which `mule install` drops. GSQL files are named by their path
relative to the repository's gsql folder (paths.GSQL_DIR).
"""

from __future__ import annotations

GRAPH_NAME = "Mule_Pattern_Learner"
CONTEXT_CONTRACT = "context_d7ea81aac6e7"
# The vertex type of the graph's one split (gsql/schema/scope_vertex.gsql). It is part of
# the graph's schema, so it keeps the name it was created with.
SCOPE_VERTEX = "Temporal_Training_Scope"

# The installed queries, named by what they do (docs/architecture.md, Naming).
CONTEXT_QUERY = "fetch_training_context"
CREATE_SCOPE_QUERY = "create_training_scope"
FINALIZE_SCOPE_QUERY = "finalize_training_scope"
POPULATION_QUERY = "list_scope_accounts"
CUTOFF_QUERY = "resolve_split_cutoffs"
HUB_QUERY = "list_hub_accounts"
REVEAL_QUERY = "reveal_mule_labels"
# The per-mule draws the reveal calls.
REVEAL_UNIFORMS_QUERY = "draw_reveal_uniforms"
# The oracle: evaluation and diagnostics only.
TRUTH_QUERY = "read_ground_truth"

# The names the queries were installed under before they were named after what they do,
# and the two queries retired then: the public Fourier wrapper, which only a deleted verification
# script called, and the population query of the removed shared_history protocol. Then the
# split's unowned-rule summary and the label-contract check, which nothing reads any more,
# and the deleted analytics queries, the context query first. `mule install` drops those
# still installed in this order, callers before the queries they call, and no other query
# (tigergraph.installer.drop_retired).
RETIRED_QUERIES = (
    "temporal_training_context",
    "temporal_fourier64",
    "zelle_pair_time64",
    "payment_pair_time64",
    "temporal_fourier64_values",
    "temporal_reveal_mule_labels",
    "temporal_reveal_uniforms",
    "temporal_create_training_scope",
    "temporal_finalize_training_scope",
    "temporal_scope_population",
    "temporal_scope_policy",
    "temporal_training_cutoffs",
    "temporal_hub_registry",
    "temporal_validate_account_supervision",
    "temporal_get_account_supervision",
    "temporal_training_population",
    "summarize_scope_policy",
    "validate_label_contract",
    "fetch_analytics_context",
    "encode_zelle_pair_gaps",
    "encode_payment_pair_gaps",
)

# The context query file.
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
# audits and the one-time reveal job (the first run reveals known mules; see
# tigergraph.reveal).
TRAINING_QUERY_FILES = (
    *QUERY_FILES,
    "evaluation/ground_truth.gsql",
    "queries/label_reveal.gsql",
)
