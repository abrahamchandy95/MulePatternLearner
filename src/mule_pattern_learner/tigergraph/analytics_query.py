"""The analytics context query's client side: its requests and the checks of its rows.

fetch_analytics_context (gsql/analytics/analytics_context.gsql) computes every feature
group, those training reads and those it never does (contract.analytics_features), for
analyses only; training never calls it. Its rows print ANALYTICS_CONTRACT and carry the
analytics node features and message fields beside the training ones, so
validate_analytics_context checks them rather than context_query.validate_context, which
knows only the training registry; both run context_query.check_context. A request names
no include flag, so the query computes every group, and it asks for no Fourier vectors.
TigerGraphAnalyticsFetcher is the fetcher `mule diagnose features` reads it through.
"""

from __future__ import annotations

from typing import Any

from ..contract.analytics_features import ANALYTICS_GROUPS
from ..contract.feature_groups import FEATURE_GROUPS
from ..contract.graph_schema import ContextKey
from ..contract.sampler_plan import SamplerPlan
from ..contract.server import ANALYTICS_CONTEXT_QUERY, ANALYTICS_CONTRACT
from .context_query import (
    KNOWN_NODE_FEATURES,
    batch_params,
    bisected,
    check_context,
    indexed_rows,
    numeric_fields,
)
from .executor import QueryExecutor

# The node features an analytics row may carry: the training query's and every analytics
# node and summary feature.
ANALYTICS_NODE_FEATURES = KNOWN_NODE_FEATURES | frozenset(
    name
    for spec in ANALYTICS_GROUPS.values()
    if spec.path in ("node", "summary")
    for name in spec.names
)
# The message fields that must be finite and nonnegative: the training query's pair and
# flow fields, and the analytics groups' pair window counts and device and IP ages.
ANALYTICS_NUMERIC = numeric_fields(FEATURE_GROUPS) + tuple(
    (group, name)
    for group, spec in ANALYTICS_GROUPS.items()
    if spec.path == "message"
    for name in spec.names
)


def validate_analytics_context(
    key: ContextKey, row: dict[str, Any], sampler: SamplerPlan, hop: int = 1
) -> int:
    """Check one analytics context against its key, clocks and ANALYTICS_CONTRACT.

    Returns the number of messages whose channel is outside the known channels.
    """
    return check_context(
        key,
        row,
        sampler,
        hop,
        contract=ANALYTICS_CONTRACT,
        query=ANALYTICS_CONTEXT_QUERY,
        node_features=ANALYTICS_NODE_FEATURES,
        numeric=ANALYTICS_NUMERIC,
        flow=True,
        require_encodings=False,
    )


def query_analytics_rows(
    executor: QueryExecutor,
    batch: list[ContextKey],
    *,
    sampler: SamplerPlan,
    hop: int = 1,
    timeout_retries: int = 1,
) -> list[dict[str, Any]]:
    """One REST call; checked ok rows or per-request status rows, in key order.

    The request carries the keys and the sampler's candidate pool of the hop, so the
    query samples the messages the training query would.
    """
    params = {
        **batch_params(batch),
        **sampler.query_params(hop),
        "scope_id": batch[0].scope_id,
        "visibility_phase": batch[0].visibility_phase,
    }
    result = executor.run(ANALYTICS_CONTEXT_QUERY, params, timeout_retries=timeout_retries)

    def check(key: ContextKey, row: dict[str, Any]) -> int:
        return validate_analytics_context(key, row, sampler, hop)

    return indexed_rows(result, batch, check)


class TigerGraphAnalyticsFetcher:
    """The analytics context query for one block of keys, checked and bisected on timeouts.

    It satisfies diagnostics.feature_table.AnalyticsFetcher.
    """

    def __init__(self, executor: QueryExecutor) -> None:
        self.executor = executor

    def request(
        self, keys: list[ContextKey], *, sampler: SamplerPlan, hop: int = 1
    ) -> tuple[list[dict[str, Any]], int]:
        """Rows of one block of keys in key order, and the REST calls they took."""

        def request(block: list[ContextKey], timeout_retries: int) -> list[dict[str, Any]]:
            return query_analytics_rows(
                self.executor, block, sampler=sampler, hop=hop, timeout_retries=timeout_retries
            )

        return bisected(request, keys, hop)
