from typing import cast

from mule_pattern_learner.tigergraph.client import Client


class FeatureFetchError(RuntimeError):
    pass


_ACCOUNT_FEATURE_QUERY = "fetch_account_features"
_HAS_PAID_FEATURE_QUERY = "fetch_has_paid_features"


def fetch_account_vertices(
    client: Client,
    account_ids: list[str],
    query_name: str = _ACCOUNT_FEATURE_QUERY,
) -> list[object]:
    """Fetch raw Account vertices ({"v_id", "attributes"}) for the given ids.

    Returns the rows in pyTigerGraph's shape, ready for build_account_features.
    Empty id list short-circuits without a server call.
    """
    if not account_ids:
        return []

    params: dict[str, object] = {"ids": [(aid,) for aid in account_ids]}
    raw = client.conn.runInstalledQuery(query_name, params)

    for block in raw:
        if not isinstance(block, dict):
            continue
        b = cast(dict[str, object], block)
        if "Accounts" in b:
            vertices = b["Accounts"]
            if not isinstance(vertices, list):
                raise FeatureFetchError(f"'Accounts' is not a list: {vertices!r}")
            return cast(list[object], vertices)

    raise FeatureFetchError(f"query {query_name!r} returned no 'Accounts' block: {raw!r}")


def fetch_has_paid_edges(
    client: Client,
    account_ids: list[str],
    query_name: str = _HAS_PAID_FEATURE_QUERY,
) -> list[object]:
    """Fetch HAS_PAID edges with both endpoints in account_ids.

    The query nests edges under each source vertex; this flattens them into one
    list of {"from_id", "to_id", "attributes"}, ready for build_edge_features.
    Empty id list short-circuits without a server call.
    """
    if not account_ids:
        return []

    params: dict[str, object] = {"ids": [(aid,) for aid in account_ids]}
    raw = client.conn.runInstalledQuery(query_name, params)

    edges: list[object] = []
    found_block = False
    for block in raw:
        if not isinstance(block, dict):
            continue
        b = cast(dict[str, object], block)
        if "Collected" not in b:
            continue
        found_block = True
        vertices = b["Collected"]
        if not isinstance(vertices, list):
            raise FeatureFetchError(f"'Collected' is not a list: {vertices!r}")
        for vertex in cast(list[object], vertices):
            if not isinstance(vertex, dict):
                raise FeatureFetchError(f"vertex is not a dict: {vertex!r}")
            v = cast(dict[str, object], vertex)
            attrs = v.get("attributes")
            if not isinstance(attrs, dict):
                raise FeatureFetchError(f"vertex missing attributes: {v!r}")
            a = cast(dict[str, object], attrs)
            out_edges = a.get("out_edges")
            if not isinstance(out_edges, list):
                raise FeatureFetchError(f"'out_edges' is not a list: {out_edges!r}")
            edges.extend(cast(list[object], out_edges))

    if not found_block:
        raise FeatureFetchError(f"query {query_name!r} returned no 'Collected' block: {raw!r}")
    return edges
