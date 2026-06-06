import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from mule_pattern_learner.features.edge_spec import (
    NUM_BIN_CHANNELS,
    NUM_SCALAR_FEATURES,
    SCALAR_FEATURE_NAMES,
    flat_edge_dim,
)
from mule_pattern_learner.features.temporal import (
    EdgeFeatures,
    build_edge_features,
    flat_edge_features,
)
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path
from mule_pattern_learner.tigergraph.settings import Settings

_EXPORT_QUERY = "export_has_paid_edges"
_DERIVE_QUERY = "derive_max_bins"
_DEMO_PAGE_SIZE = 50
_REFERENCE_EPOCH_S = 1750000000.0


def _check[T](label: str, fn: Callable[[], T]) -> T:
    print(f"{label} ... ", end="", flush=True)
    try:
        result = fn()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        raise
    print("ok")
    return result


def _gsql(client: Client, statement: str) -> str:
    return cast(str, client.conn.gsql(statement))


def _run_query(client: Client, name: str, params: dict[str, object]) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name, params))


def _install_from_file(client: Client, query_path: Path, query_name: str) -> str:
    text = query_path.read_text(encoding="utf-8")
    out = _gsql(client, text + f"\nINSTALL QUERY {query_name}\n")
    low = out.lower()
    for marker in ("error", "fail", "could not", "cannot"):
        if marker in low:
            raise RuntimeError(f"install {query_name}: output had {marker!r}:\n{out}")
    return out


def _scalar_block(raw: list[object], key: str) -> object:
    for block in raw:
        if isinstance(block, dict) and key in block:
            return cast(dict[str, object], block)[key]
    raise RuntimeError(f"key {key!r} not found in query output: {raw!r}")


def _derive_max_bins(client: Client) -> int:
    raw = _run_query(client, _DERIVE_QUERY, {})
    max_bins = _scalar_block(raw, "max_bins")
    amount_len = _scalar_block(raw, "max_amount_len")
    count_len = _scalar_block(raw, "max_count_len")
    if not (
        isinstance(max_bins, int) and isinstance(amount_len, int) and isinstance(count_len, int)
    ):
        raise RuntimeError(f"derive_max_bins returned non-ints: {raw!r}")
    if not (max_bins == amount_len == count_len):
        raise RuntimeError(
            "INCONSISTENT bin counts in data: "
            + f"num_bins={max_bins}, amount_bins.size()={amount_len}, "
            + f"count_bins.size()={count_len}. The declared num_bins disagrees "
            + "with actual list lengths; padding would truncate real data."
        )
    return max_bins


def _flatten_edges(raw: list[object]) -> list[object]:
    edges: list[object] = []
    for block in raw:
        if not (isinstance(block, dict) and "PageAccounts" in block):
            continue
        vertices_obj = cast(dict[str, object], block)["PageAccounts"]
        if not isinstance(vertices_obj, list):
            continue
        for vertex in cast(list[object], vertices_obj):
            if not isinstance(vertex, dict):
                continue
            attrs = cast(dict[str, object], vertex).get("attributes")
            if not isinstance(attrs, dict):
                continue
            out_edges = cast(dict[str, object], attrs).get("out_edges")
            if not isinstance(out_edges, list):
                continue
            edges.extend(cast(list[object], out_edges))
    return edges


def _fetch_demo_edges(client: Client) -> list[object]:
    raw = _run_query(client, _EXPORT_QUERY, {"cursor": "", "page_size": _DEMO_PAGE_SIZE})
    return _flatten_edges(raw)


def _print_overview(features: EdgeFeatures) -> None:
    print()
    print("=" * 60)
    print("EDGE FEATURE SHAPES")
    print("=" * 60)
    print(f"  edges parsed            : {features.num_edges}")
    print(f"  max_bins (derived)      : {features.max_bins}")
    print(f"  scalar_feats shape      : {tuple(features.scalar_feats.shape)}")
    print(f"      -> [E, {NUM_SCALAR_FEATURES}] = [edges, scalar features]")
    print(f"  bin_seq shape           : {tuple(features.bin_seq.shape)}")
    print(
        f"      -> [E, {features.max_bins}, {NUM_BIN_CHANNELS}] "
        + "= [edges, time bins, (log_amount, count)]"
    )
    flat = flat_edge_features(features)
    print(f"  flat edge_attr shape    : {tuple(flat.shape)}")
    print(f"      -> [E, {flat_edge_dim(features.max_bins)}] for a flat-edge GNN layer")


def _print_samples(features: EdgeFeatures, n: int) -> None:
    k = min(n, features.num_edges)
    print()
    print("=" * 60)
    print(f"FIRST {k} EDGE SAMPLES")
    print("=" * 60)
    scalar_list = features.scalar_feats.tolist()
    bins_list = features.bin_seq.tolist()
    for i in range(k):
        src = features.src_ids[i]
        dst = features.dst_ids[i]
        print(f"\n  edge {i}: {src} -> {dst}")
        row = cast(list[float], scalar_list[i])
        print("    scalar features:")
        for name, value in zip(SCALAR_FEATURE_NAMES, row):
            print(f"      {name:22s} = {value: .4f}")
        seq = cast(list[list[float]], bins_list[i])
        nonzero = [
            (j, pair[0], pair[1]) for j, pair in enumerate(seq) if pair[0] != 0.0 or pair[1] != 0.0
        ]
        print(f"    bin_seq: {len(seq)} bins, {len(nonzero)} non-empty")
        for j, log_amt, cnt in nonzero[:6]:
            print(f"      bin[{j:2d}] log_amount={log_amt: .4f}  count={cnt: .1f}")
        if len(nonzero) > 6:
            print(f"      ... ({len(nonzero) - 6} more non-empty bins)")


def main() -> int:
    settings = Settings()
    print(f"host:   {settings.host}")
    print(f"graph:  {settings.graphname}")
    print()

    derive_path = gsql_path(_DERIVE_QUERY)
    export_path = gsql_path(_EXPORT_QUERY)
    for path in (derive_path, export_path):
        if not path.is_file():
            print(f"missing GSQL file: {path}")
            return 1

    client = _check("connect + auth", lambda: Client(settings))
    _ = _check(
        "install derive_max_bins",
        lambda: _install_from_file(client, derive_path, _DERIVE_QUERY),
    )
    _ = _check(
        "install export_has_paid_edges",
        lambda: _install_from_file(client, export_path, _EXPORT_QUERY),
    )
    max_bins = _check("derive max_bins from data", lambda: _derive_max_bins(client))
    print(f"  derived max_bins = {max_bins}")

    edges = _check("fetch demo edge page", lambda: _fetch_demo_edges(client))
    print(f"  fetched {len(edges)} raw edges")
    if not edges:
        print("\nNo HAS_PAID edges returned; nothing to transform.")
        return 1

    features = _check(
        "build_edge_features (run temporal.py)",
        lambda: build_edge_features(edges, _REFERENCE_EPOCH_S, max_bins),
    )

    _print_overview(features)
    _print_samples(features, 3)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
