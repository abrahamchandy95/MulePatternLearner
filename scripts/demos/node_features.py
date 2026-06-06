from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from mule_pattern_learner.features.nodes import (
    NUM_ACCOUNT_FEATURES,
    NodeFeatures,
    Transform,
    account_feature_names,
    build_account_features,
)
from mule_pattern_learner.features.nodes import (
    _ACCOUNT_FEATURE_TRANSFORMS as ACCOUNT_FEATURE_TRANSFORMS,  # pyright: ignore[reportPrivateUsage]
)
from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path

_EXPORT_QUERY = "export_account_features"
_DEMO_PAGE_SIZE = 50
_SAMPLE_COUNT = 3


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


def _flatten_accounts(raw: list[object]) -> list[object]:
    accounts: list[object] = []
    for block in raw:
        if not (isinstance(block, dict) and "PageAccounts" in block):
            continue
        verts_obj = cast(dict[str, object], block)["PageAccounts"]
        if isinstance(verts_obj, list):
            accounts.extend(cast(list[object], verts_obj))
    return accounts


def _fetch_demo_accounts(client: Client) -> list[object]:
    raw = _run_query(client, _EXPORT_QUERY, {"cursor": "", "page_size": _DEMO_PAGE_SIZE})
    return _flatten_accounts(raw)


def _print_overview(features: NodeFeatures) -> None:
    print()
    print("=" * 64)
    print("NODE FEATURE SHAPES")
    print("=" * 64)
    print(f"  accounts parsed     : {features.num_nodes}")
    print(f"  feats shape         : {tuple(features.feats.shape)}")
    print(f"      -> [N, {NUM_ACCOUNT_FEATURES}] = [accounts, numeric features]")

    counts: dict[Transform, int] = {}
    for transform in ACCOUNT_FEATURE_TRANSFORMS.values():
        counts[transform] = counts.get(transform, 0) + 1
    print("  transform families  :")
    for transform in Transform:
        print(f"      {transform.name:9s}: {counts.get(transform, 0)} features")


def _print_samples(features: NodeFeatures, n: int) -> None:
    k = min(n, features.num_nodes)
    names = account_feature_names()
    rows = cast(list[list[object]], features.feats.tolist())
    print()
    print("=" * 64)
    print(f"FIRST {k} ACCOUNT SAMPLES (transformed feature values)")
    print("=" * 64)
    for i in range(k):
        print(f"\n  account {i}: {features.node_ids[i]}")
        row = rows[i]
        for name, value in zip(names, row):
            cell = value if isinstance(value, (int, float)) else 0.0
            transform = ACCOUNT_FEATURE_TRANSFORMS[name].name
            print(f"      {name:24s} [{transform:8s}] = {float(cell): .4f}")


def main() -> int:
    settings = Settings()
    print(f"host:   {settings.host}")
    print(f"graph:  {settings.graphname}")
    print()

    export_path = gsql_path(_EXPORT_QUERY)
    if not export_path.is_file():
        print(f"missing GSQL file: {export_path}")
        return 1

    client = _check("connect + auth", lambda: Client(settings))
    _ = _check(
        "install export_account_features",
        lambda: _install_from_file(client, export_path, _EXPORT_QUERY),
    )
    accounts = _check("fetch demo account page", lambda: _fetch_demo_accounts(client))
    print(f"  fetched {len(accounts)} accounts")
    if not accounts:
        print("\nNo accounts returned; nothing to transform.")
        return 1

    features = _check(
        "build_account_features (run node_features.py)",
        lambda: build_account_features(accounts),
    )

    _print_overview(features)
    _print_samples(features, _SAMPLE_COUNT)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
