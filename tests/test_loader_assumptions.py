from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

_PROBE_QUERY_NAME = "smoke_probe_has_paid_list"

_GSQL_ERROR_MARKERS: tuple[str, ...] = (
    "error",
    "fail",
    "could not",
    "cannot",
    "not exist",
    "syntax error",
)


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


def _run_query(client: Client, name: str) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name))


def _assert_install_clean(output: str, what: str) -> str:
    low = output.lower()
    for marker in _GSQL_ERROR_MARKERS:
        if marker in low:
            raise RuntimeError(f"{what}: GSQL output contained {marker!r}:\n{output}")
    return output


def _install_probe(client: Client) -> str:
    q = f"""
USE GRAPH {client.graphname}
CREATE OR REPLACE DISTRIBUTED QUERY {_PROBE_QUERY_NAME}() FOR GRAPH {client.graphname} SYNTAX V2 {{
  ListAccum<EDGE> @out_edges;
  P =
    SELECT a
    FROM Account:a
    ORDER BY a.id
    LIMIT 200;
  C =
    SELECT src
    FROM P:src -(HAS_PAID>:e)- Account:tgt
    ACCUM src.@out_edges += e;
  PRINT P[P.@out_edges AS out_edges];
}}
INSTALL QUERY {_PROBE_QUERY_NAME}
"""
    return _assert_install_clean(_gsql(client, q), "probe install")


def _extract_edge_attributes(edge: object) -> dict[str, object]:
    if not isinstance(edge, dict):
        raise RuntimeError(f"edge entry was not a dict: {edge!r}")
    edge_typed = cast(dict[str, object], edge)
    attrs = edge_typed.get("attributes")
    if not isinstance(attrs, dict):
        raise RuntimeError(f"edge had no attributes dict: {edge_typed!r}")
    return cast(dict[str, object], attrs)


def _run_probe(client: Client) -> dict[str, object]:
    raw = _run_query(client, _PROBE_QUERY_NAME)
    for block in raw:
        if not (isinstance(block, dict) and "P" in block):
            continue
        block_typed = cast(dict[str, object], block)
        vertices_obj = block_typed["P"]
        if not isinstance(vertices_obj, list):
            raise RuntimeError(f"P block was not a list: {vertices_obj!r}")
        vertices = cast(list[object], vertices_obj)
        for vertex in vertices:
            if not isinstance(vertex, dict):
                continue
            vertex_typed = cast(dict[str, object], vertex)
            attrs_obj = vertex_typed.get("attributes")
            if not isinstance(attrs_obj, dict):
                continue
            attrs_typed = cast(dict[str, object], attrs_obj)
            edges_obj = attrs_typed.get("out_edges")
            if not isinstance(edges_obj, list):
                continue
            edges = cast(list[object], edges_obj)
            if edges:
                return _extract_edge_attributes(edges[0])
        raise RuntimeError(
            "no account in the probe batch had an outgoing HAS_PAID edge; "
            + "cannot test LIST round-trip (is the graph loaded?)."
        )
    raise RuntimeError(f"P result block not found in query output: {raw!r}")


def _assert_list_roundtrip(row: dict[str, object]) -> tuple[int, int]:
    raw_amount: object = row.get("amount_bins")
    raw_count: object = row.get("count_bins")
    if not isinstance(raw_amount, list):
        raise TypeError(
            f"amount_bins is {type(raw_amount).__name__}, expected list. "
            + f"LIST<DOUBLE> did NOT round-trip as a JSON array. Value: {raw_amount!r}"
        )
    if not isinstance(raw_count, list):
        raise TypeError(
            f"count_bins is {type(raw_count).__name__}, expected list. "
            + f"LIST<INT> did NOT round-trip as a JSON array. Value: {raw_count!r}"
        )
    amount_bins = cast(list[object], raw_amount)
    count_bins = cast(list[object], raw_count)
    if amount_bins and isinstance(amount_bins[0], str):
        raise TypeError(f"amount_bins elements are str, expected float. First: {amount_bins[0]!r}")
    return len(amount_bins), len(count_bins)


def _drop_probe(client: Client) -> str:
    q = f"USE GRAPH {client.graphname}\nDROP QUERY {_PROBE_QUERY_NAME}"
    out = _gsql(client, q)
    low = out.lower()
    if "not exist" in low:
        return out
    return _assert_install_clean(out, "probe drop")


def _install_real_query(client: Client, gsql_path: Path) -> str:
    text = gsql_path.read_text(encoding="utf-8")
    install = "\nINSTALL QUERY export_has_paid_edges\n"
    return _assert_install_clean(_gsql(client, text + install), "export query install")


def main() -> int:
    settings = Settings()
    secret = settings.secret.get_secret_value()
    print(f"host:   {settings.host}")
    print(f"graph:  {settings.graphname}")
    print(f"secret: {secret[:6]}...{secret[-4:]}  (length {len(secret)})")
    print()

    client = _check("connect + auth", lambda: Client(settings))

    _ = _check("install isolated LIST probe", lambda: _install_probe(client))
    row = _check("run probe (fetch 1 HAS_PAID edge)", lambda: _run_probe(client))
    n_amt, n_cnt = _check(
        "LIST<DOUBLE>/LIST<INT> round-trip as Python lists",
        lambda: _assert_list_roundtrip(row),
    )
    print(f"  amount_bins length = {n_amt}, count_bins length = {n_cnt}")
    if n_amt != n_cnt:
        print(f"  note: bin-count mismatch ({n_amt} vs {n_cnt}); expected equal per edge")

    _ = _check("drop probe (cleanup)", lambda: _drop_probe(client))

    if len(sys.argv) > 1:
        gsql_path = Path(sys.argv[1])
        if not gsql_path.is_file():
            print(f"\nexport query path not found: {gsql_path}")
            return 1
        _ = _check(
            f"install real query ({gsql_path.name}) [INTO + ACCUM]",
            lambda: _install_real_query(client, gsql_path),
        )
        print("  INTO + trailing ACCUM installs cleanly on this server.")
    else:
        print("\n(skipping real-query install test; pass the .gsql path to enable)")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
