import sys
from collections.abc import Callable
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings

_VERTEX_EXPECTATIONS: tuple[tuple[str, int], ...] = (
    ("Account", 0),
    ("Party", 0),
)


def _check[T](label: str, fn: Callable[[], T]) -> T:
    """Run fn(), print pass/fail inline, re-raise on failure."""
    print(f"{label} ... ", end="", flush=True)
    try:
        result = fn()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        raise
    print("ok")
    return result


def _count_vertices(client: Client, vertex_type: str) -> int:
    """Type-narrowed wrapper around pyTigerGraph's getVertexCount.

    getVertexCount is overloaded behaviorally: a single type name returns
    int, while "*" or a list returns dict[str, int]. pyTigerGraph's hints
    don't express that split, so its declared return is
    int | dict[Unknown, Unknown]. We only ever call the single-string
    form, so the cast encodes that assumption and quiets
    reportUnknownLambdaType at every call site.
    """
    return cast(int, client.conn.getVertexCount(vertex_type))


def main() -> int:
    settings = Settings()
    secret = settings.secret.get_secret_value()
    print(f"host:   {settings.host}")
    print(f"graph:  {settings.graphname}")
    print(f"secret: {secret[:6]}...{secret[-4:]}  (length {len(secret)})")
    print()

    client = _check("connect + auth", lambda: Client(settings))

    for vertex_type, expected in _VERTEX_EXPECTATIONS:
        n = _check(
            f"count {vertex_type:>15s} (expect {expected})",
            lambda vt=vertex_type: _count_vertices(client, vt),
        )
        if n != expected:
            print(f"  note: got {n}, expectation was {expected}")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
