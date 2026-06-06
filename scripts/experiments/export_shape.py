from __future__ import annotations

import json
import sys
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings


def _run(client: Client, name: str, params: dict[str, object]) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name, params))


def _describe(obj: object, depth: int = 0, max_depth: int = 4) -> str:
    pad = "  " * depth
    if depth > max_depth:
        return pad + "...(deeper)"
    if isinstance(obj, dict):
        d = cast(dict[str, object], obj)
        lines = [f"{pad}dict keys: {list(d.keys())}"]
        for k, v in list(d.items())[:6]:
            lines.append(f"{pad}  [{k!r}]:")
            lines.append(_describe(v, depth + 2, max_depth))
        return "\n".join(lines)
    if isinstance(obj, list):
        lst = cast(list[object], obj)
        lines = [f"{pad}list len={len(lst)}"]
        if lst:
            lines.append(f"{pad}  first item:")
            lines.append(_describe(lst[0], depth + 2, max_depth))
        return "\n".join(lines)
    text = repr(obj)
    if len(text) > 80:
        text = text[:80] + "..."
    return f"{pad}{type(obj).__name__}: {text}"


def main() -> int:
    settings = Settings()
    client = Client(settings)
    print(f"graph: {settings.graphname}\n")

    raw = _run(client, "export_has_paid_edges", {"cursor": "", "page_size": 5})

    print("=" * 60)
    print("RAW STRUCTURE of export_has_paid_edges(cursor='', page_size=5)")
    print("=" * 60)
    print(f"top-level: list of {len(raw)} blocks\n")
    for i, block in enumerate(raw):
        print(f"--- block[{i}] ---")
        print(_describe(block))
        print()

    # also dump the first ~1200 chars of raw JSON so we see exact key names
    print("=" * 60)
    print("RAW JSON (truncated)")
    print("=" * 60)
    dumped = json.dumps(raw, default=str)
    print(dumped[:1200] + ("..." if len(dumped) > 1200 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
