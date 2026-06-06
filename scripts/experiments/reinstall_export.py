import sys
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path


def _gsql(client: Client, statement: str) -> str:
    return cast(str, client.conn.gsql(statement))


def _run(client: Client, name: str, params: dict[str, object]) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name, params))


def main() -> int:
    settings = Settings()
    client = Client(settings)
    graph = settings.graphname
    print(f"graph: {graph}\n")

    path = gsql_path("export_has_paid_edges")
    if not path.is_file():
        print(f"FILE NOT FOUND: {path}")
        return 1

    text = path.read_text(encoding="utf-8")

    print("=" * 60)
    print("STEP 1: what the FILE on disk contains")
    print("=" * 60)
    print(f"path: {path}")
    has_seeds = "Seeds" in text and "PageAccounts" not in text
    has_page = "PageAccounts" in text
    print(f"  contains 'PageAccounts' : {has_page}")
    print(f"  contains 'Seeds' (old)  : {'Seeds' in text}")
    print(f"  contains '20000'        : {'20000' in text}")
    if has_seeds:
        print("  --> FILE IS THE OLD VERSION. Replace it with the corrected file first.")
        return 1
    if not has_page:
        print("  --> FILE has neither marker; unexpected. Inspect it manually.")
        return 1
    print("  --> file is the NEW version (good).")

    print()
    print("=" * 60)
    print("STEP 2: DROP the installed query (force removal of stale compiled version)")
    print("=" * 60)
    drop_out = _gsql(client, f"USE GRAPH {graph}\nDROP QUERY export_has_paid_edges\n")
    print(drop_out.strip()[:400])

    print()
    print("=" * 60)
    print("STEP 3: CREATE + INSTALL fresh from the file")
    print("=" * 60)
    install_out = _gsql(client, text + "\nINSTALL QUERY export_has_paid_edges\n")
    low = install_out.lower()
    if any(m in low for m in ("error", "fail", "could not", "cannot")):
        print("INSTALL REPORTED A PROBLEM:\n" + install_out)
        return 1
    print(install_out.strip()[:400])

    print()
    print("=" * 60)
    print("STEP 4: VERIFY what the server now returns (page_size=3)")
    print("=" * 60)
    raw = _run(client, "export_has_paid_edges", {"cursor": "", "page_size": 3})
    block0 = raw[0] if raw else None
    if isinstance(block0, dict):
        keys = list(cast(dict[str, object], block0).keys())
        print(f"  block[0] key: {keys}")
        if "PageAccounts" in keys:
            verts = cast(dict[str, object], block0)["PageAccounts"]
            n = len(cast(list[object], verts)) if isinstance(verts, list) else "?"
            print(f"  PageAccounts vertices returned: {n}  (expected 3)")
            print("  SUCCESS: server now runs the NEW query.")
        elif "Seeds" in keys:
            print("  STILL 'Seeds' --> the running query is STILL stale.")
            print("  This points to a server-side caching / wrong-graph issue.")
            return 1
    else:
        print(f"  unexpected result: {raw!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
