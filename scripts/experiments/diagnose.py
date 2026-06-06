import sys
from typing import cast

from mule_pattern_learner.tigergraph.client import Client
from mule_pattern_learner.tigergraph.settings import Settings
from mule_pattern_learner.tigergraph.gsql_paths import gsql_path


def _gsql(client: Client, statement: str) -> str:
    return cast(str, client.conn.gsql(statement))


def _run(client: Client, name: str, params: dict[str, object]) -> list[object]:
    return cast(list[object], client.conn.runInstalledQuery(name, params))


def _scalar(raw: list[object], key: str) -> object:
    for block in raw:
        if isinstance(block, dict) and key in block:
            return cast(dict[str, object], block)[key]
    return None


def main() -> int:
    settings = Settings()
    print(f"host:  {settings.host}")
    print(f"graph: {settings.graphname}")
    client = Client(settings)

    path = gsql_path("diagnose_export")
    if not path.is_file():
        print(f"missing: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    out = _gsql(client, text + "\nINSTALL QUERY diagnose_export\n")
    if any(m in out.lower() for m in ("error", "fail", "could not", "cannot")):
        print("install failed:\n" + out)
        return 1
    print("installed diagnose_export ... ok")

    raw = _run(client, "diagnose_export", {"cursor": "", "page_size": 50})

    print(f"  accounts_all          = {_scalar(raw, 'accounts_all')}")
    print(f"  accounts_after_cursor = {_scalar(raw, 'accounts_after_cursor')}")
    print(f"  edges_from_page       = {_scalar(raw, 'edges_from_page')}")

    # show a couple of page account ids + their out-edge counts
    for block in raw:
        if isinstance(block, dict) and "PageAccounts" in block:
            verts_obj = cast(dict[str, object], block)["PageAccounts"]
            if isinstance(verts_obj, list):
                verts = cast(list[object], verts_obj)
                print(f"  page returned {len(verts)} account vertices; first few:")
                for v in verts[:5]:
                    if isinstance(v, dict):
                        vt = cast(dict[str, object], v)
                        vid = vt.get("v_id")
                        attrs = vt.get("attributes")
                        n_edges = 0
                        if isinstance(attrs, dict):
                            oe = cast(dict[str, object], attrs).get("out_edges")
                            if isinstance(oe, list):
                                n_edges = len(cast(list[object], oe))
                        print(f"    id={vid!r}  out_edges={n_edges}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
