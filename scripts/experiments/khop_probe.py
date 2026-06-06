import json
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
    print(f"graph: {settings.graphname}\n")

    path = gsql_path("sample_khop_neighborhood")
    if not path.is_file():
        print(f"missing: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    print("Installing sample_khop_neighborhood ...")
    out = _gsql(client, text + "\nINSTALL QUERY sample_khop_neighborhood\n")
    if any(m in out.lower() for m in ("error", "fail", "could not", "cannot", "not valid")):
        print("INSTALL FAILED:\n" + out)
        return 1
    print("installed ok\n")

    # use two known active seeds from earlier demos
    seed_ids = ["A0000000001", "A0000000003"]
    # VERTEX<T> params must be passed as 1-tuples (id,) in the current API;
    # plain strings trigger a deprecated-format fallback.
    seeds: list[tuple[str]] = [(sid,) for sid in seed_ids]
    print(f"running with seeds={seed_ids}\n")
    raw = _run(
        client,
        "sample_khop_neighborhood",
        {
            "seeds": seeds,
            "fanout_has_paid": 15,
            "fanout_account_account": 10,
            "fanout_acct_party": 5,
            "fanout_party_entity": 5,
            "fanout_entity_party": 10,
            "fanout_party_acct": 10,
            "fanout_has_paid_2": 5,
            "fanout_account_account_2": 5,
            "fanout_acct_party_2": 3,
            "fanout_party_entity_2": 3,
            "fanout_entity_party_2": 5,
            "fanout_party_acct_2": 5,
        },
    )

    # report shapes per node-set / edge-set block
    edge_type_counts: dict[str, int] = {}
    for block in raw:
        if not isinstance(block, dict):
            continue
        b = cast(dict[str, object], block)
        for key in ("account_ids", "party_ids", "entity_ids"):
            if key in b:
                ids = b[key]
                n = len(cast(list[object], ids)) if isinstance(ids, list) else "?"
                print(f"{key:14s}: {n}")
                if isinstance(ids, list):
                    print(f"  sample: {cast(list[object], ids)[:6]}")
        if "edges" in b:
            edges = b["edges"]
            if isinstance(edges, list):
                edge_list = cast(list[object], edges)
                print(f"edges total   : {len(edge_list)}")
                for e in edge_list:
                    if isinstance(e, dict):
                        et = cast(dict[str, object], e).get("e_type")
                        if isinstance(et, str):
                            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
                print("  by type:")
                for et, c in sorted(edge_type_counts.items()):
                    print(f"    {et:16s}: {c}")
                print("  identity-edge samples:")
                shown = 0
                for e in edge_list:
                    if isinstance(e, dict):
                        et = cast(dict[str, object], e).get("e_type")
                        if isinstance(et, str) and et in (
                            "Account_Party",
                            "Party_Entity",
                            "Entity_Party",
                            "Party_Account",
                        ):
                            print(f"    {e}")
                            shown += 1
                            if shown >= 4:
                                break

    print("\n--- raw JSON (first 900 chars) ---")
    print(json.dumps(raw, default=str)[:900])
    return 0


if __name__ == "__main__":
    sys.exit(main())
