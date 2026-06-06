from __future__ import annotations

from mule_pattern_learner.pyg.reindex import (
    parse_raw_result,
    reindex_neighborhood,
)


def main() -> int:
    # A tiny hand-built neighborhood that mimics one GSQL result, so the
    # global->local translation is easy to follow by eye.
    raw_result: list[object] = [
        {"account_ids": ["A0000000001", "XM00000272", "A0000043166"]},
        {"party_ids": ["C00000000113"]},
        {"entity_ids": ["59770824"]},
        {
            "edges": [
                {"from_id": "A0000000001", "to_id": "XM00000272", "e_type": "HAS_PAID"},
                {"from_id": "A0000043166", "to_id": "A0000000001", "e_type": "HAS_PAID"},
                {"from_id": "A0000000001", "to_id": "C00000000113", "e_type": "Account_Party"},
                {"from_id": "C00000000113", "to_id": "59770824", "e_type": "Party_Entity"},
            ]
        },
    ]

    print("=" * 68)
    print("STEP 1: RAW from GSQL  (global ids \u2014 what TigerGraph returns)")
    print("=" * 68)
    raw = parse_raw_result(raw_result)
    for ntype, ids in raw.node_ids.items():
        print(f"  {ntype:16s}: {ids}")
    print("  edges (global):")
    for f, t, et in raw.edges:
        print(f"    {f}  --{et}-->  {t}")

    print()
    print("=" * 68)
    print("STEP 2: the global->local maps the reindexer builds")
    print("=" * 68)
    local = reindex_neighborhood(raw)
    for ntype, ids in local.node.items():
        print(f"  {ntype}:")
        for i, gid in enumerate(ids):
            print(f"    local {i}  <->  {gid}")

    print()
    print("=" * 68)
    print("STEP 3: LOCAL edges  (row/col = positions, what the GNN consumes)")
    print("=" * 68)
    for etype in local.row:
        src_type, rel, dst_type = etype
        print(f"  edge type {etype}:")
        rows = local.row[etype]
        cols = local.col[etype]
        for r, c in zip(rows, cols):
            src_gid = local.node[src_type][r]
            dst_gid = local.node[dst_type][c]
            print(
                f"    row={r} col={c}   (i.e. {src_type}[{r}]={src_gid}"
                f"  ->  {dst_type}[{c}]={dst_gid})"
            )
        print(f"    row tensor would be: {rows}")
        print(f"    col tensor would be: {cols}")

    print()
    print("=" * 68)
    print("WHY THIS MATTERS")
    print("=" * 68)
    print("  The GNN never sees 'A0000000001'. It sees row=0 in the Account")
    print("  tensor. The edge HAS_PAID row=[0,2] col=[1,0] tells it to pass")
    print("  messages between rows of a 3-row tensor \u2014 pure array math.")
    print("  Reindexing is the translation from global ids to those rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
