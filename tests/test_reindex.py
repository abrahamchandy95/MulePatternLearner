from mule_pattern_learner.indexing.reindex import (
    RawNeighborhood,
    reindex_neighborhood,
)


def test_seeds_ordered_first() -> None:
    # Account ids come back in arbitrary (set) order; seeds scattered within
    raw = RawNeighborhood(
        node_ids={"Account": ["X", "S2", "Y", "S1", "Z"]},
        edges=[],
    )
    local = reindex_neighborhood(raw, seed_ids=["S1", "S2"])
    # the first len(seeds) Account rows must be exactly the seeds, in seed order
    assert local.node["Account"][:2] == ["S1", "S2"]
    assert set(local.node["Account"]) == {"X", "Y", "Z", "S1", "S2"}
    assert len(local.node["Account"]) == 5


def test_edge_indices_consistent_after_reorder() -> None:
    # edges must still point to the same nodes after the reorder
    raw = RawNeighborhood(
        node_ids={"Account": ["X", "S2", "Y", "S1", "Z"]},
        edges=[("S1", "Y", "HAS_PAID"), ("X", "S2", "HAS_PAID")],
    )
    local = reindex_neighborhood(raw, seed_ids=["S1", "S2"])
    order = local.node["Account"]
    # S1=0, S2=1, then X,Y,Z in original order -> X=2, Y=3, Z=4
    assert order == ["S1", "S2", "X", "Y", "Z"]
    schema = ("Account", "HAS_PAID", "Account")
    # edge S1->Y is (0, 3); edge X->S2 is (2, 1)
    assert list(zip(local.row[schema], local.col[schema])) == [(0, 3), (2, 1)]


def test_no_seed_ids_keeps_original_order() -> None:
    raw = RawNeighborhood(node_ids={"Account": ["A", "B", "C"]}, edges=[])
    local = reindex_neighborhood(raw)
    assert local.node["Account"] == ["A", "B", "C"]


def test_missing_seed_skipped() -> None:
    raw = RawNeighborhood(node_ids={"Account": ["A", "B"]}, edges=[])
    local = reindex_neighborhood(raw, seed_ids=["MISSING", "A"])
    assert local.node["Account"][:1] == ["A"]
    assert set(local.node["Account"]) == {"A", "B"}
