"""The customer hold-out: its seeded hash, and its split written on the reference graph."""

from __future__ import annotations

import pytest

from mule_pattern_learner.data.split import HELD_OUT_SHARE, held_out, splits, write_holdout
from mule_pattern_learner.testing.reference_graph import ReferenceGraph
from mule_pattern_learner.testing.world import standard_world
from mule_pattern_learner.tigergraph.queries import SplitWritten

IDS = [f"P{number:06d}" for number in range(20_000)]


def test_a_quarter_of_the_customers_are_held_out_independently_of_their_neighbours() -> None:
    chosen = splits(IDS, seed=7)
    assert abs(sum(chosen.values()) / len(IDS) - HELD_OUT_SHARE) < 0.01
    # Neighbouring ids are held out together about as often as any two ids, so the hash is
    # no rotation of the id.
    pairs = sum(chosen[a] and chosen[b] for a, b in zip(IDS, IDS[1:], strict=False))
    assert abs(pairs / (len(IDS) - 1) - HELD_OUT_SHARE**2) < 0.01


def test_the_seed_and_nothing_else_decides_a_customers_split() -> None:
    first = splits(IDS, seed=7)
    assert splits(reversed(IDS), seed=7) == first
    assert all(held_out(party_id, 7) == bool(split) for party_id, split in first.items())
    # Another seed holds out another quarter, unrelated to the first.
    other = splits(IDS, seed=8)
    both = sum(first[party_id] and other[party_id] for party_id in IDS)
    assert abs(both / len(IDS) - HELD_OUT_SHARE**2) < 0.01


def test_the_holdout_is_written_on_every_party_and_the_accounts_it_owns() -> None:
    tables = standard_world().tables()
    graph = ReferenceGraph(tables)
    parties = list(tables["Party"].id)
    chosen = splits(parties, seed=3)
    assert set(chosen.values()) == {0, 1}
    owners = dict(zip(tables["Party_Owns_Account"].to_id, tables["Party_Owns_Account"].from_id))
    assert write_holdout(graph, parties, seed=3) == SplitWritten(len(parties), len(owners))
    assert graph.party_split == chosen
    assert {account: graph.account_split[account] for account in owners} == {
        account: chosen[owner] for account, owner in owners.items()
    }
    # Accounts no customer owns keep split -1.
    assert graph.account_split["X_CASH"] == graph.account_split["G_FEES"] == -1


def test_an_unknown_party_writes_nothing() -> None:
    graph = ReferenceGraph(standard_world().tables())
    with pytest.raises(ValueError, match="unknown_party"):
        write_holdout(graph, ["P_ANN", "P_NOBODY"], seed=0)
    assert set(graph.party_split.values()) == {-1}
