"""The loaded world's identity: the hash of read_world's fields that names every cache and store
folder, and the fields PhantomLedger's world record shares with read_world; and the rest of a
cached row's key, the folder of the query versions and settings and the folder of each Monday
(docs/design.md, "Cache")."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mule_pattern_learner.config import SubgraphConfig
from mule_pattern_learner.contract import ACCOUNT_TYPES, PAYMENT_RAILS
from mule_pattern_learner.paths import IDENTITY
from mule_pattern_learner.tigergraph.queries import (
    READ_ANSWER_KEY,
    READ_MONEY_FLOW,
    READ_WORLD,
    VERSIONS,
    read_world,
)
from mule_pattern_learner.tigergraph.queries.read_world import (
    EDGE_TYPES,
    SECONDS,
    VERTEX_TYPES,
)
from mule_pattern_learner.tigergraph.world import (
    CACHED_QUERIES,
    cache_key,
    identity,
    monday_folder,
    shared_fields,
)

# read_world's answer for a small world, as TigerGraph prints it: no key for a type or value
# with no row, which the parser reads as 0.
WORLD = read_world.parse(
    [
        {
            "status": "ok",
            "version": VERSIONS[READ_WORLD],
            "vertices": {"Party": 3, "Account": 4, "Payment_Transaction": 5, "Device": 1},
            "edges": {"Party_Owns_Account": 4, "Transaction_From_Account": 5},
            "account_types": {"deposit": 3, "gl": 1},
            "payment_rails": {"ach": 4, "unknown": 1},
            "external_accounts": 1,
            "seconds": {"Party.joined_ms": 3_000, "Payment_Transaction.event_ts_ms": 9_000_000},
            "cents": {"Payment_Transaction": 12_345},
            "squared_degrees": {"Party_Owns_Account": 6, "Transaction_From_Account": 5},
            "squared_target_degrees": {"Party_Owns_Account": 4, "Transaction_From_Account": 9},
            "degree_products": {"Party_Owns_Account": 6, "Transaction_From_Account": 9},
            "first_joined_ms": 1_000,
            "first_event_ms": 1_704_067_200_000,
            "last_event_ms": 1_743_292_800_000,
        }
    ]
)


def test_the_identity_is_32_hex_digits_and_pinned() -> None:
    found = identity(WORLD)
    assert IDENTITY.fullmatch(found)
    # Golden: blake2b, 16 bytes, of the canonical JSON of WORLD. A change here renames every
    # cache and store folder.
    assert found == "92cf9e2a5d23e5995375bc0416c07822"


def test_the_identity_is_stable_under_the_order_of_every_map() -> None:
    maps = {name: value for name, value in vars(WORLD).items() if isinstance(value, dict)}
    reordered = replace(
        WORLD, **{name: dict(reversed(value.items())) for name, value in maps.items()}
    )
    assert list(reordered.vertices) != list(WORLD.vertices)
    assert identity(reordered) == identity(WORLD)


def test_any_change_of_a_count_sum_or_degree_gives_another_identity() -> None:
    changed = [
        replace(WORLD, external_accounts=2),
        replace(WORLD, vertices={**WORLD.vertices, "Token": 1}),
        replace(WORLD, seconds={**WORLD.seconds, "Account.opened_ms": 1}),
        replace(WORLD, cents={**WORLD.cents, "Zelle_Transfer": 1}),
        # One link moving between vertices of degrees 2 and 1 leaves the squared sums alone;
        # the products move.
        replace(WORLD, degree_products={**WORLD.degree_products, "Party_Owns_Account": 7}),
        replace(WORLD, squared_target_degrees={**WORLD.squared_target_degrees, "Party_Uses_IP": 1}),
        replace(WORLD, last_event_ms=WORLD.last_event_ms + 1),
    ]
    found = {identity(world) for world in changed}
    assert len(found) == len(changed) and identity(WORLD) not in found


def test_the_shared_fields_are_those_of_phantomledgers_world_record() -> None:
    shared = shared_fields(WORLD)
    expected = {
        *(f"vertices.{kind}" for kind in VERTEX_TYPES),
        *(f"edges.{kind}" for kind in EDGE_TYPES),
        *(f"account_types.{value}" for value in ACCOUNT_TYPES),
        *(f"payment_rails.{value}" for value in PAYMENT_RAILS),
        "external_accounts",
        *(f"seconds.{key}" for key in SECONDS),
        "cents.Payment_Transaction",
        "cents.Zelle_Transfer",
        "first_joined_ms",
        "first_event_ms",
        "last_event_ms",
    }
    # 7 vertex types, 15 forward edge types, 5 account types, 6 rails, 1, 15 times, 2, 3.
    assert set(shared) == expected and len(shared) == 54
    assert list(shared) == sorted(shared)
    assert shared["edges.Party_Owns_Account"] == 4 and shared["seconds.Party.joined_ms"] == 3_000
    assert shared["vertices.Token"] == 0 and shared["last_event_ms"] == 1_743_292_800_000
    # The reverse types and the degree sums are read_world's own.
    assert not any("Owned_By" in key or "degree" in key for key in shared)
    # A shared field moves with the world, and a degree sum does not reach the shared ones.
    assert shared_fields(replace(WORLD, external_accounts=2))["external_accounts"] == 2
    products = replace(WORLD, degree_products={**WORLD.degree_products, "Party_Owns_Account": 7})
    assert shared_fields(products) == shared


def test_the_cache_key_names_the_query_versions_and_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = cache_key(SubgraphConfig())
    assert IDENTITY.fullmatch(key) and key == cache_key(SubgraphConfig())
    # Two settings never share a folder.
    assert cache_key(SubgraphConfig(hub_threshold=300)) != key
    assert cache_key(SubgraphConfig(co_users=47)) != key
    # Another version of a cached query moves every cached row to another folder; a query whose
    # rows the cache does not hold moves none.
    assert READ_MONEY_FLOW in CACHED_QUERIES and READ_ANSWER_KEY not in CACHED_QUERIES
    assert READ_WORLD not in CACHED_QUERIES
    monkeypatch.setitem(VERSIONS, READ_ANSWER_KEY, "read_answer_key-000000000000")
    assert cache_key(SubgraphConfig()) == key
    monkeypatch.setitem(VERSIONS, READ_MONEY_FLOW, "read_money_flow-000000000000")
    assert cache_key(SubgraphConfig()) != key


def test_a_mondays_folder_is_its_date() -> None:
    assert monday_folder(1_711_929_600_000) == "2024-04-01"
    assert monday_folder(1_742_774_400_000) == "2025-03-24"
    # Only a Monday at 00:00 UTC has a folder.
    for t_ms in (1_711_929_600_001, 1_711_929_600_000 + 86_400_000, 1_711_926_000_000):
        with pytest.raises(ValueError, match="Monday at 00:00 UTC"):
            monday_folder(t_ms)
