"""Which configured world is loaded: the identity file whose shared fields equal read_world's,
those fields only, and its seed (docs/design.md, "The pipeline" and "Cache"); and the roles a
world plays in the rotations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, ROTATIONS
from mule_pattern_learner.data.worlds import (
    SELECT,
    TEST,
    TRAIN,
    find_world,
    match_world,
    read_record,
    roles,
)
from mule_pattern_learner.testing.calendar import (
    JOINED,
    Spy,
    calendar_world,
    identity_text,
    world_of,
)
from mule_pattern_learner.tigergraph.queries.read_world import EDGE_TYPES, World
from mule_pattern_learner.tigergraph.world import identity, shared_fields

WORLDS = DEFAULT_CONFIG.worlds
Records = Callable[[str, str], Path]


def _world() -> World:
    return world_of(calendar_world().graph())


def test_the_identity_file_matching_read_world_names_the_configured_world(
    records: Records,
) -> None:
    world = _world()
    path = records("world_1001.identity.toml", identity_text(world, 1001))
    other = records(
        "world_1002.identity.toml", identity_text(world, 1002, changed={"cents.Zelle_Transfer": 1})
    )
    found = match_world(world, [other, path], WORLDS)
    assert found.config.name == "D1" and found.config.seed == 1001
    assert found.path == path and found.world == world
    assert found.identity == identity(world)
    # The file's own keys are read as they are.
    assert found.record["seed"] == 1001 and found.record["clean_tree"] is True
    assert found.record["population"] == 30_000


def test_a_file_of_tables_reads_as_one_of_dotted_keys(records: Records) -> None:
    world = _world()
    text = identity_text(world, 2001, nested=True)
    assert "[vertices]" in text and '"vertices.Party"' not in text
    path = records("world_2001.identity.toml", text)
    assert read_record(path) == read_record(
        records("flat.identity.toml", identity_text(world, 2001))
    )
    assert match_world(world, [path], WORLDS).config.name == "A"


def test_no_matching_file_is_refused(records: Records) -> None:
    world = _world()
    with pytest.raises(ValueError, match="No identity file of 0"):
        match_world(world, [], WORLDS)
    other = records(
        "world_1002.identity.toml", identity_text(world, 1002, changed={"vertices.Account": 1})
    )
    with pytest.raises(ValueError, match="No identity file of 1 matches"):
        match_world(world, [other], WORLDS)


def test_two_matching_files_are_refused(records: Records) -> None:
    world = _world()
    first = records("world_1001.identity.toml", identity_text(world, 1001))
    second = records("world_1002.identity.toml", identity_text(world, 1002))
    with pytest.raises(ValueError, match="all match the loaded world"):
        match_world(world, [first, second], WORLDS)


@pytest.mark.parametrize(
    "field",
    [
        "vertices.Account",
        "edges.Transfer_To_Account",
        "account_types.deposit",
        "payment_rails.check",
        "external_accounts",
        "seconds.Account.opened_ms",
        "seconds.Party_Uses_IP.valid_to_ms",
        "cents.Payment_Transaction",
        "first_joined_ms",
        "first_event_ms",
        "last_event_ms",
    ],
)
def test_a_file_differing_in_one_shared_field_does_not_match(records: Records, field: str) -> None:
    world = _world()
    value = shared_fields(world)[field]
    path = records(
        "world_1001.identity.toml", identity_text(world, 1001, changed={field: value + 1})
    )
    with pytest.raises(ValueError, match="No identity file of 1 matches"):
        match_world(world, [path], WORLDS)


def test_read_worlds_own_fields_are_not_compared(records: Records) -> None:
    world = _world()
    path = records("world_1001.identity.toml", identity_text(world, 1001))
    # The degree sums are read_world's own, and neither side holds a reverse edge type's count.
    other = replace(
        world,
        squared_degrees={name: 7 for name in EDGE_TYPES},
        squared_target_degrees={name: 8 for name in EDGE_TYPES},
        degree_products={name: 9 for name in EDGE_TYPES},
    )
    assert identity(other) != identity(world)
    found = match_world(other, [path], WORLDS)
    assert found.identity == identity(other)
    assert not any(name.startswith("edges.") and "Owned_By" in name for name in read_record(path))


def test_a_file_without_a_shared_key_or_with_another_is_refused(records: Records) -> None:
    world = _world()
    # payment_rails.check counts 0, and PhantomLedger writes zero counts too.
    assert shared_fields(world)["payment_rails.check"] == 0
    missing = records(
        "missing.identity.toml", identity_text(world, 1001, dropped=("payment_rails.check",))
    )
    with pytest.raises(ValueError, match=r"missing \['payment_rails.check'\]"):
        match_world(world, [missing], WORLDS)
    extra = records(
        "extra.identity.toml", identity_text(world, 1001, changed={"vertices.Address": 0})
    )
    with pytest.raises(ValueError, match=r"unknown \['vertices.Address'\]"):
        match_world(world, [extra], WORLDS)
    text = identity_text(world, 1001, changed={"external_accounts": "1"})
    with pytest.raises(ValueError, match="not an integer"):
        match_world(world, [records("text.identity.toml", text)], WORLDS)


def test_a_key_written_both_ways_is_refused(records: Records) -> None:
    world = _world()
    text = identity_text(world, 1001) + f"[vertices]\nParty = {world.vertices['Party']}\n"
    with pytest.raises(ValueError, match="holds the key vertices.Party twice"):
        match_world(world, [records("twice.identity.toml", text)], WORLDS)


def test_a_seed_no_world_is_configured_with_is_refused(records: Records) -> None:
    world = _world()
    path = records("world_77.identity.toml", identity_text(world, 77))
    with pytest.raises(ValueError, match="of seed 77"):
        match_world(world, [path], WORLDS)
    unseeded = records("x.identity.toml", identity_text(world, 1001, dropped=("seed",)))
    with pytest.raises(ValueError, match="holds no integer seed"):
        match_world(world, [unseeded], WORLDS)


def test_find_world_runs_read_world_alone(records: Records) -> None:
    graph = calendar_world().graph()
    path = records("world_1003.identity.toml", identity_text(world_of(graph), 1003))
    spy = Spy(graph)
    assert find_world(spy, WORLDS, [path]).config.name == "D3"
    assert spy.names() == ["read_world"]


def test_a_value_outside_the_contract_is_refused(records: Records) -> None:
    w = calendar_world()
    w.customer("loan", JOINED, account_type="loan")
    with pytest.raises(ValueError, match=r"account_types of \['loan'\], outside the contract"):
        find_world(Spy(w.graph()), WORLDS, [])


def test_the_roles_of_each_world() -> None:
    assert roles("A", ROTATIONS) == {TRAIN, SELECT, TEST}
    assert roles("D1", ROTATIONS) == {TRAIN}
    assert roles("D2", ROTATIONS) == {SELECT}
    assert roles("D3", ROTATIONS) == {TEST}
    # The 5,000-person test world and the bench world play no role.
    assert roles("T", ROTATIONS) == roles("P100", ROTATIONS) == frozenset()
