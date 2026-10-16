"""The world's identity and its lookback check, on read_world responses written by hand.

Each change below is what read_world prints when the world changes as the test's name says:
the counts by rail move, a link time sum moves by whole seconds, a payment type's cents
move, and a rewired edge moves the squared degrees of its type.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from mule_pattern_learner.tigergraph.queries import READ_WORLD, VERSIONS
from mule_pattern_learner.tigergraph.world import (
    check_lookback,
    data_start,
    identity,
    read_world,
)

START_MS = 1_672_531_200_000  # 2023-01-01 00:00 UTC


def _head(**changes: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": VERSIONS[READ_WORLD],
        "vertices": {"Party": 2, "Account": 3, "Device": 1},
        "edges": {"Party_Owns_Account": 3, "Party_Uses_Device": 2},
        "account_types": {"deposit": 2, "unknown": 1},
        "payment_rails": {"ach": 4},
        "external_accounts": 1,
        "seconds": {
            "Party.joined_ms": 3_345_062_400,
            "Party_Uses_Device.valid_from_ms": 3_345_062_460,
        },
        "cents": {"Payment_Transaction": 123_456},
        "squared_degrees": {"Party_Owns_Account": 5, "Device_Used_By_Party": 4},
        "first_joined_ms": START_MS,
        "first_event_ms": START_MS + 60_000,
        "last_event_ms": START_MS + 700 * 86_400_000,
        **changes,
    }


class WorldExecutor:
    """Answers read_world with one printed row."""

    def __init__(self, head: dict[str, Any]) -> None:
        self.head = head

    def run(
        self,
        name: str,
        params: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int | None = None,
        timeout_retries: int = 1,
    ) -> list[dict[str, Any]]:
        assert (name, params) == (READ_WORLD, {})
        return [self.head]


def _identity(**changes: Any) -> str:
    return identity(read_world(WorldExecutor(_head(**changes))))


def test_the_same_world_has_the_same_identity_whatever_the_order_printed() -> None:
    reordered = _head(vertices={"Device": 1, "Account": 3, "Party": 2})
    assert identity(read_world(WorldExecutor(reordered))) == _identity()
    assert len(_identity()) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"payment_rails": {"ach": 3, "card": 1}},
        {
            "seconds": {
                "Party.joined_ms": 3_345_062_400,
                "Party_Uses_Device.valid_from_ms": 3_345_062_461,
            }
        },
        {"cents": {"Payment_Transaction": 123_457}},
        {"squared_degrees": {"Party_Owns_Account": 5, "Device_Used_By_Party": 2}},
    ],
    ids=["one rail", "one valid_from_ms", "one amount", "one edge's endpoint"],
)
def test_a_changed_world_has_another_identity(changes: dict[str, Any]) -> None:
    assert _identity(**changes) != _identity()


def test_the_lookback_must_lie_inside_the_data() -> None:
    world = read_world(WorldExecutor(_head()))
    assert data_start(world) == START_MS
    check_lookback(world, date(2024, 1, 1), 365)
    with pytest.raises(ValueError, match="warm-up is shorter than the lookback"):
        check_lookback(world, date(2023, 12, 31), 365)
    later = read_world(WorldExecutor(_head(first_joined_ms=START_MS + 1)))
    assert data_start(later) == START_MS + 1
    with pytest.raises(ValueError, match="365-day lookback before 2024-01-01"):
        check_lookback(later, date(2024, 1, 1), 365)
