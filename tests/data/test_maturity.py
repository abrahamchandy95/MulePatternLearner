"""The delay from onset to discovery of the reference graph's revealed mules, and the gap it
implies, worked out by hand.

At T, the standard world's revealed mule is Mia: found 55 days after her onset, 60 days
after which came T. The world here adds three more found before T: Far after 10 days (its
onset 100 days before T), Mid after 30 (onset 50 days before) and Near after 5 (onset 20
days before). Near, so close to T, was revealed only because it was found within 20 days,
and Mid only within 50, so the long delays count for more than their plain share.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mule_pattern_learner.data.maturity import (
    DISCOVERED_SHARE,
    delay_distribution,
    implied_gap_days,
)
from mule_pattern_learner.data.supervision import read_supervision
from mule_pattern_learner.testing.reference_graph import DAY_MS
from mule_pattern_learner.testing.world import Mule, T, WorldBuilder, standard_world
from mule_pattern_learner.tigergraph.queries import Phase

D = DAY_MS


def _found(world: WorldBuilder, account_id: str, onset_ms: int, found_ms: int) -> None:
    """Adds a customer whose account is a mule from onset_ms, closed when found at found_ms."""
    party = account_id.replace("A_", "P_")
    world.party(party, T - 300 * D)
    mule = Mule(5, "first_layer", "pass_through", onset_ms, found_ms, found_ms, "trace")
    world.account(account_id, party, T - 300 * D, closed_ms=found_ms, mule=mule)


def _world() -> WorldBuilder:
    world = standard_world()
    for account_id, onset, found in (("A_FAR", 100, 90), ("A_MID", 50, 20), ("A_NEAR", 20, 15)):
        _found(world, account_id, T - onset * D, T - found * D)
    return world


def _revealed(world: WorldBuilder) -> pd.DataFrame:
    return read_supervision(world.graph(), T, Phase.TRAINING)


def test_one_revealed_mule_is_found_within_its_own_delay() -> None:
    within = delay_distribution(_revealed(standard_world()), T)
    assert within.to_dict() == {55 * D: 1.0}
    assert implied_gap_days(_revealed(standard_world()), T) == 55


def test_the_estimate_gives_the_delays_the_newest_mules_could_not_show_their_weight() -> None:
    """Going down from the longest delay, each step keeps the share not found at it among
    the mules at risk there: those found as fast or faster that could have taken as long.
    At 55 days, Far and Mia (1/2 kept); at 30, Far and Mid (1/2); at 10, Far and Near (1/2);
    at 5, Near alone (none kept). The plain shares would be 1/4, 1/2, 3/4 and 1."""
    within = delay_distribution(_revealed(_world()), T)
    assert within.to_dict() == {5 * D: 0.125, 10 * D: 0.25, 30 * D: 0.5, 55 * D: 1.0}
    assert DISCOVERED_SHARE == 0.9
    revealed = _revealed(_world())
    assert implied_gap_days(revealed, T) == 55
    assert implied_gap_days(revealed, T, share=0.5) == 30
    assert implied_gap_days(revealed, T, share=0.2) == 10
    assert implied_gap_days(revealed, T, share=0.125) == 5


def test_a_mule_not_yet_found_changes_nothing() -> None:
    """Only revealed mules count: Max and Roy, found after T, are not among them, and
    neither is a mule found later still."""
    world = _world()
    _found(world, "A_SLOW", T - 200 * D, T + D)
    pd.testing.assert_series_equal(
        delay_distribution(_revealed(world), T), delay_distribution(_revealed(_world()), T)
    )


def test_the_gap_is_rounded_up_to_whole_days() -> None:
    revealed = _revealed(standard_world())
    later = revealed.assign(discovered_ms=revealed.discovered_ms + 1)
    assert implied_gap_days(later, T) == 56


def test_no_mules_a_mule_found_after_the_cutoff_or_a_bad_share_is_refused() -> None:
    revealed = _revealed(standard_world())
    with pytest.raises(ValueError, match="No revealed mule"):
        delay_distribution(revealed.iloc[:0], T)
    with pytest.raises(ValueError, match="before the cutoff"):
        delay_distribution(revealed, T - 5 * D)
    for share in (0.0, 1.5):
        with pytest.raises(ValueError, match="share"):
            implied_gap_days(revealed, T, share=share)
