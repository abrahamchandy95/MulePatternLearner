"""Supervision of the reference graph's world, read into a frame at a cutoff."""

from __future__ import annotations

import pytest

from mule_pattern_learner.data.supervision import read_supervision
from mule_pattern_learner.testing.reference_graph import DAY_MS
from mule_pattern_learner.testing.world import Mule, T, standard_world
from mule_pattern_learner.tigergraph.queries import Phase

D = DAY_MS


def test_supervision_is_the_phases_mules_found_before_the_cutoff_by_account() -> None:
    world = standard_world()
    world.party("P_HELD", T - 100 * D, held_out=True)
    found = Mule(1, "first_layer", "pass_through", T - 50 * D, T - 4 * D, T - 3 * D, "trace")
    world.account("A_HELD", "P_HELD", T - 100 * D, closed_ms=T - 3 * D, mule=found)
    graph = world.graph()
    known = read_supervision(graph, T, Phase.TRAINING)
    assert list(known.columns) == [
        "account_id",
        "discovered_ms",
        "discovery_channel",
        "mule_onset_ms",
    ]
    assert [str(dtype) for dtype in known.dtypes] == ["str", "int64", "str", "int64"]
    assert known.to_dict("records") == [
        {
            "account_id": "A_MIA",
            "discovered_ms": T - 5 * D,
            "discovery_channel": "victim_report",
            "mule_onset_ms": T - 60 * D,
        }
    ]
    # The held-out mule only in the test phase; Max and Roy once the cutoff passes them.
    assert list(read_supervision(graph, T, Phase.TEST).account_id) == ["A_HELD", "A_MIA"]
    later = read_supervision(graph, T + 30 * D + 1, Phase.TRAINING)
    assert list(later.account_id) == ["A_MAX", "A_MIA", "A_ROY"]
    # Before any discovery the frame is empty, with the same columns and types.
    empty = read_supervision(graph, T - 100 * D, Phase.TEST)
    assert empty.empty and list(empty.dtypes) == list(known.dtypes)


@pytest.mark.parametrize("onset", [0, T - 2 * D])
def test_a_mule_found_without_an_onset_before_its_discovery_is_refused(onset: int) -> None:
    world = standard_world()
    world.party("P_ODD", T - 100 * D)
    odd = Mule(2, "first_layer", "collector", onset, T - 2 * D, T - 3 * D, "monitoring")
    world.account("A_ODD", "P_ODD", T - 100 * D, closed_ms=T - 3 * D, mule=odd)
    with pytest.raises(ValueError, match="A_ODD"):
        read_supervision(world.graph(), T, Phase.TRAINING)
