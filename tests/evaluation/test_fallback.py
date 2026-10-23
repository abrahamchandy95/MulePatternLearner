"""The test-role fallback list (docs/design.md, "If whole-world scoring is too slow"): every true
mule on its active and early-warning Mondays and a weighted keyed sample of the rest; and the
Keys chosen by their reader guard: selection on a world after a test-role fallback fetch of it,
with the answer key erased, reads byte-identical inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.data.examples import monday, selection_frame
from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.evaluation.answer_key import read_answer_key
from mule_pattern_learner.evaluation.fallback import by_monday, fallback_keys
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import NO_RING, T
from mule_pattern_learner.training.trainer import SelectionWorld

W = 7 * 86_400_000
MONDAYS = (T, T + W, T + 2 * W)
DATA = DEFAULT_CONFIG.data
WORLD = "0123456789abcdef0123456789abcdef"


def _populations(graph: ReferenceGraph) -> dict[int, list[str]]:
    return {t_ms: [row.account_id for row in list_population(graph, t_ms)] for t_ms in MONDAYS}


def test_the_list_holds_every_true_mule_on_its_active_and_early_warning_mondays(
    graph: ReferenceGraph,
) -> None:
    keys = fallback_keys(read_answer_key(Spy(graph)), _populations(graph), DATA)
    kept = keys[keys["weight"] == 1.0]
    found = {t_ms: sorted(accounts) for t_ms, accounts in by_monday(kept).items()}
    ring = ["a_hr1", "a_hr2", "a_hr3"]
    assert found == {
        # Ring 7's four mules and the drop are active at T only; ring 11's three from T to
        # T + 2 weeks; the drop's last payment is exactly 7 days before T + 1 week.
        T: sorted(["a_collector", "a_drop", "a_mule", "a_mule2", "a_mule3", *ring]),
        # a_late's onset is 10 days after T: T + 1 week is its early-warning Monday.
        T + W: sorted(["a_drop", "a_late", *ring]),
        T + 2 * W: sorted(["a_late", *ring]),
    }
    sampled = keys[keys["weight"] != 1.0]
    assert (sampled["weight"] == DATA.fallback_weight).all() and not sampled.empty
    assert not set(zip(sampled["t_ms"], sampled["account_id"], strict=True)) & set(
        zip(kept["t_ms"], kept["account_id"], strict=True)
    )


def _erase_answer_key(tables: Mapping[str, pd.DataFrame]) -> ReferenceGraph:
    """The world with the answer key erased: is_mule equal to pu_label, no ring, and no
    activity period but an identified mule's, so that only what read_labels reads is left."""
    accounts = tables["Account"].copy()
    internal = ~accounts["is_external"]
    accounts.loc[internal, "is_mule"] = accounts.loc[internal, "pu_label"]
    accounts["mule_ring_id"] = NO_RING
    hidden = accounts["pu_label"] != 1
    accounts.loc[hidden, "mule_onset_ms"] = 0
    accounts.loc[hidden, "mule_last_ms"] = 0
    return ReferenceGraph({**tables, "Account": accounts})


def _selection(graph: ReferenceGraph) -> pd.DataFrame:
    labels = read_identified(Spy(graph))
    days = [
        selection_frame(monday(t_ms, accounts, labels.mules), labels.accounts, DATA)
        for t_ms, accounts in _populations(graph).items()
    ]
    return pd.concat(days, ignore_index=True)


def _bytes(batches: list[Batch]) -> list[tuple[object, ...]]:
    found: list[tuple[object, ...]] = []
    for batch in batches:
        data = batch.data
        tensors: Mapping[str, torch.Tensor] = {
            **{f"x.{kind}": data[kind].x for kind in data.node_types},
            **{f"edges.{relation}": data[relation].edge_index for relation in data.edge_types},
            **{
                f"attrs.{relation}": data[relation].edge_attr
                for relation in data.edge_types
                if "edge_attr" in data[relation]
            },
            "scored": data.scored_index,
            "community": data.community,
        }
        found.append(
            (batch.keys, {name: value.numpy().tobytes() for name, value in tensors.items()})
        )
    return found


def test_selection_after_a_fallback_fetch_reads_byte_identical_inputs(
    tables: Mapping[str, pd.DataFrame], graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    erased = _erase_answer_key(tables)
    keys = _selection(erased)
    assert keys.equals(_selection(graph)) and keys["positive"].sum() == 3
    fresh = Cache(tmp_path / "fresh")
    fetch(Spy(erased), fresh, by_monday(keys), parity=parity)
    # The other cache first fills the test-role fallback list, which the truth chose.
    fallback = fallback_keys(read_answer_key(Spy(graph)), _populations(graph), DATA)
    fetched = Cache(tmp_path / "fetched")
    fetch(Spy(graph), fetched, by_monday(fallback), parity=parity)
    spy = Spy(erased)
    fetch(spy, fetched, by_monday(keys), parity=parity)
    assert spy.calls, "the selection keys need rows the fallback list did not fetch"
    first = _bytes(list(SelectionWorld(fresh, WORLD, keys, 16).batches))
    second = _bytes(list(SelectionWorld(fetched, WORLD, keys, 16).batches))
    # 34 scorable accounts each Monday; a_mule is out of the ranking at T + 1 week and T + 2
    # weeks, a_drop at T + 2 weeks: batches of 16 of 34, 33 and 32 ranked accounts.
    assert first == second and len(first) == 3 + 3 + 2
