"""The test-role fallback list (docs/design.md, "If whole-world scoring is too slow"): every true
mule at its active and early-warning times and a weighted keyed sample of the rest; and the
Keys chosen by their reader guard: selection on a world after a test-role fallback fetch of it,
with the answer key erased, reads byte-identical inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from mule_pattern_learner.config import FULL_DESIGN_CONFIG
from mule_pattern_learner.data.examples import selection_frame, week
from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.evaluation.answer_key import AnswerKey, read_answer_key
from mule_pattern_learner.evaluation.fallback import by_week, fallback_keys
from mule_pattern_learner.evaluation.sample import audit_sample
from mule_pattern_learner.subgraph.build import Batch
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import NO_RING, T
from mule_pattern_learner.tigergraph.queries.read_answer_key import AccountTruth
from mule_pattern_learner.training.trainer import SelectionWorld

W = 7 * 86_400_000
# The world's three scoring times, a week apart at 00:00 UTC, as these tests give them by hand.
TIMES = (T, T + W, T + 2 * W)
# The full design's fallback, a sampled account standing for 10: the reference world holds 224
# accounts a week, too few for the proof of concept's weight of 6,000 to sample any.
DATA = FULL_DESIGN_CONFIG.data
WORLD = "0123456789abcdef0123456789abcdef"


def _populations(graph: ReferenceGraph) -> dict[int, list[str]]:
    return {t_ms: [row.account_id for row in list_population(graph, t_ms)] for t_ms in TIMES}


def test_the_list_holds_every_true_mule_at_its_active_and_early_warning_times(
    graph: ReferenceGraph,
) -> None:
    keys = fallback_keys(read_answer_key(Spy(graph)), _populations(graph), DATA, TIMES)
    kept = keys[keys["weight"] == 1.0]
    found = {t_ms: sorted(accounts) for t_ms, accounts in by_week(kept).items()}
    ring = ["a_hr1", "a_hr2", "a_hr3"]
    assert found == {
        # Ring 7's four mules and the drop are active at T only; ring 11's three from T to
        # T + 2 weeks; the drop's last payment is exactly 7 days before T + 1 week.
        T: sorted(["a_collector", "a_drop", "a_mule", "a_mule2", "a_mule3", *ring]),
        # a_late's onset is 10 days after T: T + 1 week is its early-warning time.
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
        selection_frame(week(t_ms, accounts, labels.mules, TIMES), labels.accounts, DATA)
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
    fetch(Spy(erased), fresh, by_week(keys), parity=parity)
    # The other cache first fills the test-role fallback list, which the truth chose.
    fallback = fallback_keys(read_answer_key(Spy(graph)), _populations(graph), DATA, TIMES)
    fetched = Cache(tmp_path / "fetched")
    fetch(Spy(graph), fetched, by_week(fallback), parity=parity)
    spy = Spy(erased)
    fetch(spy, fetched, by_week(keys), parity=parity)
    assert spy.calls, "the selection keys need rows the fallback list did not fetch"
    first = _bytes(list(SelectionWorld(fresh, WORLD, keys, 16).batches))
    second = _bytes(list(SelectionWorld(fetched, WORLD, keys, 16).batches))
    # 34 scorable accounts each week; a_mule is out of the ranking at T + 1 week and T + 2
    # weeks, a_drop at T + 2 weeks: batches of 16 of 34, 33 and 32 ranked accounts.
    assert first == second and len(first) == 3 + 3 + 2


def _truth(account: str, mule: bool = False) -> AccountTruth:
    return AccountTruth(
        account,
        int(mule),
        0 if mule else -1,
        T if mule else 0,
        T + W if mule else 0,
        0,
        "deposit",
        True,
    )


def test_the_audit_sample_lies_inside_the_samples_the_selection_and_test_worlds_fetch() -> None:
    """Proof of concept (docs/design.md, "Proof-of-concept settings (25 October 2026)"): the
    audit's non-mule account-weeks are the lowest keyed hashes of the year, and the fallback
    sample of a world is every account-week under a hash threshold, so while the audit asks for
    fewer than the fallback samples hold, every one of them is in the sample the selection and
    test worlds fetch, and the audit fetches no non-mule account-week twice."""
    accounts = [f"a_{index:05d}" for index in range(2_000)]
    mules = {"a_00007", "a_00900"}
    key = AnswerKey(tuple(_truth(account, account in mules) for account in accounts))
    populations = {t_ms: accounts for t_ms in TIMES}
    # 6,000 account-weeks at weight 40 are about 150 in the sample.
    data = replace(FULL_DESIGN_CONFIG.data, fallback_weight=40)
    fallback = fallback_keys(key, populations, data, TIMES)
    sampled = fallback[fallback["weight"] == 40.0]
    held = set(zip(sampled["t_ms"], sampled["account_id"], strict=True))
    assert 100 < len(held) < 200
    for size in (20, 100, len(held) - 10):
        audit = audit_sample(key, populations, size, data.draw_key, TIMES)
        asked = audit[~audit["positive"]]
        assert len(asked) == size
        assert set(zip(asked["t_ms"], asked["account_id"], strict=True)) <= held
        # The mules are in the audit's sample in every week of their activity, all of them.
        assert set(audit.loc[audit["positive"], "account_id"]) == mules


def test_a_mule_with_no_time_in_the_week_after_it_is_kept_and_sampled_where_the_gap_ends() -> None:
    """Scoring times 7, 9 and 7 days apart: a mule active from T + 5 days to an hour later has
    no scoring time in the week after it, so it is active, and sampled by the audit, at T + 13
    days alone, the time that ends the 9-day gap holding its payments, whose window is the gap;
    and the fallback list holds it there and at T + 4 days, its early-warning time."""
    d = 86_400_000
    times = (T - 3 * d, T + 4 * d, T + 13 * d, T + 20 * d)
    quiet = AccountTruth("a_quiet", 1, 5, T + 5 * d, T + 5 * d + 3_600_000, 0, "deposit", True)
    key = AnswerKey((quiet, *(_truth(f"a_{index:02d}") for index in range(30))))
    populations = {t_ms: [truth.account_id for truth in key.truths] for t_ms in times}
    data = replace(FULL_DESIGN_CONFIG.data, fallback_weight=2)
    listed = fallback_keys(key, populations, data, times)
    kept = listed[(listed["account_id"] == "a_quiet") & (listed["weight"] == 1.0)]
    assert kept["t_ms"].tolist() == [T + 4 * d, T + 13 * d]
    audit = audit_sample(key, populations, 10, data.draw_key, times)
    assert audit.loc[audit["positive"], "t_ms"].tolist() == [T + 13 * d]
