"""The feature audit (docs/design.md, "The feature audit"): the audited columns of a built batch,
and the three outcomes of a column: no separation, an almost perfect one, and a near-pure region
that a minority of mules holds, which keeps its ROC AUC near 0.5 and stops the run anyway."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.feature_audit import (
    COLUMNS,
    audit_columns,
    feature_audit,
    near_pure,
    tails_measured,
)
from mule_pattern_learner.subgraph.build import build, read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, HOUR_MS, T

AUDIT = replace(DEFAULT_CONFIG.audit, replicates=100)


def test_the_audited_columns_of_a_built_batch(
    graph: ReferenceGraph, parity: Path, tmp_path: Path
) -> None:
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: ["a_hr2", "a_o00"]}, parity=parity)
    found = audit_columns(build(read_rows(cache, [(T, "a_hr2"), (T, "a_o00")]), "w"))
    assert found.shape == (2, len(COLUMNS)) and len(COLUMNS) == 234
    value = {name: (found[0, i], found[1, i]) for i, name in enumerate(COLUMNS)}
    # a_hr2 received 200.00 from a_hr1 and paid 200.00 to a_hr3 in the same second, 4 days 12
    # hours before T, from the ring's device, on which the three members are enrolled.
    ago = 4 * DAY_MS + 12 * HOUR_MS
    assert value["account.in_amount_7d"][0] == 20_000 and value["account.out_7d"][0] == 1
    assert value["pays_in_mean.amount"][0] == 20_000 and value["pays_in_max.age_ms"][0] == ago
    assert value["pays_in_mean.rail=zelle"][0] == 1
    assert value["device_max.customers_90d"][0] == 3 and value["customer.co_users"][0] == 2
    assert value["seen_max.since_last_use_ms"][0] == ago
    assert value["community.members"][0] == 3
    # a_o00 paid a shop 20.00 by card 33 hours before T; it received nothing, and holds no
    # device or IP, so their aggregates are 0.
    assert value["pays_out_mean.amount"][1] == 2_000 and value["pays_out_max.rail=card"][1] == 1
    assert value["pays_out_mean.age_ms"][1] == 33 * HOUR_MS
    assert value["pays_in_max.amount"][1] == 0 and value["device_mean.customers_90d"][1] == 0
    assert value["community.members"][1] == 1 and value["customer.accounts_owned"][1] == 1


# The audit sample at production proportions: 1,200 active true-mule account-Mondays, each
# weighing 1, in 400 rings of three, and 20,000 non-mule ones each standing for 115, so that
# the sample stands for 2.3 million account-Mondays and the mules are 1,200 / 2,301,200 = 0.052%
# of the weight. In production about 12,000 of 23 million are mules, 0.052% as well: under the
# 0.1% of a tail, so a tail holds every mule that a column marks and room for 1,100 ordinary
# account-Mondays beside them.
MULES, ORDINARY, STANDS_FOR = 1_200, 20_000, 115.0


def _sample() -> tuple[NDArray[np.bool_], NDArray[np.float64], list[str]]:
    positive = np.array([True] * MULES + [False] * ORDINARY)
    weight = np.where(positive, 1.0, STANDS_FOR)
    keys = [f"ring:{i // 3}" if mule else f"account:{i}" for i, mule in enumerate(positive)]
    return positive, weight, keys


def _held_by_fifteen_percent_of_the_mules(positive: NDArray[np.bool_]) -> NDArray[np.bool_]:
    held = np.zeros(len(positive), dtype=bool)
    held[: MULES * 15 // 100] = True
    return held


def test_a_marker_a_minority_of_mules_hold_stops_the_run_whatever_its_kind() -> None:
    positive, weight, keys = _sample()
    rng = np.random.default_rng(0)
    held = _held_by_fifteen_percent_of_the_mules(positive)
    # Each marker is on 15% of the mules, 180 of them, and on no ordinary account: its ROC AUC
    # is 0.5 + 0.15 / 2 = 0.575, nowhere near 0.95, but the region where it is set holds 15% of
    # the mules (at least 1%) and nothing else.
    flag = held.astype(np.float64)
    count = np.where(held, np.log1p(1.0), 0.0)
    continuous = np.where(held, 1 + rng.random(len(positive)), 0.0)
    # Controls that must pass: noise, a count that ordinary accounts hold as often as mules
    # (the mules' marker value is on 5% of the ordinary rows too, 1,000 rows of weight 115), and
    # an ordinary-looking count with one level more than ordinary accounts reach among mules
    # only, 5% of the ordinary rows sitting at the level below it.
    noise = rng.normal(size=len(positive))
    shared = np.where(held | (~positive & (rng.random(len(positive)) < 0.05)), np.log1p(1.0), 0.0)
    levels = np.where(rng.random(len(positive)) < 0.05, 2.0, rng.integers(0, 2, len(positive)))
    below = np.where(held, 3.0, levels)
    perfect = positive + rng.random(len(positive)) * 0.5
    matrix = np.stack([flag, count, continuous, noise, shared, below, perfect], axis=1)
    columns = ("flag", "count", "continuous", "noise", "shared", "below", "perfect")
    found = feature_audit(matrix, positive, weight, keys, AUDIT, seed=0, columns=columns)
    table = found.table.set_index("column")
    assert table.loc["flag", "roc_auc"] == pytest.approx(0.575)
    assert table.loc["count", "roc_auc"] == pytest.approx(0.575)
    for name in ("flag", "count", "continuous"):
        assert not table.loc[name, "separates"]
    # The same marker as a 0/1 flag, as a count and as a continuous value, whatever its kind: the
    # flag also has its value, 1.
    assert table.loc["flag", "near_pure"] == "value=1,top"
    assert table.loc["count", "near_pure"] == "top"
    assert table.loc["continuous", "near_pure"] == "top"
    # Ordinary rows that hold the marker value too, and a tie of ordinary rows at the level
    # beneath it, hold the mules' region off: in neither is a region mostly mules, except the
    # one value only mules reach.
    assert table.loc["shared", "near_pure"] == "" and table.loc["noise", "near_pure"] == ""
    assert table.loc["below", "near_pure"] == "top"
    assert table.loc["perfect", "separates"] and table.loc["perfect", "roc_auc"] == 1.0
    assert not found.passed
    assert found.stopping == ("flag", "count", "continuous", "below", "perfect")
    controls = [columns.index(name) for name in ("noise", "shared")]
    clean = feature_audit(
        matrix[:, controls], positive, weight, keys, AUDIT, seed=0, columns=("noise", "shared")
    )
    assert clean.passed and clean.stopping == ()


def test_a_region_must_hold_one_percent_of_the_mules_and_be_mostly_mules() -> None:
    positive, weight, _ = _sample()
    # One mule in a region of its own: 1 of 1,200 mules, under the 1% (12) a region needs.
    few = np.zeros(len(positive))
    few[0] = 1
    assert near_pure(few, positive, weight, AUDIT) == []
    # 180 mules (15%) with 2 ordinary account-Mondays of weight 115 beside them: 230 of 410, so
    # under half of the region; with 1, 180 of 295, over half.
    marked = np.zeros(len(positive))
    marked[:180] = 1
    marked[MULES : MULES + 2] = 1
    assert near_pure(marked, positive, weight, AUDIT) == []
    marked[MULES + 1] = 0
    assert near_pure(marked, positive, weight, AUDIT) == ["value=1", "top"]
    # Weights count: counted one each, the same two ordinary rows would not outweigh the mules,
    # and the 182 rows at 1 are more than the 21 of the top 0.1% of 21,200, so only the value is
    # a region.
    marked[MULES + 1] = 1
    assert near_pure(marked, positive, np.ones(len(positive)), AUDIT) == ["value=1"]


def test_a_tie_at_the_edge_of_a_tail_is_left_out_of_it() -> None:
    positive, weight, _ = _sample()
    # Ordinary rows sit on three levels, 70% at 0, 25% at 1 and 5% at 2; 180 mules sit above
    # them at 3. The weight of the top 0.1% is 2,301, which the tie at 2 (about 115,000) passes,
    # so the edge of the top tail is 2 and the tail holds what lies above it: only the mules.
    rng = np.random.default_rng(1)
    values = rng.choice([0.0, 1.0, 2.0], size=len(positive), p=[0.70, 0.25, 0.05])
    values[:180] = 3.0
    assert near_pure(values, positive, weight, AUDIT) == ["top"]
    # With the mules at the level of the tie they share it with 1,000 ordinary rows, so no
    # region is mostly mules, the tie being no part of the tail.
    values[:180] = 2.0
    assert near_pure(values, positive, weight, AUDIT) == []
    # The bottom tail likewise: 180 mules at -1 beneath ordinary rows at 0, 1 and 2.
    values[:180] = -1.0
    assert near_pure(values, positive, weight, AUDIT) == ["bottom"]


def test_a_tail_the_sample_cannot_measure_is_reported_and_stops_nothing() -> None:
    # D1's audit in the proof of concept: 188 mules and 300 sampled non-mules, each standing for
    # 2,445,228 / 300, about 8,151 account-Mondays, more than the 0.1% tail (about 2,445). The
    # top tail's edge would be the highest sampled non-mule, and 3 mules above every one of them
    # would make a region of mules alone, though nothing tells the classes apart.
    positive = np.array([True] * 188 + [False] * 300)
    weight = np.where(positive, 1.0, 2_445_228 / 300)
    keys = [f"account:{i}" for i in range(len(positive))]
    values = 1 + np.random.default_rng(2).random(len(positive))
    values[:3] = (5.0, 6.0, 7.0)
    assert not tails_measured(positive, weight, AUDIT)
    assert near_pure(values, positive, weight, AUDIT) == []
    found = feature_audit(values[:, None], positive, weight, keys, AUDIT, seed=0, columns=("v",))
    assert found.passed and not found.table.loc[0, "tails_measured"]
    # A flag's values are still read: a flag only mules hold is a region.
    flag = np.zeros(len(positive))
    flag[:10] = 1
    assert near_pure(flag, positive, weight, AUDIT) == ["value=1"]
    # The full design's sample measures them: 1,200 mules and 20,000 non-mules of weight 115.
    measured, measured_weight, _ = _sample()
    assert tails_measured(measured, measured_weight, AUDIT)
