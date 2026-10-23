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


def _sample() -> tuple[NDArray[np.bool_], list[str]]:
    """200 active true-mule account-Mondays in 100 rings, and 5,000 non-mule ones."""
    positive = np.array([True] * 200 + [False] * 5_000)
    keys = [f"ring:{i // 2}" if mule else f"account:{i}" for i, mule in enumerate(positive)]
    return positive, keys


def test_a_near_pure_flag_of_few_mules_stops_the_run() -> None:
    positive, keys = _sample()
    rng = np.random.default_rng(0)
    weight = np.ones(len(positive))
    # The planted flag is 1 on 30 of the 200 mules, 15%, and on no ordinary account: its ROC
    # AUC is 0.5 + 0.15 / 2 = 0.575, nowhere near 0.95, but the region where it is 1 holds 15%
    # of the mules (at least 1%) and nothing else.
    flag = np.zeros(len(positive))
    flag[:30] = 1
    noise = rng.normal(size=len(positive))
    perfect = positive.astype(np.float64) + rng.random(len(positive)) * 0.5
    matrix = np.stack([flag, noise, perfect], axis=1)
    found = feature_audit(
        matrix, positive, weight, keys, AUDIT, seed=0, columns=("flag", "noise", "perfect")
    )
    table = found.table.set_index("column")
    assert table.loc["flag", "roc_auc"] == pytest.approx(0.575)
    assert not table.loc["flag", "separates"] and table.loc["flag", "near_pure"] == "value=1,top"
    assert table.loc["noise", "no_separation"] and table.loc["noise", "near_pure"] == ""
    assert not table.loc["noise", "separates"]
    assert table.loc["perfect", "separates"] and table.loc["perfect", "roc_auc"] == 1.0
    assert not found.passed and found.stopping == ("flag", "perfect")
    clean = feature_audit(matrix[:, 1:2], positive, weight, keys, AUDIT, seed=0, columns=("noise",))
    assert clean.passed and clean.stopping == ()


def test_a_region_must_hold_one_percent_of_the_mules_and_be_mostly_mules() -> None:
    positive = np.array([True] * 200 + [False] * 5_000)
    weight = np.ones(len(positive))
    # One mule in a region of its own: 0.5% of the mules, under the 1% a region needs.
    few = np.zeros(len(positive))
    few[0] = 1
    assert "value=1" not in near_pure(few, positive, weight, AUDIT)
    # 30 mules among 40 ordinary accounts: 15% of the mules, but under half of the region.
    mixed = np.zeros(len(positive))
    mixed[:30] = 1
    mixed[200:240] = 1
    assert near_pure(mixed, positive, weight, AUDIT) == []
    # Weights count: the 40 ordinary accounts sampled at 1 in 10 stand for 400.
    sampled = np.where(positive, 1.0, 10.0)
    mixed[200:204], mixed[204:240] = 1, 0
    assert near_pure(mixed, positive, sampled, AUDIT) == []
    assert near_pure(mixed, positive, weight, AUDIT) == ["value=1", "top"]


def test_the_tails_hold_their_share_of_the_weight_with_ties_inside() -> None:
    positive = np.array([True] * 20 + [False] * 1_980)
    weight = np.ones(len(positive))
    # The top 0.1% of 2,000 account-Mondays is 2 of them; here 20 tie at the top, all mules.
    values = np.arange(len(positive), dtype=np.float64)
    values[:20] = 10_000
    assert near_pure(values, positive, weight, AUDIT) == ["top"]
    assert near_pure(-values, positive, weight, AUDIT) == ["bottom"]
