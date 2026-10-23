"""The missed-mule report (docs/design.md, "The missed-mule report"): who is missed and who is
weak on planted ranks, and what the report holds of each, read from the reference world."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mule_pattern_learner.evaluation.answer_key import ACTIVE, read_answer_key
from mule_pattern_learner.evaluation.feature_audit import COLUMNS
from mule_pattern_learner.evaluation.groups import group_rows
from mule_pattern_learner.evaluation.missed import MISSED, WEAK, missed_mules, write_report
from mule_pattern_learner.evaluation.rankings import Ranking, evaluation_frame, mule_table
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import T


def test_missed_and_weak_mules_on_planted_ranks() -> None:
    mules = pd.DataFrame(
        {
            "account_id": ["x1", "x2", "x3", "x4", "x5"],
            "active_mondays": [2, 3, 2, 0, 1],
            "ever_at_5pct": [0.0, 0.6, 1.0, 0.0, 1.0],
            "median_rank": [0.3, 0.2, 0.01, np.nan, 0.05],
        }
    )
    found = missed_mules(mules, 0.05)
    # x1 is never in the top 5%; x2 is, on some Monday, but its median rank is outside it; x3
    # ranks high; x4 has no active Monday; x5's median is exactly at 5%, which is inside.
    assert found[["account_id", "outcome"]].values.tolist() == [["x1", MISSED], ["x2", WEAK]]


def test_the_report_of_a_missed_mule(graph: ReferenceGraph, parity: Path, tmp_path: Path) -> None:
    accounts = ["a_hr1", "a_hr2", "a_hr3", *(f"a_o{i:02d}" for i in range(12))]
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, {T: accounts}, parity=parity)
    # a_hr3 scores under every ordinary account; its ring-mates above them all. Each ordinary
    # account stands for 100, so the top 5% (60.15 of 1,203) holds both ring-mates whole.
    logits = [5.0, 4.0, -5.0, *(float(i) / 10 for i in range(12))]
    scores = pd.DataFrame({"t_ms": [T] * len(accounts), "account_id": accounts, "logit": logits})
    keys = scores.loc[:, ["t_ms", "account_id"]].assign(weight=[1.0] * 3 + [100.0] * 12)
    frame = evaluation_frame(scores, keys, read_answer_key(Spy(graph)), frozenset())
    ranking = Ranking.of(frame, (frame["status"] == ACTIVE).to_numpy())
    reported = missed_mules(mule_table(frame, ranking, frozenset()), 0.05)
    assert reported["account_id"].tolist() == ["a_hr3"]
    groups = group_rows(frame, frozenset(), frozenset(), {})
    write_report(tmp_path, cache, "w", frame, reported, groups, {"rotation": "D1D2D3"})
    folder = tmp_path / "missed"
    mules = pd.read_parquet(folder / "mules.parquet")
    assert mules["groups"].tolist() == ["hidden_ring,hop_beyond"]
    assert mules["rotation"].tolist() == ["D1D2D3"]
    rows = pd.read_parquet(folder / "rows.parquet")
    assert len(rows) == len(COLUMNS) and set(rows["account_id"]) == {"a_hr3"}
    decoded = dict(zip(rows["column"], rows["value"], strict=True))
    # a_hr3 was paid 200.00 by a_hr2 4 days 12 hours before T.
    assert decoded["account.in_amount_7d"] == 20_000 and decoded["account.in_7d"] == 1
    payments = pd.read_parquet(folder / "subgraph_payments.parquet")
    assert set(payments["mule_id"]) == {"a_hr3"} and "a_hr2" in set(payments["sender_id"])
    devices = pd.read_parquet(folder / "subgraph_devices.parquet")
    assert "d_hidden" in set(devices["node_id"])
    mates = pd.read_parquet(folder / "ring_mates.parquet")
    assert sorted(mates["mate_id"]) == ["a_hr1", "a_hr2"]
    assert (mates["rank"] < 0.1).all() and not mates["identified"].any()
