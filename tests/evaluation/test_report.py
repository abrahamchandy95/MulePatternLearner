"""mule evaluate's results (docs/design.md, "Gates", "Evaluation" and "Results"): when a result
is reported, the tables a rotation's evaluation writes, the manifest that refuses a dirty tree
for a reported result, and the event log."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG, WorldConfig
from mule_pattern_learner.data.labels import read_identified
from mule_pattern_learner.data.population import list_population
from mule_pattern_learner.evaluation.answer_key import read_answer_key
from mule_pattern_learner.evaluation.audit import CLOCK_PROBE, FEATURE_AUDIT, GATE_RECORD
from mule_pattern_learner.evaluation.label_checks import IDENTIFICATION_HORIZON, LABEL_CHECKS
from mule_pattern_learner.evaluation.rankings import HEADLINE
from mule_pattern_learner.evaluation.report import (
    EVENTS_FILE,
    MadeUnder,
    Run,
    ScoredWorld,
    Verdict,
    append_event,
    evaluate,
    manifest,
    rotation_spread,
    selected,
    verdict,
    write_manifest,
)
from mule_pattern_learner.evaluation.store import AuditRecord, Store
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.subgraph.fetch import fetch
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.testing.world import DAY_MS, T
from mule_pattern_learner.tigergraph.queries import VERSIONS

W = 7 * DAY_MS
MONDAYS = (T, T + W, T + 2 * W)
CURRENT = MadeUnder(dict(VERSIONS), "c" * 32, "s" * 32)
PASSED = {GATE_RECORD: True, **dict.fromkeys(LABEL_CHECKS, True), IDENTIFICATION_HORIZON: True}


def _world(name: str, kind: str, index: int) -> tuple[WorldConfig, str]:
    return WorldConfig(name, 5000 + index, 1_000, kind), f"{index:032x}"


ROTATION = [_world(name, "reported", i) for i, name in enumerate("ABC")]
DEVELOPMENT = [_world(name, "development", 10 + i) for i, name in enumerate(("D1", "D2", "D3"))]


def _record(
    world: tuple[WorldConfig, str], checks: Mapping[str, bool], current: MadeUnder = CURRENT
) -> None:
    config, identity = world
    Store(identity).write_record(
        AuditRecord(
            config.name,
            config.kind,
            identity,
            current.cache_key,
            current.settings,
            dict(current.versions),
            dict(checks),
        )
    )


def _all_passing() -> None:
    for world in ROTATION:
        _record(world, {**PASSED, CLOCK_PROBE: True})
    for world in DEVELOPMENT:
        _record(world, {**PASSED, CLOCK_PROBE: False, FEATURE_AUDIT: True})


def test_a_result_is_reported_when_every_gate_passed(data_dir: Path) -> None:
    _all_passing()
    found = verdict(ROTATION, DEVELOPMENT, CURRENT)
    assert found.reported and found.reasons == ()


def test_a_missing_or_failed_record_makes_results_development_only(data_dir: Path) -> None:
    _all_passing()
    _record(ROTATION[1], {**PASSED, CLOCK_PROBE: False})
    _record(DEVELOPMENT[2], {**PASSED, FEATURE_AUDIT: False})
    (Store(ROTATION[2][1]).folder / "audit.toml").unlink()
    found = verdict(ROTATION, DEVELOPMENT, CURRENT)
    assert not found.reported
    assert found.reasons == (
        "B: clock_probe did not pass",
        "C has no audit record",
        "D3: the feature audit did not pass",
    )


def test_records_of_other_versions_or_development_worlds_are_not_reported(
    data_dir: Path,
) -> None:
    _all_passing()
    other = replace(CURRENT, versions={**CURRENT.versions, "read_hubs": "read_hubs-000000000000"})
    _record(DEVELOPMENT[0], {**PASSED, FEATURE_AUDIT: True}, other)
    assert verdict(ROTATION, DEVELOPMENT, CURRENT).reasons == (
        "D1 was audited under other query versions or settings",
    )
    found = verdict(DEVELOPMENT, DEVELOPMENT[1:], CURRENT)
    assert "D2 is a development world" in found.reasons and not found.reported
    assert "no world of the development rotation is given" in verdict(ROTATION, [], CURRENT).reasons


def _manifest(reasons: tuple[str, ...], dirty: bool) -> dict[str, object]:
    return manifest(
        verdict_found=Verdict(reasons),
        identities={"C": "0" * 32},
        wall_seconds={"evaluate": 1.5},
        tigergraph_calls=0,
        cache_hits=7,
        chosen_from={"hub_threshold": "bench histograms of D1"},
        informed_by=[],
        git=("abc1234", dirty),
    )


def test_the_manifest_refuses_a_dirty_tree_for_a_reported_result() -> None:
    with pytest.raises(ValueError, match="clean tree"):
        _manifest((), dirty=True)
    found = _manifest(("C has no audit record",), dirty=True)
    assert found["dirty"] and not found["reported"] and found["commit"] == "abc1234"
    assert found["versions"] == VERSIONS
    assert found["not_reported_because"] == ["C has no audit record"]
    for name in ("gpus", "gpu_count", "cuda", "torch", "torch_geometric", "xgboost"):
        assert name in found
    assert _manifest((), dirty=False)["reported"]


def test_the_manifest_and_events_are_written(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"commit": "abc1234", "reported": False})
    assert json.loads((tmp_path / "manifest.json").read_text())["commit"] == "abc1234"
    append_event(tmp_path, "evaluate", "started")
    append_event(tmp_path, "evaluate", "stopped", ValueError("no model file"))
    lines = [json.loads(line) for line in (tmp_path / EVENTS_FILE).read_text().splitlines()]
    assert [line["event"] for line in lines] == ["started", "stopped"]
    assert lines[0]["error"] is None and lines[1]["error"] == "ValueError: no model file"


def _noise(name: str) -> float:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest()) / 2**32


def test_a_rotations_evaluation(
    graph: ReferenceGraph, parity: Path, data_dir: Path, tmp_path: Path
) -> None:
    identity = "ab" * 16
    store = Store(identity)
    store.write_answer_key(read_answer_key(Spy(graph)))
    store.write_labels(read_identified(Spy(graph)))
    populations = {
        t_ms: [row.account_id for row in list_population(graph, t_ms)] for t_ms in MONDAYS
    }
    cache = Cache(tmp_path / "cache")
    fetch(Spy(graph), cache, populations, parity=parity)
    keys = pd.DataFrame(
        [(t_ms, account, 1.0) for t_ms, accounts in populations.items() for account in accounts],
        columns=["t_ms", "account_id", "weight"],
    )
    world = ScoredWorld.of(identity, cache, keys)
    mules = {truth.account_id for truth in world.key.mules} - {"a_hr3"}

    def run(ranker: str, seed: int, score: float, strength: float) -> Run:
        logits = [
            strength * (account in mules) + _noise(f"{ranker}{seed}{t_ms}{account}")
            for t_ms, account in zip(keys["t_ms"], keys["account_id"], strict=True)
        ]
        return Run(ranker, seed, score, keys.loc[:, ["t_ms", "account_id"]].assign(logit=logits))

    runs = [
        run("gnn", 0, 0.3, 2.0),
        run("gnn", 1, 0.5, 2.0),
        run("gnn_scored_alone", 0, 0.2, 0.0),
        run("gnn_scored_alone", 1, 0.2, 0.0),
    ]
    assert selected(runs) is runs[1]
    audit = replace(DEFAULT_CONFIG.audit, replicates=30)
    folder = tmp_path / "results"
    tables = evaluate(
        world, runs, audit, 0, folder, rotation="D1D2D3", verdict_found=Verdict(("C failed",))
    )
    metrics = tables["metrics"]
    chosen = metrics[metrics["selected"] & metrics["metric"].isin(HEADLINE)]
    assert set(chosen["seed"]) == {1} and len(chosen) == len(HEADLINE)
    assert np.isfinite(chosen.loc[chosen["metric"] == "roc_auc", ["low", "high"]]).all(axis=None)
    assert metrics.loc[~metrics["selected"], "low"].isna().all()
    comparisons = {
        (str(row["ranker"]), str(row["metric"])): row
        for row in tables["comparisons"].to_dict("records")
    }
    assert {ranker for ranker, _ in comparisons} == {"gnn_scored_alone"}
    # The GNN ranks the planted mules far above the ablation, which ranks them at random.
    assert comparisons[("gnn_scored_alone", "roc_auc")]["difference"] > 0.3
    assert comparisons[("gnn_scored_alone", "roc_auc")]["shown"]
    assert len(tables["per_monday"]) == 4 * 3
    spread = {
        (str(row["ranker"]), str(row["group"]), str(row["metric"])): row
        for row in tables["spread"].to_dict("records")
    }
    assert spread[("gnn", "all", "roc_auc")]["max"] >= spread[("gnn", "all", "roc_auc")]["min"]
    assert "hidden_ring" in set(tables["groups"]["group"])
    assert "groups" in tables["mules"].columns and "selection_correlation" not in tables
    written = pd.read_parquet(folder / "metrics.parquet")
    assert written["development_only"].all() and (written["rotation"] == "D1D2D3").all()
    reported = pd.read_parquet(folder / "missed" / "mules.parquet")
    assert "a_hr3" in set(reported["account_id"]) and reported["development_only"].all()


def test_the_spread_over_rotations() -> None:
    def table(values: list[float], best: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ranker": "gnn",
                "seed": range(len(values)),
                "group": "all",
                "metric": "roc_auc",
                "value": values,
                "selected": [seed == best for seed in range(len(values))],
            }
        )

    found = rotation_spread({"ABC": table([0.6, 0.8], 1), "BCA": table([0.7, 0.9], 0)})
    rows = {str(row["of"]): row for row in found.to_dict("records")}
    # The seeds' means are 0.7 and 0.8; the selected models scored 0.8 and 0.7.
    assert rows["seed_mean"]["mean"] == pytest.approx(0.75) and rows["seed_mean"]["rotations"] == 2
    assert rows["selected"]["min"] == pytest.approx(0.7) and rows["selected"][
        "max"
    ] == pytest.approx(0.8)
