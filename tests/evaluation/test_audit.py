"""mule audit on a loaded world through a spy executor backed by the reference graph, and what
the mule train command reads of its pass records (docs/design.md, "The pipeline" and "Gates")."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mule_pattern_learner.config import (
    DEFAULT_CONFIG,
    FULL_DESIGN_CONFIG,
    RunConfig,
    WorldConfig,
)
from mule_pattern_learner.evaluation.audit import (
    CLOCK_GAIN,
    FEATURE_AUDIT,
    GATE_RECORD,
    NO_POSITIVE_HUB,
    audit,
    audit_settings,
    train_may_run,
)
from mule_pattern_learner.evaluation.clock_invariance import COLUMNS, week_start_keys
from mule_pattern_learner.evaluation.label_checks import IDENTIFICATION_HORIZON, LABEL_CHECKS
from mule_pattern_learner.evaluation.store import AuditRecord, Store
from mule_pattern_learner.subgraph.build import read_rows
from mule_pattern_learner.subgraph.cache import Cache
from mule_pattern_learner.testing.calendar import Spy
from mule_pattern_learner.testing.reference import ReferenceGraph
from mule_pattern_learner.tigergraph.queries import (
    FETCH_ALLOWLIST,
    READ_ANSWER_KEY,
    READ_WORLD,
    VERSIONS,
)
from mule_pattern_learner.tigergraph.world import cache_key

Records = Callable[..., tuple[Path, Path]]
EVERY_WORLD = {GATE_RECORD, *LABEL_CHECKS, IDENTIFICATION_HORIZON, CLOCK_GAIN}


def test_an_audit_of_a_development_world(
    records: Records, graph: ReferenceGraph, parity: Path, data_dir: Path, config: RunConfig
) -> None:
    identity, gates = records(1001)
    spy = Spy(graph)
    done = audit(spy, config, identity_files=[identity], gate_file=gates, parity=parity)
    record = done.record
    assert (record.world, record.kind) == ("D1", "development")
    assert set(record.checks) == EVERY_WORLD | {FEATURE_AUDIT, NO_POSITIVE_HUB}
    assert record.checks[GATE_RECORD] and all(record.checks[name] for name in LABEL_CHECKS)
    assert record.checks[NO_POSITIVE_HUB]
    # Nine true mules are fewer mule clusters than the gain test's minimum of 30: it fails, which
    # on a development world does not stop training.
    assert not record.checks[CLOCK_GAIN]
    gain = Store(record.identity).table(CLOCK_GAIN).iloc[0]
    assert gain["mule_clusters"] < config.audit.clock_min_mule_clusters
    assert record.versions == VERSIONS and record.cache_key == cache_key(config.subgraph)
    assert record.settings == audit_settings(config)
    store = Store(record.identity)
    assert store.record() == record and store.labels().accounts == {"a_mule", "a_drop"}
    assert len(store.answer_key().mules) == 9
    assert not (store.folder / "fallback.parquet").exists()
    # D1 trains in every rotation it is in and never tests: no clock sample, no rows at week starts.
    assert store.clock_sample() is None
    for table in (
        "label_counts",
        "mules_never_scored",
        "identification_horizon",
        "clock_gain",
        "feature_audit",
        "positive_hubs",
        "ring_leaf_share",
        "card_spending",
    ):
        assert not store.table(table).empty or table in ("mules_never_scored", "positive_hubs")
    # Ring 11's three mules share d_hidden at the scoring times of their activity: three ring
    # nodes, none a leaf.
    assert store.table("ring_leaf_share").to_dict("records") == [
        {"ring_nodes": 3, "leaves": 0, "leaf_share": 0.0}
    ]
    # read_world first; the answer key alone outside the fetch allowlist.
    assert spy.names()[0] == READ_WORLD and spy.names().count(READ_ANSWER_KEY) == 1
    assert set(spy.names()) <= FETCH_ALLOWLIST | {READ_ANSWER_KEY}


def test_an_audit_of_a_reported_test_world_with_the_fallback(
    records: Records, graph: ReferenceGraph, parity: Path, data_dir: Path, config: RunConfig
) -> None:
    identity, gates = records(2003, passed={"S-TIM-1": False})
    fallback_config = replace(config, data=replace(config.data, scoring="fallback"))
    done = audit(
        Spy(graph), fallback_config, identity_files=[identity], gate_file=gates, parity=parity
    )
    record = done.record
    # World C tests in the rotation (A, B, C): its fallback list is built and stored.
    assert (record.world, record.kind) == ("C", "reported")
    assert set(record.checks) == EVERY_WORLD and not record.checks[GATE_RECORD]
    assert done.reasons[GATE_RECORD] == (
        "the code gate S-TIM-1 failed",
        "the world check S-TIM-1 failed",
    )
    fallback = Store(record.identity).fallback()
    assert {"a_hr1", "a_late", "a_mule"} <= set(fallback["account_id"])


def test_an_audit_of_a_test_world_stores_its_sample_at_the_week_starts_and_fetches_their_rows(
    records: Records, graph: ReferenceGraph, parity: Path, data_dir: Path, config: RunConfig
) -> None:
    identity, gates = records(2003)
    spy = Spy(graph)
    done = audit(spy, config, identity_files=[identity], gate_file=gates, parity=parity)
    record = done.record
    # World C tests in the rotation (A, B, C), under whole scoring as here: the sample is stored
    # with the start of each row's week, whatever the way the test world is scored.
    assert record.world == "C" and not (Store(record.identity).folder / "fallback.parquet").exists()
    stored = Store(record.identity).clock_sample()
    assert stored is not None and tuple(stored.columns) == COLUMNS and not stored.empty
    times = config.calendar.scoring_times(2003)
    starts = dict(zip(times, config.calendar.week_starts(), strict=True))
    assert set(stored["t_ms"]) <= set(times)
    assert stored["week_ms"].tolist() == [starts[t_ms] for t_ms in stored["t_ms"]]
    # Each scoring time is inside the week that starts at its row's week start, and does not
    # fall on it.
    assert ((stored["t_ms"] - stored["week_ms"]).between(0, 7 * 86_400_000 - 1)).all()
    # The sample's rows are the audit sample's, mules and a sample of the rest, and every row's
    # account is scorable at the start of its week in this world, whose accounts are all old.
    assert stored["positive"].any() and (~stored["positive"]).any()
    assert stored["at_week_start"].all()
    # The rows at the week starts are in the cache, which holds the rows at the scoring times
    # beside them.
    cache = Cache.of(record.identity, config.subgraph)
    keys = week_start_keys(stored)
    rows = read_rows(cache, list(zip(keys["t_ms"], keys["account_id"], strict=True)))
    assert len(rows) == len(keys) == len(stored)
    assert {row.t_ms for row in rows} == set(starts.values())
    sampled = read_rows(cache, list(zip(stored["t_ms"], stored["account_id"], strict=True)))
    assert len(sampled) == len(stored)


def _worlds(*kinds: str) -> list[tuple[WorldConfig, str]]:
    return [
        (WorldConfig(f"W{index}", 9000 + index, 1_000, kind), f"{index:032x}")
        for index, kind in enumerate(kinds)
    ]


def _write(world: tuple[WorldConfig, str], **checks: bool) -> None:
    config, identity = world
    Store(identity).write_record(
        AuditRecord(config.name, config.kind, identity, "c" * 32, "s" * 32, dict(VERSIONS), checks)
    )


@pytest.mark.parametrize(
    ("kind", "failed", "stops"),
    [
        ("development", FEATURE_AUDIT, True),
        ("development", "label.truth_join", True),
        ("development", CLOCK_GAIN, False),
        ("development", IDENTIFICATION_HORIZON, False),
        ("development", GATE_RECORD, False),
        ("reported", CLOCK_GAIN, True),
        ("reported", GATE_RECORD, True),
    ],
)
def test_train_may_run_reads_pass_or_fail_alone(
    kind: str, failed: str, stops: bool, data_dir: Path
) -> None:
    worlds = _worlds(kind, kind, kind)
    for world in worlds:
        _write(world, **{failed: world is not worlds[1]})
    expected = (f"W1: {failed}",) if stops else ()
    assert train_may_run(worlds) == expected


def test_a_world_without_a_record_or_with_another_worlds_stops_training(data_dir: Path) -> None:
    worlds = _worlds("development", "development")
    _write(worlds[0], **{FEATURE_AUDIT: True})
    assert train_may_run(worlds) == ("W1: no audit record",)
    identity = worlds[1][1]
    Store(identity).write_record(
        AuditRecord("D9", "development", identity, "c" * 32, "s" * 32, dict(VERSIONS), {})
    )
    assert train_may_run(worlds) == ("W1: the audit record is of the world D9",)


def test_the_default_audit_settings_are_the_designs() -> None:
    assert DEFAULT_CONFIG.audit.clock_gain_upper == 0.02
    assert DEFAULT_CONFIG.audit.clock_invariance_margin == 0.05
    assert DEFAULT_CONFIG.audit.clock_min_mule_clusters == 30
    assert DEFAULT_CONFIG.audit.separation_auc == 0.95
    assert FULL_DESIGN_CONFIG.audit.nonmule_sample == 200_000
    assert DEFAULT_CONFIG.audit.nonmule_sample == 300


def test_the_hash_of_the_audit_settings_holds_the_minimum_of_mule_clusters(
    config: RunConfig,
) -> None:
    """A record made under another minimum of mule clusters is not one a verdict accepts."""
    other = replace(config, audit=replace(config.audit, clock_min_mule_clusters=31))
    assert audit_settings(other) != audit_settings(config)
