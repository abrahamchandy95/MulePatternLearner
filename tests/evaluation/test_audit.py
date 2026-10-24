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
    CLOCK_PROBE,
    FEATURE_AUDIT,
    GATE_RECORD,
    NO_POSITIVE_HUB,
    audit,
    audit_settings,
    train_may_run,
)
from mule_pattern_learner.evaluation.label_checks import IDENTIFICATION_HORIZON, LABEL_CHECKS
from mule_pattern_learner.evaluation.store import AuditRecord, Store
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
EVERY_WORLD = {GATE_RECORD, *LABEL_CHECKS, IDENTIFICATION_HORIZON, CLOCK_PROBE}


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
    # Nine true mules are too few for the probe's interval: it fails, which on a development
    # world does not stop training.
    assert not record.checks[CLOCK_PROBE]
    assert record.versions == VERSIONS and record.cache_key == cache_key(config.subgraph)
    assert record.settings == audit_settings(config)
    store = Store(record.identity)
    assert store.record() == record and store.labels().accounts == {"a_mule", "a_drop"}
    assert len(store.answer_key().mules) == 9
    assert not (store.folder / "fallback.parquet").exists()
    for table in (
        "label_counts",
        "mules_never_scored",
        "identification_horizon",
        "clock_probe",
        "feature_audit",
        "positive_hubs",
        "ring_leaf_share",
        "card_spending",
    ):
        assert not store.table(table).empty or table in ("mules_never_scored", "positive_hubs")
    # Ring 11's three mules share d_hidden on its three active Mondays: three ring nodes, none
    # a leaf.
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
        ("development", CLOCK_PROBE, False),
        ("development", IDENTIFICATION_HORIZON, False),
        ("development", GATE_RECORD, False),
        ("reported", CLOCK_PROBE, True),
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
    assert DEFAULT_CONFIG.audit.clock_probe_upper == 0.55
    assert DEFAULT_CONFIG.audit.separation_auc == 0.95
    assert FULL_DESIGN_CONFIG.audit.nonmule_sample == 200_000
    assert DEFAULT_CONFIG.audit.nonmule_sample == 300
