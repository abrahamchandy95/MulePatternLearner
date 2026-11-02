"""mark_crowds' text against its Python reader: it takes the parameters its builder gives, its
registered version is the hash of its text, it declares the settings' thresholds and lookback,
and each set it prints has the fields of a Crowd, each read from the set's own vertex or
accumulators."""

from __future__ import annotations

import pytest

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.testing.gsql_rules import (
    field_rule,
    parameter_rule,
    projection_fields,
    projection_items,
    source_rule,
    version_rule,
)
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import (
    MARK_CROWDS,
    QUERY_FILES,
    REGISTRY,
    VERSIONS,
    mark_crowds,
)
from mule_pattern_learner.tigergraph.queries.mark_crowds import Crowd
from mule_pattern_learner.tigergraph.queries.rows import LOOKBACK_MS

DEFINITION = definitions((GSQL_DIR / QUERY_FILES[MARK_CROWDS]).read_text())[MARK_CROWDS]
SUBGRAPH = DEFAULT_CONFIG.subgraph
T_MS = 1_711_929_600_000  # 2024-04-01 00:00 UTC


def test_it_takes_the_parameters_its_builder_gives_and_its_version_is_its_hash() -> None:
    assert REGISTRY[MARK_CROWDS].params is mark_crowds.params
    assert parameter_rule(MARK_CROWDS, DEFINITION, mark_crowds.params(T_MS)) == []
    assert version_rule(MARK_CROWDS, DEFINITION, VERSIONS[MARK_CROWDS]) == []


def test_its_thresholds_and_lookback_are_the_settings() -> None:
    assert f"INT CROWD_THRESHOLD = {SUBGRAPH.crowd_threshold};" in DEFINITION
    assert f"INT HUB_THRESHOLD = {SUBGRAPH.hub_threshold};" in DEFINITION
    assert f"UINT LOOKBACK_MS = {LOOKBACK_MS};" in DEFINITION


@pytest.mark.parametrize("vertex_set", ["device_crowds", "ip_crowds"])
def test_each_printed_set_has_the_fields_of_a_crowd(vertex_set: str) -> None:
    assert REGISTRY[MARK_CROWDS].rows == (Crowd,)
    printed = projection_fields(DEFINITION, vertex_set)
    assert field_rule(f"{MARK_CROWDS} {vertex_set}", printed, Crowd) == []
    items = projection_items(DEFINITION, vertex_set)
    assert source_rule(f"{MARK_CROWDS} {vertex_set}", items) == []
