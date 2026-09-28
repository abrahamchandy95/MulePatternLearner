"""The reveal's draws and defaults, which its GSQL shares."""

import re

import numpy as np

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.contract.discovery import REVEAL_DEFAULTS, reveal_uniforms
from mule_pattern_learner.contract.server import REVEAL_QUERY
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph.gsql_text import definitions

REVEAL_FILE = GSQL_DIR / "queries/label_reveal.gsql"


def test_hash_mirror_is_pinned_uniform_and_stream_independent() -> None:
    # Pinned: the GSQL draw_reveal_uniforms must return exactly these values.
    assert reveal_uniforms(123456789, 42, 3) == [
        0.8522561180182994,
        0.339975114371616,
        0.2213301638706262,
    ]
    assert reveal_uniforms(1, 1042, 1) == [0.856376556379896]
    draws = np.array([reveal_uniforms(k, 42, 2) for k in range(60_000_000, 60_020_000)])
    assert ((draws > 0) & (draws < 1)).all()
    assert abs(draws.mean() - 0.5) < 0.01 and abs(draws.var() - 1 / 12) < 0.003
    assert abs(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]) < 0.03
    assert abs(np.corrcoef(draws[:-1, 0], draws[1:, 0])[0, 1]) < 0.03
    # Every intermediate product stays inside a signed 64-bit integer, as in GSQL.
    assert (2147483647 - 1) ** 2 + 1013904223 < 2**63


def test_reveal_defaults_are_the_query_defaults() -> None:
    query = definitions(REVEAL_FILE.read_text())[REVEAL_QUERY]
    header = query.split("(", 1)[1].split(") FOR GRAPH", 1)[0]
    declared = re.findall(r"\b(?:INT|DOUBLE)\s+(\w+)\s*=\s*([-\d.]+)", header)
    assert {name: float(value) for name, value in declared} == REVEAL_DEFAULTS
    assert DEFAULT_CONFIG.scope.reveal_per_split == REVEAL_DEFAULTS["budget"]
    assert DEFAULT_CONFIG.scope.reveal_salt == REVEAL_DEFAULTS["salt"]
