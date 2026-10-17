"""The version restamper, on a query file and a registry of the test's own."""

from __future__ import annotations

from pathlib import Path

import pytest

from mule_pattern_learner.testing.gsql_rules import version_rule
from mule_pattern_learner.testing.versions import REGISTRY_FILE, restamp, stamped
from mule_pattern_learner.tigergraph.installer import definitions
from mule_pattern_learner.tigergraph.queries import REGISTRY

QUERY = """USE GRAPH Mule_Pattern_Learner

CREATE OR REPLACE QUERY read_items(UINT t_ms) FOR GRAPH Mule_Pattern_Learner SYNTAX V2 {
    PRINT "ok" AS status, "read_items-000000000000" AS version;
}
"""
REGISTRY_TEXT = 'ITEMS = Query("read_items", "queries", "read_items-000000000000")\n'


def test_the_restamper_edits_the_registry_package() -> None:
    assert REGISTRY_FILE.name == "__init__.py" and REGISTRY_FILE.parent.name == "queries"
    assert "REGISTRY" in REGISTRY_FILE.read_text() and REGISTRY


def test_a_query_and_its_registry_entry_get_the_hash_of_its_text(tmp_path: Path) -> None:
    file, registry = tmp_path / "read_items.gsql", tmp_path / "__init__.py"
    file.write_text(QUERY)
    registry.write_text(REGISTRY_TEXT)
    version = restamp("read_items", file, registry)
    assert version.startswith("read_items-") and version != "read_items-000000000000"
    assert registry.read_text() == REGISTRY_TEXT.replace("read_items-000000000000", version)
    definition = definitions(file.read_text())["read_items"]
    assert version_rule("read_items", definition, version) == []
    # The hash leaves the version out, so stamping again changes nothing.
    assert restamp("read_items", file, registry) == version
    # A change of text, not of layout or comments, is a new version.
    file.write_text(file.read_text().replace("PRINT", "/* the answer */\n    PRINT"))
    assert restamp("read_items", file, registry) == version
    file.write_text(file.read_text().replace('"ok"', '"OK"'))
    assert restamp("read_items", file, registry) != version


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (QUERY.replace("read_items-000000000000", "read_items-0"), "print its version once"),
        (QUERY.replace("version;", 'version, "read_items-000000000000" AS other;'), "once"),
        (QUERY.replace("QUERY read_items", "QUERY read_other"), "defines"),
    ],
)
def test_a_query_without_one_version_of_its_own_is_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stamped("read_items", text)


def test_a_registry_without_one_entry_of_the_query_is_refused(tmp_path: Path) -> None:
    file, registry = tmp_path / "read_items.gsql", tmp_path / "__init__.py"
    file.write_text(QUERY)
    registry.write_text("ITEMS = None\n")
    with pytest.raises(ValueError, match="one version of read_items"):
        restamp("read_items", file, registry)
    # Neither file was changed.
    assert file.read_text() == QUERY and registry.read_text() == "ITEMS = None\n"
