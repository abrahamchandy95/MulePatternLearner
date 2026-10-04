"""Context scopes and the Account loading columns against the canonical DDL."""

from __future__ import annotations

import re

import pytest

from mule_pattern_learner.contract import graph_schema
from mule_pattern_learner.contract.graph_schema import context_scope
from mule_pattern_learner.paths import GSQL_DIR, REPOSITORY_ROOT


def test_strict_claim_fails_before_preparation_or_training() -> None:
    for missing in (None, ""):
        with pytest.raises(ValueError, match="frozen TigerGraph scope id"):
            context_scope(missing)
    assert context_scope("scope") == "scope"


def test_account_schema_contract_matches_canonical_ddl() -> None:
    ddl = (GSQL_DIR / "schema/schema.gsql").read_text()
    block = ddl.split("ADD VERTEX Account (", 1)[1].split(") WITH", 1)[0]
    fields = re.findall(
        r"^\s*(?:PRIMARY_ID )?(\w+)\s+(?:STRING|BOOL|UINT|INT)", block, re.MULTILINE
    )
    # The schema declares is_mule last, after mule_label_source, while the CSV/PSV input
    # has it sixth; the loading job maps the named columns to the attribute order.
    load = graph_schema.ACCOUNT_LOAD_COLUMNS
    storage = [name for name in load if name != "is_mule"] + ["is_mule"]
    assert fields == storage
    assert re.search(r"is_mule INT DEFAULT 0", block)
    loader = (GSQL_DIR / "schema/account_loading.gsql").read_text()
    columns = re.findall(r'\$"(\w+)"', loader)
    assert columns == storage
    header = loader.split("DEFINE HEADER account_header =", 1)[1].split(";", 1)[0]
    assert re.findall(r'"(\w+)"', header) == graph_schema.ACCOUNT_LOAD_COLUMNS


def test_the_labels_reference_maps_each_loaded_column_to_its_attribute() -> None:
    # A Kafka or UI loading mapping is drawn from this table: every column of the export,
    # at its position, to the attribute of its name.
    page = (REPOSITORY_ROOT / "docs/reference/labels.md").read_text()
    rows = re.findall(r"^\| (\d+) \| `(\w+)` \| `(\w+)`", page, re.MULTILINE)
    load = graph_schema.ACCOUNT_LOAD_COLUMNS
    assert [(int(position), column) for position, column, _ in rows] == list(enumerate(load))
    assert all(column == attribute for _, column, attribute in rows)
    # A positional mapping lists the columns in the schema's attribute order, as the
    # loading job does: the positions the reference gives.
    loader = (GSQL_DIR / "schema/account_loading.gsql").read_text()
    positions = [load.index(name) for name in re.findall(r'\$"(\w+)"', loader)]
    assert positions == [0, 1, 2, 3, 4, *range(6, 15), 5]
    assert "`$0`, `$1`, `$2`, `$3`, `$4`, `$6` to\n`$14`, then `$5`" in page
    assert "mule_ring_id" in page and "-1, no ring" in page
