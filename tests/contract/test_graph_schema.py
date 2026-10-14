"""Context keys' visibility phases, and the loading job against the canonical DDL."""

from __future__ import annotations

import re

import pytest

from mule_pattern_learner.contract import graph_schema
from mule_pattern_learner.contract.graph_schema import NO_SPLIT, SPLIT_PHASE, ContextKey
from mule_pattern_learner.paths import GSQL_DIR, REPOSITORY_ROOT


def test_a_context_key_sees_a_phase_of_the_split_or_no_split() -> None:
    # Phase 0 is no split (scoring); 1 to 3 are the phases of the graph's one split.
    assert ContextKey("Account", "A1", 10, 1000).visibility_phase == NO_SPLIT == 0
    for phase in (NO_SPLIT, *SPLIT_PHASE.values()):
        assert ContextKey("Account", "A1", 10, 1000, phase).visibility_phase == phase
    for phase in (-1, 4):
        with pytest.raises(ValueError, match="no split=0, train=1, validation=2 or test=3"):
            ContextKey("Account", "A1", 10, 1000, phase)


# The loading job of PhantomLedger's export, without its comments.
LOADING = re.sub(r"/\*.*?\*/", "", (GSQL_DIR / "schema/loading.gsql").read_text(), flags=re.S)
# Each LOAD statement: its file variable, type, VALUES list and USING clause.
LOADS = re.findall(
    r"LOAD (\w+) TO (?:VERTEX|EDGE) (\w+) VALUES \((.*?)\)\s*USING (.*?);", LOADING, re.S
)


def loaded_columns(type_name: str) -> list[str]:
    """The columns the loading job maps to a type's attributes, in their order."""
    (values,) = [values for _, name, values, _ in LOADS if name == type_name]
    return re.findall(r'\$"(\w+)"', values)


def schema_types() -> dict[str, list[str]]:
    """What a load maps for each type of schema.gsql, in order.

    A vertex type's primary id and attributes; an edge type's endpoints (FROM, TO) and
    attributes, its discriminator included.
    """
    ddl = re.sub(r"/\*.*?\*/", "", (GSQL_DIR / "schema/schema.gsql").read_text(), flags=re.S)
    types = {}
    for kind, name, body in re.findall(
        r"ADD (VERTEX|DIRECTED EDGE) (\w+) \((.*?)\) WITH", ddl, re.S
    ):
        attributes = re.findall(r"(\w+) (?:STRING|BOOL|UINT|INT|DOUBLE)\b", body)
        types[name] = attributes if kind == "VERTEX" else ["FROM", "TO", *attributes]
    return types


def test_the_loading_job_loads_every_attribute_once_from_its_named_column() -> None:
    headers = {
        name: re.findall(r'"(\w+)"', columns)
        for name, columns in re.findall(r"DEFINE HEADER (\w+) =(.*?);", LOADING, re.S)
    }
    files = re.findall(r"DEFINE FILENAME (\w+);", LOADING)
    types = schema_types()
    assert len(types) == 27 and len(files) == 27
    # One mapping per vertex and edge type, each from a file of its own named after the
    # table, and none for a type the schema lacks: the reverse edges come from REVERSE_EDGE.
    assert sorted(name for _, name, _, _ in LOADS) == sorted(types)
    assert [file for file, *_ in LOADS] == files == [f"{name.lower()}_csv" for _, name, *_ in LOADS]
    for _, name, values, using in LOADS:
        # Nothing but named columns, each in the table's header, mapping every attribute
        # from its same-named column (an edge's endpoints from from_id and to_id) and no
        # attribute the schema lacks.
        assert re.fullmatch(r"[\s,]*", re.sub(r'\$"\w+"', "", values)), name
        columns = loaded_columns(name)
        assert columns == [{"FROM": "from_id", "TO": "to_id"}.get(a, a) for a in types[name]]
        header = re.findall(r'USER_DEFINED_HEADER="(\w+)"', using)
        assert len(header) == 1 and set(columns) <= set(headers[header[0]]), name
        # Each file's header line is skipped, and its columns are separated by commas.
        assert 'HEADER="true"' in using and 'SEPARATOR=","' in using, name


# The columns of the Account file the schema keeps no attribute for, which the job skips.
SKIPPED = (
    "is_mule_masked",
    "mule_label_effective_seq",
    "mule_label_effective_ts_ms",
    "mule_label_available_seq",
)


def test_account_schema_contract_matches_canonical_ddl() -> None:
    ddl = (GSQL_DIR / "schema/schema.gsql").read_text()
    block = ddl.split("ADD VERTEX Account (", 1)[1].split(") WITH", 1)[0]
    fields = re.findall(
        r"^\s*(?:PRIMARY_ID )?(\w+)\s+(?:STRING|BOOL|UINT|INT)", block, re.MULTILINE
    )
    # The schema keeps every column but SKIPPED, and declares is_mule last, after
    # mule_label_source, while the file has it sixth; the loading job maps the named
    # columns to the attribute order.
    load = graph_schema.ACCOUNT_LOAD_COLUMNS
    storage = [name for name in load if name not in (*SKIPPED, "is_mule")] + ["is_mule"]
    assert fields == storage
    assert re.search(r"is_mule INT DEFAULT 0", block)
    assert loaded_columns("Account") == storage
    # The header names every column of the file, so each one is read where it is.
    header = LOADING.split("DEFINE HEADER account_header =", 1)[1].split(";", 1)[0]
    assert re.findall(r'"(\w+)"', header) == graph_schema.ACCOUNT_LOAD_COLUMNS


def test_the_labels_reference_maps_each_loaded_column_to_its_attribute() -> None:
    # A Kafka or UI loading mapping is drawn from this table: every column of the export,
    # at its position, to the attribute of its name, or to none.
    page = (REPOSITORY_ROOT / "docs/reference/labels.md").read_text()
    rows = re.findall(r"^\| (\d+) \| `(\w+)` \| (`\w+`|none, skipped)", page, re.MULTILINE)
    load = graph_schema.ACCOUNT_LOAD_COLUMNS
    assert [(int(position), column) for position, column, _ in rows] == list(enumerate(load))
    for _, column, attribute in rows:
        assert attribute == ("none, skipped" if column in SKIPPED else f"`{column}`"), column
    # A positional mapping lists the columns in the schema's attribute order, as the
    # loading job does: the positions the reference gives.
    positions = [load.index(name) for name in loaded_columns("Account")]
    assert positions == [0, 1, 2, 3, 4, 6, 8, 12, 13, 14, 5]
    assert "`$0`, `$1`, `$2`, `$3`, `$4`, `$6`, `$8`, `$12`, `$13`, `$14`, then `$5`" in page
    assert "mule_ring_id" in page and "-1, no ring" in page
