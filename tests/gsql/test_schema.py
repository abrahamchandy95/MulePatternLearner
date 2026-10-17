"""The schema, read as text, against the export contract (docs/design.md, "The export
contract" and "Schema and loading"), and the rule that no GSQL file of the repository
writes to the graph.

The schema must hold exactly one type per contract table, each with exactly its table's
columns, in order, with their graph types and every default written out; and be the five
statements the loader runs from the one after CREATE GRAPH.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from mule_pattern_learner.contract import ENDS, ID, TABLES, Kind, Table
from mule_pattern_learner.paths import GSQL_DIR, REPOSITORY_ROOT
from mule_pattern_learner.testing.gsql_rules import writes
from mule_pattern_learner.tigergraph.installer import COMMENTS
from mule_pattern_learner.tigergraph.queries import GRAPH_NAME

SCHEMA_FILE = GSQL_DIR / "schema" / "schema.gsql"
SCHEMA = SCHEMA_FILE.read_text()
# The owner's demo queries: untracked files, which no test reads.
DEMO_DIR = GSQL_DIR / "demo"
# Every attribute's default, by its graph type: 0 for times and amount, false for
# is_external, the empty string for account_type and payment_rail, and -1 for is_mule,
# pu_label and mule_ring_id (docs/design.md, "Schema and loading").
DEFAULTS = {"UINT": "0", "DOUBLE": "0", "BOOL": '"false"', "STRING": '""', "INT": "-1"}
VERTEX_OPTIONS = 'PRIMARY_ID_AS_ATTRIBUTE="true", STATS="OUTDEGREE_BY_EDGETYPE"'
JOB = "create_schema"


def _code(text: str) -> str:
    """text without its comments, each run of whitespace one space."""
    return " ".join(COMMENTS.sub(" ", text).split())


def _items(body: str) -> list[str]:
    """The comma-separated items of a type's parenthesized body, at the top level."""
    items, depth, quoted, current = list[str](), 0, False, ""
    for char in body:
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and char == "," and depth == 0:
            items.append(current.strip())
            current = ""
            continue
        current += char
    items.append(current.strip())
    return items


def _statements(text: str) -> tuple[list[str], list[str]]:
    """The schema's top-level statements, the job's body as `{...}`, and the job's own
    statements."""
    code = _code(text)
    start = code.index("{")
    end = code.rindex("}")
    job = [part.strip() for part in code[start + 1 : end].split(";")]
    assert job[-1] == "", "the job's last statement ends with a semicolon"
    outer = code[:start] + "{...}" + code[end + 1 :]
    statements = re.split(r" (?=CREATE |USE |RUN |DROP )", outer)
    return statements, job[:-1]


STATEMENTS, JOB_STATEMENTS = _statements(SCHEMA)


def _types(job: list[str]) -> dict[str, tuple[str, list[str], str]]:
    """Each type the job adds: its kind (VERTEX or EDGE), its declarations in order, and
    what follows WITH."""
    found: dict[str, tuple[str, list[str], str]] = {}
    for statement in job:
        match = re.fullmatch(r"ADD (VERTEX|DIRECTED EDGE) (\w+) \((.*)\) WITH (.*)", statement)
        assert match is not None, f"not a vertex or directed edge: {statement}"
        kind, name, body, options = match.groups()
        assert name not in found, f"{name} is added twice"
        found[name] = ("VERTEX" if kind == "VERTEX" else "EDGE", _items(body), options)
    return found


TYPES = _types(JOB_STATEMENTS)


def _expected(table: Table) -> tuple[str, list[str], str]:
    """What the schema must declare for table, from the contract and the design's
    defaults."""
    declarations: list[str] = []
    for column in table.columns:
        if column == ID:
            declarations.append("PRIMARY_ID id STRING")
        elif column == ENDS[0]:
            declarations.append(f"FROM {table.source}")
        elif column == ENDS[1]:
            declarations.append(f"TO {table.target}")
        elif table.kind is Kind.LINK and column.name == "valid_from_ms":
            declarations.append(f"DISCRIMINATOR(valid_from_ms {column.graph_type})")
        else:
            default = DEFAULTS[column.graph_type]
            declarations.append(f"{column.name} {column.graph_type} DEFAULT {default}")
    if table.kind is Kind.VERTEX:
        return "VERTEX", declarations, VERTEX_OPTIONS
    return "EDGE", declarations, f'REVERSE_EDGE="{table.reverse}"'


def test_the_schema_is_the_five_statements_the_loader_runs() -> None:
    assert STATEMENTS == [
        f"CREATE GRAPH {GRAPH_NAME} ()",
        f"USE GRAPH {GRAPH_NAME}",
        f"CREATE SCHEMA_CHANGE JOB {JOB} FOR GRAPH {GRAPH_NAME} {{...}}",
        f"RUN SCHEMA_CHANGE JOB {JOB}",
        f"DROP JOB {JOB}",
    ]


def test_the_schema_has_exactly_the_7_vertex_and_15_edge_types_of_the_contract() -> None:
    vertices = [name for name, (kind, _, _) in TYPES.items() if kind == "VERTEX"]
    edges = [name for name, (kind, _, _) in TYPES.items() if kind == "EDGE"]
    assert (len(vertices), len(edges)) == (7, 15)
    assert list(TYPES) == [table.name for table in TABLES]
    reverses = {re.findall(r'REVERSE_EDGE="(\w+)"', TYPES[name][2])[0] for name in edges}
    # 30 edge types with the reverses, every name distinct.
    assert len(set(edges) | reverses) == 30 and not reverses & set(vertices)


def test_each_type_declares_its_tables_columns_in_order_with_types_and_defaults() -> None:
    for table in TABLES:
        assert TYPES[table.name] == _expected(table), table.name


def test_the_label_and_record_attributes_are_declared_as_the_design_writes_them() -> None:
    assert TYPES["Account"][1] == [
        "PRIMARY_ID id STRING",
        'account_type STRING DEFAULT ""',
        'is_external BOOL DEFAULT "false"',
        "opened_ms UINT DEFAULT 0",
        "closed_ms UINT DEFAULT 0",
        "is_mule INT DEFAULT -1",
        "pu_label INT DEFAULT -1",
        "mule_ring_id INT DEFAULT -1",
        "mule_onset_ms UINT DEFAULT 0",
        "mule_last_ms UINT DEFAULT 0",
    ]
    assert TYPES["Party_Owns_Account"][1] == [
        "FROM Party",
        "TO Account",
        "DISCRIMINATOR(valid_from_ms UINT)",
        "valid_to_ms UINT DEFAULT 0",
    ]
    assert TYPES["Transfer_To_Account"][1] == [
        "FROM Zelle_Transfer",
        "TO Account",
        "event_ts_ms UINT DEFAULT 0",
    ]


def test_nothing_left_out_of_the_export_is_in_the_schema() -> None:
    code = _code(SCHEMA)
    for name in (
        "split",
        "discovered_ms",
        "discovery_channel",
        "mule_layer",
        "mule_flow",
        "Address",
        "Party_Has_Address",
        "Account_Uses_Device",
        "Transaction_From_Token",
        "Transaction_To_Token",
    ):
        assert not re.search(rf"\b{name}\b", code), name


def test_the_write_check_finds_every_form_of_a_write() -> None:
    assert writes("CREATE LOADING JOB load_party FOR GRAPH G { }") == ["loading job"]
    assert writes('INSERT INTO Party VALUES ("p", 1);') == ["insert"]
    assert writes("DELETE s FROM Account:s WHERE s.closed_ms > 0;") == ["delete"]
    assert writes("UPDATE s FROM Account:s SET s.opened_ms = 1;") == [
        "update",
        "attribute assignment",
    ]
    assert writes("Written = SELECT p FROM Parties:p POST-ACCUM p.split = 1;") == [
        "attribute assignment"
    ]
    assert writes("X = SELECT a FROM Account:a ACCUM a.count += 1;") == ["attribute assignment"]
    # A "#" or "//" inside a status message starts no comment, so it hides no write.
    for message in ("refused: #1", "a//b"):
        hidden = (
            f'PRINT "{message}";\n'
            "X = SELECT a FROM Account:a POST-ACCUM a.pu_label = 1;\n"
            'PRINT "done";'
        )
        assert writes(hidden) == ["attribute assignment"], message
    reads = (
        "/* delete nothing, insert nothing */ X = SELECT a FROM Account:a\n"
        "  WHERE a.opened_ms < t_ms AND a.closed_ms == 0 AND a.is_mule != 1\n"
        '    AND a.pu_label >= 0 AND a.mule_ring_id <= 3 AND a.account_type == "a.b = c"\n'
        "  ACCUM a.@seen += 1, @@total += a.amount; // update nothing\n"
        "INT page = 10;"
    )
    assert writes(reads) == []


def _gsql_files() -> list[Path]:
    return sorted(path for path in GSQL_DIR.rglob("*.gsql") if DEMO_DIR not in path.parents)


def test_no_gsql_file_writes_to_the_graph() -> None:
    files = _gsql_files()
    assert SCHEMA_FILE in files
    for path in files:
        assert writes(path.read_text()) == [], path.relative_to(GSQL_DIR)


def test_every_gsql_file_passes_gsql_lsp_against_the_schema() -> None:
    lsp = shutil.which("gsql-lsp")
    if lsp is None:
        pytest.skip("gsql-lsp is not on PATH, so no file can be checked offline")
    files = [str(path.relative_to(REPOSITORY_ROOT)) for path in _gsql_files()]
    result = subprocess.run(
        [lsp, "check", "--format", "json", *files],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
