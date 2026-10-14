"""The GSQL files, read as text: the schema, the loading job, and what the queries read.

No server compiles them here, so these tests hold the files to the design and to the Python
that calls them: the types and attributes of the design's graph table, the 27 tables of
PhantomLedger loaded by name, every attribute a query reads declared on its type, labels
read only where the design allows, and parameters and printed fields equal to the Python
request builders and dataclasses.
"""

from __future__ import annotations

from dataclasses import fields
import re

from mule_pattern_learner.config import DEFAULT_CONFIG
from mule_pattern_learner.evaluation.answer_key import AnswerKeyRow
from mule_pattern_learner.paths import GSQL_DIR
from mule_pattern_learner.tigergraph.installer import (
    COMMENTS,
    parameter_names,
    repository_queries,
    text_version,
)
from mule_pattern_learner.tigergraph.queries import (
    FETCH_SUBGRAPHS,
    GRAPH_NAME,
    LIST_POPULATION,
    NODES,
    QUERY_FILES,
    READ_ANSWER_KEY,
    READ_SUPERVISION,
    RELATIONS,
    SUBGRAPH_VERSION,
    WRITE_SPLIT,
    AccountNode,
    IdentityNode,
    LinkEdge,
    PartyNode,
    PaymentEdge,
    Phase,
    PopulationAccount,
    SentEdge,
    SubgraphRequest,
    Supervision,
    population_params,
    split_params,
    subgraph_params,
    supervision_params,
)

SCHEMA = (GSQL_DIR / "schema" / "schema.gsql").read_text()
LOADING = (GSQL_DIR / "schema" / "loading.gsql").read_text()
QUERIES = repository_queries()

# The design's graph table, in the schema's order.
VERTICES = {
    "Party": ["id", "joined_ms", "split"],
    "Account": [
        "id",
        "account_type",
        "is_external",
        "opened_ms",
        "closed_ms",
        "split",
        "discovered_ms",
        "discovery_channel",
        "mule_onset_ms",
        "is_mule",
        "mule_ring_id",
        "mule_layer",
        "mule_flow",
        "mule_last_ms",
    ],
    "Token": ["id"],
    "Device": ["id"],
    "IP": ["id"],
    "Address": ["id"],
    "Payment_Transaction": ["id", "amount", "payment_rail", "event_ts_ms"],
    "Zelle_Transfer": ["id", "amount", "event_ts_ms"],
}
SUPERVISION = {"discovered_ms", "discovery_channel", "mule_onset_ms"}
ANSWER_KEY = {"is_mule", "mule_ring_id", "mule_layer", "mule_flow", "mule_last_ms"}
# The edge types of the graph before the redesign (commit 4df5b09): name, ends, reverse.
ASSOCIATIONS = {
    "Party_Owns_Account": ("Party", "Account", "Account_Owned_By_Party"),
    "Party_Uses_Token": ("Party", "Token", "Token_Used_By_Party"),
    "Token_Bound_To_Account": ("Token", "Account", "Account_Bound_From_Token"),
    "Party_Uses_Device": ("Party", "Device", "Device_Used_By_Party"),
    "Account_Uses_Device": ("Account", "Device", "Device_Used_By_Account"),
    "Party_Uses_IP": ("Party", "IP", "IP_Used_By_Party"),
    "Party_Has_Address": ("Party", "Address", "Address_Used_By_Party"),
}
EVENTS = {
    "Transfer_From_Account": ("Zelle_Transfer", "Account", "Account_Sent_Zelle_Transfer"),
    "Transfer_To_Account": ("Zelle_Transfer", "Account", "Account_Received_Zelle_Transfer"),
    "Transfer_From_Token": ("Zelle_Transfer", "Token", "Token_Sent_Transfer"),
    "Transfer_To_Token": ("Zelle_Transfer", "Token", "Token_Received_Transfer"),
    "Transfer_Used_Device": ("Zelle_Transfer", "Device", "Device_Used_By_Transfer"),
    "Transfer_Used_IP": ("Zelle_Transfer", "IP", "IP_Used_By_Transfer"),
    "Transaction_From_Account": (
        "Payment_Transaction",
        "Account",
        "Account_Initiated_Transaction",
    ),
    "Transaction_To_Account": ("Payment_Transaction", "Account", "Account_Received_Transaction"),
    "Transaction_From_Token": ("Payment_Transaction", "Token", "Token_Sent_Transaction"),
    "Transaction_To_Token": ("Payment_Transaction", "Token", "Token_Received_Transaction"),
    "Transaction_Used_Device": ("Payment_Transaction", "Device", "Device_Used_By_Transaction"),
    "Transaction_Used_IP": ("Payment_Transaction", "IP", "IP_Observed_In_Transaction"),
}
# PhantomLedger's table headers (use case mule-temporal), in its column order.
TABLES = {
    "Party": ["id", "joined_ms"],
    "Account": [
        "id",
        "account_type",
        "is_external",
        "opened_ms",
        "closed_ms",
        "is_mule",
        "mule_ring_id",
        "mule_layer",
        "mule_flow",
        "mule_onset_ms",
        "mule_last_ms",
        "discovered_ms",
        "discovery_channel",
    ],
    "Token": ["id"],
    "Device": ["id"],
    "IP": ["id"],
    "Address": ["id"],
    "Payment_Transaction": ["id", "amount", "payment_rail", "event_ts_ms"],
    "Zelle_Transfer": ["id", "amount", "event_ts_ms"],
    **{name: ["from_id", "to_id", "valid_from_ms", "valid_to_ms"] for name in ASSOCIATIONS},
    **{name: ["from_id", "to_id", "event_ts_ms"] for name in EVENTS},
}


def _declared(body: str) -> dict[str, str]:
    """Attribute name to its declaration, from a type's parenthesized body."""
    found: dict[str, str] = {}
    for line in (part.strip() for part in body.split("\n")):
        line = line.rstrip(",")
        if line.startswith("PRIMARY_ID "):
            found[line.split()[1]] = line
        elif line.startswith("DISCRIMINATOR("):
            found[line[len("DISCRIMINATOR(") :].split()[0]] = line
        elif line and not line.startswith("FROM "):
            found[line.split()[0]] = line
    return found


def _schema() -> tuple[dict[str, dict[str, str]], dict[str, tuple[str, str, str]]]:
    """The schema's types, each with its attributes' declarations, and its edge types."""
    types: dict[str, dict[str, str]] = {}
    edges: dict[str, tuple[str, str, str]] = {}
    for name, body in re.findall(r"ADD VERTEX (\w+) \((.*?)\) WITH", SCHEMA, re.S):
        types[name] = _declared(body)
    for name, body, reverse in re.findall(
        r'ADD DIRECTED EDGE (\w+) \((.*?)\) WITH REVERSE_EDGE="(\w+)";', SCHEMA, re.S
    ):
        ends = re.search(r"FROM (\w+), TO (\w+)", body)
        assert ends is not None
        edges[name] = (ends[1], ends[2], reverse)
        types[name] = types[reverse] = _declared(body)
    return types, edges


TYPES, EDGES = _schema()


def test_the_schema_holds_the_designs_types_and_attributes() -> None:
    assert {name: list(TYPES[name]) for name in VERTICES} == VERTICES
    assert EDGES == {**ASSOCIATIONS, **EVENTS}
    for name in ASSOCIATIONS:
        assert TYPES[name] == {
            "valid_from_ms": "DISCRIMINATOR(valid_from_ms UINT)",
            "valid_to_ms": "valid_to_ms UINT DEFAULT 0",
        }
    for name in EVENTS:
        assert TYPES[name] == {"event_ts_ms": "event_ts_ms UINT"}
    assert TYPES["Party"]["split"] == TYPES["Account"]["split"] == "split INT DEFAULT -1"
    assert TYPES["Account"]["is_mule"] == "is_mule INT DEFAULT -1"
    assert f"CREATE GRAPH {GRAPH_NAME} ()" in SCHEMA


def test_the_loading_job_loads_every_phantomledger_table_by_its_column_names() -> None:
    headers = {
        name: re.findall(r'"(\w+)"', columns)
        for name, columns in re.findall(r"DEFINE HEADER (\w+) = (.*?);", LOADING, re.S)
    }
    loads = re.findall(
        r'LOAD (\w+) TO (VERTEX|EDGE) (\w+) VALUES \((.*?)\)\s*USING HEADER="true", '
        r'SEPARATOR=",", USER_DEFINED_HEADER="(\w+)";',
        LOADING,
        re.S,
    )
    assert "CREATE LOADING JOB load_graph" in LOADING
    assert sorted(load[2] for load in loads) == sorted(TABLES)
    for file, kind, table, values, header in loads:
        assert f"DEFINE FILENAME {file};" in LOADING and file == f"{table.lower()}_csv"
        assert headers[header] == TABLES[table]
        loaded = [value.strip() for value in values.split(",")]
        attributes = list(TYPES[table])
        if kind == "EDGE":
            attributes = ["from_id", "to_id", *attributes]
        # Every attribute from its same-named column, but split, which the job leaves.
        expected = [f'$"{name}"' if name != "split" else "_" for name in attributes]
        assert loaded == expected, table
        assert {name for name in attributes if name != "split"} == set(TABLES[table]), table


def _tuples(definition: str) -> dict[str, list[str]]:
    """The field names of each tuple type a query declares."""
    return {
        name: [field.split()[-1] for field in body.split(",")]
        for body, name in re.findall(r"TYPEDEF TUPLE\s*<([^<>]*)>\s*(\w+);", definition)
    }


def _statements(definition: str) -> list[str]:
    body = COMMENTS.sub("", definition)
    return [" ".join(part.split()) for part in body[body.index("{") + 1 :].split(";")]


def _types(source: str, sets: dict[str, set[str]]) -> set[str]:
    """The types of a pattern step's source: `Type`, `(T1|T2)` or a vertex set variable."""
    names = {name.strip().rstrip(">") for name in source.strip("()").split("|")}
    if len(names) == 1 and next(iter(names)) in sets:
        return sets[next(iter(names))]
    assert names <= set(TYPES), f"unknown types {names}"
    return names


def _aliases(statement: str, sets: dict[str, set[str]]) -> dict[str, set[str]]:
    """The aliases a SELECT's pattern declares, with their types."""
    pattern = re.split(r"\b(?:WHERE|ACCUM|POST-ACCUM|ORDER BY|LIMIT)\b", statement)[0]
    aliases: dict[str, set[str]] = {}
    for source, alias in re.findall(r"-\((\(?[\w>| ]+\)?)\s*:\s*(\w+)\)-", pattern):
        aliases[alias] = _types(source, sets)
    vertices = re.sub(r"-\((\(?[\w>| ]+\)?)\s*:\s*(\w+)\)-", " ", pattern)
    for source, alias in re.findall(r"(\([\w| ]+\)|\w+)\s*:\s*(\w+)", vertices):
        aliases[alias] = _types(source, sets)
    return aliases


def _reads(definition: str) -> list[tuple[str, str, set[str]]]:
    """Every attribute read of a query: (alias, attribute, the alias's types).

    Vertex set variables take their types from to_vertex_set, a SELECT's chosen alias, a
    UNION or another variable. A name.field whose name is no alias is a tuple's field,
    and the field must be one a TYPEDEF declares.
    """
    tuple_fields = {field for fields in _tuples(definition).values() for field in fields}
    sets: dict[str, set[str]] = {}
    reads: list[tuple[str, str, set[str]]] = []
    for statement in _statements(definition):
        aliases = dict(sets)
        assigned = re.match(r"(\w+) = (.*)", statement)
        select = re.search(r"\bSELECT (\w+) FROM (.*)", statement)
        if select is not None:
            aliases.update(_aliases(select[2], sets))
        for name, attribute in re.findall(r"(?<![@\w])(\w+)\.(\w+)\b(?!\s*\()", statement):
            if name in aliases:
                reads.append((name, attribute, aliases[name]))
            else:
                assert attribute in tuple_fields, f"{name}.{attribute} reads an unknown alias"
        if assigned is None:
            continue
        target, value = assigned[1], assigned[2]
        made = re.match(r'to_vertex_set\(.*, "(\w+)"\)$', value)
        if select is not None:
            sets[target] = aliases[select[1]]
        elif made is not None:
            sets[target] = {made[1]}
        elif re.fullmatch(r"\w+( UNION \w+)*", value) and value.split()[0] in sets:
            sets[target] = set().union(*(sets[name] for name in value.split(" UNION ")))
    return reads


def test_every_attribute_a_query_reads_is_declared_on_its_type() -> None:
    for name, definition in QUERIES.items():
        reads = _reads(definition)
        assert reads, name
        for alias, attribute, types in reads:
            for kind in types:
                declared = {*TYPES[kind], "type"}
                assert attribute in declared, f"{name}: {alias}.{attribute} is not on {kind}"


def test_only_the_supervision_and_answer_key_queries_read_labels() -> None:
    for name, definition in QUERIES.items():
        words = set(re.findall(r"\w+", COMMENTS.sub("", definition)))
        if name == READ_ANSWER_KEY:
            assert words & ANSWER_KEY
        elif name == READ_SUPERVISION:
            assert words & SUPERVISION and not words & ANSWER_KEY
        else:
            assert not words & (SUPERVISION | ANSWER_KEY), name


def _conjuncts(condition: str) -> list[str]:
    """The parts of a condition joined by AND outside parentheses."""
    parts, depth, current = list[str](), 0, ""
    for token in re.findall(r"\(|\)|[^()]+", condition):
        depth += token.count("(") - token.count(")")
        if depth == 0 and token not in "()":
            pieces = token.split(" AND ")
            current += pieces[0]
            for piece in pieces[1:]:
                parts.append(current.strip())
                current = piece
        else:
            current += token
    return [*parts, current.strip()]


def test_read_supervision_always_applies_its_cutoff() -> None:
    definition = QUERIES[READ_SUPERVISION]
    assert parameter_names(definition) == ["cutoff_ms", "held_out_visible"]
    assert not re.search(r"cutoff_ms\s*=", definition.split("{")[0])
    statements = _statements(definition)
    selects = [statement for statement in statements if " SELECT " in statement]
    assert len(selects) == 1
    found = re.match(r"(\w+) = SELECT (\w+) FROM Account:\2 WHERE (.*)", selects[0])
    assert found is not None
    result, alias, condition = found.groups()
    assert {f"{alias}.discovered_ms > 0", f"{alias}.discovered_ms < cutoff_ms"} <= set(
        _conjuncts(condition)
    )
    # Supervision leaves the query only as that selection's rows.
    for statement in statements:
        if statement not in selects and set(re.findall(r"\w+", statement)) & SUPERVISION:
            assert statement.startswith(f"PRINT {result}[")


def test_the_subgraph_query_prints_the_version_of_its_text() -> None:
    definition = QUERIES[FETCH_SUBGRAPHS]
    assert f'"{SUBGRAPH_VERSION}" AS version' in definition
    assert text_version("subgraphs-", definition, SUBGRAPH_VERSION) == SUBGRAPH_VERSION


def test_query_parameters_are_the_request_builders_keys() -> None:
    request = SubgraphRequest(("A1",), 1_720_000_000_000, Phase.TRAINING)
    page = ["after_id", "batch_size"]
    expected = {
        FETCH_SUBGRAPHS: list(subgraph_params(request, DEFAULT_CONFIG.subgraph)),
        LIST_POPULATION: [*population_params(1, Phase.TEST), *page],
        READ_SUPERVISION: list(supervision_params(1, Phase.TEST)),
        WRITE_SPLIT: list(split_params({"P1": 0})),
        READ_ANSWER_KEY: page,
    }
    assert {name: parameter_names(text) for name, text in QUERIES.items()} == expected


def _printed(definition: str, result: str) -> list[str]:
    projection = re.search(rf"PRINT {result}\[(.*?)\] AS accounts;", definition, re.S)
    assert projection is not None
    return re.findall(r"AS (\w+)", projection[1])


def test_printed_fields_are_the_parsers_fields() -> None:
    subgraphs = QUERIES[FETCH_SUBGRAPHS]
    rows = {
        "AccountRow": AccountNode,
        "PartyRow": PartyNode,
        "IdentityRow": IdentityNode,
        "PaymentRow": PaymentEdge,
        "LinkRow": LinkEdge,
        "SentRow": SentEdge,
    }
    for row, kind in rows.items():
        assert _tuples(subgraphs)[row] == [field.name for field in fields(kind)], row
    per_seed = subgraphs[subgraphs.rindex("PRINT @@seed_list") :]
    assert re.findall(r"AS (\w+)", per_seed) == ["seed", *NODES, *RELATIONS]
    pages = {
        LIST_POPULATION: ("Page", PopulationAccount),
        READ_SUPERVISION: ("Discovered", Supervision),
        READ_ANSWER_KEY: ("Page", AnswerKeyRow),
    }
    for name, (result, kind) in pages.items():
        assert _printed(QUERIES[name], result) == [field.name for field in fields(kind)], name
    assert (
        '"ok" AS status, Parties.size() AS parties, @@accounts AS accounts'
        in (QUERIES[WRITE_SPLIT])
    )


def test_every_query_file_is_for_the_graph_in_syntax_v2() -> None:
    for name, relative in QUERY_FILES.items():
        text = (GSQL_DIR / relative).read_text()
        assert text.startswith(f"USE GRAPH {GRAPH_NAME}\n"), relative
        assert f"FOR GRAPH {GRAPH_NAME} SYNTAX V2 {{" in QUERIES[name], relative
