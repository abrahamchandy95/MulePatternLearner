"""The GSQL files, read as text: the schema, the loading job, and what the queries read.

No server compiles them here, so these tests hold the files to the design and to the Python
that calls them: the types and attributes of the design's graph table, the 27 tables of
PhantomLedger loaded by name, every attribute a query reads declared on its type, labels
read only where the design allows, held-out customers seen only when the phase allows,
nothing from after the as-of time, and parameters and printed fields equal to the Python
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


PAYMENTS = {"Payment_Transaction", "Zelle_Transfer"}
LINKS = {*ASSOCIATIONS, *(reverse for _, _, reverse in ASSOCIATIONS.values())}
# The edges between a payment and its sender, and its receiver, both ways.
SENDER_EDGES = {
    name
    for edge in ("Transaction_From_Account", "Transfer_From_Account")
    for name in (edge, EVENTS[edge][2])
}
RECEIVER_EDGES = {
    name
    for edge in ("Transaction_To_Account", "Transfer_To_Account")
    for name in (edge, EVENTS[edge][2])
}
# The test that hides the held-out customers (split 1) unless held_out_visible.
VISIBLE = "(held_out_visible OR {}.split != 1)"


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


EDGE_STEP = re.compile(r"-\((\(?[\w>| ]+\)?)\s*:\s*(\w+)\)-")
type Step = tuple[str, set[str]]


def _chain(select: str, sets: dict[str, set[str]]) -> tuple[list[Step], list[Step]]:
    """A SELECT's pattern in order: its vertex steps, the first its source, and the edge
    steps between them, each (alias, types)."""
    pattern = re.split(r"\b(?:WHERE|ACCUM|POST-ACCUM|ORDER BY|LIMIT)\b", select)[0]
    parts = EDGE_STEP.split(pattern)
    vertices: list[Step] = []
    for text in parts[::3]:
        found = re.fullmatch(r"\s*(\([\w| ]+\)|\w+)\s*:\s*(\w+)\s*", text)
        assert found is not None, f"unknown pattern step {text!r}"
        vertices.append((found[2], _types(found[1], sets)))
    edges = [
        (alias, _types(source, sets))
        for source, alias in zip(parts[1::3], parts[2::3], strict=True)
    ]
    return vertices, edges


def _scopes(definition: str) -> list[tuple[str, dict[str, set[str]]]]:
    """Each statement of a query, with the types of the vertex set variables before it.

    A variable takes its types from to_vertex_set, a SELECT's chosen alias, a UNION or
    another variable.
    """
    sets: dict[str, set[str]] = {}
    scoped: list[tuple[str, dict[str, set[str]]]] = []
    for statement in _statements(definition):
        scoped.append((statement, dict(sets)))
        assigned = re.match(r"(\w+) = (.*)", statement)
        if assigned is None:
            continue
        target, value = assigned[1], assigned[2]
        select = re.match(r"SELECT (\w+) FROM (.*)", value)
        made = re.match(r'to_vertex_set\(.*, "(\w+)"\)$', value)
        if select is not None:
            sets[target] = dict(_chain(select[2], sets)[0])[select[1]]
        elif made is not None:
            sets[target] = {made[1]}
        elif re.fullmatch(r"\w+( UNION \w+)*", value) and value.split()[0] in sets:
            sets[target] = set().union(*(sets[name] for name in value.split(" UNION ")))
    return scoped


def _reads(definition: str) -> list[tuple[str, str, set[str]]]:
    """Every attribute read of a query: (alias, attribute, the alias's types).

    A name.field whose name is no alias is a tuple's field, and the field must be one a
    TYPEDEF declares.
    """
    tuple_fields = {field for fields in _tuples(definition).values() for field in fields}
    reads: list[tuple[str, str, set[str]]] = []
    for statement, sets in _scopes(definition):
        aliases = dict(sets)
        select = re.search(r"\bSELECT (\w+) FROM (.*)", statement)
        if select is not None:
            vertices, edges = _chain(select[2], sets)
            aliases.update(vertices + edges)
        for name, attribute in re.findall(r"(?<![@\w])(\w+)\.(\w+)\b(?!\s*\()", statement):
            if name in aliases:
                reads.append((name, attribute, aliases[name]))
            else:
                assert attribute in tuple_fields, f"{name}.{attribute} reads an unknown alias"
    return reads


def _selects(definition: str) -> list[tuple[str, list[Step], list[Step], set[str]]]:
    """Each SELECT of a query: its statement, its pattern's vertex and edge steps, and the
    conjuncts of its WHERE clause."""
    found: list[tuple[str, list[Step], list[Step], set[str]]] = []
    for statement, sets in _scopes(definition):
        select = re.search(r"\bSELECT \w+ FROM (.*)", statement)
        if select is None:
            continue
        where = re.search(r" WHERE (.*?)(?= (?:POST-ACCUM|ACCUM|ORDER BY|LIMIT)\b|$)", statement)
        conjuncts = set(_conjuncts(where[1])) if where else set[str]()
        vertices, edges = _chain(select[1], sets)
        found.append((statement, vertices, edges, conjuncts))
    return found


def test_every_attribute_a_query_reads_is_declared_on_its_type() -> None:
    for name, definition in QUERIES.items():
        reads = _reads(definition)
        assert reads, name
        for alias, attribute, types in reads:
            for kind in types:
                declared = {*TYPES[kind], "type"}
                assert attribute in declared, f"{name}: {alias}.{attribute} is not on {kind}"


def _vertex_sets(definition: str) -> set[str]:
    """The names of a query's vertex set variables, wherever they are assigned."""
    names: set[str] = set()
    for statement in _statements(definition):
        found = re.search(
            r"\b(\w+) = (?:SELECT\b|to_vertex_set\(|(\w+)(?: UNION \w+)*$)", statement
        )
        if found is not None and (found[2] is None or found[2] in names):
            names.add(found[1])
    return names


def _printed_sets(definition: str) -> list[tuple[str, str | None]]:
    """Each vertex set a PRINT prints: its name and its projection, or None when the whole
    set is printed, with every attribute. Printing its size is no print of it."""
    names = _vertex_sets(definition)
    printed: list[tuple[str, str | None]] = []
    for statement in _statements(definition):
        start = re.search(r"\bPRINT\b", statement)
        if start is None:
            continue
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', statement[start.start() :])
        for name, projection in re.findall(r"\b(\w+)\[([^\[\]]*)\]", text):
            if name in names:
                printed.append((name, projection))
        rest = re.sub(r"\b\w+\[[^\[\]]*\]|\b\w+\.size\(\)", " ", text)
        printed.extend((name, None) for name in re.findall(r"\w+", rest) if name in names)
    return printed


def test_only_the_supervision_and_answer_key_queries_read_labels() -> None:
    """A query reads a label by its name, or by printing a whole vertex set, which carries
    every attribute. So a query prints a vertex set only through a projection, and the
    projections, like the rest of the text, name only the labels the query may read."""
    labels = {READ_SUPERVISION: SUPERVISION, READ_ANSWER_KEY: SUPERVISION | ANSWER_KEY}
    # What each of the two prints at least.
    printed_labels = {READ_SUPERVISION: SUPERVISION, READ_ANSWER_KEY: ANSWER_KEY}
    for name, definition in QUERIES.items():
        printed = _printed_sets(definition)
        whole = [vertex_set for vertex_set, projection in printed if projection is None]
        assert not whole, f"{name} prints all of {whole}"
        words = set(re.findall(r"\w+", COMMENTS.sub("", definition)))
        projected = {
            word for _, projection in printed for word in re.findall(r"\w+", projection or "")
        }
        forbidden = (SUPERVISION | ANSWER_KEY) - labels.get(name, set())
        assert not words & forbidden and not projected & forbidden, name
        assert printed_labels.get(name, set()) <= projected, name


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


def test_the_population_and_supervision_see_only_the_phases_customers() -> None:
    customer = "({0}.split == 0 OR (held_out_visible AND {0}.split == 1))"
    for name in (LIST_POPULATION, READ_SUPERVISION):
        [(_, [(alias, types)], _, conjuncts)] = _selects(QUERIES[name])
        assert types == {"Account"} and customer.format(alias) in conjuncts, name
        if name == LIST_POPULATION:
            open_at_t = {
                f"{alias}.opened_ms <= as_of_ms",
                f"({alias}.closed_ms == 0 OR {alias}.closed_ms > as_of_ms)",
            }
            assert open_at_t <= conjuncts


def test_the_subgraph_query_reaches_only_visible_customers() -> None:
    """Unless held_out_visible, no account or party of split 1 reaches a subgraph or a feature.

    The seeds are checked, and a pattern's source is a set of visible nodes. Every other
    account a pattern binds is tested with VISIBLE, and so is every party but an account's
    owner (write_split gives an account its owner's split). A payment whose pattern binds
    only one of its accounts checks the other: a receiver through @blocked, which marks the
    payments to hidden accounts, and a sender through @passed, which only an inflow from a
    visible sender gets. Only the selections that set @blocked bind hidden accounts.
    """
    definition = QUERIES[FETCH_SUBGRAPHS]
    seeds = (
        r"\w+ = SELECT (\w+) FROM Seeds:\1 WHERE \1\.opened_ms > as_of_ms "
        r"OR \(NOT held_out_visible AND \1\.split == 1\) "
    )
    assert any(re.match(seeds, statement) for statement in _statements(definition))
    for statement, vertices, edges, conjuncts in _selects(definition):
        blocker = statement.startswith("IF NOT held_out_visible THEN ") and statement.endswith(
            ".@blocked += TRUE"
        )
        for index, (alias, types) in enumerate(vertices[1:], 1):
            where = f"{statement[:50]}...: {alias}"
            owner = vertices[index - 1][1] == {"Account"} and edges[index - 1][1] == {
                "Account_Owned_By_Party"
            }
            if types == {"Account"} or (types == {"Party"} and not owner):
                hidden = blocker and f"{alias}.split == 1" in conjuncts
                assert hidden or VISIBLE.format(alias) in conjuncts, where
            elif types <= PAYMENTS:
                bound = [
                    edges[edge][1]
                    for edge, vertex in ((index - 1, index - 1), (index, index + 1))
                    if vertex < len(vertices) and vertices[vertex][1] == {"Account"}
                ]
                if len(bound) == 1:
                    assert bound[0] <= SENDER_EDGES | RECEIVER_EDGES, where
                    other = "NOT {}.@blocked" if bound[0] <= SENDER_EDGES else "{}.@passed"
                    assert other.format(alias) in conjuncts, where


def test_the_subgraph_query_sees_only_what_existed_before_its_time() -> None:
    """Payments with window_start <= event_ts_ms < t, and links with valid_from_ms <= t.

    A payment may instead be bounded by another payment of its pattern (a pair's earlier
    payments, a receiver's later outflows), and from below by since_30d, inside the window.
    A pattern's source is a set selected so before.
    """
    definition = QUERIES[FETCH_SUBGRAPHS]
    windows = {
        "window_start = as_of_ms - lookback_days * day_ms",
        "since_30d = as_of_ms - 30 * day_ms",
    }
    assert windows <= set(_statements(definition))
    for statement, vertices, edges, conjuncts in _selects(definition):
        payments = [alias for alias, types in vertices if types <= PAYMENTS]
        for alias, types in vertices[1:]:
            if types <= PAYMENTS:
                others = [f"{other}.event_ts_ms" for other in payments if other != alias]
                upper = {f"{alias}.event_ts_ms < {bound}" for bound in ["as_of_ms", *others]}
                lower = {
                    f"{alias}.event_ts_ms >= {bound}"
                    for bound in ["window_start", "since_30d", *others]
                }
                assert upper & conjuncts and lower & conjuncts, f"{statement[:50]}...: {alias}"
        for alias, types in edges:
            if types <= LINKS:
                assert f"{alias}.valid_from_ms <= as_of_ms" in conjuncts, f"{statement[:50]}..."


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
