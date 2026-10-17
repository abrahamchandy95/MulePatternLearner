"""The rules every query's GSQL text must pass, read without a server (docs/design.md, "Leak
rules and how tests enforce them" and "Query plumbing").

Each rule takes a query's name and its definition (installer.definitions: from QUERY to its
closing brace) and returns what breaks it, one line per problem, so a passing query gives [].
The query steps call them from tests/gsql/; tests/gsql/test_text_rules.py holds every rule to
passing and failing snippets, the seven planted slips of the October 17 build among them; and
testing.mutate runs the tests against one-line mutants of every written query. Beside these
rules, gsql-lsp checks the same files against gsql/schema/, and the live compile test compiles
them on the server.

The rules read the text, so the queries keep these forms, with keywords in capitals:

- t is the parameter t_ms. Every bound at t is written against t_ms itself, in one of these
  forms, and nowhere else does t_ms appear but as `t_ms - X` (a duration or a window's
  start) or in a test of the parameters, which reads no attribute:
  - a payment in a window before t: `p.event_ts_ms >= t_ms - X AND p.event_ts_ms < t_ms`,
    X a name or an integer, on the payment vertex's alias, as conjuncts of its WHERE clause,
    or bounded by another payment of the pattern;
  - a visible link: `l.valid_from_ms < t_ms AND (l.valid_to_ms == 0 OR l.valid_to_ms >
    t_ms - lookback_ms)`, and a current one the same with `t_ms - 2592000000` (30 days);
  - an account that exists: `a.opened_ms < t_ms`, required wherever an account is reached
    through its ownership;
  - an open account, in list_accounts alone: `(a.closed_ms == 0 OR a.closed_ms >= t_ms)`;
  - a known start s: `s > t_ms - lookback_ms`, `s > q.joined_ms`, and an own payment before
    it, `p.event_ts_ms < s`; a start new over w, a window shorter than the lookback:
    `s >= t_ms - W`.
- valid_to_ms appears only in the visible and current forms, closed_ms only in the open
  form, joined_ms only after `>` in a known start, and opened_ms only in the exists form
  (read_answer_key also prints it). read_world, which has no t, reads every time into the
  world identity's sums and is held to none of the time rules.
- The leaf test of a device or IP d is the conjunct `NOT d.@leaf` of the WHERE clause of a
  SELECT that steps from d to its users (customers or payments). The one step without it
  counts the crowd itself, accumulating only into accumulators named `@crowd...`.
- A vertex is read through a pattern's alias or a vertex set; a name.field that is neither
  is a tuple's field, which a TYPEDEF declares.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
import re

from ..contract import LABEL_COLUMNS, TABLES, Kind
from ..tigergraph.installer import COMMENTS, definitions, parameter_names, text_version
from ..tigergraph.queries import (
    GRAPH_NAME,
    LIST_ACCOUNTS,
    READ_ANSWER_KEY,
    READ_LABELS,
    READ_WORLD,
)

# The graph's types, from the export contract, which tests/gsql/test_schema.py holds the
# schema to: each vertex and edge type's attributes (a reverse edge type has its forward
# type's), and the link types, ownership and the enrolments, with their reverses.
VERTEX_ATTRIBUTES = {
    table.name: frozenset(table.column_names) for table in TABLES if table.kind is Kind.VERTEX
}
EDGE_ATTRIBUTES = {
    name: frozenset(table.column_names[2:])
    for table in TABLES
    if table.kind is not Kind.VERTEX
    for name in (table.name, table.reverse)
    if name is not None
}
LINK_TYPES = frozenset(
    name
    for table in TABLES
    if table.kind is Kind.LINK
    for name in (table.name, table.reverse)
    if name is not None
)
OWNERSHIP = frozenset({"Party_Owns_Account", "Account_Owned_By_Party"})
ACCOUNT = frozenset({"Account"})
PAYMENTS = frozenset({"Payment_Transaction", "Zelle_Transfer"})
NODES = frozenset({"Device", "IP"})
USERS = frozenset({"Party", *PAYMENTS})

# Which label columns each query may name (docs/design.md, the Labels guard): read_labels
# reads pu_label and the identified mules' dates; read_answer_key the truth and every mule's
# dates. No other query names one.
LABELS_ALLOWED = {
    READ_LABELS: frozenset({"pu_label", "mule_onset_ms", "mule_last_ms"}),
    READ_ANSWER_KEY: frozenset({"is_mule", "mule_ring_id", "mule_onset_ms", "mule_last_ms"}),
}
# A query name starts with a verb.
VERBS = ("list_", "read_", "profile_")
# The queries with no t: read_world reads the whole world, read_answer_key every account.
NO_AS_OF = frozenset({READ_WORLD, READ_ANSWER_KEY})

# A string literal or a comment, matched in one pass from the left with string literals
# first, so a "#" or "//" inside a string never starts a comment and a quote inside a
# comment never starts a string.
LITERALS_AND_COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|' + COMMENTS.pattern, re.S)
# The forms of a write: a loading job, an insert, a delete, or an attribute set (an
# assignment to `alias.attribute`, plain or accumulating; accumulators are `alias.@name`).
WRITES = {
    "loading job": re.compile(r"\bLOADING\s+JOB\b", re.I),
    "insert": re.compile(r"\bINSERT\b", re.I),
    "delete": re.compile(r"\bDELETE\b", re.I),
    "update": re.compile(r"\bUPDATE\b", re.I),
    "attribute assignment": re.compile(r"\b\w+\s*\.\s*\w+\s*[+-]?=(?!=)"),
}

TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|POST-ACCUM|@@?\w+(?:\.\w+)*|\w+(?:\.@?\w+)*'
    r"|<=|>=|==|!=|->|\+=|-=|\S"
)
COMPARISONS = frozenset({"<", "<=", ">", ">=", "==", "!="})
# The tokens that end an operand: a comparison's operands lie between them.
BOUNDARIES = frozenset(
    {
        *("AND", "OR", "NOT", "WHERE", "THEN", "IF", "WHEN", "ELSE", "CASE", "END"),
        *("ACCUM", "POST-ACCUM", "DO", "WHILE", "FOREACH", "IN", "SELECT", "FROM"),
        *("PRINT", "AS", "ORDER", "BY", "LIMIT", "HAVING", "RETURN", "ASC", "DESC"),
        *("(", ")", ",", ";", "=", "+=", "-=", "->", "[", "]", "{", "}", ":", "|"),
    }
)
# The keywords that start a clause of a SELECT after its pattern.
CLAUSES = frozenset({"WHERE", "ACCUM", "POST-ACCUM", "ORDER", "LIMIT", "HAVING", "SAMPLE", "PER"})
# The attributes of times and dates of record whose every comparison the time rule reads.
WATCHED = re.compile(
    r"^(?:\w+)\.(event_ts_ms|valid_from_ms|valid_to_ms|opened_ms|closed_ms|joined_ms)$"
)
# A name other than t_ms, and a start: a link's, or one a variable or accumulator holds.
NAME = r"(?!t_ms\b)[A-Za-z_]\w*"
START = rf"(?:{NAME}\.valid_from_ms|{NAME}|{NAME}\.@\w+|@@\w+)"
# The comparisons with t, or with a date of record or a link's time, that a query may make.
ALLOWED = tuple(
    re.compile(form)
    for form in (
        rf"{NAME}\.event_ts_ms < t_ms",
        rf"{NAME}\.event_ts_ms >= t_ms - (?:{NAME}|\d+)",
        rf"{NAME}\.opened_ms < t_ms",
        rf"(?:{START}|{NAME}\.event_ts_ms) > {NAME}\.joined_ms",
        rf"{START} > t_ms - lookback_ms",
        rf"{NAME}\.valid_from_ms >= t_ms - (?:(?!lookback_ms\b){NAME}|\d+)",
        rf"(?:{NAME}\.event_ts_ms|{NAME}|{NAME}\.@\w+|@@\w+) < {NAME}\.valid_from_ms",
    )
)
# A visible or current link, and an open account, whole.
LINK_FORM = re.compile(
    r"(?<![\w.@])(\w+)\.valid_from_ms < t_ms AND \( \1\.valid_to_ms == 0 OR "
    r"\1\.valid_to_ms > t_ms - (?:lookback_ms|2592000000) \)"
)
OPEN_FORM = re.compile(r"\( (\w+)\.closed_ms == 0 OR \1\.closed_ms >= t_ms \)")


def without_comments(text: str) -> str:
    """text without its comments; string literals stay."""
    return LITERALS_AND_COMMENTS.sub(lambda m: m[0] if m[0].startswith('"') else " ", text)


def tokens(text: str) -> list[str]:
    """The tokens of a GSQL text without its comments: each string literal, name, attribute
    (`a.amount`, `a.@seen`, `@@total`) and operator one token."""
    return TOKEN.findall(without_comments(text))


def writes(text: str) -> list[str]:
    """The forms of a write a text holds, outside its comments and string literals."""
    code = LITERALS_AND_COMMENTS.sub(lambda m: '""' if m[0].startswith('"') else " ", text)
    return [form for form, pattern in WRITES.items() if pattern.search(code)]


def _body(definition: str) -> list[str]:
    """The tokens of a query's body, between its outer braces."""
    found = tokens(definition)
    if "{" not in found or "}" not in found:
        raise ValueError("A query definition has no body")
    start = found.index("{")
    end = len(found) - 1 - found[::-1].index("}")
    return found[start + 1 : end]


def _split(items: Sequence[str], separator: str) -> list[list[str]]:
    """items split at separator outside brackets."""
    parts: list[list[str]] = [[]]
    depth = 0
    for item in items:
        if item in ("(", "[", "{"):
            depth += 1
        elif item in (")", "]", "}"):
            depth -= 1
        if item == separator and depth == 0:
            parts.append([])
        else:
            parts[-1].append(item)
    return parts


def statements(definition: str) -> list[list[str]]:
    """The statements of a query's body, as tokens; a block's statements are its own."""
    return [part for part in _split(_body(definition), ";") if part]


def parameters(definition: str) -> dict[str, list[str]]:
    """Each parameter of a query and the tokens of its type, in order."""
    found = tokens(definition)
    start = found.index("(")
    depth, end = 0, start
    for index in range(start, len(found)):
        depth += {"(": 1, ")": -1}.get(found[index], 0)
        if depth == 0:
            end = index
            break
    result: dict[str, list[str]] = {}
    for part in _split_angles(found[start + 1 : end]):
        declared = part[: part.index("=")] if "=" in part else part
        if declared:
            result[declared[-1]] = declared[:-1]
    return result


def _split_angles(items: Sequence[str]) -> list[list[str]]:
    """items split at commas outside angle brackets and parentheses."""
    parts: list[list[str]] = [[]]
    depth = 0
    for item in items:
        depth += {"<": 1, "(": 1, ">": -1, ")": -1}.get(item, 0)
        if item == "," and depth == 0:
            parts.append([])
        else:
            parts[-1].append(item)
    return [part for part in parts if part]


def _vertex_parameters(definition: str) -> dict[str, frozenset[str]]:
    """The vertex types of a query's VERTEX<T> and SET<VERTEX<T>> parameters."""
    found: dict[str, frozenset[str]] = {}
    for name, kind in parameters(definition).items():
        if "VERTEX" in kind:
            at = kind.index("VERTEX")
            if kind[at + 1 : at + 2] == ["<"]:
                found[name] = frozenset({kind[at + 2]})
    return found


@dataclass(frozen=True)
class Step:
    """A vertex of a pattern: its alias, its types, and whether it is a vertex set chosen
    before rather than a type."""

    alias: str | None
    types: frozenset[str]
    chosen: bool


@dataclass(frozen=True)
class EdgeStep:
    """An edge of a pattern: its alias and its edge types."""

    alias: str | None
    types: frozenset[str]


@dataclass(frozen=True)
class Select:
    """A SELECT: its statement, each path of its pattern (vertex and edge steps alternating),
    its WHERE clause's conjuncts and its ACCUM and POST-ACCUM clauses."""

    statement: tuple[str, ...]
    paths: tuple[tuple[tuple[Step, ...], tuple[EdgeStep, ...]], ...]
    conjuncts: frozenset[str]
    accumulations: tuple[str, ...]

    @property
    def vertices(self) -> list[Step]:
        return [step for steps, _ in self.paths for step in steps]

    @property
    def edges(self) -> list[EdgeStep]:
        return [edge for _, edges in self.paths for edge in edges]

    def aliases(self) -> dict[str, frozenset[str]]:
        """Each alias of the pattern and its types."""
        found: dict[str, frozenset[str]] = {}
        for step in [*self.vertices, *self.edges]:
            if step.alias is not None:
                found[step.alias] = step.types
        return found


def _clauses(statement: Sequence[str]) -> dict[str, list[str]]:
    """A SELECT's clauses by keyword, from FROM on, split at its clause keywords outside
    brackets."""
    at = list(statement).index("FROM")
    found: dict[str, list[str]] = {"FROM": []}
    current, depth = "FROM", 0
    for item in statement[at + 1 :]:
        if item in ("(", "[", "{"):
            depth += 1
        elif item in (")", "]", "}"):
            depth -= 1
        if depth == 0 and item in CLAUSES:
            current = item
            found.setdefault(current, [])
            continue
        found[current].append(item)
    return found


def _vertex(items: Sequence[str], at: int, sets: dict[str, frozenset[str]]) -> tuple[Step, int]:
    """The vertex step at items[at], and the index after it."""
    if items[at] == "(":
        close = list(items).index(")", at)
        names = [item for item in items[at + 1 : close] if item != "|"]
        at = close + 1
    else:
        names = [items[at]]
        at += 1
    alias = None
    if at + 1 < len(items) and items[at] == ":":
        alias, at = items[at + 1], at + 2
    chosen = len(names) == 1 and names[0] in sets
    types = sets[names[0]] if chosen else frozenset(names)
    return Step(alias, types, chosen), at


def _edge(items: Sequence[str]) -> EdgeStep:
    """The edge step inside `-( ... )-`."""
    alias = None
    if len(items) >= 2 and items[-2] == ":":
        alias, items = items[-1], items[:-2]
    names = frozenset(item for item in items if re.fullmatch(r"[A-Za-z_]\w*", item))
    return EdgeStep(alias, names)


def _path(
    items: Sequence[str], sets: dict[str, frozenset[str]]
) -> tuple[tuple[Step, ...], tuple[EdgeStep, ...]]:
    """A path of a pattern: its vertex steps and the edge steps between them."""
    steps: list[Step] = []
    edges: list[EdgeStep] = []
    step, at = _vertex(items, 0, sets)
    steps.append(step)
    while at < len(items) and items[at] == "-" and at + 1 < len(items) and items[at + 1] == "(":
        depth, close = 0, at + 1
        for index in range(at + 1, len(items)):
            depth += {"(": 1, ")": -1}.get(items[index], 0)
            if depth == 0:
                close = index
                break
        edges.append(_edge(items[at + 2 : close]))
        if close + 1 >= len(items) or items[close + 1] != "-":
            raise ValueError(f"A pattern's edge does not end with )-: {' '.join(items)}")
        step, at = _vertex(items, close + 2, sets)
        steps.append(step)
    if at != len(items):
        raise ValueError(f"A pattern holds what is not a step: {' '.join(items[at:])}")
    return tuple(steps), tuple(edges)


def _assigned(statement: Sequence[str]) -> tuple[str, list[str]] | None:
    """The vertex set a statement assigns, and the tokens of its value."""
    for index in range(1, len(statement) - 1):
        if statement[index] == "=" and re.fullmatch(r"[A-Za-z_]\w*", statement[index - 1]):
            before = statement[index - 2] if index >= 2 else None
            if before in (None, "THEN", "DO", "ELSE"):
                return statement[index - 1], list(statement[index + 1 :])
    return None


def scopes(
    definition: str,
) -> list[tuple[list[str], dict[str, frozenset[str]], Select | None]]:
    """Each statement of a query with the types of the vertex sets assigned before it, and
    its SELECT, if it is one."""
    sets: dict[str, frozenset[str]] = dict(_vertex_parameters(definition))
    found: list[tuple[list[str], dict[str, frozenset[str]], Select | None]] = []
    for statement in statements(definition):
        select = None
        if "SELECT" in statement:
            clauses = _clauses(statement)
            paths = tuple(_path(path, sets) for path in _split(clauses["FROM"], ","))
            where = clauses.get("WHERE", [])
            conjuncts = frozenset(" ".join(part) for part in _split_and(where) if part)
            accumulations = tuple(clauses.get("ACCUM", []) + clauses.get("POST-ACCUM", []))
            select = Select(tuple(statement), paths, conjuncts, accumulations)
        found.append((statement, dict(sets), select))
        assigned = _assigned(statement)
        if assigned is None:
            continue
        target, value = assigned
        if select is not None and value[:1] == ["SELECT"]:
            chosen = value[1]
            sets[target] = select.aliases().get(chosen, frozenset())
        elif value[:1] == ["{"] and value[-1:] == ["}"]:
            kinds: set[str] = set()
            for item in _split(value[1:-1], ","):
                name = "".join(item)
                if name.endswith(".*"):
                    kinds.add(name[:-2])
                elif name in sets:
                    kinds |= sets[name]
            sets[target] = frozenset(kinds)
        elif value[:2] == ["to_vertex_set", "("] and value[-2:-1] and value[-2].startswith('"'):
            sets[target] = frozenset({value[-2].strip('"')})
        elif value and all(
            item in sets if index % 2 == 0 else item in ("UNION", "INTERSECT", "MINUS")
            for index, item in enumerate(value)
        ):
            sets[target] = frozenset().union(*(sets[item] for item in value[::2]))
    return found


def _split_and(items: Sequence[str]) -> list[list[str]]:
    """A condition's conjuncts: its parts joined by AND outside brackets."""
    return _split(items, "AND")


def selects(definition: str) -> list[Select]:
    """Every SELECT of a query."""
    return [select for _, _, select in scopes(definition) if select is not None]


# The rules.


def name_rule(name: str) -> list[str]:
    """A query's name starts with a verb (the owner's standing rule)."""
    return [] if name.startswith(VERBS) else [f"{name} does not start with {' or '.join(VERBS)}"]


def header_rule(name: str, text: str) -> list[str]:
    """A query's file is for the graph, and its query in syntax V2."""
    problems: list[str] = []
    if not text.startswith(f"USE GRAPH {GRAPH_NAME}\n"):
        problems.append(f"{name}'s file does not start with USE GRAPH {GRAPH_NAME}")
    if not re.search(rf"FOR GRAPH {GRAPH_NAME} SYNTAX V2 \{{", text):
        problems.append(f"{name} is not FOR GRAPH {GRAPH_NAME} SYNTAX V2")
    return problems


def write_rule(name: str, definition: str) -> list[str]:
    """A query writes nothing: MulePatternLearner writes nothing to the graph."""
    return [f"{name} writes to the graph: {form}" for form in writes(definition)]


def label_rule(name: str, definition: str) -> list[str]:
    """A query names only the label columns the Labels guard lets it read."""
    words = set(re.findall(r"\w+", without_comments(definition)))
    named = words & set(LABEL_COLUMNS)
    return [
        f"{name} names the label column {column}"
        for column in sorted(named - LABELS_ALLOWED.get(name, frozenset()))
    ]


def _atoms(items: Sequence[str]) -> list[list[str]]:
    """The operands and comparisons of a text: its tokens split at BOUNDARIES."""
    found: list[list[str]] = [[]]
    for item in items:
        if item in BOUNDARIES:
            found.append([])
        else:
            found[-1].append(item)
    return [atom for atom in found if atom]


def time_rule(name: str, definition: str) -> list[str]:
    """Every comparison with t, or of a date of record or a link's time, is one allowed form;
    t_ms appears otherwise only as `t_ms - X` or in a test of the parameters; valid_to_ms,
    closed_ms, joined_ms and opened_ms are never a value (opened_ms is printed by
    read_answer_key alone)."""
    if name == READ_WORLD:
        return []
    code = " ".join(_body(definition))
    code = LINK_FORM.sub("TRUE", code)
    if name == LIST_ACCOUNTS:
        code = OPEN_FORM.sub("TRUE", code)
    never_values = {"valid_to_ms", "closed_ms", "joined_ms"}
    if name != READ_ANSWER_KEY:
        never_values.add("opened_ms")
    problems: list[str] = []
    for atom in _atoms(code.split(" ")):
        text = " ".join(atom)
        read = {match[1] for item in atom if (match := WATCHED.match(item))}
        if not read and "t_ms" not in atom:
            continue
        if any(item in COMPARISONS for item in atom):
            if any(form.fullmatch(text) for form in ALLOWED):
                continue
            if not read or (read == {"event_ts_ms"} and "t_ms" not in atom):
                continue
            if re.fullmatch(rf"{NAME}\.valid_from_ms < t_ms", text):
                problems.append(
                    f"{name} compares {text} without the link's end test: write the visible "
                    "or the current form"
                )
            else:
                problems.append(f"{name} compares {text}, which is no allowed form at t")
            continue
        alone = [
            index
            for index, item in enumerate(atom)
            if item == "t_ms" and atom[index + 1 : index + 2] != ["-"]
        ]
        if alone:
            problems.append(f"{name} reads t_ms in {text}, other than as t_ms - X")
        for column in sorted(read & never_values):
            problems.append(f"{name} reads {column} as a value in {text}")
    return problems


def bound_rule(name: str, definition: str) -> list[str]:
    """Every SELECT bounds what it reads at t: each payment it steps to by a window before t
    or by another payment of its pattern, each link it steps along by the visible or current
    form, and each account it reaches through its ownership by `opened_ms < t_ms`. A vertex
    set chosen before was bounded where it was chosen. read_world and read_answer_key have
    no t."""
    if name in NO_AS_OF:
        return []
    problems: list[str] = []
    lower = re.compile(rf"(\w+)\.event_ts_ms >= (?:t_ms - (?:{NAME}|\d+)|(\w+)\.event_ts_ms)")
    upper = re.compile(r"(\w+)\.event_ts_ms < (?:t_ms|(\w+)\.(?:event_ts_ms|valid_from_ms))")
    link = re.compile(
        r"\( (\w+)\.valid_to_ms == 0 OR \1\.valid_to_ms > t_ms - (?:lookback_ms|2592000000) \)"
    )
    for select in selects(definition):
        where = f"{name}: {' '.join(select.statement)[:80]}"
        aliases = select.aliases()
        for step in select.vertices:
            if step.chosen or not step.types or not step.types <= PAYMENTS:
                continue
            if step.alias is None:
                problems.append(f"{where} steps to a payment without an alias")
                continue
            bounds = [lower.fullmatch(c) for c in select.conjuncts] + [
                upper.fullmatch(c) for c in select.conjuncts
            ]
            ends = {
                (match.re is upper, match[2])
                for match in bounds
                if match is not None and match[1] == step.alias
            }
            for is_upper in (False, True):
                found = [other for kind, other in ends if kind is is_upper]
                if not any(other is None or other in aliases for other in found):
                    side = "below t" if is_upper else "from a window's start"
                    problems.append(f"{where} does not bound {step.alias} {side}")
        for edge in select.edges:
            if not edge.types & LINK_TYPES:
                continue
            visible = edge.alias is not None and (
                f"{edge.alias}.valid_from_ms < t_ms" in select.conjuncts
                and any(
                    (match := link.fullmatch(c)) is not None and match[1] == edge.alias
                    for c in select.conjuncts
                )
            )
            if not visible:
                problems.append(f"{where} steps along a link that is not visible or current")
        for steps, edges in select.paths:
            for index, step in enumerate(steps):
                beside = [edges[i].types for i in (index - 1, index) if 0 <= i < len(edges)]
                owned = any(types & OWNERSHIP for types in beside)
                if step.chosen or step.types != ACCOUNT or not owned:
                    continue
                if step.alias is None or f"{step.alias}.opened_ms < t_ms" not in select.conjuncts:
                    problems.append(f"{where} reaches an account that may not exist at t")
    return problems


def projection_rule(name: str, definition: str) -> list[str]:
    """Every PRINT of a vertex set is a projection: a whole set prints every attribute, the
    labels among them."""
    problems: list[str] = []
    for statement, sets, _ in scopes(definition):
        if "PRINT" not in statement:
            continue
        printed = statement[statement.index("PRINT") + 1 :]
        for item in _split(printed, ","):
            if item and item[0] in sets and item[1:2] != ["["]:
                problems.append(f"{name} prints all of {item[0]}")
    return problems


def heap_rule(name: str, definition: str) -> list[str]:
    """Every HeapAccum declares its capacity, so no list grows without a bound."""
    problems: list[str] = []
    for statement in statements(definition):
        for index, item in enumerate(statement):
            if item != "HeapAccum":
                continue
            rest = statement[index:]
            opened = rest.index("(") if "(" in rest else None
            first: list[str] = []
            if opened is not None and ")" in rest[opened:]:
                arguments = rest[opened + 1 : rest.index(")", opened)]
                first = _split(arguments, ",")[0]
            if not first or {"ASC", "DESC"} & set(first):
                problems.append(f"{name} declares a HeapAccum without its capacity")
    return problems


def _accumulator(item: str) -> str | None:
    """The accumulator a token names, if any: `@@total`, or `@seen` of `a.@seen`."""
    if item.startswith('"'):
        return None
    match = re.search(r"@@?\w+", item)
    return match[0] if match else None


def accumulator_rule(name: str, definition: str) -> list[str]:
    """Every accumulator a query uses is declared."""
    declared: set[str] = set()
    used: set[str] = set()
    for statement in statements(definition):
        names = {found for item in statement if (found := _accumulator(item))}
        if re.fullmatch(r"\w+Accum", statement[0]):
            declared |= names
        else:
            used |= names
    return [f"{name} uses the undeclared {accumulator}" for accumulator in sorted(used - declared)]


def tuples(definition: str) -> dict[str, list[str]]:
    """The field names of each tuple type a query declares, in order."""
    found: dict[str, list[str]] = {}
    for statement in statements(definition):
        if statement[:3] == ["TYPEDEF", "TUPLE", "<"] and statement[-2] == ">":
            found[statement[-1]] = [part[-1] for part in _split_angles(statement[3:-2])]
    return found


def tuple_rule(name: str, definition: str) -> list[str]:
    """Every tuple a query builds is built with all its fields."""
    declared = tuples(definition)
    problems: list[str] = []
    for statement in statements(definition):
        for index, item in enumerate(statement[:-1]):
            if item in declared and statement[index + 1] == "(":
                depth, close = 0, index + 1
                for end in range(index + 1, len(statement)):
                    depth += {"(": 1, ")": -1}.get(statement[end], 0)
                    if depth == 0:
                        close = end
                        break
                arguments = [part for part in _split(statement[index + 2 : close], ",") if part]
                if len(arguments) != len(declared[item]):
                    problems.append(f"{name} builds {item} with {len(arguments)} fields")
    return problems


def attribute_rule(name: str, definition: str) -> list[str]:
    """Every attribute a query reads is declared on every type of its alias; a name.field
    whose name is no alias is a tuple's field."""
    fields_of_tuples = {field for names in tuples(definition).values() for field in names}
    problems: list[str] = []
    for statement, sets, select in scopes(definition):
        aliases = dict(sets)
        if select is not None:
            aliases.update(select.aliases())
        for index, item in enumerate(statement):
            match = re.fullmatch(r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)", item)
            if match is None or statement[index + 1 : index + 2] == ["("]:
                continue
            alias, attribute = match[1], match[2]
            if alias not in aliases:
                if attribute not in fields_of_tuples:
                    problems.append(f"{name} reads {item} of no alias or tuple")
                continue
            for kind in sorted(aliases[alias]):
                declared = VERTEX_ATTRIBUTES.get(kind) or EDGE_ATTRIBUTES.get(kind)
                if declared is None or attribute not in {*declared, "type"}:
                    problems.append(f"{name} reads {item}, which {kind} does not have")
    return problems


def leaf_rule(name: str, definition: str) -> list[str]:
    """Every step from a device or IP to its users excludes the leaves by `NOT d.@leaf`, but
    the step that counts the crowd, which accumulates into `@crowd...` alone."""
    problems: list[str] = []
    for select in selects(definition):
        for steps, edges in select.paths:
            for index, edge in enumerate(edges):
                node, users = steps[index], steps[index + 1]
                if not node.types or not node.types <= NODES or not users.types & USERS:
                    continue
                if node.alias is not None and f"NOT {node.alias}.@leaf" in select.conjuncts:
                    continue
                targets = [
                    _accumulator(select.accumulations[at - 1])
                    for at, item in enumerate(select.accumulations)
                    if item in ("+=", "=") and at > 0 and "@" in select.accumulations[at - 1]
                ]
                if targets and all(
                    target is not None and target.lstrip("@").startswith("crowd")
                    for target in targets
                ):
                    continue
                problems.append(
                    f"{name} steps from {node.alias} to its users without the leaf test "
                    f"(edge {sorted(edge.types)})"
                )
    return problems


def version_rule(name: str, definition: str, version: str | None) -> list[str]:
    """A written query prints the version the registry holds, which is the hash of its text."""
    if version is None:
        return [f"{name} has no version in the registry"]
    problems: list[str] = []
    if f'"{version}" AS version' not in definition:
        problems.append(f"{name} does not print {version} AS version")
    elif text_version(f"{name}-", definition, version) != version:
        problems.append(f"{name}'s text has changed since {version}: run testing.versions")
    return problems


def parameter_rule(name: str, definition: str, built: Iterable[str]) -> list[str]:
    """A query's parameters are the keys its Python builder gives, in order."""
    found, expected = parameter_names(definition), list(built)
    return [] if found == expected else [f"{name} takes {found}, not {expected}"]


def projection_fields(definition: str, vertex_set: str) -> list[str]:
    """The names a PRINT of a projection of vertex_set prints, in order."""
    for statement in statements(definition):
        if statement[:1] == ["PRINT"] and statement[1:3] == [vertex_set, "["]:
            close = statement.index("]")
            return [
                part[-1]
                for part in _split(statement[3:close], ",")
                if len(part) >= 2 and part[-2] == "AS"
            ]
    raise ValueError(f"No PRINT of a projection of {vertex_set}")


def status_fields(definition: str) -> list[str]:
    """What a query's PRINT of its "ok" status prints after the status and the version."""
    for statement in statements(definition):
        if statement[:4] == ["PRINT", '"ok"', "AS", "status"]:
            printed = [part[-1] for part in _split(statement[1:], ",") if part[-2:-1] == ["AS"]]
            return printed[2:]
    raise ValueError('No PRINT "ok" AS status')


def field_rule(what: str, printed: Sequence[str], kind: type) -> list[str]:
    """The fields a query prints are those of the dataclass its parser reads them into."""
    expected = [field.name for field in fields(kind)]
    return [] if list(printed) == expected else [f"{what} prints {list(printed)}, not {expected}"]


def common_problems(name: str, text: str) -> list[str]:
    """What breaks any rule that reads the query's file alone: every rule but the version,
    the parameters and the printed fields, which also read the Python above it."""
    defined = definitions(text)
    if list(defined) != [name]:
        return [f"The file of {name} defines {list(defined)}"]
    definition = defined[name]
    return [
        *name_rule(name),
        *header_rule(name, text),
        *write_rule(name, definition),
        *label_rule(name, definition),
        *time_rule(name, definition),
        *bound_rule(name, definition),
        *projection_rule(name, definition),
        *heap_rule(name, definition),
        *accumulator_rule(name, definition),
        *tuple_rule(name, definition),
        *attribute_rule(name, definition),
        *leaf_rule(name, definition),
    ]
