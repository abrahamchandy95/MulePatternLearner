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
    X `lookback_ms` or an integer, on the payment vertex's alias, as conjuncts of its WHERE
    clause; or bounded by other payments of its pattern, `p.event_ts_ms >= x.event_ts_ms`
    or `p.event_ts_ms < x.event_ts_ms`, so long as each chain of such bounds ends at a
    window's start below and at t, or at the start of a link that began before t, above;
  - a visible link: `l.valid_from_ms < t_ms AND (l.valid_to_ms == 0 OR l.valid_to_ms >
    t_ms - lookback_ms)`, and a current one the same with `t_ms - 2592000000` (30 days);
  - an account that exists: `a.opened_ms < t_ms`, required wherever an account is reached
    through its ownership, and on every other account a query with t reaches but as a
    payment's sender or receiver, which exists before the payment by the loader's check;
  - an open account, in list_accounts alone: `(a.closed_ms == 0 OR a.closed_ms >= t_ms)`;
  - a known start s: `s > t_ms - lookback_ms`, `s > q.joined_ms`, and an own payment before
    it, `p.event_ts_ms < s`; a start new over w, a window shorter than the lookback:
    `s >= t_ms - W`, W an integer.
- Every window and duration lies inside the lookback (the Lookback guard): an integer taken
  from t_ms is at most the lookback, 7,776,000,000 ms. lookback_ms, crowd_threshold and
  hub_threshold are each a parameter or declared once as `UINT name = value`, with the
  setting's value, and never set again.
- valid_to_ms appears only in the visible and current forms, closed_ms only in the open
  form, joined_ms only after `>` in a known start, and opened_ms only in the exists form
  (read_answer_key also prints it). read_world, which has no t, reads every time into the
  world identity's sums and is held to none of the time rules.
- A query with t refuses account parameters that do not exist at t, in this form, with
  Scored the parameter or a set assigned `{parameter}`:
  `Existing = SELECT s FROM Scored:s WHERE s.opened_ms < t_ms;`, then
  `IF Existing.size() != Scored.size() THEN PRINT "<status>" AS status ...; RETURN; END;`.
- No query reads the clock: no date or time function, and no remainder but a test of the
  parameter itself, `t_ms % N != R` or `t_ms % N == R` with N and R integers.
- No query reads around the rules: no built-in that reads a vertex's edges or attributes by
  name (outdegree, neighbors, neighborAttribute, edgeAttribute, getAttr, evaluate) but
  read_hubs' all-time prefilter of the account hub test, `a.outdegree("T") + ... >
  hub_threshold` over an account's payment edges, which changes the cost and never an
  answer; no library algorithm (`tg_...`) and no other query called; and no step to a Token
  or along its links but in read_world, since no token is read.
- The leaf test of a device or IP d is the conjunct `NOT d.@leaf` of the WHERE clause of a
  SELECT that steps between d and its users (customers or payments), in either order. Two
  steps go without it: one from users chosen before to the devices or IPs of their type that
  ends there (a customer's own devices, the devices seen on chosen payments), and the crowd
  count, which selects d and reads or writes no accumulator but d's own named `@crowd...`.
  The leaf flag is set in one form alone, before it is tested and after a crowd count filled
  the accumulator it reads: `Leaves = SELECT d FROM Nodes:d POST-ACCUM d.@leaf +=
  d.@crowd_customers.size() > crowd_threshold` (or `> 8`). A `@crowd...` accumulator is
  read only through its size.
- A vertex is read through a pattern's alias or a vertex set; a name.field that is neither
  is a tuple's field, which a TYPEDEF declares.
- Every step of a pattern names a vertex set or the schema's types: no wildcard (`_` or
  `ANY`), no edge without a type and no edge of several hops (`.` or `*`), so that every
  vertex a query reaches is a step the rules see. A vertex set is assigned, with or without
  a declared type (`All (Account) = ...`), by a SELECT, from a vertex parameter or vertex
  accumulator (`{@@kept}`, declared VERTEX<T>), as a whole type (`{Account.*}`), by
  `to_vertex_set(..., "T")` or by a set expression of UNION, INTERSECT and MINUS, with or
  without parentheses. A set of a whole type or from to_vertex_set was bounded nowhere, so a
  step on it is bounded like a step on the type itself.
- A PRINT prints a vertex set only as a projection, `Set[Set.id AS id, ...]`. A printed name
  that is not a parameter of another type than vertices, a local scalar or a tuple is taken
  for a vertex set, so a set the rules cannot follow is never printed whole.
- A field of a projection or of the "ok" status row reads what it is named for:
  `Set.name AS name`, `Set.@name AS name` or `@@name AS name`, and an id `Set.id AS
  account_id` or `Set.id AS node_id`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
import re

from ..config import DEFAULT_CONFIG
from ..contract import LABEL_COLUMNS, TABLES, Kind
from ..tigergraph.installer import COMMENTS, definitions, parameter_names, text_version
from ..tigergraph.queries import (
    GRAPH_NAME,
    LIST_ACCOUNTS,
    QUERY_FILES,
    READ_ANSWER_KEY,
    READ_HUBS,
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
# The roles that join a payment to its sender or receiver, with their reverses: the account
# at their end was open before the payment, by the loader's check.
PAYMENT_ACCOUNTS = frozenset(
    name
    for table in TABLES
    if table.kind is Kind.ROLE and table.target == "Account"
    for name in (table.name, table.reverse)
    if name is not None
)
# An account's payment edges, whose all-time degrees read_hubs' prefilter sums.
ACCOUNT_PAYMENTS = frozenset(
    table.reverse
    for table in TABLES
    if table.kind is Kind.ROLE and table.target == "Account" and table.reverse is not None
)
PAYMENTS = frozenset({"Payment_Transaction", "Zelle_Transfer"})
NODES = frozenset({"Device", "IP"})
USERS = frozenset({"Party", *PAYMENTS})
# A token and its links, which no query reads but read_world (the No dates of record guard).
TOKEN = "Token"
TOKEN_EDGES = frozenset(
    name
    for table in TABLES
    if table.kind is not Kind.VERTEX and TOKEN in (table.source, table.target)
    for name in (table.name, table.reverse)
    if name is not None
)

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

# The lookback, the longest any window or duration reaches back from t, and the thresholds
# of the leaf and hub tests: the settings' values, which a query declares by these names.
DAY_MS = 86_400_000
SUBGRAPH = DEFAULT_CONFIG.subgraph
LOOKBACK_MS = SUBGRAPH.lookback_days * DAY_MS
SETTINGS = {
    "lookback_ms": LOOKBACK_MS,
    "crowd_threshold": SUBGRAPH.crowd_threshold,
    "hub_threshold": SUBGRAPH.hub_threshold,
}
# The built-ins that read a vertex's edges or attributes around the rules: all-time degrees
# and neighbours, and attributes named by a string; in lower case, as GSQL reads them.
STRUCTURE_CALLS = frozenset(
    {"outdegree", "neighbors", "neighborattribute", "edgeattribute", "getattr", "evaluate"}
)
# The built-ins that read the clock or a calendar, beside every datetime_ function.
CLOCK_CALLS = frozenset(
    {
        *("now", "year", "month", "day", "hour", "minute", "second", "weekday", "week"),
        *("dayofweek", "dayofmonth", "dayofyear", "epoch_to_datetime", "to_datetime"),
    }
)
# The types of a query's local scalars, beside its tuple types.
SCALAR_TYPES = frozenset({"INT", "UINT", "FLOAT", "DOUBLE", "STRING", "BOOL", "DATETIME"})
# The words a PRINT may hold that name no vertex set.
PRINT_WORDS = frozenset({"TRUE", "FALSE", "NULL", "AND", "OR", "NOT", "IN", "IS"})
# The printed names that read a vertex's id; every other printed name reads an attribute or
# an accumulator of its own name.
ID_NAMES = frozenset({"account_id", "node_id"})

# A string literal or a comment, matched in one pass from the left with string literals
# first, so a "#" or "//" inside a string never starts a comment and a quote inside a
# comment never starts a string.
LITERALS_AND_COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|' + COMMENTS.pattern, re.S)
# The forms of a write: a loading job, an insert, a delete, an UPDATE statement, or an
# attribute set (an assignment to `alias.attribute`, plain or accumulating, or a vertex's or
# edge's setAttr; accumulators are `alias.@name`). A ListAccum's update method changes no
# graph data, so only the statement `UPDATE alias FROM` counts. gsql-lsp reads method names
# in any case, so setAttr is matched in any case too.
WRITES = {
    "loading job": re.compile(r"\bLOADING\s+JOB\b", re.I),
    "insert": re.compile(r"\bINSERT\b", re.I),
    "delete": re.compile(r"\bDELETE\b", re.I),
    "update": re.compile(r"(?<![.\w@])UPDATE\s+\w+\s+FROM\b", re.I),
    "attribute assignment": re.compile(r"\b\w+\s*\.\s*\w+\s*[+-]?=(?!=)|\.\s*(?i:setAttr)\s*\("),
}

TOKENS = re.compile(
    r'"(?:\\.|[^"\\])*"|POST-ACCUM|@@?\w+(?:\.\w+)*|\w+(?:\.@?\w+)*'
    r"|<=|>=|==|!=|->|\+=|-=|\S"
)
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
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
# The tokens a statement may start with before the vertex set it assigns.
STARTS = (None, "THEN", "DO", "ELSE")
SET_OPERATORS = frozenset({"UNION", "INTERSECT", "MINUS"})
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
        rf"{NAME}\.event_ts_ms >= t_ms - (?:lookback_ms|\d+)",
        rf"{NAME}\.opened_ms < t_ms",
        rf"(?:{START}|{NAME}\.event_ts_ms) > {NAME}\.joined_ms",
        rf"{START} > t_ms - lookback_ms",
        rf"{NAME}\.valid_from_ms >= t_ms - \d+",
        rf"(?:{NAME}\.event_ts_ms|{NAME}|{NAME}\.@\w+|@@\w+) < {NAME}\.valid_from_ms",
    )
)
# A visible or current link, and an open account, whole.
LINK_FORM = re.compile(
    r"(?<![\w.@])(\w+)\.valid_from_ms < t_ms AND \( \1\.valid_to_ms == 0 OR "
    r"\1\.valid_to_ms > t_ms - (?:lookback_ms|2592000000) \)"
)
OPEN_FORM = re.compile(r"\( (\w+)\.closed_ms == 0 OR \1\.closed_ms >= t_ms \)")
# The one statement that sets the leaf flag, and the test of the parameter t that takes a
# remainder.
LEAF_FORM = re.compile(
    r"\w+ = SELECT (\w+) FROM \w+ : \1 (?:ACCUM|POST-ACCUM) \1\.@leaf \+?= "
    rf"\1\.@crowd\w*\.size \( \) > (?:crowd_threshold|{SUBGRAPH.crowd_threshold})"
)
PARAMETER_REMAINDER = re.compile(r"(?:^| )t_ms % \d+ (?:!=|==) \d+(?= |$)")


def without_comments(text: str) -> str:
    """text without its comments; string literals stay."""
    return LITERALS_AND_COMMENTS.sub(lambda m: m[0] if m[0].startswith('"') else " ", text)


def tokens(text: str) -> list[str]:
    """The tokens of a GSQL text without its comments: each string literal, name, attribute
    (`a.amount`, `a.@seen`, `@@total`) and operator one token."""
    return TOKENS.findall(without_comments(text))


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


def _vertex_accumulators(definition: str) -> dict[str, frozenset[str]]:
    """The vertex type of each global accumulator declared to hold VERTEX<T>, such as
    `SetAccum<VERTEX<Account>> @@kept`; one declared to hold an untyped VERTEX has none."""
    found: dict[str, frozenset[str]] = {}
    for statement in statements(definition):
        if not re.fullmatch(r"\w+Accum", statement[0]) or "VERTEX" not in statement:
            continue
        at = statement.index("VERTEX")
        types = (
            frozenset({statement[at + 2]}) if statement[at + 1 : at + 2] == ["<"] else frozenset()
        )
        for item in statement:
            if re.fullmatch(r"@@\w+", item):
                found[item] = types
    return found


@dataclass(frozen=True)
class Step:
    """A vertex of a pattern: its alias, its types, and whether it is a vertex set chosen
    before rather than a type. A set is chosen when its members came from a SELECT, a vertex
    parameter or a vertex accumulator, or from chosen sets alone; a set of a whole type
    (`{Payment_Transaction.*}`) or from to_vertex_set is not, and is bounded like the type
    itself."""

    alias: str | None
    types: frozenset[str]
    chosen: bool


@dataclass(frozen=True)
class EdgeStep:
    """An edge of a pattern: its alias, its edge types, and whether it spans several hops
    (a chain `A>.B>` or a repetition `*`), which hides the vertices between them."""

    alias: str | None
    types: frozenset[str]
    hops: bool = False


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

    @property
    def selected(self) -> str | None:
        """The alias it selects."""
        rest = list(self.statement[self.statement.index("SELECT") + 1 :])
        if rest[:1] == ["DISTINCT"]:
            rest = rest[1:]
        return rest[0] if rest else None

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


def _vertex(
    items: Sequence[str], at: int, sets: dict[str, frozenset[str]], chosen: frozenset[str]
) -> tuple[Step, int]:
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
    is_set = len(names) == 1 and names[0] in sets
    types = sets[names[0]] if is_set else frozenset(names)
    return Step(alias, types, is_set and names[0] in chosen), at


def _edge(items: Sequence[str]) -> EdgeStep:
    """The edge step inside `-( ... )-`."""
    alias = None
    if len(items) >= 2 and items[-2] == ":":
        alias, items = items[-1], items[:-2]
    names = frozenset(item for item in items if IDENTIFIER.fullmatch(item))
    hops = any(item == "*" or "." in item for item in items)
    return EdgeStep(alias, names, hops)


def _path(
    items: Sequence[str], sets: dict[str, frozenset[str]], chosen: frozenset[str]
) -> tuple[tuple[Step, ...], tuple[EdgeStep, ...]]:
    """A path of a pattern: its vertex steps and the edge steps between them."""
    steps: list[Step] = []
    edges: list[EdgeStep] = []
    step, at = _vertex(items, 0, sets, chosen)
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
        step, at = _vertex(items, close + 2, sets, chosen)
        steps.append(step)
    if at != len(items):
        raise ValueError(f"A pattern holds what is not a step: {' '.join(items[at:])}")
    return tuple(steps), tuple(edges)


def _assigned(statement: Sequence[str]) -> tuple[str, frozenset[str], list[str]] | None:
    """The vertex set a statement assigns, the types it declares for it (`All (Account) =`;
    none for ANY or no declaration), and the tokens of its value."""
    for index, item in enumerate(statement):
        if item != "=" or index == 0:
            continue
        at, declared = index - 1, frozenset[str]()
        if statement[at] == ")" and "(" in statement[:at]:
            opened = max(i for i in range(at) if statement[i] == "(")
            names = [name for name in statement[opened + 1 : at] if name != "|"]
            if not names or not all(IDENTIFIER.fullmatch(name) for name in names):
                continue
            at, declared = opened - 1, frozenset(names) - {"ANY"}
        if at < 0 or not IDENTIFIER.fullmatch(statement[at]):
            continue
        if (statement[at - 1] if at >= 1 else None) in STARTS:
            return statement[at], declared, list(statement[index + 1 :])
    return None


def _set_expression(
    value: Sequence[str], sets: dict[str, frozenset[str]], chosen: set[str]
) -> tuple[frozenset[str], bool] | None:
    """The types of a set expression (named sets joined by UNION, INTERSECT and MINUS, with
    or without parentheses), and whether every set in it was chosen; None for another
    value."""
    names = [item for item in value if item not in ("(", ")")]
    if not names or value.count("(") != value.count(")"):
        return None
    if (
        not all(
            item in sets if index % 2 == 0 else item in SET_OPERATORS
            for index, item in enumerate(names)
        )
        or len(names) % 2 == 0
    ):
        return None
    kept = names[::2]
    return frozenset().union(*(sets[item] for item in kept)), all(item in chosen for item in kept)


def scopes(
    definition: str,
) -> list[tuple[list[str], dict[str, frozenset[str]], Select | None]]:
    """Each statement of a query with the types of the vertex sets assigned before it, and
    its SELECT, if it is one. A set whose members were chosen before (see Step) is marked
    chosen in its SELECT's steps."""
    sets: dict[str, frozenset[str]] = dict(_vertex_parameters(definition))
    chosen = set(sets)
    held = _vertex_accumulators(definition)
    found: list[tuple[list[str], dict[str, frozenset[str]], Select | None]] = []
    for statement in statements(definition):
        select = None
        if "SELECT" in statement:
            clauses = _clauses(statement)
            paths = tuple(
                _path(path, sets, frozenset(chosen)) for path in _split(clauses["FROM"], ",")
            )
            where = clauses.get("WHERE", [])
            conjuncts = frozenset(" ".join(part) for part in _split_and(where) if part)
            accumulations = tuple(clauses.get("ACCUM", []) + clauses.get("POST-ACCUM", []))
            select = Select(tuple(statement), paths, conjuncts, accumulations)
        found.append((statement, dict(sets), select))
        assigned = _assigned(statement)
        if assigned is None:
            continue
        target, declared, value = assigned
        kinds: frozenset[str] | None = None
        picked = False
        if select is not None and value[:1] == ["SELECT"]:
            kinds, picked = select.aliases().get(value[1], frozenset()), True
        elif value[:1] == ["{"] and value[-1:] == ["}"]:
            found_kinds: set[str] = set()
            whole = False
            for item in _split(value[1:-1], ","):
                name = "".join(item)
                if name.endswith(".*"):
                    found_kinds.add(name[:-2])
                    whole = True
                elif name in sets:
                    found_kinds |= sets[name]
                    whole |= name not in chosen
                elif name in held:
                    found_kinds |= held[name]
                else:
                    whole = True
            kinds, picked = frozenset(found_kinds), not whole
        elif value[:2] == ["to_vertex_set", "("] and value[-2:-1] and value[-2].startswith('"'):
            kinds = frozenset({value[-2].strip('"')})
        elif (expression := _set_expression(value, sets, chosen)) is not None:
            kinds, picked = expression
        # A value the rules cannot follow leaves the name no set they know.
        chosen.discard(target)
        sets.pop(target, None)
        if kinds is not None:
            sets[target] = kinds or declared
            if picked:
                chosen.add(target)
    return found


def _split_and(items: Sequence[str]) -> list[list[str]]:
    """A condition's conjuncts: its parts joined by AND outside brackets."""
    return _split(items, "AND")


def selects(definition: str) -> list[Select]:
    """Every SELECT of a query."""
    return [select for _, _, select in scopes(definition) if select is not None]


def _calls(items: Sequence[str]) -> list[tuple[int, str]]:
    """Each call in a text, by the index of its name's token and its name in lower case: a
    function's, or a method's, the last part of `a.outdegree`."""
    return [
        (index, item.rsplit(".", 1)[-1].lower())
        for index, item in enumerate(items[:-1])
        if items[index + 1] == "(" and re.fullmatch(r"[A-Za-z_][\w.@]*", item)
    ]


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


def window_rule(name: str, definition: str) -> list[str]:
    """Every window and duration lies inside the lookback (the Lookback guard): no integer
    taken from t_ms exceeds it, a link start new over a window is newer than its start, and
    lookback_ms and the thresholds the rules read by name hold the settings' values."""
    body = _body(definition)
    code = " ".join(body)
    problems: list[str] = []
    for match in re.finditer(r"(?<![\w.@])t_ms - (\d+)(?![\w.])", code):
        if int(match[1]) > LOOKBACK_MS:
            problems.append(f"{name} reads {match[0]}, before the lookback's start")
    for match in re.finditer(r"\.valid_from_ms >= t_ms - (\d+)(?![\w.])", code):
        if int(match[1]) >= LOOKBACK_MS:
            problems.append(
                f"{name} reads {match[0][1:]}, a start new over the whole lookback: write the "
                "known start, > t_ms - lookback_ms"
            )
    declared = parameters(definition)
    found = statements(definition)
    for setting, value in SETTINGS.items():
        assigned = [
            statement
            for statement in found
            if any(
                item == setting and statement[index + 1 : index + 2] in (["="], ["+="], ["-="])
                for index, item in enumerate(statement)
            )
        ]
        if setting in declared:
            if assigned:
                problems.append(f"{name} sets its parameter {setting}")
        elif setting in body and assigned != [["UINT", setting, "=", str(value)]]:
            problems.append(
                f"{name} reads {setting} without declaring it once as UINT {setting} = "
                f"{value} or taking it as a parameter"
            )
    return problems


def _bounded(
    alias: str, ends: dict[str, set[str | None]], chosen: set[str], seen: frozenset[str]
) -> bool:
    """Whether a chain of bounds from alias reaches t's side: a bound of its own (None), a
    vertex set chosen before, or another alias bounded in turn; a cycle bounds nothing."""
    if alias in chosen or None in ends.get(alias, set()):
        return True
    return any(
        other is not None and other not in seen and _bounded(other, ends, chosen, seen | {alias})
        for other in ends.get(alias, set())
    )


def bound_rule(name: str, definition: str) -> list[str]:
    """Every SELECT bounds what it reads at t: each payment it steps to by a window before t,
    or by other payments of its pattern whose bounds lead to one; each link it steps along by
    the visible or current form; and each account it reaches through its ownership by
    `opened_ms < t_ms`. In a query with t, so is every other account it reaches but as a
    payment's sender or receiver. A vertex set chosen before was bounded where it was chosen.
    read_world and read_answer_key have no t."""
    if name in NO_AS_OF:
        return []
    as_of = "t_ms" in parameters(definition)
    problems: list[str] = []
    lower = re.compile(r"(\w+)\.event_ts_ms >= (?:t_ms - (?:lookback_ms|\d+)|(\w+)\.event_ts_ms)")
    upper = re.compile(r"(\w+)\.event_ts_ms < (?:t_ms|(\w+)\.(event_ts_ms|valid_from_ms))")
    link = re.compile(
        r"\( (\w+)\.valid_to_ms == 0 OR \1\.valid_to_ms > t_ms - (?:lookback_ms|2592000000) \)"
    )
    for select in selects(definition):
        where = f"{name}: {' '.join(select.statement)[:80]}"
        aliases = select.aliases()
        chosen = {step.alias for step in select.vertices if step.chosen and step.alias}
        below: dict[str, set[str | None]] = {}
        above: dict[str, set[str | None]] = {}
        for conjunct in select.conjuncts:
            if (match := lower.fullmatch(conjunct)) and (match[2] is None or match[2] in aliases):
                below.setdefault(match[1], set()).add(match[2])
            if (match := upper.fullmatch(conjunct)) and (match[2] is None or match[2] in aliases):
                if match[3] != "valid_from_ms":
                    above.setdefault(match[1], set()).add(match[2])
                elif f"{match[2]}.valid_from_ms < t_ms" in select.conjuncts:
                    above.setdefault(match[1], set()).add(None)
        for step in select.vertices:
            if step.chosen or not step.types & PAYMENTS:
                continue
            if step.alias is None:
                problems.append(f"{where} steps to a payment without an alias")
                continue
            for ends, side in ((below, "from a window's start"), (above, "below t")):
                if not _bounded(step.alias, ends, chosen, frozenset()):
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
                paid = any(types & PAYMENT_ACCOUNTS for types in beside)
                if step.chosen or not step.types & ACCOUNT or not (owned or (as_of and not paid)):
                    continue
                if step.alias is None or f"{step.alias}.opened_ms < t_ms" not in select.conjuncts:
                    problems.append(f"{where} reaches an account that may not exist at t")
    return problems


def refusal_rule(name: str, definition: str) -> list[str]:
    """A query with t refuses every account parameter that does not exist at t, with a status
    row (docs/design.md, "Query plumbing"), in the form the module's docstring gives and
    before it prints its "ok" status, which its parser reads first: a vertex parameter is
    otherwise taken as bounded where it was chosen."""
    if "t_ms" not in parameters(definition):
        return []
    found = statements(definition)
    ok = next(
        (at for at, statement in enumerate(found) if '"ok"' in statement and "PRINT" in statement),
        len(found),
    )
    problems: list[str] = []
    for parameter, types in _vertex_parameters(definition).items():
        if not types & ACCOUNT:
            continue
        names = {parameter} | {
            assigned[0]
            for statement in found
            if (assigned := _assigned(statement)) is not None
            and assigned[2] == ["{", parameter, "}"]
        }
        refused = False
        for index, statement in enumerate(found):
            existing = re.fullmatch(
                r"(\w+) = SELECT (\w+) FROM (\w+) : \2 WHERE (.*)", " ".join(statement)
            )
            if existing is None or existing[3] not in names:
                continue
            where = _clauses(statement).get("WHERE", [])
            if f"{existing[2]}.opened_ms < t_ms" not in {" ".join(p) for p in _split_and(where)}:
                continue
            for test, after in zip(found[index + 1 : ok], found[index + 2 :], strict=False):
                condition = re.fullmatch(
                    rf"IF {existing[1]}\.size \( \) (?:<|!=) (\w+)\.size \( \) THEN PRINT "
                    r'"(\w+)" AS status\b.*',
                    " ".join(test),
                )
                if (
                    condition is not None
                    and condition[1] in names
                    and condition[2] != "ok"
                    and after == ["RETURN"]
                ):
                    refused = True
        if not refused:
            problems.append(
                f"{name} does not refuse accounts of {parameter} that do not exist at t"
            )
    return problems


def pattern_rule(name: str, definition: str) -> list[str]:
    """Every step of a pattern can be placed: a vertex set chosen before, or vertex types and
    edge types of the schema, one hop each. A wildcard (`_` or `ANY`), an edge without a type,
    an edge of several hops, or a name that is neither a vertex set the rules can follow nor a
    type could reach a payment, a link or an account that no bound covers. No query but
    read_world steps to a Token or along its links."""
    problems: list[str] = []
    for select in selects(definition):
        where = f"{name}: {' '.join(select.statement)[:80]}"
        for step in select.vertices:
            if TOKEN in step.types and name != READ_WORLD:
                problems.append(f"{where} steps to a Token, which only read_world reads")
            if step.chosen:
                continue
            unknown = sorted(step.types - VERTEX_ATTRIBUTES.keys())
            if unknown or not step.types:
                problems.append(
                    f"{where} steps to {' or '.join(unknown) or 'a set of no known type'}, "
                    "which is no vertex type of the schema"
                )
        for edge in select.edges:
            unknown = sorted(edge.types - EDGE_ATTRIBUTES.keys())
            if unknown or not edge.types:
                problems.append(
                    f"{where} steps along {' or '.join(unknown) or 'an edge of no type'}, "
                    "which is no edge type of the schema"
                )
            if edge.hops:
                problems.append(
                    f"{where} steps along several edges at once: name every vertex it reaches"
                )
            if edge.types & TOKEN_EDGES and name != READ_WORLD:
                problems.append(
                    f"{where} steps along {sorted(edge.types & TOKEN_EDGES)}, a Token's link, "
                    "which only read_world reads"
                )
    return problems


def _values(definition: str) -> frozenset[str]:
    """The names a query declares for what is not a vertex set: its parameters of other
    types than vertices, and its local scalars and tuples."""
    names = {name for name, kind in parameters(definition).items() if "VERTEX" not in kind}
    types = SCALAR_TYPES | tuples(definition).keys()
    for statement in statements(definition):
        if statement[0] in types and len(statement) > 1:
            names |= {part[0] for part in _split(statement[1:], ",") if part}
    return frozenset(names)


def projection_rule(name: str, definition: str) -> list[str]:
    """Every PRINT of a vertex set is a projection: a whole set prints every attribute, the
    labels among them. A printed name that is no value the query declares is taken for a
    vertex set."""
    values = _values(definition)
    problems: list[str] = []
    for statement in statements(definition):
        if "PRINT" not in statement:
            continue
        for item in _split(statement[statement.index("PRINT") + 1 :], ","):
            if item[-2:-1] == ["AS"]:
                item = item[:-2]
            if item[1:2] == ["["]:
                continue
            problems += [
                f"{name} prints all of {token}"
                for at, token in enumerate(item)
                if IDENTIFIER.fullmatch(token)
                and token not in values
                and token not in PRINT_WORDS
                and item[at + 1 : at + 2] != ["("]
            ]
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
    whose name is no alias is a tuple's field. A call (`a.outdegree(...)`) is no attribute:
    structure_rule reads those."""
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


def _prefilter(conjunct: str) -> int:
    """How many all-time degrees a conjunct of read_hubs' prefilter form sums (`a.outdegree(
    "T") + ... > hub_threshold`, over the account's payment edges), or 0 for another
    conjunct."""
    left, _, right = conjunct.rpartition(" > ")
    if right not in ("hub_threshold", str(SUBGRAPH.hub_threshold)):
        return 0
    parts = [re.fullmatch(r'(\w+)\.outdegree \( "(\w+)" \)', part) for part in left.split(" + ")]
    if any(part is None or part[2] not in ACCOUNT_PAYMENTS for part in parts):
        return 0
    return len(parts) if len({part[1] for part in parts if part is not None}) == 1 else 0


def structure_rule(name: str, definition: str) -> list[str]:
    """No query reads a vertex's edges or attributes around the rules: an all-time degree or
    neighbour list reads payments and links after t, and an attribute named by a string
    escapes the time and label rules. read_hubs' prefilter alone reads all-time degrees,
    since an account with no more payments than the hub threshold cannot be a hub."""
    problems: list[str] = []
    for statement, _, select in scopes(definition):
        calls = [call for _, call in _calls(statement) if call in STRUCTURE_CALLS]
        allowed = 0
        if name == READ_HUBS and select is not None:
            allowed = sum(_prefilter(conjunct) for conjunct in select.conjuncts)
        for call in sorted(set(calls)):
            if call == "outdegree" and calls.count(call) <= allowed:
                continue
            problems.append(f"{name} calls {call}, which reads the graph around the rules")
    return problems


def clock_rule(name: str, definition: str) -> list[str]:
    """No query reads the clock (the Clock guard): no date or time function, and no remainder
    but a test of the parameter t_ms itself."""
    body = _body(definition)
    problems = [
        f"{name} calls {call}, which reads the clock"
        for _, call in _calls(body)
        if call in CLOCK_CALLS or call.startswith("datetime")
    ]
    allowed = len(PARAMETER_REMAINDER.findall(" ".join(body)))
    if body.count("%") > allowed:
        problems.append(f"{name} takes a remainder, which reads the clock of a time")
    return problems


def call_rule(name: str, definition: str) -> list[str]:
    """No query calls a library algorithm, which reads the whole graph with no as-of time
    (the design's B6: never a library algorithm), or another query, which the rules do not
    read with it."""
    problems: list[str] = []
    for _, call in _calls(_body(definition)):
        if call.startswith("tg_"):
            problems.append(f"{name} calls the library algorithm {call}")
        elif call in QUERY_FILES:
            problems.append(f"{name} calls the query {call}")
    return problems


def _crowd_count(select: Select, node: Step) -> bool:
    """Whether a SELECT is a crowd count of node: it selects node and reads or writes no
    accumulator but node's own named `@crowd...`."""
    if node.alias is None or select.selected != node.alias:
        return False
    named = [item for item in select.accumulations if "@" in item]
    return bool(named) and all(
        re.fullmatch(rf"{re.escape(node.alias)}\.@crowd\w*", item) for item in named
    )


def leaf_rule(name: str, definition: str) -> list[str]:
    """Every step between a device or IP and its users excludes the leaves by `NOT d.@leaf`,
    but a step from users chosen before to their devices or IPs and the crowd count; the
    leaf flag is set in its one form from a crowd count before it is tested; and a crowd
    accumulator is read through its size alone."""
    problems: list[str] = []
    counted: set[str] = set()
    flagged = False
    for statement, _, select in scopes(definition):
        declaration = re.fullmatch(r"\w+Accum", statement[0]) is not None
        text = " ".join(statement)
        for index, item in enumerate(statement):
            after = statement[index + 1 : index + 2]
            if "@crowd" in item and not declaration:
                written = re.fullmatch(r"(?:\w+\.)?@@?crowd\w*", item) and after in (["+="], ["="])
                sized = re.fullmatch(r"(?:\w+\.)?@@?crowd\w*\.size", item) and after == ["("]
                if not (written or sized):
                    problems.append(f"{name} reads {item} other than through its size")
            if re.fullmatch(r"\w+\.@leaf", item) and after in (["+="], ["="]):
                form = LEAF_FORM.fullmatch(text)
                crowd = re.search(r"\.(@crowd\w*)\.size", text)
                if form is None or crowd is None or crowd[1] not in counted:
                    problems.append(
                        f"{name} sets the leaf flag other than from a crowd count, in the form "
                        f"`Leaves = SELECT d FROM Nodes:d POST-ACCUM d.@leaf += "
                        f"d.@crowd_customers.size() > crowd_threshold`: {text[:80]}"
                    )
                else:
                    flagged = True
            if re.fullmatch(r"\w+\.@leaf", item) and statement[index - 1 : index] == ["NOT"]:
                if not flagged:
                    problems.append(f"{name} tests {item} before a crowd count sets it")
        if select is None:
            continue
        adjacent: Counter[object] = Counter()
        for number, (steps, edges) in enumerate(select.paths):
            for index in range(len(edges)):
                for at in (index, index + 1):
                    adjacent[steps[at].alias or (number, at)] += 1
        for number, (steps, edges) in enumerate(select.paths):
            for index, edge in enumerate(edges):
                ends = ((index, index + 1), (index + 1, index))
                pair = next(
                    (
                        (node, users)
                        for node, users in ends
                        if steps[node].types & NODES and steps[users].types & USERS
                    ),
                    None,
                )
                if pair is None:
                    continue
                node, users = steps[pair[0]], steps[pair[1]]
                if (
                    users.chosen
                    and not node.chosen
                    and adjacent[node.alias or (number, pair[0])] == 1
                ):
                    continue
                if node.alias is not None and f"NOT {node.alias}.@leaf" in select.conjuncts:
                    continue
                if _crowd_count(select, node):
                    counted |= {
                        item.split(".", 1)[1] for item in select.accumulations if "@crowd" in item
                    }
                    continue
                problems.append(
                    f"{name} steps between {node.alias} and its users without the leaf test "
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


def _printed(items: Sequence[str]) -> list[tuple[str, str]]:
    """Each `expression AS name` of a PRINT's items: the expression's tokens joined by spaces,
    and the name."""
    return [
        (" ".join(part[:-2]), part[-1])
        for part in _split(items, ",")
        if len(part) >= 2 and part[-2] == "AS"
    ]


def projection_items(definition: str, vertex_set: str) -> list[tuple[str, str]]:
    """What a PRINT of a projection of vertex_set prints, in order: each expression and the
    name it is printed as."""
    for statement in statements(definition):
        if statement[:1] == ["PRINT"] and statement[1:3] == [vertex_set, "["]:
            return _printed(statement[3 : statement.index("]")])
    raise ValueError(f"No PRINT of a projection of {vertex_set}")


def status_items(definition: str) -> list[tuple[str, str]]:
    """What a query's PRINT of its "ok" status prints after the status and the version: each
    expression and the name it is printed as."""
    for statement in statements(definition):
        if statement[:4] == ["PRINT", '"ok"', "AS", "status"]:
            return _printed(statement[1:])[2:]
    raise ValueError('No PRINT "ok" AS status')


def projection_fields(definition: str, vertex_set: str) -> list[str]:
    """The names a PRINT of a projection of vertex_set prints, in order."""
    return [name for _, name in projection_items(definition, vertex_set)]


def status_fields(definition: str) -> list[str]:
    """What a query's PRINT of its "ok" status prints after the status and the version."""
    return [name for _, name in status_items(definition)]


def source_rule(what: str, printed: Sequence[tuple[str, str]]) -> list[str]:
    """Each printed name reads what it names: its expression ends in `.name`, `.@name` or
    `@@name`, or, for an id (ID_NAMES), in `.id`. The names alone match the dataclass whatever
    they read, so without this two fields could swap their sources unseen."""
    problems: list[str] = []
    for expression, name in printed:
        last = expression.rsplit(" ", 1)[-1]
        named = rf"(?:\w+\.@?|@@){re.escape(name)}"
        if not re.fullmatch(named, last) and not (
            name in ID_NAMES and re.fullmatch(r"\w+\.id", last)
        ):
            problems.append(
                f"{what} prints {expression or 'nothing'} as {name}, which does not read {name}"
            )
    return problems


def field_rule(what: str, printed: Sequence[str], kind: type) -> list[str]:
    """The fields a query prints are those of the dataclass its parser reads them into."""
    expected = [field.name for field in fields(kind)]
    return [] if list(printed) == expected else [f"{what} prints {list(printed)}, not {expected}"]


# Readers of one query's text, shared by the tests that hold a query clause by clause: each
# takes the query's definition, and most the name a SELECT assigns, its target.


def normal(text: str) -> str:
    """A GSQL text as the rules read it: its tokens joined by single spaces."""
    return " ".join(tokens(text))


def select_of(definition: str, target: str) -> Select:
    """The SELECT that assigns target."""
    for statement, _, select in scopes(definition):
        if select is not None and statement[:2] == [target, "="]:
            return select
    raise AssertionError(f"no SELECT assigns {target}")


def accumulation_text(definition: str, target: str) -> str:
    """The ACCUM and POST-ACCUM of the SELECT that assigns target, as the rules read them."""
    return " ".join(select_of(definition, target).accumulations)


def declared_locals(definition: str, target: str) -> list[tuple[str, str]]:
    """The local variables a SELECT's clauses declare, with their types, in order."""
    found = re.findall(
        r"(?:^|, )(INT|UINT|STRING|BOOL|DOUBLE|FLOAT) (\w+) =",
        accumulation_text(definition, target),
    )
    return [(kind, name) for kind, name in found]


VERTEX_ACCUMULATOR = re.compile(r"\w+\.@(\w+)(\.\w+)?")


def select_flows(definition: str) -> list[tuple[str, set[str], set[str]]]:
    """Each assigned SELECT of a query in file order, with the vertex accumulators (`a.@name`)
    its ACCUM and POST-ACCUM clauses write (`+=` or `=` on the accumulator itself) and read
    (any other use, as in `a.@in_s.get(n)` or `FOREACH x IN a.@inflows`)."""
    found: list[tuple[str, set[str], set[str]]] = []
    for _, _, select in scopes(definition):
        if select is None:
            continue
        items = list(select.accumulations)
        written: set[str] = set()
        read: set[str] = set()
        for at, item in enumerate(items):
            match = VERTEX_ACCUMULATOR.fullmatch(item)
            if match is None:
                continue
            if match[2] is None and items[at + 1 : at + 2] in (["+="], ["="]):
                written.add(match[1])
            else:
                read.add(match[1])
        found.append((select.statement[0], written, read))
    return found


def dataflow_problems(definition: str) -> list[str]:
    """Every accumulator a SELECT reads is written only by SELECTs strictly before it, and
    has at least one. POST-ACCUM clauses of one SELECT run in parallel, and a SELECT moved
    before the one that fills what it reads walks empty lists and still compiles."""
    flows = select_flows(definition)
    problems: list[str] = []
    for index, (target, _, read) in enumerate(flows):
        for name in sorted(read):
            writers = [at for at, (_, written, _) in enumerate(flows) if name in written]
            if not writers or max(writers) >= index:
                by = [flows[at][0] for at in writers] or ["no SELECT"]
                problems.append(f"{target} reads @{name}, written by {', '.join(by)}")
    return problems


def moved_select(definition: str, target: str, after: str) -> str:
    """The query's text with the SELECT that assigns target moved to follow the one that
    assigns after."""

    def statement(name: str) -> re.Pattern[str]:
        return re.compile(rf"^    {name} = SELECT .*?;\n", re.MULTILINE | re.DOTALL)

    moved = statement(target).search(definition)
    stays = statement(after).search(definition)
    assert moved is not None and stays is not None
    without = definition.replace(moved[0], "", 1)
    end = statement(after).search(without)
    assert end is not None
    return without[: end.end()] + moved[0] + without[end.end() :]


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
        *window_rule(name, definition),
        *pattern_rule(name, definition),
        *bound_rule(name, definition),
        *refusal_rule(name, definition),
        *projection_rule(name, definition),
        *heap_rule(name, definition),
        *accumulator_rule(name, definition),
        *tuple_rule(name, definition),
        *attribute_rule(name, definition),
        *structure_rule(name, definition),
        *clock_rule(name, definition),
        *call_rule(name, definition),
        *leaf_rule(name, definition),
    ]
