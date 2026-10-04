"""The scope's vertex and edge types, as gsql/schema/scope_vertex.gsql declares them and a graph has them.

`mule install` compares the two (installer.scope_schema): a graph without the types gets
them from the file, and one whose types differ, such as a graph created before the scope
vertex recorded its split shares, has them replaced while it holds no scope vertex. Each
type is compared by its attributes' names and types in order, the vertex type's primary
id first, since create_training_scope inserts a scope by position; the edge type also by
the vertex types it connects and its reverse edge. Defaults are not compared. A graph's
types are read from TigerGraph's schema (getSchema), whose types this module reads by the
names the file declares and no other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any

from ..contract.server import GRAPH_NAME
from ..paths import GSQL_DIR
from .gsql_text import COMMENTS

# The schema change that adds the scope's types, relative to GSQL_DIR.
SCOPE_TYPES_FILE = "schema/scope_vertex.gsql"

# Attribute names and types, in their order.
Attributes = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ScopeTypes:
    """The scope vertex type and its membership edge type, as they are compared.

    Types are upper case. A type a graph lacks reads as one without attributes,
    endpoints or reverse edge.
    """

    vertex: str
    # The primary id first.
    vertex_attributes: Attributes
    edge: str
    edge_attributes: Attributes
    # The (from, to) vertex types the edge type connects, sorted.
    endpoints: tuple[tuple[str, str], ...]
    reverse_edge: str

    @property
    def names(self) -> tuple[str, str, str]:
        """The names a query uses the types by: the vertex, the edge and its reverse."""
        return (self.vertex, self.edge, self.reverse_edge)


def _declared(body: str) -> Attributes:
    """(name, TYPE) of each `[PRIMARY_ID] name TYPE [DEFAULT value]` of a declaration."""
    attributes = []
    for part in body.split(","):
        words = part.split()
        if words[:1] == ["PRIMARY_ID"]:
            words = words[1:]
        if words:
            attributes.append((words[0], words[1].upper()))
    return tuple(attributes)


def parsed_scope_types(text: str) -> ScopeTypes:
    """The scope types a schema change job of scope_vertex.gsql's form adds."""
    text = COMMENTS.sub("", text)
    vertex = re.search(r"ADD VERTEX (\w+) \((.*?)\) WITH", text, re.S)
    edge = re.search(r'ADD DIRECTED EDGE (\w+) \((.*?)\) WITH REVERSE_EDGE="(\w+)"', text, re.S)
    pairs = list(re.finditer(r"FROM (\w+), TO (\w+)", edge[2])) if edge else []
    if vertex is None or edge is None or not pairs:
        raise ValueError("The schema change adds no scope vertex type and edge type")
    return ScopeTypes(
        vertex=vertex[1],
        vertex_attributes=_declared(vertex[2]),
        edge=edge[1],
        edge_attributes=_declared(edge[2][pairs[-1].end() :]),
        endpoints=tuple(sorted((pair[1], pair[2]) for pair in pairs)),
        reverse_edge=edge[3],
    )


def declared_scope_types() -> ScopeTypes:
    """The scope types gsql/schema/scope_vertex.gsql declares."""
    return parsed_scope_types((GSQL_DIR / SCOPE_TYPES_FILE).read_text())


def _named(types: Iterable[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    return next((entry for entry in types if entry.get("Name") == name), None)


def _attribute(entry: Mapping[str, Any]) -> tuple[str, str]:
    return str(entry["AttributeName"]), str(entry["AttributeType"]["Name"]).upper()


def endpoints(edge: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The (from, to) vertex types of an edge type in TigerGraph's schema, sorted."""
    pairs = edge.get("EdgePairs") or [
        {"From": edge.get("FromVertexTypeName"), "To": edge.get("ToVertexTypeName")}
    ]
    return tuple(sorted((str(pair["From"]), str(pair["To"])) for pair in pairs))


def graph_scope_types(schema: Mapping[str, Any], declared: ScopeTypes) -> ScopeTypes | None:
    """The scope types of a graph's schema (getSchema), by the declared names.

    None when the graph has neither the vertex type nor the edge type.
    """
    vertex = _named(schema.get("VertexTypes", []), declared.vertex)
    edge = _named(schema.get("EdgeTypes", []), declared.edge)
    if vertex is None and edge is None:
        return None
    vertex_attributes: Attributes = ()
    if vertex is not None:
        primary = vertex.get("PrimaryId")
        key = [_attribute(primary)] if primary else []
        named = {name for name, _ in key}
        others = [_attribute(entry) for entry in vertex.get("Attributes", [])]
        vertex_attributes = (*key, *(item for item in others if item[0] not in named))
    return ScopeTypes(
        vertex=declared.vertex,
        vertex_attributes=vertex_attributes,
        edge=declared.edge,
        edge_attributes=tuple(_attribute(e) for e in (edge or {}).get("Attributes", [])),
        endpoints=endpoints(edge) if edge is not None else (),
        reverse_edge=str((edge or {}).get("Config", {}).get("REVERSE_EDGE", "")),
    )


def _listed(attributes: Attributes) -> str:
    return ", ".join(f"{name} {kind}" for name, kind in attributes) or "none"


def _attribute_difference(name: str, found: Attributes, declared: Attributes) -> list[str]:
    """How a type's attributes differ from the declared ones, in words; [] when they match."""
    if found == declared:
        return []
    if not found:
        return [f"{name} is missing"]
    lacking = [item for item in declared if item not in found]
    if found == tuple(item for item in declared if item not in lacking):
        return [f"{name} lacks {', '.join(attribute for attribute, _ in lacking)}"]
    return [f"{name} has the attributes ({_listed(found)}), not ({_listed(declared)})"]


def scope_type_differences(found: ScopeTypes, declared: ScopeTypes) -> list[str]:
    """How a graph's scope types differ from the declared ones, in words; [] when they match."""
    differences = _attribute_difference(
        found.vertex, found.vertex_attributes, declared.vertex_attributes
    )
    if not found.endpoints:
        return [*differences, f"{found.edge} is missing"]
    differences += _attribute_difference(
        found.edge, found.edge_attributes, declared.edge_attributes
    )
    if found.endpoints != declared.endpoints:
        connects = ", ".join(f"{a} to {b}" for a, b in found.endpoints)
        differences.append(f"{found.edge} connects {connects}")
    if found.reverse_edge != declared.reverse_edge:
        reverse = found.reverse_edge or "none"
        differences.append(f"{found.edge} has the reverse edge {reverse}")
    return differences


def foreign_scope_edges(schema: Mapping[str, Any], declared: ScopeTypes) -> list[str]:
    """Edge types of a graph's schema, other than the declared one, that reach the scope vertex."""
    return sorted(
        str(edge["Name"])
        for edge in schema.get("EdgeTypes", [])
        if edge.get("Name") != declared.edge
        and any(declared.vertex in pair for pair in endpoints(edge))
    )


def uses_scope_types(definition: str, declared: ScopeTypes) -> bool:
    """Whether a query definition names a scope type (its comments aside)."""
    text = COMMENTS.sub("", definition)
    return any(re.search(rf"\b{re.escape(name)}\b", text) for name in declared.names)


def drop_job(found: ScopeTypes) -> str:
    """The schema change job that drops a graph's scope types: the edge type, then the vertex."""
    drops = [f"  DROP EDGE {found.edge};"] if found.endpoints else []
    if found.vertex_attributes:
        drops.append(f"  DROP VERTEX {found.vertex};")
    return "\n".join(
        [
            f"USE GRAPH {GRAPH_NAME}",
            f"CREATE SCHEMA_CHANGE JOB drop_training_scope FOR GRAPH {GRAPH_NAME} {{",
            *drops,
            "}",
            "RUN SCHEMA_CHANGE JOB drop_training_scope",
            "DROP JOB drop_training_scope",
            "",
        ]
    )
