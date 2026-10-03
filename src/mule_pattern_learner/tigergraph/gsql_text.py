"""Read GSQL files, split them into query definitions and compare their text."""

from __future__ import annotations

import re

from ..paths import GSQL_DIR

# GSQL comments, which neither the compared text nor a query's calls include.
COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*|#[^\n]*", re.S)


def normalized(source: str) -> str:
    source = COMMENTS.sub("", source)
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[^\s"]+', source)
    return "".join(token if token.startswith('"') else token.lower() for token in tokens)


def definitions(source: str) -> dict[str, str]:
    starts = list(re.finditer(r"CREATE (?:OR REPLACE )?QUERY (\w+)", source, re.I))
    result = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(source)
        result[match[1]] = source[match.start() : end].split("USE GRAPH")[0].strip()
    return result


def calls(definition: str, name: str) -> bool:
    """Whether a query definition calls the query `name` (its comments aside)."""
    return re.search(rf"\b{re.escape(name)}\s*\(", COMMENTS.sub("", definition)) is not None


def parameter_names(definition: str) -> set[str]:
    """Names in `CREATE QUERY name(TYPE a, TYPE b = default, ...)`."""
    match = re.search(r"QUERY\s+\w+\s*\(", definition, re.I)
    if match is None:
        raise ValueError("Query definition has no parameter list")
    depth, quoted, current, parts = 1, False, "", []
    for char in definition[match.end() :]:
        if quoted:
            quoted = char != '"'
        elif char == '"':
            quoted = True
        elif char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            parts.append(current)
            current = ""
            continue
        current += char
    else:
        raise ValueError("Unterminated query parameter list")
    parts.append(current)
    names = set()
    for part in parts:
        declaration = part.split("=", 1)[0].split()
        if declaration:
            names.add(declaration[-1])
    return names


def repository_queries(files: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Query name -> (GSQL file, definition text), in file order; files are under GSQL_DIR."""
    result: dict[str, tuple[str, str]] = {}
    for path in files:
        for name, text in definitions((GSQL_DIR / path).read_text()).items():
            result[name] = (path, text)
    return result
