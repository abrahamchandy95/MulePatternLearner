"""Layout-free views of GSQL text, for tests that check what a query file says.

A formatter may change any whitespace of a query file, so tests compare text with every
run of whitespace as one space (flat) and find a block's end by the indentation of its
header line (block, blocks, split_block), never by an exact number of spaces. The graph
tests run a query file's text under INTERPRET (as_interpreted).
"""

from __future__ import annotations

from collections.abc import Iterator
import re

# The Fourier encoding the context query calls for every message age and gap; no code
# but the tests names it.
FOURIER_QUERY = "encode_fourier64"


def as_interpreted(text: str) -> str:
    """A repository `CREATE OR REPLACE QUERY` text as the equivalent INTERPRET QUERY text."""
    body = re.split(r"CREATE OR REPLACE QUERY \w+\(", text, maxsplit=1)[1]
    return "INTERPRET QUERY (" + body


def flat(text: str) -> str:
    """Text with every run of whitespace as one space."""
    return " ".join(text.split())


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _end(lines: list[str], start: int) -> int:
    """The `END;` line indented as line start, after it."""
    indent = _indent(lines[start])
    return next(
        i
        for i in range(start + 1, len(lines))
        if lines[i].strip() == "END;" and _indent(lines[i]) == indent
    )


def split_block(text: str, header: str) -> tuple[str, str, str]:
    """The text before the first block that header opens, inside it, and after its END.

    The block ends at the first `END;` line indented as the header's line, so blocks
    nested inside it are part of it.
    """
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if header in line)
    end = _end(lines, start)
    head = lines[start]
    cut = head.index(header) + len(header)
    before = "".join(lines[:start]) + head[: head.index(header)]
    inside = head[cut:] + "".join(lines[start + 1 : end])
    return before, inside, "".join(lines[end + 1 :])


def block(text: str, header: str) -> str:
    """The inside of the first block that header opens."""
    return split_block(text, header)[1]


def blocks(text: str, header: str) -> Iterator[str]:
    """The inside of every block that header opens, in order."""
    lines = text.splitlines(keepends=True)
    for start, line in enumerate(lines):
        if header in line:
            end = _end(lines, start)
            yield line[line.index(header) + len(header) :] + "".join(lines[start + 1 : end])
