"""The planted-slip harness: one-line mutants of each written query, each run against the
tests, and the mutants no test caught (docs/design.md, "Leak rules and how tests enforce
them": seven planted slips once passed every text test). It is run in every query review,
never by pytest:

    .venv/bin/python -m mule_pattern_learner.testing.mutate [query ...]

With no query named it runs every written query. For each mutant it changes one line of the
query's file, restamps the query's version (testing.versions), so that the version rule does
not catch every mutant, runs the GSQL and TigerGraph tests, and puts the file and the registry
back. A mutant no test caught is a slip the tests let through: either a test is missing, or the
change cannot alter an answer, which the review then says.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random
import re
import subprocess
import sys

from ..paths import GSQL_DIR, REPOSITORY_ROOT
from ..tigergraph.installer import COMMENTS
from ..tigergraph.queries import written_files
from .versions import REGISTRY_FILE, restamp

# At most this many mutants per query, drawn with SEED when there are more.
MUTANTS_PER_QUERY = 60
SEED = 20_261_017
# The tests a mutant runs against: every test that reads a query's text.
TESTS = ("tests/gsql", "tests/tigergraph")


@dataclass(frozen=True)
class Mutant:
    """A one-line change: the line's number (from 1), what changed, and the new line."""

    line: int
    what: str
    text: str


def _swaps(old: str, new: str, what: str) -> Callable[[str], list[tuple[str, str]]]:
    """Each line with one match of the pattern old replaced by new (which may name its
    groups, as `\\1`), one match at a time."""

    def change(line: str) -> list[tuple[str, str]]:
        return [
            (what, line[: match.start()] + match.expand(new) + line[match.end() :])
            for match in re.finditer(old, line)
        ]

    return change


def _drop_conjunct(line: str) -> list[tuple[str, str]]:
    """The line without one of its conjuncts, when the conjunct's brackets close in it."""
    parts = line.split(" AND ")
    found: list[tuple[str, str]] = []
    for index in range(1, len(parts)):
        dropped = parts[index]
        if dropped.count("(") == dropped.count(")"):
            kept = [*parts[:index], *parts[index + 1 :]]
            found.append((f"drop AND {dropped.strip()}", " AND ".join(kept)))
    stripped = line.strip()
    if stripped.startswith("AND ") and stripped.count("(") == stripped.count(")"):
        found.append((f"drop the line {stripped}", ""))
    return found


def _numbers(line: str) -> list[tuple[str, str]]:
    """The line with one integer literal, outside names and version strings, one more."""
    return [
        (
            f"{match[0]} becomes {int(match[0]) + 1}",
            f"{line[: match.start()]}{int(match[0]) + 1}{line[match.end() :]}",
        )
        for match in re.finditer(r"(?<![\w.\-\"])\d+(?![\w.])", line)
        if int(match[0]) >= 2
    ]


CHANGES: tuple[Callable[[str], list[tuple[str, str]]], ...] = (
    _swaps(" < ", " <= ", "< becomes <="),
    _swaps(" <= ", " < ", "<= becomes <"),
    _swaps(" > ", " >= ", "> becomes >="),
    _swaps(" >= ", " > ", ">= becomes >"),
    _swaps(" == ", " != ", "== becomes !="),
    _swaps(" != ", " == ", "!= becomes =="),
    _swaps(r"t_ms - lookback_ms", "t_ms", "a window from t itself"),
    _swaps(r"t_ms - lookback_ms", "t_ms - 2592000000", "the lookback becomes 30 days"),
    _swaps(r"t_ms - 2592000000", "t_ms - lookback_ms", "30 days become the lookback"),
    _swaps(r"\bNOT ", "", "drop NOT"),
    _swaps(r"\bASC\b", "DESC", "ASC becomes DESC"),
    _swaps(r"\bDESC\b", "ASC", "DESC becomes ASC"),
    _swaps(r"\bMinAccum\b", "MaxAccum", "MinAccum becomes MaxAccum"),
    _swaps(r"\bMaxAccum\b", "MinAccum", "MaxAccum becomes MinAccum"),
    _swaps(r"\.account_type\b", ".pu_label", "a label read for account_type"),
    _swaps(r"HeapAccum<(\w+)>\(\w+, ", "HeapAccum<\\1>(", "drop a heap's capacity"),
    _swaps(r"PRINT (\w+)\[[^\]]*\]", "PRINT \\1", "print a whole vertex set"),
    _drop_conjunct,
    _numbers,
)


def _commented(text: str) -> set[int]:
    """The numbers of the lines whose every character but space is in a comment."""
    covered = [False] * len(text)
    for match in COMMENTS.finditer(text):
        covered[match.start() : match.end()] = [True] * (match.end() - match.start())
    lines: set[int] = set()
    start = 0
    for number, line in enumerate(text.split("\n"), 1):
        if all(covered[start + at] for at, char in enumerate(line) if not char.isspace()):
            lines.add(number)
        start += len(line) + 1
    return lines


def mutants(text: str, *, limit: int = MUTANTS_PER_QUERY, seed: int = SEED) -> list[Mutant]:
    """One-line mutants of a query file: every change of CHANGES on every line of code after
    the query's CREATE but the one printing its version, at most limit drawn with seed, in
    line order."""
    lines = text.split("\n")
    first = next((number for number, line in enumerate(lines, 1) if line.startswith("CREATE ")), 1)
    skipped = _commented(text)
    found: list[Mutant] = []
    for number, line in enumerate(lines, 1):
        if number < first or number in skipped or " AS version" in line:
            continue
        for change in CHANGES:
            for what, new in change(line):
                if new.strip() != line.strip() or new == "":
                    found.append(Mutant(number, what, new))
    if len(found) > limit:
        found = sorted(random.Random(seed).sample(found, limit), key=lambda m: (m.line, m.what))
    return found


def applied(text: str, mutant: Mutant) -> str:
    """text with the mutant's line in place of its own."""
    lines = text.split("\n")
    lines[mutant.line - 1] = mutant.text
    return "\n".join(lines)


def _tests_pass() -> bool:
    command = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    return result.returncode == 0


def survivors(name: str, relative: str) -> tuple[int, list[Mutant]]:
    """The mutants of a query, and those no test caught; the file and the registry are put
    back whatever happens."""
    path = GSQL_DIR / relative
    original, registry = path.read_text(), REGISTRY_FILE.read_text()
    found = mutants(original)
    caught: list[Mutant] = []
    survived: list[Mutant] = []
    try:
        for mutant in found:
            path.write_text(applied(original, mutant))
            try:
                restamp(name, path, REGISTRY_FILE)
            except ValueError:
                caught.append(mutant)
                continue
            (caught if not _tests_pass() else survived).append(mutant)
            path.write_text(original)
            REGISTRY_FILE.write_text(registry)
    finally:
        path.write_text(original)
        REGISTRY_FILE.write_text(registry)
    return len(found), survived


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mutants of the named queries, or of every written one; print the survivors."""
    written = written_files()
    names = list(sys.argv[1:] if argv is None else argv) or list(written)
    unknown = [name for name in names if name not in written]
    if unknown:
        print(f"Not a written query: {', '.join(unknown)}")
        return 2
    if not _tests_pass():
        print("The tests fail before any mutant: fix them first")
        return 2
    for name in names:
        count, survived = survivors(name, written[name])
        print(f"{name}: {count} mutants, {len(survived)} not caught")
        for mutant in survived:
            print(f"  NOT CAUGHT line {mutant.line}, {mutant.what}: {mutant.text.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
