"""The version restamper: stamps each written query's version, the hash of its text, in the
registry (docs/design.md, "Query plumbing"). No query prints its version. After an edit to a
query, run

    .venv/bin/python -m mule_pattern_learner.testing.versions

which prints each query's new version. A query's registry entry is first written with a
placeholder of its name and twelve zeros, "<name>-000000000000", which the restamper replaces.
An old query that still prints its version gets the printed string replaced as well.
"""

from __future__ import annotations

from pathlib import Path
import re

from ..paths import GSQL_DIR
from ..tigergraph import queries
from ..tigergraph.installer import definitions, text_version
from ..tigergraph.queries import written_files

# The registry, whose entries hold each written query's version.
REGISTRY_FILE = Path(queries.__file__)


def _versions(name: str) -> re.Pattern[str]:
    """A version string of name, quoted."""
    return re.compile(rf'"({re.escape(name)}-[0-9a-f]{{12}})"')


def stamped(name: str, text: str) -> tuple[str, str]:
    """A query file's text and its version, the hash of its text; the version an old query
    still prints is replaced by it."""
    defined = definitions(text)
    if list(defined) != [name]:
        raise ValueError(f"The file of {name} defines {list(defined)}")
    if len(_versions(name).findall(text)) > 1:
        raise ValueError(f"{name} prints its version more than once")
    version = text_version(f"{name}-", defined[name])
    return _versions(name).sub(f'"{version}"', text), version


def restamp(name: str, file: Path, registry: Path = REGISTRY_FILE) -> str:
    """Stamp the version of name in the registry, and in its file when an old query still
    prints it; the new version."""
    text, version = stamped(name, file.read_text())
    source = registry.read_text()
    updated, count = _versions(name).subn(f'"{version}"', source)
    if count != 1:
        raise ValueError(f'The registry must hold one version of {name}, as "{name}-<12 hex>"')
    file.write_text(text)
    registry.write_text(updated)
    return version


def main() -> None:
    """Restamp every written query and print its version."""
    for name, relative in written_files().items():
        print(name, restamp(name, GSQL_DIR / relative))


if __name__ == "__main__":
    main()
