"""The import contracts that list every package but a few leave none out.

import-linter checks only the source modules a contract names, so a package missing
from such a list goes unchecked. reporting and diagnostics do not exist yet; the
contracts name them in the ".**" form, which matches nothing until they are created.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import mule_pattern_learner
from mule_pattern_learner.paths import REPOSITORY_ROOT

PACKAGE = "mule_pattern_learner"
# The packages of the design record's tree that later steps create.
PLANNED = frozenset({"reporting", "diagnostics"})
# The contracts that list every top-level module but those they allow.
ENUMERATED = {
    "Only reporting draws": {"reporting"},
    "Fakes stay out of the package": {"testing"},
    "Only diagnostics uses the verification mirrors": {"diagnostics", "reference"},
}


def top_level_modules() -> set[str]:
    root = Path(mule_pattern_learner.__file__).parent
    modules = {path.stem for path in root.glob("*.py") if path.stem != "__init__"}
    return modules | {path.name for path in root.iterdir() if (path / "__init__.py").exists()}


def contracts() -> dict[str, dict[str, object]]:
    config = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    listed = config["tool"]["importlinter"]["contracts"]
    return {contract["name"]: contract for contract in listed}


def test_every_enumerated_contract_lists_every_package_but_those_it_allows() -> None:
    known = contracts()
    for name, allowed in ENUMERATED.items():
        sources = known[name]["source_modules"]
        assert isinstance(sources, list)
        listed = {str(module).removeprefix(PACKAGE + ".").removesuffix(".**") for module in sources}
        assert listed == (top_level_modules() | PLANNED) - allowed, name
