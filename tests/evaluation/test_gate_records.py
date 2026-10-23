"""PhantomLedger's records of a world as mule audit checks them (docs/design.md, "Gates"): GATES
names every gate of docs/features.md's gate tables by its entry id, and a record passes only with
the fingerprint of the loaded world, a clean and certified build, every code gate passed and every
gate mapped to a feature present and passed."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

import pytest

from mule_pattern_learner.evaluation.gate_records import (
    GATES,
    MAPPED,
    UNMAPPED,
    check_records,
    read_gates,
)
from mule_pattern_learner.paths import REPOSITORY_ROOT
from mule_pattern_learner.testing.calendar import world_of
from mule_pattern_learner.testing.reference import ReferenceGraph

Records = Callable[..., tuple[Path, Path]]
FEATURES = REPOSITORY_ROOT / "docs" / "features.md"


def _rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]


def _entries(text: str) -> list[str]:
    """The entry ids in the first column of the tables of a part of docs/features.md, a range
    such as S-IDN-1 to S-IDN-5 as each id in it."""
    found: list[str] = []
    for cells in _rows(text):
        if cells[0] == "Entry":
            continue
        ranged = re.fullmatch(r"(S-[A-Z]+-)(\d+) to \1(\d+)", cells[0])
        if ranged is None:
            found.append(cells[0])
        else:
            first, last = int(ranged.group(2)), int(ranged.group(3))
            found += [f"{ranged.group(1)}{number}" for number in range(first, last + 1)]
    return found


def _tables() -> tuple[list[str], list[str], list[str]]:
    """The entry ids of docs/features.md's dossier rows, residual rows and PhantomLedger's added
    world checks."""
    text = FEATURES.read_text()
    start = text.index("## Generator gates each feature depends on")
    added = text.index("## Gates the features add")
    ours = text.index("**MulePatternLearner's checks**")
    residual = text.index("**Shortcuts noted but not yet researched**")
    return (
        _entries(text[start:residual]),
        _entries(text[residual:added]),
        _entries(text[added:ours]),
    )


def test_gates_names_every_entry_of_the_gate_tables_of_features_md() -> None:
    dossier, residual, world_checks = _tables()
    assert (len(dossier), len(residual), len(world_checks)) == (63, 13, 11)
    assert all(re.fullmatch(r"S-[A-Z]+-\d+", entry) for entry in dossier)
    assert all(re.fullmatch(r"R-[A-Z0-9-]+", entry) for entry in residual)
    assert all(re.fullmatch(r"G-[A-Z0-9-]+", entry) for entry in world_checks)
    # Every id once, in the order of the tables, and no other.
    assert list(GATES) == [*dossier, *residual, *world_checks]
    assert len(set(GATES)) == len(GATES) == 87


def test_only_gates_reaching_no_feature_are_unmapped() -> None:
    # S-RAIL-4 reaches none, S-MEM-4 labels only, and S-IDN-1 to S-IDN-5 labels and
    # evaluation only.
    assert UNMAPPED == {"S-RAIL-4", "S-MEM-4", *(f"S-IDN-{n}" for n in range(1, 6))}
    assert MAPPED == set(GATES) - UNMAPPED and len(MAPPED) == 80
    # The rows of features.md say so: what each reaches is not a feature.
    reaches = {cells[0]: cells[2] for cells in _rows(FEATURES.read_text()) if len(cells) == 3}
    assert reaches["S-RAIL-4"].startswith("none") and reaches["S-MEM-4"] == "labels"
    assert reaches["S-IDN-1 to S-IDN-5"].startswith("labels and evaluation only")


def test_records_of_the_loaded_world_pass(records: Records, graph: ReferenceGraph) -> None:
    identity, gates = records(1001)
    assert check_records(world_of(graph), identity, gates) == ()


def test_records_of_another_world_fail(records: Records, graph: ReferenceGraph) -> None:
    identity, gates = records(1001, changed={"vertices.Account": 1})
    assert check_records(world_of(graph), identity, gates) == (
        "the fingerprint differs from read_world's on ['vertices.Account']",
    )


def test_a_dirty_or_uncertified_build_fails(records: Records, graph: ReferenceGraph) -> None:
    identity, gates = records(1001, changed={"clean_tree": False, "code_gates_certified": False})
    assert check_records(world_of(graph), identity, gates) == (
        "PhantomLedger was not built from a clean tree",
        "PhantomLedger's code gates are not certified",
    )


def test_one_failed_mapped_gate_fails(records: Records, graph: ReferenceGraph) -> None:
    identity, gates = records(1001, passed={"S-TIM-4": False})
    assert check_records(world_of(graph), identity, gates) == (
        "the code gate S-TIM-4 failed",
        "the world check S-TIM-4 failed",
    )


def test_a_failed_world_check_reaching_no_feature_passes(
    records: Records, graph: ReferenceGraph, tmp_path: Path
) -> None:
    identity, gates = records(1001)
    # S-IDN-3's world check fails alone: it reaches labels and evaluation only.
    text = gates.read_text().replace(
        'entry = "S-IDN-3"\nkind = "world"\npass = true',
        'entry = "S-IDN-3"\nkind = "world"\npass = false',
    )
    gates.write_text(text)
    assert check_records(world_of(graph), identity, gates) == ()
    # Its code gate may not fail.
    gates.write_text(
        text.replace(
            'entry = "S-IDN-3"\nkind = "code"\npass = true',
            'entry = "S-IDN-3"\nkind = "code"\npass = false',
        )
    )
    assert check_records(world_of(graph), identity, gates) == ("the code gate S-IDN-3 failed",)


def test_a_missing_gate_or_gate_record_fails(records: Records, graph: ReferenceGraph) -> None:
    identity, gates = records(1001)
    kept = [block for block in gates.read_text().split("\n\n") if '"R-HOURS"' not in block]
    gates.write_text("\n\n".join(kept))
    assert check_records(world_of(graph), identity, gates) == ("the gate record lacks R-HOURS",)
    gates.unlink()
    assert check_records(world_of(graph), identity, gates) == ("there is no gate record",)


@pytest.mark.parametrize(
    "text",
    [
        '[[gate]]\nentry = "S-TIM-1"\nkind = "world"\npass = true\nvalue = 0.3\n',
        '[[gate]]\nentry = "S-TIM-1"\nkind = "both"\npass = true\n',
        '[[gate]]\nentry = "S-TIM-1"\nkind = "world"\npass = "yes"\n',
        '[[gate]]\nentry = "S-TIM-1"\nkind = "world"\npass = true\n'
        '[[gate]]\nentry = "S-TIM-1"\nkind = "world"\npass = false\n',
        'bound = 3\n[[gate]]\nentry = "S-TIM-1"\nkind = "world"\npass = true\n',
    ],
)
def test_a_gate_record_holding_anything_but_pass_flags_is_refused(
    text: str, tmp_path: Path
) -> None:
    path = tmp_path / "world_1001.gates.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_gates(path)
