"""PhantomLedger's records of a world, as mule audit checks them before a result may be reported
(docs/design.md, "Gates" and "What PhantomLedger must provide").

For every world PhantomLedger writes a world record and a gate record, and the loader writes
them, after a passing verification, into the records folder .env names (MULE_PL_RECORDS_DIR):
world_<seed>.identity.toml, the world record's keys, which data.worlds matches to the loaded
world, and world_<seed>.gates.toml, the gate record. A reported result needs, for each world:

- the identity file's fingerprint equal to read_world's on the fields the two share
  (tigergraph.world.shared_fields), every one of them present;
- the identity file's clean_tree and code_gates_certified true: PhantomLedger built from a
  clean tree, with every code gate of its commit certified;
- in the gate record, every code gate passed, and every gate GATES maps to a feature present
  and passed. A world check of a gate that reaches no feature (labels or evaluation only, or
  nothing) may fail without stopping a reported result, as docs/features.md's gate tables say.

The gate record holds one row per gate, pass or fail only, never a value or a bound:

    [[gate]]
    entry = "S-TIM-1"
    kind = "world"
    pass = true

A gate that runs both as a code gate and as a world check has one row of each kind. Its pass
flags are computed from the truth, so only evaluation/ and testing/ open it (paths.pl_gate_record).

GATES names every gate of docs/features.md's gate tables by its entry id, the first column of
its row: the dossier's own ids, and the R- and G- ids that docs/features.md gives the residual
checks and the world checks the features add. PhantomLedger's gate record must name its gates by
the same ids, and a gate it lacks reads as missing, so no result is reported.
tests/evaluation/test_gate_records.py holds GATES to docs/features.md's tables, id for id.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import tomllib

from ..data.worlds import read_record, shared_record
from ..tigergraph.queries.read_world import World
from ..tigergraph.world import shared_fields

# The kinds of gate a gate record holds.
CODE = "code"
WORLD = "world"
KINDS = (CODE, WORLD)
# The identity file's keys that say how PhantomLedger was built.
CLEAN_TREE = "clean_tree"
CERTIFIED = "code_gates_certified"

# Every gate of docs/features.md's gate tables, by the entry id in the first column of its row:
# the dossier's S- ids (S-IDN-1 to S-IDN-5 each apart), then the R- ids of the residual checks and
# the G- ids of the world checks the features add, which the dossier has no ids for.
GATES: tuple[str, ...] = (
    "S-TIM-1",
    "S-TIM-2",
    "S-TIM-3",
    "S-TIM-4",
    "S-TIM-5",
    "S-TIM-6",
    "S-TIM-7",
    "S-TIM-8",
    "S-TIM-9",
    "S-TIM-10",
    "S-AMT-1",
    "S-AMT-2",
    "S-AMT-3",
    "S-AMT-4",
    "S-AMT-5",
    "S-AMT-6",
    "S-AMT-7",
    "S-AMT-8",
    "S-AMT-9",
    "S-AMT-10",
    "S-AMT-11",
    "S-RAIL-1",
    "S-RAIL-2",
    "S-RAIL-3",
    "S-RAIL-4",
    "S-RAIL-5",
    "S-RAIL-6",
    "S-RAIL-7",
    "S-RAIL-8",
    "S-RAIL-9",
    "S-RAIL-10",
    "S-ID-1",
    "S-ID-2",
    "S-ID-3",
    "S-ID-4",
    "S-ID-5",
    "S-ID-6",
    "S-ID-7",
    "S-ID-8",
    "S-ID-9",
    "S-MEM-1",
    "S-MEM-2",
    "S-MEM-3",
    "S-MEM-4",
    "S-MEM-5",
    "S-MEM-6",
    "S-MEM-7",
    "S-MEM-8",
    "S-MEM-9",
    "S-LIFE-1",
    "S-LIFE-2",
    "S-LIFE-3",
    "S-LIFE-4",
    "S-LIFE-5",
    "S-LIFE-6",
    "S-LIFE-7",
    "S-LIFE-8",
    "S-LIFE-9",
    "S-IDN-1",
    "S-IDN-2",
    "S-IDN-3",
    "S-IDN-4",
    "S-IDN-5",
    "R-BURSTS",
    "R-REPOSTS",
    "R-MOVE-SIZES",
    "R-RAIL-2",
    "R-MULTI-RING",
    "R-SHELLS",
    "R-HOURS",
    "R-PAYDAYS",
    "R-CASH-CAP",
    "R-COVER-VOLUME",
    "R-CONTROLLER-COVER",
    "R-SECOND-JOBS",
    "R-ATO-CLOCK",
    "G-TENURES",
    "G-OPENINGS",
    "G-PAYMENT-IDS",
    "G-EXTERNAL-PAIRS",
    "G-1970",
    "G-NEWNESS",
    "G-SAME-OWNER",
    "G-SHARED-LINKS",
    "G-EXTERNAL-SHARES",
    "G-ORIGIN-C",
    "G-COVER-VOLUME",
)
# The gates whose rows reach no feature: S-RAIL-4 reaches none, S-MEM-4 labels only, and
# S-IDN-1 to S-IDN-5 labels and evaluation only. A world check of one may fail without stopping a
# reported result.
UNMAPPED = frozenset({"S-RAIL-4", "S-MEM-4", *(f"S-IDN-{number}" for number in range(1, 6))})
# The gates mapped to a feature: each must be in the gate record and pass.
MAPPED = frozenset(GATES) - UNMAPPED


@dataclass(frozen=True)
class GateRow:
    """A row of a gate record: a gate's entry, its kind and whether it passed."""

    entry: str
    kind: str
    passed: bool


def read_gates(path: Path) -> tuple[GateRow, ...]:
    """The rows of a gate record, once each holds exactly an entry, a kind of KINDS and a pass
    flag, and no entry and kind come twice."""
    found = tomllib.loads(path.read_text())
    if set(found) - {"gate"}:
        raise ValueError(f"{path} holds {sorted(set(found) - {'gate'})} beside its gates")
    items: object = found.get("gate", [])
    if not isinstance(items, list):
        raise ValueError(f"{path} holds its gates as {type(items).__name__}, not a list")
    rows: list[GateRow] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"entry", "kind", "pass"}
            or not isinstance(item["entry"], str)
            or item["kind"] not in KINDS
            or not isinstance(item["pass"], bool)
        ):
            raise ValueError(f"{path} holds a gate {item!r}, not an entry, a kind and a pass flag")
        rows.append(GateRow(item["entry"], item["kind"], item["pass"]))
    keys = [(row.entry, row.kind) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} holds a gate of one kind twice")
    return tuple(rows)


def fingerprint_failures(world: World, record: Mapping[str, object], path: Path) -> list[str]:
    """Why an identity file's fingerprint is not read_world's: its shared fields missing or
    unknown, or the names of those that differ. Names only: the fields hold counts and sums."""
    expected = shared_fields(world)
    try:
        shared = shared_record(record, expected, path)
    except ValueError:
        return ["the identity file lacks fields of the fingerprint, or holds others"]
    differing = [name for name in expected if shared[name] != expected[name]]
    return [f"the fingerprint differs from read_world's on {differing}"] if differing else []


def check_records(world: World, identity_file: Path, gate_file: Path) -> tuple[str, ...]:
    """Why PhantomLedger's records of the world fail what a reported result needs, by name;
    empty when they pass. world is read_world's fields of the loaded world, identity_file the
    identity file data.worlds matched to it, and gate_file its gate record."""
    record = read_record(identity_file)
    failures = fingerprint_failures(world, record, identity_file)
    if record.get(CLEAN_TREE) is not True:
        failures.append("PhantomLedger was not built from a clean tree")
    if record.get(CERTIFIED) is not True:
        failures.append("PhantomLedger's code gates are not certified")
    if not gate_file.is_file():
        return (*failures, "there is no gate record")
    rows = read_gates(gate_file)
    failures += [
        f"the code gate {row.entry} failed" for row in rows if row.kind == CODE and not row.passed
    ]
    failures += [
        f"the world check {row.entry} failed"
        for row in rows
        if row.kind == WORLD and not row.passed and row.entry in MAPPED
    ]
    present = {row.entry for row in rows}
    failures += [f"the gate record lacks {entry}" for entry in sorted(MAPPED - present)]
    return tuple(failures)
