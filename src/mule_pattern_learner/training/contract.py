"""The contract lock and the model files (docs/design.md, "Training and selection").

A model file holds what its ranker learned beside the lock it was built under: every query
version, every layout of a batch with how each kind of column is encoded and the Fourier
basis, the community, degree and ablation flags, the seed count, and the subgraph settings
(lookback, caps, time-ordered lists, community, hub and crowd thresholds). Scoring with a
model built under another lock would feed it numbers that do not mean what it learned, so
load refuses such a model and names every field that differs.

How each kind of column is encoded is read from the encoder itself, never written beside it:
the lock holds what model.encode.encode makes of a fixed probe of every kind, so changing the
amount or count transform, or the way a duration reaches the Fourier basis, changes the lock
and refuses the models built before it.

The lock is a flat mapping of dotted field names to plain values (strings, numbers, booleans
and lists of them), so that two locks compare field by field and a difference names its
field. A model file is written through a temporary file and a rename, so it appears whole or
not at all, and is read back with torch.load's weights_only, which runs no pickled code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any
import uuid

import torch

from ..config import SubgraphConfig, TrainingConfig
from ..model.encode import BatchLayout, Column, Kind, Layout, encode
from ..model.time_encoding import FREQUENCIES, SCALE, WIDTH
from ..subgraph.layouts import DEGREES, LAYOUT
from ..tigergraph.queries import VERSIONS

# The version of the model files' format, which a change to their content raises.
FORMAT = 1
# The raw rows that probe how each kind of column is encoded, each row one raw column of the
# kind (a duration's two raw columns are its presence bit and its integer milliseconds): a flag
# and a share at their ends, a count of 7, 12,345 cents, and a duration absent, of one hour and
# of 90 days.
PROBES: dict[Kind, tuple[tuple[float, ...], ...]] = {
    Kind.FLAG: ((0.0,), (1.0,)),
    Kind.COUNT: ((0.0,), (7.0,)),
    Kind.AMOUNT: ((0.0,), (12_345.0,)),
    Kind.SHARE: ((0.0,), (0.25,), (1.0,)),
    Kind.DURATION: ((0.0, 0.0), (1.0, 3_600_000.0), (1.0, 7_776_000_000.0)),
}
# The decimals an encoded probe is kept to: the last bits of a float32 sine differ between
# machines, and a model trained on one must still load on another, while any change of
# transform moves the probe by far more than this.
PROBE_DECIMALS = 2

type Lock = dict[str, Any]


def _plain(value: object) -> Any:
    """value as JSON would give it back: tuples become lists, so a lock read from a file
    equals one computed anew."""
    return json.loads(json.dumps(value))


def _layout(layout: Layout) -> list[list[str]]:
    return [[column.name, column.kind.value] for column in layout.columns]


def _relation(relation: tuple[str, str, str]) -> str:
    return ".".join(relation)


def layout_fields(layout: BatchLayout = LAYOUT) -> Lock:
    """The lock's fields of a batch layout: each node type's and each relation's columns with
    their kinds, the relations, the community block's columns and the scored node type."""
    found: Lock = {f"layout.node.{kind}": _layout(rows) for kind, rows in layout.nodes.items()}
    found["layout.relations"] = [_relation(relation) for relation in layout.relations]
    found |= {
        f"layout.edge.{_relation(relation)}": _layout(rows)
        for relation, rows in layout.edges.items()
    }
    found["layout.community"] = _layout(layout.community)
    found["layout.scored"] = layout.scored
    return found


def encoding_fields() -> Lock:
    """The lock's fields of how each kind of raw column is encoded: for every kind, its probe
    rows with what model.encode.encode makes of each, to PROBE_DECIMALS decimals, computed on
    the CPU."""
    found: Lock = {}
    for kind in Kind:
        rows = PROBES[kind]
        raw = torch.tensor(rows, dtype=torch.float64)
        encoded = encode(raw, Layout((Column(kind.value, kind),)), "cpu")
        found[f"encoding.{kind.value}"] = [
            [list(row), [round(number, PROBE_DECIMALS) for number in out.tolist()]]
            for row, out in zip(rows, encoded, strict=True)
        ]
    return found


def lock(
    ranker: str,
    *,
    community: bool,
    subgraph: SubgraphConfig,
    training: TrainingConfig,
    extra: Mapping[str, object] | None = None,
) -> Lock:
    """The lock of a ranker: its name, which says which ablation it is, whether it reads the
    community block, and the fields of this module's docstring; extra holds a ranker's own,
    such as the columns of an XGBoost vector."""
    found: Lock = {"format": FORMAT, "ranker": ranker, "community": community}
    found |= {
        f"degrees.{kind}": [relation[1] for relation in relations]
        for kind, relations in DEGREES.items()
    }
    found |= {f"versions.{name}": version for name, version in sorted(VERSIONS.items())}
    found |= layout_fields()
    found |= encoding_fields()
    found |= {
        "fourier.frequencies": FREQUENCIES.tolist(),
        "fourier.scale": SCALE,
        "fourier.width": WIDTH,
    }
    found["seeds"] = len(training.seeds)
    found |= {f"settings.{name}": value for name, value in asdict(subgraph).items()}
    found |= {f"{ranker}.{name}": value for name, value in (extra or {}).items()}
    return _plain(found)


def differences(stored: Mapping[str, object], current: Mapping[str, object]) -> list[str]:
    """The fields in which two locks differ, sorted, each with both values: a field one of
    them lacks reads as missing there."""
    found: list[str] = []
    for name in sorted(set(stored) | set(current)):
        held = stored.get(name, "missing")
        now = current.get(name, "missing")
        if held != now:
            found.append(f"{name} (the model's {_short(held)}, now {_short(now)})")
    return found


def _short(value: object, length: int = 60) -> str:
    text = json.dumps(value)
    return text if len(text) <= length else text[: length - 3] + "..."


def save(path: Path, current: Lock, content: Mapping[str, object]) -> None:
    """Write a model file at path: content under the lock, through a temporary file and a
    rename."""
    if "lock" in content or "format" in content:
        raise ValueError("A model file's content names neither its lock nor its format")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save({"format": FORMAT, "lock": dict(current), **content}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path: Path, current: Lock, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """The content of the model file at path, refused when it was built under another lock
    than current, with every field that differs named."""
    stored: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    if stored.get("format") != FORMAT:
        raise ValueError(f"{path} is not a model file of format {FORMAT}")
    found = differences(stored["lock"], current)
    if found:
        raise ValueError(
            f"{path} was built for other query versions, layouts or settings; these fields "
            f"differ: {'; '.join(found)}"
        )
    return {name: value for name, value in stored.items() if name not in ("format", "lock")}
