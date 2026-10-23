"""The contract lock and the model files (docs/design.md, "Training and selection").

A model file holds what its ranker learned beside the lock it was built under: every query
version, every layout of a batch with how each kind of column is encoded and the Fourier
basis, the community, degree and ablation flags, the seed count, and the subgraph settings
(lookback, caps, time-ordered lists, community, hub and crowd thresholds). Scoring with a
model built under another lock would feed it numbers that do not mean what it learned, so
load refuses such a model and names every field that differs.

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
from ..model.encode import BatchLayout, Kind, Layout
from ..model.time_encoding import FREQUENCIES, SCALE, WIDTH
from ..subgraph.layouts import DEGREES, LAYOUT
from ..tigergraph.queries import VERSIONS

# The version of the model files' format, which a change to their content raises.
FORMAT = 1
# How each kind of raw column is encoded (model.encode).
ENCODINGS = {
    Kind.FLAG: "as is",
    Kind.COUNT: "log(1 + n)",
    Kind.AMOUNT: "log(1 + cents / 100)",
    Kind.SHARE: "as is",
    Kind.DURATION: "presence bit and 64 Fourier numbers of log1p(seconds) / log1p(10 years)",
}

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
    found |= {f"encoding.{kind.value}": text for kind, text in ENCODINGS.items()}
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
